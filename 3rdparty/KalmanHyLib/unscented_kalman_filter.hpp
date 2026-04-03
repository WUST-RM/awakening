// Copyright 2025 Xiaojian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>

namespace kalman_hybird_lib {

template<int N_X, int N_Z, class PredictFunc, class MeasureFunc>
class UnscentedKalmanFilter {
public:
    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

    using UpdateQFunc = std::function<MatrixXX()>;
    using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1&)>;

    /// Residual function type: user provides residual = f(z_pred, z_meas)
    using ResidualFunc = std::function<MatrixZ1(const MatrixZ1& z_pred, const MatrixZ1& z_meas)>;

    explicit UnscentedKalmanFilter(
        const PredictFunc& f,
        const MeasureFunc& h,
        const UpdateQFunc& u_q,
        const UpdateRFunc& u_r,
        const MatrixXX& P0,
        double alpha = 1e-3,
        double beta = 2.0,
        double kappa = 0.0
    ) noexcept:
        f(f),
        h(h),
        update_Q(u_q),
        update_R(u_r),
        P_post(P0) {
        lambda = alpha * alpha * (N_X + kappa) - N_X;
        gamma = std::sqrt(N_X + lambda);

        weights_mean[0] = lambda / (N_X + lambda);
        weights_cov[0] = weights_mean[0] + (1 - alpha * alpha + beta);
        for (int i = 1; i < 2 * N_X + 1; ++i) {
            weights_mean[i] = weights_cov[i] = 1.0 / (2 * (N_X + lambda));
        }

        Xsig_pred.setZero();

        // default residual: simple subtraction (z_meas - z_pred)
        residual_func_ = [](const MatrixZ1& z_pred, const MatrixZ1& z_meas) -> MatrixZ1 {
            return z_meas - z_pred;
        };
    }

    void setState(const MatrixX1& x0) noexcept {
        x_post = x0;
    }

    void setResidualFunc(const ResidualFunc& func) {
        residual_func_ = func;
    }

    void setIterationNum(int num) {
        iteration_num_ = std::max(1, num);
    }

    const MatrixXX& getPriorCovariance() const noexcept {
        return P_pri;
    }

    const MatrixXX& getPosteriorCovariance() const noexcept {
        return P_post;
    }

    MatrixX1 predict() noexcept {
        Q = update_Q();
        generateSigmaPoints(x_post, P_post, Xsig);

        for (int i = 0; i < 2 * N_X + 1; ++i)
            Xsig_pred.col(i) = f(Xsig.col(i));

        x_pri.setZero();
        for (int i = 0; i < 2 * N_X + 1; ++i)
            x_pri += weights_mean[i] * Xsig_pred.col(i);

        P_pri.setZero();
        for (int i = 0; i < 2 * N_X + 1; ++i) {
            auto dx = Xsig_pred.col(i) - x_pri;
            P_pri += weights_cov[i] * dx * dx.transpose();
        }
        P_pri += Q;

        x_post = x_pri;
        return x_pri;
    }
    MatrixX1 update(const MatrixZ1& z) noexcept {
        R = update_R(z);

        // Initialize iterative update state with prior mean
        MatrixX1 x_iter = x_pri;
        double prev_res_norm = std::numeric_limits<double>::max();

        for (int iter = 0; iter < iteration_num_; ++iter) {
            // Generate sigma points around current estimate
            generateSigmaPoints(x_iter, P_pri, Xsig);

            // Predict measurement sigma points
            Eigen::Matrix<double, N_Z, 2 * N_X + 1> Zsig;
            for (int i = 0; i < 2 * N_X + 1; ++i)
                Zsig.col(i) = h(Xsig.col(i));

            // Predicted measurement mean
            MatrixZ1 z_pred = MatrixZ1::Zero();
            for (int i = 0; i < 2 * N_X + 1; ++i)
                z_pred += weights_mean[i] * Zsig.col(i);

            // Innovation covariance S
            MatrixZZ S = MatrixZZ::Zero();
            for (int i = 0; i < 2 * N_X + 1; ++i) {
                MatrixZ1 dz = residual_func_(z_pred, Zsig.col(i));
                S += weights_cov[i] * dz * dz.transpose();
            }
            S += R;

            // Cross covariance Tc
            MatrixXZ Tc = MatrixXZ::Zero();
            for (int i = 0; i < 2 * N_X + 1; ++i) {
                MatrixX1 dx = Xsig.col(i) - x_iter;
                MatrixZ1 dz = residual_func_(z_pred, Zsig.col(i));
                Tc += weights_cov[i] * dx * dz.transpose();
            }

            // Kalman gain
            MatrixXZ K_iter = Tc * S.inverse();

            // Residual with damping / clamping
            MatrixZ1 residual = residual_func_(z_pred, z);
            for (int i = 0; i < N_Z; ++i) {
                if (!std::isfinite(residual[i]))
                    residual[i] = 0.0;
            }

            double alpha = 1.0;
            double cur_res_norm = residual.norm();
            if (cur_res_norm > prev_res_norm)
                alpha = 0.5; // simple damping

            // Update state
            MatrixX1 x_new = x_iter + alpha * K_iter * residual;
            for (int i = 0; i < N_X; ++i) {
                if (!std::isfinite(x_new[i]))
                    x_new[i] = x_iter[i];
            }

            if (cur_res_norm < 1e-4 || std::abs(prev_res_norm - cur_res_norm) < 1e-6)
                break;

            x_iter = x_new;
            prev_res_norm = cur_res_norm;
        }

        x_post = x_iter;
        for (int i = 0; i < N_X; ++i)
            if (!std::isfinite(x_post[i]))
                x_post[i] = 0.0;

        // Recompute final Kalman gain and covariance
        generateSigmaPoints(x_post, P_pri, Xsig);
        Eigen::Matrix<double, N_Z, 2 * N_X + 1> Zsig_final;
        for (int i = 0; i < 2 * N_X + 1; ++i)
            Zsig_final.col(i) = h(Xsig.col(i));

        MatrixZ1 z_pred_final = MatrixZ1::Zero();
        for (int i = 0; i < 2 * N_X + 1; ++i)
            z_pred_final += weights_mean[i] * Zsig_final.col(i);

        MatrixZZ S_final = MatrixZZ::Zero();
        for (int i = 0; i < 2 * N_X + 1; ++i) {
            MatrixZ1 dz = residual_func_(z_pred_final, Zsig_final.col(i));
            S_final += weights_cov[i] * dz * dz.transpose();
        }
        S_final += R;

        MatrixXZ Tc_final = MatrixXZ::Zero();
        for (int i = 0; i < 2 * N_X + 1; ++i) {
            MatrixX1 dx = Xsig.col(i) - x_post;
            MatrixZ1 dz = residual_func_(z_pred_final, Zsig_final.col(i));
            Tc_final += weights_cov[i] * dx * dz.transpose();
        }

        K = Tc_final * S_final.inverse();

        // Joseph form covariance update
        P_post = (MatrixXX::Identity() - K * S_final * K.transpose());
        P_post = 0.5 * (P_post + P_post.transpose());

        return x_post;
    }

private:
    PredictFunc f; ///< Process model function
    MeasureFunc h; ///< Measurement model function
    UpdateQFunc update_Q; ///< Process noise covariance updater
    UpdateRFunc update_R; ///< Measurement noise covariance updater

    double lambda = 0; ///< UKF scaling parameter lambda
    double gamma = 0; ///< Square root of (N_X + lambda)
    std::array<double, 2 * N_X + 1> weights_mean {}; ///< Sigma point weights for mean
    std::array<double, 2 * N_X + 1> weights_cov {}; ///< Sigma point weights for covariance

    Eigen::Matrix<double, N_X, 2 * N_X + 1> Xsig; ///< Sigma points matrix
    Eigen::Matrix<double, N_X, 2 * N_X + 1> Xsig_pred; ///< Predicted sigma points matrix

    MatrixXX Q = MatrixXX::Zero(); ///< Process noise covariance
    MatrixXX P_pri = MatrixXX::Identity(); ///< Prior covariance
    MatrixXX P_post = MatrixXX::Identity(); ///< Posterior covariance

    MatrixZZ R = MatrixZZ::Zero(); ///< Measurement noise covariance
    MatrixXZ K = MatrixXZ::Zero(); ///< Kalman gain

    MatrixX1 x_pri = MatrixX1::Zero(); ///< Predicted (prior) state
    MatrixX1 x_post = MatrixX1::Zero(); ///< Updated (posterior) state

    // angle_dims_ removed per request; use residual_func_ to handle wrapping if needed

    int iteration_num_ = 1; ///< Number of iterations during update (>=1)

    /// Residual function used to compute measurement differences (default = z_meas - z_pred)
    ResidualFunc residual_func_;

    void generateSigmaPoints(
        const MatrixX1& x,
        const MatrixXX& P,
        Eigen::Matrix<double, N_X, 2 * N_X + 1>& Xsig_out
    ) {
        Eigen::Matrix<double, N_X, N_X> A = P.llt().matrixL();
        Xsig_out.col(0) = x;
        for (int i = 0; i < N_X; ++i) {
            Xsig_out.col(i + 1) = x + gamma * A.col(i);
            Xsig_out.col(i + 1 + N_X) = x - gamma * A.col(i);
        }
    }
};

} // namespace kalman_hybird_lib
