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
#include <ceres/jet.h>
#include <functional>

namespace kalmanLib {

template<int N_X, int N_Z, class PredicFunc, class MeasureFunc>
class AdaptiveExtendedKalmanFilter {
public:
    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

    /// @brief Function type that returns the prior process noise covariance Q.
    using UpdateQFunc = std::function<MatrixXX()>;
    /// @brief Function type that returns the measurement noise covariance R given a measurement.
    using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1&)>;

    /// @brief Residual function type: residual = f(z_pred, z_meas)
    using ResidualFunc = std::function<MatrixZ1(const MatrixZ1& z_pred, const MatrixZ1& z_meas)>;

    AdaptiveExtendedKalmanFilter() = default;

    explicit AdaptiveExtendedKalmanFilter(
        const PredicFunc& f,
        const MeasureFunc& h,
        const UpdateQFunc& u_q,
        const UpdateRFunc& u_r,
        const MatrixXX& P0
    ) noexcept:
        f(f),
        h(h),
        update_Q(u_q),
        update_R(u_r),
        P_post(P0) {
        F.setZero();
        H.setZero();
        // default residual: simple difference z_meas - z_pred
        residual_func_ = [](const MatrixZ1& z_pred, const MatrixZ1& z_meas) -> MatrixZ1 {
            return z_meas - z_pred;
        };
    }

    void setState(const MatrixX1& x0) noexcept {
        x_post = x0;
    }

    void setPredictFunc(const PredicFunc& f) noexcept {
        this->f = f;
    }

    void setMeasureFunc(const MeasureFunc& h) noexcept {
        this->h = h;
    }

    void setIterationNum(int num) {
        iteration_num_ = num;
    }

    void setSmallnoise(double n) {
        small_noise_ = n;
    }

    void enableAdaptiveQ(bool enable) {
        adaptive_Q_enabled = enable;
    }

    void enableAdaptiveR(bool enable) {
        adaptive_R_enabled = enable;
    }

    void setResidualAlpha(double a) {
        alpha = std::clamp(a, 0.0, 1.0);
    }

    void setAdaptiveQRatio(double beta) {
        beta_Q = std::clamp(beta, 0.0, 1.0);
    }

    void setAdaptiveRRatio(double beta) {
        beta_R = std::clamp(beta, 0.0, 1.0);
    }

    void setResidualFunc(const ResidualFunc& func) {
        residual_func_ = func;
    }

    MatrixX1 predict() noexcept {
        // Auto-differentiate process model to compute F jacobian
        ceres::Jet<double, N_X> x_e_jet[N_X];
        for (int i = 0; i < N_X; ++i) {
            x_e_jet[i].a = x_post[i];
            x_e_jet[i].v.setZero();
            x_e_jet[i].v[i] = 1.0;
        }

        ceres::Jet<double, N_X> x_p_jet[N_X];
        f(x_e_jet, x_p_jet);

        for (int i = 0; i < N_X; ++i) {
            x_pri[i] = std::isfinite(x_p_jet[i].a) ? x_p_jet[i].a : 0.0;
            F.block(i, 0, 1, N_X) = x_p_jet[i].v.transpose();
        }

        // Adaptive process noise Q
        if (adaptive_Q_enabled) {
            process_noise_est_ = x_pri - x_post;
            process_noise_est_ =
                process_noise_est_.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.0; });
            MatrixXX Q_adapt = process_noise_est_ * process_noise_est_.transpose();
            MatrixXX Q_prior = update_Q();
            Q = beta_Q * Q_adapt + (1.0 - beta_Q) * Q_prior;
            Q += small_noise_ * MatrixXX::Identity();
        } else {
            Q = update_Q();
        }

        // Covariance propagation
        P_pri = F * P_post * F.transpose() + Q;
        P_pri = 0.5 * (P_pri + P_pri.transpose());

        x_post = x_pri;
        return x_pri;
    }

    MatrixX1 update(const MatrixZ1& z) noexcept {
        MatrixX1 x_iter = x_post;
        MatrixXZ K_local; // 保存最后一次增益
        double prev_res_norm = std::numeric_limits<double>::max();

        for (int iter = 0; iter < iteration_num_; ++iter) {
            // Auto-differentiate measurement model to compute H jacobian
            ceres::Jet<double, N_X> x_p_jet[N_X];
            for (int i = 0; i < N_X; ++i) {
                x_p_jet[i].a = x_iter[i];
                x_p_jet[i].v.setZero();
                x_p_jet[i].v[i] = 1.0;
            }

            ceres::Jet<double, N_X> z_p_jet[N_Z];
            h(x_p_jet, z_p_jet);

            MatrixZ1 z_pri;
            for (int i = 0; i < N_Z; ++i) {
                z_pri[i] = std::isfinite(z_p_jet[i].a) ? z_p_jet[i].a : 0.0;
                H.block(i, 0, 1, N_X) = z_p_jet[i].v.transpose();
            }

            // Compute residual using user-supplied residual_func_ (default: z - z_pri)
            MatrixZ1 residual = residual_func_(z_pri, z);

            // Clamp and sanitize residual
            for (int i = 0; i < N_Z; ++i) {
                if (!std::isfinite(residual[i]))
                    residual[i] = 0.0;
            }
            last_residual_ = residual;

            // Adaptive measurement noise R
            if (adaptive_R_enabled) {
                MatrixZZ R_adapt = residual * residual.transpose();
                MatrixZZ R_prior = update_R(z);
                R = beta_R * R_adapt + (1.0 - beta_R) * R_prior;
            } else {
                R = update_R(z);
            }
            R += small_noise_ * MatrixZZ::Identity();

            // Kalman gain and state update
            MatrixZZ S = H * P_pri * H.transpose() + R;
            S += small_noise_ * MatrixZZ::Identity();
            MatrixXZ K = P_pri * H.transpose() * S.inverse();
            K_local = K; // 保存最后一次增益W
            double alpha = 1.0;
            double cur_res_norm = residual.norm();
            if (cur_res_norm > prev_res_norm) {
                alpha = 0.5; // 阻尼
            }

            // Update estimate
            MatrixX1 x_new = x_iter + alpha * K * residual;
            // sanitize x_new
            for (int i = 0; i < N_X; ++i) {
                if (!std::isfinite(x_new[i]))
                    x_new[i] = x_iter[i];
            }

            if (cur_res_norm < 1e-4 || std::abs(prev_res_norm - cur_res_norm) < 1e-6)
                break;

            x_iter = x_new;
            prev_res_norm = cur_res_norm;
        }

        // Finalize post-update state
        x_post = x_iter;
        for (int i = 0; i < N_X; ++i) {
            if (!std::isfinite(x_post[i]))
                x_post[i] = 0.0;
        }

        // Updated covariance using Joseph form
        P_post = (MatrixXX::Identity() - K_local * H) * P_pri
                * (MatrixXX::Identity() - K_local * H).transpose()
            + K_local * R * K_local.transpose();
        P_post = 0.5 * (P_post + P_post.transpose());

        return x_post;
    }

private:
    PredicFunc f; ///< Process model functor
    MeasureFunc h; ///< Measurement model functor
    UpdateQFunc update_Q; ///< Function to obtain Q_prior
    UpdateRFunc update_R; ///< Function to obtain R_prior(z)

    MatrixXX F = MatrixXX::Zero(); ///< State transition jacobian
    MatrixZX H = MatrixZX::Zero(); ///< Measurement jacobian
    MatrixXX Q = MatrixXX::Zero(); ///< Process noise covariance
    MatrixZZ R = MatrixZZ::Zero(); ///< Measurement noise covariance
    MatrixXX P_pri = MatrixXX::Identity(); ///< Prior covariance
    MatrixXX P_post = MatrixXX::Identity(); ///< Posterior covariance
    MatrixXZ K = MatrixXZ::Zero(); ///< Kalman gain
    MatrixX1 x_pri = MatrixX1::Zero(); ///< Prior state
    MatrixX1 x_post = MatrixX1::Zero(); ///< Posterior state

    MatrixZ1 last_residual_ = MatrixZ1::Zero(); ///< Last measurement residual
    MatrixX1 process_noise_est_ = MatrixX1::Zero(); ///< Estimated process noise

    // angle_dims_ removed per your request

    int iteration_num_ = 1; ///< Number of update iterations
    bool adaptive_Q_enabled = false; ///< Enable adaptive Q
    bool adaptive_R_enabled = false; ///< Enable adaptive R
    double alpha = 0.5; ///< Residual blending factor for R
    double beta_Q = 1.0; ///< Blending factor for Q adaptation
    double beta_R = 1.0; ///< Blending factor for R adaptation
    double small_noise_ = 1e-6; ///< Small noise floor for stability

    // Residual function (default = z_meas - z_pred)
    ResidualFunc residual_func_;
};

} // namespace kalmanLib
