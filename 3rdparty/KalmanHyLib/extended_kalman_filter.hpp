#pragma once

#include <Eigen/Dense>
#include <ceres/jet.h>
#include <cmath>
#include <functional>
#include <limits>

namespace kalman_hybird_lib {

template<int N_X, int N_Z, class PredicFunc, class MeasureFunc>
class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter() = default;

    // Alias for square matrices and vectors of appropriate dimensions
    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

    // Functor types for updating process noise Q and measurement noise R
    using UpdateQFunc = std::function<MatrixXX()>;
    using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1& z)>;

    // Residual function type: user provides residual = f(z_pred, z_meas)
    using ResidualFunc = std::function<MatrixZ1(const MatrixZ1& z_pred, const MatrixZ1& z_meas)>;

    explicit ExtendedKalmanFilter(
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
        F.setZero(); // Initialize process Jacobian
        H.setZero(); // Initialize measurement Jacobian

        // default residual: simple difference z_meas - z_pred
        cal_residual = [](const MatrixZ1& z_pred, const MatrixZ1& z_meas) -> MatrixZ1 {
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
    void setResidualFunc(const ResidualFunc& func) {
        cal_residual = func;
    }

    const MatrixXX& getPriorCovariance() const noexcept {
        return P_pri;
    }

    const MatrixXX& getPosteriorCovariance() const noexcept {
        return P_post;
    }

    double getResidualNorm() const noexcept {
        return last_residual_.norm();
    }

    void setUpdateQ(const UpdateQFunc& u_q) {
        this->update_Q = u_q;
    }

    void setUpdateR(const UpdateRFunc& u_r) {
        this->update_R = u_r;
    }

    MatrixX1 predict() noexcept {
        // Convert state to Ceres Jet for auto-diff
        ceres::Jet<double, N_X> x_e_jet[N_X];
        for (int i = 0; i < N_X; ++i) {
            x_e_jet[i].a = x_post[i]; // value
            x_e_jet[i].v.setZero(); // derivative vector
            x_e_jet[i].v[i] = 1.0; // set seed for Jacobian
        }

        // Evaluate process model
        ceres::Jet<double, N_X> x_p_jet[N_X];
        f(x_e_jet, x_p_jet);

        // Extract predicted state and Jacobian F
        for (int i = 0; i < N_X; ++i) {
            x_pri[i] = std::isfinite(x_p_jet[i].a) ? x_p_jet[i].a : 0.0;
            F.block(i, 0, 1, N_X) = x_p_jet[i].v.transpose();
        }

        // Compute process noise
        Q = update_Q();

        // Predict covariance: P_prior = F * P_post * F^T + Q
        P_pri = F * P_post * F.transpose() + Q;
        // Symmetrize
        P_pri = 0.5 * (P_pri + P_pri.transpose());

        // Update posterior state for next cycle
        x_post = x_pri;
        return x_pri;
    }

    MatrixX1 update(const MatrixZ1& z) noexcept {
        // Start from the prior/posterior state
        MatrixX1 x_iter = x_post;
        MatrixZ1 z_pri; // predicted measurement from last iter
        MatrixZZ S; // innovation covariance from last iter
        MatrixXZ K_local; // Kalman gain from last iter

        double prev_res_norm = std::numeric_limits<double>::max();

        // Optional iterative refinement
        for (int iter = 0; iter < iteration_num_; ++iter) {
            // Build Ceres Jet for current guess x_iter
            ceres::Jet<double, N_X> x_p_jet[N_X];
            for (int i = 0; i < N_X; ++i) {
                x_p_jet[i].a = x_iter[i];
                x_p_jet[i].v.setZero();
                x_p_jet[i].v[i] = 1.0;
            }

            // Evaluate measurement model
            ceres::Jet<double, N_X> z_p_jet[N_Z];
            h(x_p_jet, z_p_jet);

            // Extract predicted measurement and Jacobian H
            for (int i = 0; i < N_Z; ++i) {
                z_pri[i] = std::isfinite(z_p_jet[i].a) ? z_p_jet[i].a : 0.0;
                H.block(i, 0, 1, N_X) = z_p_jet[i].v.transpose();
            }

            // Compute measurement noise
            R = update_R(z);

            // Innovation covariance S = H P_prior H^T + R
            S = H * P_pri * H.transpose() + R;
            S += small_noise_ * MatrixZZ::Identity();

            // Compute Kalman gain K = P_prior H^T S^{-1}
            MatrixXZ K = P_pri * H.transpose() * S.inverse();
            K_local = K; // save last K

            // Compute residual using user-supplied cal_residual (default: z - z_pri)
            MatrixZ1 residual = cal_residual(z_pri, z);

            // Clamp and sanitize residual
            for (int i = 0; i < N_Z; ++i) {
                if (!std::isfinite(residual[i]))
                    residual[i] = 0.0;
            }

            double alpha = 1.0;
            double cur_res_norm = residual.norm();
            if (cur_res_norm > prev_res_norm) {
                alpha = 0.5; // 阻尼
            }

            // Update estimate
            MatrixX1 x_new = x_iter + alpha * K * residual;

            // Ensure finite
            for (int i = 0; i < N_X; ++i) {
                if (!std::isfinite(x_new[i]))
                    x_new[i] = x_iter[i];
            }
            if (cur_res_norm < 1e-4 || std::abs(prev_res_norm - cur_res_norm) < 1e-6)
                break;

            x_iter = x_new;
            prev_res_norm = cur_res_norm;
            last_residual_ = residual;
        }

        // Final posterior state
        x_post = x_iter;
        for (int i = 0; i < N_X; ++i) {
            if (!std::isfinite(x_post[i]))
                x_post[i] = 0.0;
        }

        P_post = (MatrixXX::Identity() - K_local * H) * P_pri
                * (MatrixXX::Identity() - K_local * H).transpose()
            + K_local * R * K_local.transpose();
        P_post = 0.5 * (P_post + P_post.transpose()); // Symmetrize

        return x_post;
    }

private:
    PredicFunc f; // Process model function
    MeasureFunc h; // Measurement model function
    UpdateQFunc update_Q; // Function to update process noise covariance
    UpdateRFunc update_R; // Function to update measurement noise covariance
    ResidualFunc cal_residual;
    MatrixXX F = MatrixXX::Zero(); // Process Jacobian
    MatrixZX H = MatrixZX::Zero(); // Measurement Jacobian
    MatrixXX Q = MatrixXX::Zero(); // Process noise covariance
    MatrixZZ R = MatrixZZ::Zero(); // Measurement noise covariance

    MatrixXX P_pri = MatrixXX::Identity(); // Prior covariance
    MatrixXX P_post = MatrixXX::Identity(); // Posterior covariance
    MatrixXZ K = MatrixXZ::Zero(); // Kalman gain

    MatrixX1 x_pri = MatrixX1::Zero(); // Prior state
    MatrixX1 x_post = MatrixX1::Zero(); // Posterior state

    MatrixZ1 last_residual_ = MatrixZ1::Zero();

    int iteration_num_ = 1; // GN iterations in update
    double small_noise_ = 1e-6;

    // residual function (default = z_meas - z_pred)
};

} // namespace kalman_hybird_lib
