#pragma once

#include <Eigen/Dense>
#include <ceres/jet.h>
#include <cmath>
#include <functional>
#include <limits>

namespace kalman_hybird_lib {
/**
  * @brief Error-State Extended Kalman Filter with covariance accessors and online NIS/NEES anomaly detection.
  *
  * @tparam N_X Dimension of the state vector.
  * @tparam N_Z Dimension of the measurement vector.
  * @tparam PredicFunc Functor type for the process model.
  * @tparam MeasureFunc Functor type for the measurement model.
  */
template<int N_X, int N_Z, class PredicFunc, class MeasureFunc>
class ErrorStateEKF {
public:
    ErrorStateEKF() = default;
    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

    using UpdateQFunc = std::function<MatrixXX()>;
    using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1&)>;
    using InjectFunc = std::function<void(const MatrixX1&, MatrixX1&)>;
    using ResidualFunc = std::function<MatrixZ1(const MatrixZ1& z_pred, const MatrixZ1& z_meas)>;

    explicit ErrorStateEKF(
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
        P_delta(P0),
        P_delta_pri(P0) {
        F.setZero();
        H.setZero();

        cal_residual = [](const MatrixZ1& z_pred, const MatrixZ1& z_meas) -> MatrixZ1 {
            return z_meas - z_pred;
        };
        inject_state = [](const MatrixX1& delta, MatrixX1& nominal) {
            for (int i = 0; i < N_X; i++) {
                nominal[i] += delta[i];
            }
        };
    }

    void setInjectFunc(const InjectFunc& inject_func) {
        inject_state = inject_func;
    }

    void setState(const MatrixX1& x0) noexcept {
        x_nominal = x0;
        delta_x.setZero();
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
        return P_delta_pri;
    }

    const MatrixXX& getPosteriorCovariance() const noexcept {
        return P_delta;
    }

    const MatrixX1& getState() const noexcept {
        return x_nominal;
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
        // Auto-diff for process model
        ceres::Jet<double, N_X> x_jet[N_X];
        for (int i = 0; i < N_X; ++i) {
            x_jet[i].a = x_nominal[i];
            x_jet[i].v.setZero();
            x_jet[i].v[i] = 1.0;
        }
        ceres::Jet<double, N_X> x_pred_jet[N_X];
        f(x_jet, x_pred_jet);

        MatrixX1 x_pri;
        for (int i = 0; i < N_X; ++i) {
            x_pri[i] = x_pred_jet[i].a;
            F.block(i, 0, 1, N_X) = x_pred_jet[i].v.transpose();
        }

        // Update error covariance prior
        Q = update_Q();
        P_delta_pri = F * P_delta * F.transpose() + Q;
        P_delta_pri = 0.5 * (P_delta_pri + P_delta_pri.transpose());

        // update buffers
        P_delta = P_delta_pri;
        x_nominal = x_pri;
        delta_x.setZero();

        return x_pri;
    }

    MatrixX1 update(const MatrixZ1& z) noexcept {
        return update(z, h);
    }
    MatrixX1 update(const MatrixZ1& z, const MeasureFunc& _h) noexcept {
        MatrixX1 delta_iter = delta_x;
        MatrixXX P_iter = P_delta;
        MatrixZZ S_last = MatrixZZ::Zero(); // store last S for NIS
        MatrixZ1 residual_last = MatrixZ1::Zero();

        double prev_res_norm = std::numeric_limits<double>::max();
        MatrixXZ K_last;

        for (int iter = 0; iter < iteration_num_; ++iter) {
            // Inject error into nominal
            MatrixX1 x_full = x_nominal;
            if (inject_state)
                inject_state(delta_iter, x_full);

            // Auto-diff for measurement model
            ceres::Jet<double, N_X> x_jet[N_X];
            for (int i = 0; i < N_X; ++i) {
                x_jet[i].a = x_full[i];
                x_jet[i].v.setZero();
                x_jet[i].v[i] = 1.0;
            }
            ceres::Jet<double, N_X> z_jet[N_Z];
            _h(x_jet, z_jet);

            MatrixZ1 z_pred;
            for (int i = 0; i < N_Z; ++i) {
                z_pred[i] = z_jet[i].a;
                H.block(i, 0, 1, N_X) = z_jet[i].v.transpose();
            }

            R = update_R(z);
            MatrixZZ S = H * P_iter * H.transpose() + R;
            S += small_noise_ * MatrixZZ::Identity();
            MatrixXZ K = P_iter * H.transpose() * S.inverse();
            K_last = K;

            MatrixZ1 residual = cal_residual(z_pred, z);

            for (int i = 0; i < N_Z; ++i) {
                if (!std::isfinite(residual[i]))
                    residual[i] = 0.0;
            }

            double alpha = 1.0;
            double cur_res_norm = residual.norm();
            if (cur_res_norm > prev_res_norm) {
                alpha = 0.5; // 阻尼
            }

            delta_iter += alpha * K * residual;

            double old_res_norm = prev_res_norm;
            prev_res_norm = cur_res_norm;

            last_residual_ = residual;
            S_last = S;
            residual_last = residual;

            if (cur_res_norm < 1e-4 || std::abs(old_res_norm - cur_res_norm) < 1e-6) {
                break;
            }
        }

        if (inject_state)
            inject_state(delta_iter, x_nominal);
        delta_x.setZero();

        P_delta = (MatrixXX::Identity() - K_last * H) * P_iter
                * (MatrixXX::Identity() - K_last * H).transpose()
            + K_last * R * K_last.transpose();
        P_delta = 0.5 * (P_delta + P_delta.transpose());

        return x_nominal;
    }

private:
    PredicFunc f;
    MeasureFunc h;
    UpdateQFunc update_Q;
    UpdateRFunc update_R;
    InjectFunc inject_state;
    ResidualFunc cal_residual;

    MatrixXX F = MatrixXX::Zero();
    MatrixZX H = MatrixZX::Zero();
    MatrixXX Q = MatrixXX::Zero();
    MatrixZZ R = MatrixZZ::Zero();

    MatrixX1 x_nominal = MatrixX1::Zero();
    MatrixX1 delta_x = MatrixX1::Zero();
    MatrixXX P_delta = MatrixXX::Identity();
    MatrixXX P_delta_pri = MatrixXX::Identity();

    MatrixZ1 last_residual_ = MatrixZ1::Zero();

    int iteration_num_ = 1;
    double small_noise_ = 1e-6;
};

} // namespace kalman_hybird_lib
