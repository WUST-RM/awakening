#pragma once
#include "KalmanHyLib/error_state_extended_kalman_filter.hpp"
#include "rclcpp/rclcpp.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <array>
#include <ceres/jet.h>

namespace awakening::radar_detect {
static constexpr int X_N = 6, Z_N = 3;
using VecZ = Eigen::Matrix<double, Z_N, 1>;
using VecX = Eigen::Matrix<double, X_N, 1>;
namespace idx {
    enum { CX, VCX, CY, VCY, CZ, VCZ };
    enum { YPD_Y, YPD_P, YPD_D };
} // namespace idx
struct Predict {
    template<typename T>
    void operator()(const T x0[X_N], T x1[X_N]) const {
        for (int i = 0; i < X_N; i++) {
            x1[i] = x0[i];
        }
        if (use_vel) {
            x1[idx::CX] += x0[idx::VCX] * T(dt);
            x1[idx::CY] += x0[idx::VCY] * T(dt);
            x1[idx::CZ] += x0[idx::VCZ] * T(dt);
        }
    }
    void f(const VecX& x0, VecX& x1) const {
        assert(x0.size() == X_N);
        assert(x1.size() == X_N);
        operator()(x0.data(), x1.data());
    }

    double dt;
    bool use_vel = true;
};
template<typename T>
T normalize_angle_t(T angle) {
    T two_pi = T(2.0 * M_PI);
    return angle - two_pi * floor((angle + T(M_PI)) / two_pi);
}

struct Measure {
    Measure() = default;
    template<typename T>
    void operator()(const T x[X_N], T z[Z_N]) const {
        // Compute armor position
        auto [_x, _y, _z] = h_xyz(x);
        T xy_dist = ceres::sqrt(_x * _x + _y * _y);
        T dist = ceres::sqrt(xy_dist * xy_dist + _z * _z);
        // Observation model
        z[idx::YPD_Y] = ceres::atan2(_y, _x); // yaw
        z[idx::YPD_P] = ceres::atan2(_z, xy_dist); // pitch
        z[idx::YPD_D] = dist; // distance
    }
    template<typename T>
    std::tuple<T, T, T> h_xyz(const T x[X_N]) const {
        T _x = x[idx::CX];
        T _y = x[idx::CY];
        T _z = x[idx::CZ];
        return { _x, _y, _z };
    }
    void h(const VecX& x, VecZ& z) const {
        assert(x.size() == X_N);
        assert(z.size() == Z_N);
        operator()(x.data(), z.data());
    }
};

using ESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Z_N, Predict, Measure>;
struct State {
    VecX x;
    TimePoint timestamp;
    inline void predict(const TimePoint& t) {
        auto dt = std::chrono::duration<double>(t - timestamp).count();
        predict(dt);
    }
    inline void predict(double dt) {
        Predict p {
            .dt = dt,
        };
        p.f(x, x);
        timestamp +=
            std::chrono::duration_cast<TimePoint::duration>(std::chrono::duration<double>(dt));
    }
    inline void set_pos(const Vec3& p) noexcept {
        x[idx::CX] = p.x();
        x[idx::CY] = p.y();
        x[idx::CZ] = p.z();
    }
    inline Vec3 pos() const noexcept {
        return Vec3(x[idx::CX], x[idx::CY], x[idx::CZ]);
    }
    inline void set_vel(const Vec3& v) noexcept {
        x[idx::VCX] = v.x();
        x[idx::VCY] = v.y();
        x[idx::VCZ] = v.z();
    }
    inline Vec3 vel() const noexcept {
        return Vec3(x[idx::VCX], x[idx::VCY], x[idx::VCZ]);
    }
};
struct PointTarget {
    struct PointTargetConfig {
        int esekf_iter_num = 1;
        double yp_r = 0.0015;
        double dis_r = 0.02;
        Eigen::Vector3d qxyz = { 10.0, 10.0, 10.0 };
        void load(const YAML::Node& config) {
            esekf_iter_num = config["esekf_iter_num"].as<int>();
            yp_r = config["yp_r"].as<double>();
            dis_r = config["dis_r"].as<double>();
            auto qxyz_ = config["qxyz"].as<std::vector<double>>();
            qxyz = Eigen::Vector3d(qxyz_[0], qxyz_[1], qxyz_[2]);
        }
    };
    PointTarget() {}
    PointTarget(
        const PointTargetConfig& cfg,
        const Eigen::Vector3d& init_p,
        const TimePoint& t,
        bool use_vel = true
    ) {
        target_config = cfg;
        this->use_vel = use_vel;
        Eigen::DiagonalMatrix<double, X_N> p0;
        p0.diagonal() << 1, 64, 1, 64, 1, 64;
        const auto f = Predict { .dt = 0.05, .use_vel = use_vel };
        const auto h = Measure();
        const auto u_q = [this]() {
            Eigen::Matrix<double, X_N, X_N> q;
            return q;
        };

        const auto u_r = [this](const Eigen::Matrix<double, Z_N, 1>& z) {
            Eigen::Matrix<double, Z_N, Z_N> r;
            return r;
        };
        esekf = ESEKF(f, h, u_q, u_r, p0);
        esekf->setResidualFunc([this](
                                   const Eigen::Matrix<double, Z_N, 1>& z_pred,
                                   const Eigen::Matrix<double, Z_N, 1>& z
                               ) {
            Eigen::Matrix<double, Z_N, 1> r = z - z_pred;
            r[0] = angles::shortest_angular_distance(z_pred[idx::YPD_Y],
                                                     z[idx::YPD_Y]); // yaw
            return r;
        });
        esekf->setIterationNum(target_config.esekf_iter_num);
        esekf->setInjectFunc([this](
                                 const Eigen::Matrix<double, X_N, 1>& delta,
                                 Eigen::Matrix<double, X_N, 1>& nominal
                             ) {
            for (int i = 0; i < X_N; i++) {
                nominal[i] += delta[i];
            }
        });
        state.x = Eigen::VectorXd::Zero(X_N);
        state.x << init_p.x(), 0, init_p.y(), 0, init_p.z(), 0;
        esekf->setState(state.x);
        last_update = t;
        state.timestamp = t;
        is_inited = true;
    }
    Eigen::Matrix<double, Z_N, Z_N>
    compute_measurement_covariance(const Eigen::Matrix<double, Z_N, 1>& z) const noexcept {
        Eigen::Matrix<double, Z_N, Z_N> r;

        // clang-format off
        r <<target_config.yp_r, 0, 0, 
                0, target_config.yp_r , 0, 
                0, 0, target_config.dis_r;
        // clang-format on
        return r;
    }
    Eigen::Matrix<double, X_N, X_N> compute_process_noise(double dt) const noexcept {
        Eigen::Matrix<double, X_N, X_N> q;
        Eigen::Vector3d q_xyz;

        q_xyz = target_config.qxyz;

        const double t = dt;
        const double q_x_x = pow(t, 4) / 4 * q_xyz.x(), q_x_vx = pow(t, 3) / 2 * q_xyz.x(),
                     q_vx_vx = pow(t, 2) * q_xyz.x();
        const double q_y_y = pow(t, 4) / 4 * q_xyz.y(), q_y_vy = pow(t, 3) / 2 * q_xyz.y(),
                     q_vy_vy = pow(t, 2) * q_xyz.y();
        const double q_z_z = pow(t, 4) / 4 * q_xyz.z(), q_z_vz = pow(t, 3) / 2 * q_xyz.z(),
                     q_vz_vz = pow(t, 2) * q_xyz.z();

        // clang-format off
            //      xc      v_xc    yc      v_yc    zc      v_zc    
            q <<    q_x_x,  q_x_vx, 0,      0,      0,      0,      
                    q_x_vx, q_vx_vx,0,      0,      0,      0,      
                    0,      0,      q_y_y,  q_y_vy, 0,      0,      
                    0,      0,      q_y_vy, q_vy_vy,0,      0,      
                    0,      0,      0,      0,      q_z_z,  q_z_vz, 
                    0,      0,      0,      0,      q_z_vz, q_vz_vz;
        // clang-format on
        return q;
    }
    void predict_ekf(const TimePoint& t) {
        auto dt = std::chrono::duration<double>(t - state.timestamp).count();

        esekf->setPredictFunc(Predict { .dt = dt, .use_vel = use_vel });
        const auto u_q = [dt, this]() { return compute_process_noise(dt); };
        esekf->setUpdateQ(u_q);
        state.x = esekf->predict();
        state.timestamp = t;
    }
    double last_ypd_y_;
    [[nodiscard]] Eigen::Matrix<double, Z_N, 1> getMeasure(const Eigen::Vector3d& p) noexcept {
        double ypd_y = std::atan2(p.y(), p.x());
        ypd_y = this->last_ypd_y_ + angles::shortest_angular_distance(this->last_ypd_y_, ypd_y);
        this->last_ypd_y_ = ypd_y;
        const double ypd_p = std::atan2(p.z(), std::sqrt(p.x() * p.x() + p.y() * p.y()));
        const double ypd_d = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
        return Eigen::Vector3d(ypd_y, ypd_p, ypd_d);
    }
    void update(const Eigen::Vector3d& p, const TimePoint& t) {
        const auto u_r = [this](const Eigen::Matrix<double, Z_N, 1>& z) {
            return this->compute_measurement_covariance(z);
        };
        esekf->setUpdateR(u_r);
        auto measurement = getMeasure(p);
        esekf->setMeasureFunc(Measure());
        state.x = esekf->update(measurement);
        last_update = t;
        state.timestamp = t;
    }
    void set_ekf_state(const State& s) {
        state = s;
        esekf->setState(state.x);
    }
    State state;
    TimePoint last_update;
    bool is_inited = false;
    PointTargetConfig target_config;
    bool use_vel = true;
    std::optional<ESEKF> esekf;
};

} // namespace awakening::radar_detect