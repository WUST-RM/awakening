#pragma once
#include "KalmanHyLib/kalman_hybird_lib.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "tasks/base/web.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>
#include <yaml-cpp/node/node.h>
namespace awakening {
namespace wheel_motion_model {
    constexpr int X_N = 6;
    constexpr int Z_N = 3;

    using VecX = Eigen::Matrix<double, X_N, 1>;
    using VecZ = Eigen::Matrix<double, Z_N, 1>;
    namespace idx {
        enum {
            X,
            VX,
            Y,
            VY,
            Z,
            VZ,
        };
        enum { ZVX, ZVY, ZVZ };

    } // namespace idx
    struct Predict {
        double dt { 0.0 };

        template<typename T>
        void operator()(const T x0[X_N], T x1[X_N]) const {
            std::copy(x0, x0 + X_N, x1);

            x1[idx::X] += x0[idx::VX] * T(dt);
            x1[idx::Y] += x0[idx::VY] * T(dt);
            x1[idx::Z] += x0[idx::VZ] * T(dt);

            clamp(x1);
        }

        template<typename T>
        void clamp(T x[X_N]) const {}
        void f(const VecX& x0, VecX& x1) const {
            assert(x0.size() == X_N);
            assert(x1.size() == X_N);
            operator()(x0.data(), x1.data());
        }
    };
    struct Measure {
        template<typename T>
        void operator()(const T x[X_N], T z[Z_N]) const {
            z[idx::ZVX] = x[idx::VX];
            z[idx::ZVY] = x[idx::VY];
            z[idx::ZVZ] = x[idx::VZ];
        }
    };
    using RobotStateESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Z_N, Predict, Measure>;
    struct State {
        VecX x;
        TimePoint timestamp;

        void predict(const TimePoint& t) {
            auto dt = std::chrono::duration<double>(t - timestamp).count();
            predict(dt);
        }
        void predict(double dt) {
            Predict p {
                .dt = dt,
            };
            p.f(x, x);
            timestamp +=
                std::chrono::duration_cast<TimePoint::duration>(std::chrono::duration<double>(dt));
        }

        Vec3 pos() const noexcept {
            return Vec3(x[idx::X], x[idx::Y], x[idx::Z]);
        }

        Vec3 vel() const noexcept {
            return Vec3(x[idx::VX], x[idx::VY], x[idx::VZ]);
        }
    };
} // namespace wheel_motion_model
class WheelOdometry {
public:
    struct Params {
        Vec3 q_xyz;
        Vec3 r_vxyz;
        void load(const YAML::Node& config) {
            auto q_xyz_vec = config["q_xyz"].as<std::vector<double>>();
            q_xyz = Vec3(q_xyz_vec[0], q_xyz_vec[1], q_xyz_vec[2]);

            auto r_vxyz_vec = config["r_vxyz"].as<std::vector<double>>();
            r_vxyz = Vec3(r_vxyz_vec[0], r_vxyz_vec[1], r_vxyz_vec[2]);
        }
    } params_;
    WheelOdometry(const YAML::Node& config, const TimePoint& t) {
        params_.load(config);
        Eigen::DiagonalMatrix<double, wheel_motion_model::X_N> p0;
        p0.diagonal() << 1, 64, 1, 64, 1, 64;
        const auto u_q = [this]() {
            Eigen::Matrix<double, wheel_motion_model::X_N, wheel_motion_model::X_N> q;
            return q;
        };
        const auto u_r = [this](const Eigen::Matrix<double, wheel_motion_model::Z_N, 1>& z) {
            Eigen::Matrix<double, wheel_motion_model::Z_N, wheel_motion_model::Z_N> r;
            return r;
        };
        esekf = wheel_motion_model::RobotStateESEKF(
            wheel_motion_model::Predict {
                .dt = 0.005,
            },
            wheel_motion_model::Measure {},
            u_q,
            u_r,
            p0
        );
        esekf.value().setResidualFunc(
            [this](
                const Eigen::Matrix<double, wheel_motion_model::Z_N, 1>& z_pred,
                const Eigen::Matrix<double, wheel_motion_model::Z_N, 1>& z
            ) {
                Eigen::Matrix<double, wheel_motion_model::Z_N, 1> r = z - z_pred;
                return r;
            }
        );
        esekf.value().setIterationNum(1);
        esekf.value().setInjectFunc(
            [this](
                const Eigen::Matrix<double, wheel_motion_model::X_N, 1>& delta,
                Eigen::Matrix<double, wheel_motion_model::X_N, 1>& nominal
            ) {
                for (int i = 0; i < wheel_motion_model::X_N; i++) {
                    nominal[i] += delta[i];
                }
            }
        );
        state.x = Eigen::VectorXd::Zero(wheel_motion_model::X_N);
        state.x << 0, 0, 0, 0, 0, 0;
        esekf.value().setState(state.x);
        is_inited = true;
        last_update = t;
    }
    Eigen::Matrix<double, wheel_motion_model::X_N, wheel_motion_model::X_N> process_noise(double dt
    ) const noexcept {
        Eigen::Matrix<double, wheel_motion_model::X_N, wheel_motion_model::X_N> q;
        Vec3 q_xyz = params_.q_xyz;

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
    Eigen::Matrix<double, wheel_motion_model::Z_N, wheel_motion_model::Z_N>
    measurement_covariance(const Eigen::Matrix<double, wheel_motion_model::Z_N, 1>& z
    ) const noexcept {
        Eigen::Matrix<double, wheel_motion_model::Z_N, wheel_motion_model::Z_N> r;

        Vec3 r_vxyz = params_.r_vxyz;
        // clang-format off
        r <<r_vxyz.x(), 0 ,0,
            0, r_vxyz.y(), 0,
            0, 0, r_vxyz.z();
        // clang-format on

        return r;
    }
    void predict_ekf(const TimePoint& timestamp) {
        if (!esekf) {
            throw std::runtime_error("ESEKF is not initialized");
        }
        auto dt = std::chrono::duration<double>(timestamp - state.timestamp).count();
        esekf.value().setPredictFunc(wheel_motion_model::Predict { .dt = dt });
        esekf.value().setUpdateQ([&]() { return process_noise(dt); });
        state.x = esekf.value().predict();
        state.timestamp = timestamp;
    }
    void update(const Vec3& v, const TimePoint& t) {
        if (!esekf) {
            throw std::runtime_error("ESEKF is not initialized");
        }
        esekf.value().setUpdateR([&](const Eigen::Matrix<double, wheel_motion_model::Z_N, 1>& z) {
            return measurement_covariance(z);
        });
        wheel_motion_model::Measure measure;
        wheel_motion_model::VecZ z;
        z<<v.x(),v.y(),v.z();
        esekf.value().setMeasureFunc(measure);
        state.x = esekf.value().update(z);
        last_update = t;
        state.timestamp = t;
    }
    void write_log() {
        web::write_log("wheel_odometry", [&](auto& j) {
            j["timestamp"] = static_cast<int>(
                std::chrono::duration<double>(last_update.time_since_epoch()).count()
            );
            
           
            j["x"] = web::val(state.pos().x());
            j["y"] = web::val(state.pos().y());
            j["z"] = web::val(state.pos().z());
            j["vx"] = web::val(state.vel().x());
            j["vy"] = web::val(state.vel().y());
            j["vz"] = web::val(state.vel().z());

        });
    }
    TimePoint last_update;
    bool is_inited = false;
    wheel_motion_model::State state;
    std::optional<wheel_motion_model::RobotStateESEKF> esekf;
};
} // namespace awakening