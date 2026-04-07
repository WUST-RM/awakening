#pragma once
#include "KalmanHyLib/kalman_hybird_lib.hpp"
#include "tasks/auto_aim/type.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <algorithm>
#include <ceres/ceres.h>
#include <chrono>
#include <optional>
#include <vector>
namespace awakening::armor_motion_model {

enum class MotionModel { CONSTANT_VELOCITY, CONSTANT_ROTATION, CONSTANT_VEL_ROT };

constexpr int X_N = 11;
constexpr int Z_N = 4;

using VecX = Eigen::Matrix<double, X_N, 1>;
using VecZ = Eigen::Matrix<double, Z_N, 1>;

namespace idx {
    enum { CX, VCX, CY, VCY, CZ, VCZ, YAW, VYAW, R, P1, P2 };
    constexpr int L = P1;
    constexpr int H = P2;
    constexpr int OUTPOST01DZ = P1;
    constexpr int OUTPOST02DZ = P2;
    enum { YPD_Y, YPD_P, YPD_D, ORI_YAW };
} // namespace idx

template<typename T>
inline T normalize_angle(T a) {
    const T two_pi = T(2.0 * M_PI);
    return a - two_pi * floor((a + T(M_PI)) / two_pi);
}

struct Predict {
    double dt { 0.0 };
    MotionModel model { MotionModel::CONSTANT_VEL_ROT };

    template<typename T>
    void operator()(const T x0[X_N], T x1[X_N]) const {
        std::copy(x0, x0 + X_N, x1);

        if (model != MotionModel::CONSTANT_ROTATION) {
            x1[idx::CX] += x0[idx::VCX] * T(dt);
            x1[idx::CY] += x0[idx::VCY] * T(dt);
            x1[idx::CZ] += x0[idx::VCZ] * T(dt);
        } else {
            x1[idx::VCX] = x1[idx::VCY] = x1[idx::VCZ] = T(0);
        }

        if (model != MotionModel::CONSTANT_VELOCITY) {
            x1[idx::YAW] += x0[idx::VYAW] * T(dt);
        } else {
            x1[idx::VYAW] = T(0);
        }

        clamp(x1);
    }

    template<typename T>
    static void clamp(T x[X_N]) {
        auto& r = x[idx::R];
        auto& l = x[idx::L];
        auto& h = x[idx::H];

        r = std::clamp(r, T(0.1), T(0.5));
        if (r + l < T(0.1) || r + l > T(0.5)) {
            r = T(0.25);
            l = T(0);
        }

        h = std::clamp(h, T(-0.5), T(0.5));
    }
    void f(const VecX& x0, VecX& x1) const {
        assert(x0.size() == X_N);
        assert(x1.size() == X_N);
        operator()(x0.data(), x1.data());
    }
};

struct Measure {
    struct Ctx {
        int armor_num { 4 };
        int id { 0 };
    } ctx;

    template<typename T>
    void operator()(const T x[X_N], T z[Z_N]) const {
        T ax, ay, az, yaw;
        armor_pose(x, ax, ay, az, yaw);

        T xy = ceres::sqrt(ax * ax + ay * ay);
        T dist = ceres::sqrt(xy * xy + az * az);

        z[idx::YPD_Y] = ceres::atan2(ay, ax);
        z[idx::YPD_P] = ceres::atan2(az, xy);
        z[idx::YPD_D] = dist;
        z[idx::ORI_YAW] = yaw;
    }
    void h(const VecX& x, VecZ& z) const {
        assert(x.size() == X_N);
        assert(z.size() == Z_N);
        operator()(x.data(), z.data());
    }
    template<typename T>
    T get_armor_r(const T x[X_N]) const {
        const bool use_lh = (ctx.armor_num == 4) && (ctx.id & 1);
        return use_lh ? x[idx::R] + x[idx::L] : x[idx::R];
    }
    template<typename T>
    void armor_pose(const T x[X_N], T& ax, T& ay, T& az, T& yaw) const {
        yaw = normalize_angle(x[idx::YAW] + T(ctx.id) * T(2.0 * M_PI / ctx.armor_num));

        const bool outpost = (ctx.armor_num == 3);
        const bool use_lh = (ctx.armor_num == 4) && (ctx.id & 1);

        const T r = get_armor_r(x);

        ax = x[idx::CX] - ceres::cos(yaw) * r;
        ay = x[idx::CY] - ceres::sin(yaw) * r;

        if (outpost) {
            az = (ctx.id == 0)  ? x[idx::CZ]
                : (ctx.id == 1) ? x[idx::CZ] + x[idx::OUTPOST01DZ]
                : (ctx.id == 2) ? x[idx::CZ] + x[idx::OUTPOST02DZ]
                                : x[idx::CZ];
        } else {
            az = use_lh ? x[idx::CZ] + x[idx::H] : x[idx::CZ];
        }
    }
};
struct State {
    VecX x;
    TimePoint timestamp;
    int frame_id = 0;
    std::optional<ISO3> oldest_in_new;
    double oldest_yaw;
    std::vector<Vec4> get_armors_xyza(auto_aim::ArmorClass armor_number) const {
        std::vector<Vec4> r;
        int armor_num = armor_num_by_armor_class(armor_number);
        r.reserve(armor_num);
        for (int i = 0; i < armor_num; ++i) {
            Measure::Ctx ctx;
            ctx.id = i;
            ctx.armor_num = armor_num;
            Measure m {
                .ctx = ctx,
            };
            double ax, ay, az, ayaw;
            m.armor_pose(x.data(), ax, ay, az, ayaw);
            ISO3 pose;
            r.push_back({ ax, ay, az, ayaw });
        }
        return r;
    }

    std::vector<ISO3> get_armors_pose(auto_aim::ArmorClass armor_number) const {
        std::vector<ISO3> r;
        const double armor_pitch = (armor_number == auto_aim::ArmorClass::OUTPOST)
            ? -auto_aim::FIFTTEN_DEGREE_RAD
            : auto_aim::FIFTTEN_DEGREE_RAD;
        int armor_num = armor_num_by_armor_class(armor_number);
        r.reserve(armor_num);
        State tmp = *this;
        if (oldest_in_new) {
            tmp.transform(oldest_in_new.value().inverse(), frame_id);
            tmp.x[idx::YAW] = oldest_yaw;
        }
        for (int i = 0; i < armor_num; ++i) {
            Measure::Ctx ctx;
            ctx.id = i;
            ctx.armor_num = armor_num;
            Measure m {
                .ctx = ctx,
            };
            double ax, ay, az, ayaw;
            m.armor_pose(tmp.x.data(), ax, ay, az, ayaw);
            ISO3 pose;
            auto p = Vec3 { ax, ay, az };
            pose.translation() = p;
            Mat3 R = utils::euler2matrix(Vec3 { ayaw, armor_pitch, 0 }, utils::EulerOrder::ZYX);
            pose.linear() = R;
            if (oldest_in_new) {
                pose = oldest_in_new.value() * pose;
            }
            r.push_back(pose);
        }

        return r;
    }
    void transform(const ISO3& old_in_new, int new_frame_id) {
        oldest_yaw = x[idx::YAW];
        auto old_pose = ISO3::Identity();
        Vec3 p_new = old_in_new * pos();
        x[idx::CX] = p_new.x();
        x[idx::CY] = p_new.y();
        x[idx::CZ] = p_new.z();

        auto old_vel = vel();
        Vec3 v_new = old_in_new.linear() * vel();
        x[idx::VCX] = v_new.x();
        x[idx::VCY] = v_new.y();
        x[idx::VCZ] = v_new.z();

        double yaw_offset = std::atan2(old_in_new.linear()(1, 0), old_in_new.linear()(0, 0));
        x[idx::YAW] += yaw_offset;

        frame_id = new_frame_id;
        if (!oldest_in_new) {
            oldest_in_new = old_in_new;
            return;
        }
        oldest_in_new.value() = old_in_new * oldest_in_new.value();
    }
    void predict(const TimePoint& t) {
        auto dt = std::chrono::duration<double>(t - timestamp).count();
        predict(dt);
    }
    void predict(double dt) {
        Predict p { .dt = dt, .model = MotionModel::CONSTANT_VEL_ROT };
        p.f(x, x);
        timestamp +=
            std::chrono::duration_cast<TimePoint::duration>(std::chrono::duration<double>(dt));
    }
    double get_armor_r(int id, auto_aim::ArmorClass armor_number) {
        Measure::Ctx ctx {
            .armor_num = armor_num_by_armor_class(armor_number),
            .id = id,
        };
        Measure m {
            .ctx = ctx,
        };
        return m.get_armor_r(x.data());
    }

    Vec3 pos() const noexcept {
        return Vec3(x[idx::CX], x[idx::CY], x[idx::CZ]);
    }

    Vec3 vel() const noexcept {
        return Vec3(x[idx::VCX], x[idx::VCY], x[idx::VCZ]);
    }

    double yaw() const noexcept {
        return x[idx::YAW];
    }
    double vyaw() const noexcept {
        return x[idx::VYAW];
    }

    double r() const noexcept {
        return x[idx::R];
    }
    double l() const noexcept {
        return x[idx::L];
    }
    double h() const noexcept {
        return x[idx::H];
    }
    double outpost01DZ() const noexcept {
        return x[idx::OUTPOST01DZ];
    }
    double outpost02DZ() const noexcept {
        return x[idx::OUTPOST02DZ];
    }
};

using RobotStateEKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;

using RobotStateESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Z_N, Predict, Measure>;

} // namespace awakening::armor_motion_model