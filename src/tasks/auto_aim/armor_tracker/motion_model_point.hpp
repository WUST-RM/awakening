#pragma once
#include "KalmanHyLib/kalman_hybird_lib.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <algorithm>
#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>
namespace awakening::armor_point_motion_model {

constexpr int X_N = 11;
constexpr int Z_N = 8;

using VecX = Eigen::Matrix<double, X_N, 1>;
using VecZ = Eigen::Matrix<double, Z_N, 1>;

namespace idx {
    enum { CX, VCX, CY, VCY, CZ, VCZ, YAW, VYAW, R, P1, P2 };
    constexpr int L = P1;
    constexpr int H = P2;
    constexpr int OUTPOST01DZ = P1;
    constexpr int OUTPOST02DZ = P2;
    enum {
        LEFT_TOP_X,
        LEFT_TOP_Y,
        LEFT_BOTTOM_X,
        LEFT_BOTTOM_Y,
        RIGHT_BOTTOM_X,
        RIGHT_BOTTOM_Y,
        RIGHT_TOP_X,
        RIGHT_TOP_Y
    };
} // namespace idx

template<typename T>
inline T normalize_angle(T a) {
    const T two_pi = T(2.0 * M_PI);
    return a - two_pi * floor((a + T(M_PI)) / two_pi);
}

struct Predict {
    double dt { 0.0 };

    auto_aim::ArmorClass armor_number = auto_aim::ArmorClass::UNKNOWN;

    template<typename T>
    inline void operator()(const T x0[X_N], T x1[X_N]) const {
        std::copy(x0, x0 + X_N, x1);

        if (armor_number != auto_aim::ArmorClass::OUTPOST
            && armor_number != auto_aim::ArmorClass::BASE) {
            x1[idx::CX] += x0[idx::VCX] * T(dt);
            x1[idx::CY] += x0[idx::VCY] * T(dt);
            x1[idx::CZ] += x0[idx::VCZ] * T(dt);
        } else {
            x1[idx::VCX] = x1[idx::VCY] = x1[idx::VCZ] = T(0);
        }

        if (armor_number != auto_aim::ArmorClass::BASE) {
            x1[idx::YAW] += x0[idx::VYAW] * T(dt);
        }

        clamp(x1);
    }

    template<typename T>
    inline void clamp(T x[X_N]) const {
        auto& r = x[idx::R];
        auto& l = x[idx::L];
        auto& h = x[idx::H];
        auto& vyaw = x[idx::VYAW];
        if (armor_number != auto_aim::ArmorClass::OUTPOST) {
            if (r + l < T(0.1) || r + l > T(0.5)) {
                r = T(0.25);
                l = T(0);
            }

            if (ceres::abs(h) > T(0.5)) {
                h = T(0.0);
            }
        } else {
            x[idx::VCZ] = T(0.0);
            r = T(0.27);
        }

        if (ceres::abs(vyaw) > T(20.0)) {
            vyaw = T(0.0);
        }
    }
    inline void f(const VecX& x0, VecX& x1) const {
        assert(x0.size() == X_N);
        assert(x1.size() == X_N);
        operator()(x0.data(), x1.data());
    }
};

template<typename T>
inline void project_points_jets(
    const std::vector<cv::Point3f>& obj_pts,
    const Eigen::Transform<T, 3, Eigen::Isometry>& pose_cam,
    const cv::Mat& K,
    const cv::Mat& dist_coeffs,
    std::vector<Eigen::Matrix<T, 2, 1>>& img_pts_jet
) {
    const Eigen::Matrix<T, 3, 3>& R = pose_cam.linear();
    const Eigen::Matrix<T, 3, 1>& t = pose_cam.translation();

    const T fx = T(K.at<double>(0, 0));
    const T fy = T(K.at<double>(1, 1));
    const T cx = T(K.at<double>(0, 2));
    const T cy = T(K.at<double>(1, 2));

    auto get_dist = [&](int i) -> double {
        return (dist_coeffs.rows == 1) ? dist_coeffs.at<double>(0, i)
                                       : dist_coeffs.at<double>(i, 0);
    };

    const int n_dist = dist_coeffs.rows * dist_coeffs.cols;

    const T k1 = n_dist > 0 ? T(get_dist(0)) : T(0);
    const T k2 = n_dist > 1 ? T(get_dist(1)) : T(0);
    const T p1 = n_dist > 2 ? T(get_dist(2)) : T(0);
    const T p2 = n_dist > 3 ? T(get_dist(3)) : T(0);
    const T k3 = n_dist > 4 ? T(get_dist(4)) : T(0);

    img_pts_jet.clear();
    img_pts_jet.reserve(obj_pts.size());

    for (const auto& pt3: obj_pts) {
        Eigen::Matrix<T, 3, 1> Pw(T(pt3.x), T(pt3.y), T(pt3.z));

        Eigen::Matrix<T, 3, 1> Pc = R * Pw + t;

        T Xc = Pc(0);
        T Yc = Pc(1);
        T Zc = Pc(2);

        if (ceres::abs(Zc) < T(1e-8)) {
            continue;
        }

        T xp = Xc / Zc;
        T yp = Yc / Zc;

        T r2 = xp * xp + yp * yp;
        T r4 = r2 * r2;
        T r6 = r4 * r2;

        T radial = T(1) + k1 * r2 + k2 * r4 + k3 * r6;

        T xd = xp * radial + T(2) * p1 * xp * yp + p2 * (r2 + T(2) * xp * xp);

        T yd = yp * radial + p1 * (r2 + T(2) * yp * yp) + T(2) * p2 * xp * yp;

        T u = fx * xd + cx;
        T v = fy * yd + cy;

        img_pts_jet.emplace_back(u, v);
    }
}

struct Measure {
    struct Ctx {
        int armor_num { 4 };
        int id { 0 };
        ISO3 camera_cv_in_odom = ISO3::Identity();
        CameraInfo camera_info;
        auto_aim::ArmorClass armor_number = auto_aim::ArmorClass::UNKNOWN;
    } ctx;

    template<typename T>
    inline void operator()(const T x[X_N], T z[Z_N]) const {
        T ax, ay, az, yaw;
        armor_pose(x, ax, ay, az, yaw);
        Eigen::Transform<T, 3, Eigen::Isometry> pose_in_odom;
        pose_in_odom.translation() << ax, ay, az;

        const T armor_pitch = (ctx.armor_number == auto_aim::ArmorClass::OUTPOST)
            ? T(-auto_aim::FIFTTEN_DEGREE_RAD)
            : T(auto_aim::FIFTTEN_DEGREE_RAD);

        Eigen::Quaternion<T> q_yaw(Eigen::AngleAxis<T>(yaw, Eigen::Vector3<T>::UnitZ()));
        Eigen::Quaternion<T> q_pitch(Eigen::AngleAxis<T>(armor_pitch, Eigen::Vector3<T>::UnitY()));
        Eigen::Quaternion<T> q_roll(Eigen::AngleAxis<T>(T(0.0), Eigen::Vector3<T>::UnitX()));
        pose_in_odom.linear() = (q_yaw * q_pitch * q_roll).toRotationMatrix();

        Eigen::Transform<T, 3, Eigen::Isometry> camera_cv_in_odom_jet;
        camera_cv_in_odom_jet.matrix() = ctx.camera_cv_in_odom.matrix().template cast<T>();

        auto pose_in_camera_cv = camera_cv_in_odom_jet.inverse() * pose_in_odom;

        std::vector<cv::Point3f> object_points = getArmorKeyPoints3D<cv::Point3f>(ctx.armor_number);

        std::vector<Eigen::Matrix<T, 2, 1>> img_pts_jet;
        project_points_jets(
            object_points,
            pose_in_camera_cv,
            ctx.camera_info.camera_matrix,
            ctx.camera_info.distortion_coefficients,
            img_pts_jet
        );

        z[idx::LEFT_TOP_X] =
            img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::LEFT_TOP)].x();
        z[idx::LEFT_TOP_Y] =
            img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::LEFT_TOP)].y();
        z[idx::LEFT_BOTTOM_X] =
            img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::LEFT_BOTTOM)].x();
        z[idx::LEFT_BOTTOM_Y] =
            img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::LEFT_BOTTOM)].y();
        z[idx::RIGHT_TOP_X] =
            img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::RIGHT_TOP)].x();
        z[idx::RIGHT_TOP_Y] =
            img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::RIGHT_TOP)].y();
        z[idx::RIGHT_BOTTOM_X] =
            img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::RIGHT_BOTTOM)].x();
        z[idx::RIGHT_BOTTOM_Y] =
            img_pts_jet[std::to_underlying(auto_aim::ArmorKeyPointsIndex::RIGHT_BOTTOM)].y();
    }

    inline void h(const VecX& x, VecZ& z) const {
        operator()(x.data(), z.data());
    }

    template<typename T>
    inline T get_armor_r(const T x[X_N]) const {
        const bool use_lh = (ctx.armor_num == 4) && (ctx.id & 1);
        return use_lh ? x[idx::R] + x[idx::L] : x[idx::R];
    }

    template<typename T>
    inline void armor_pose(const T x[X_N], T& ax, T& ay, T& az, T& yaw) const {
        yaw = normalize_angle(x[idx::YAW] + T(ctx.id) * T(2.0 * M_PI / ctx.armor_num));

        const bool outpost = (ctx.armor_number == auto_aim::ArmorClass::OUTPOST);
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

    inline std::vector<Vec4> get_armors_xyza(auto_aim::ArmorClass armor_number) const {
        std::vector<Vec4> r;
        int armor_num = armor_num_by_armor_class(armor_number);
        r.reserve(armor_num);
        for (int i = 0; i < armor_num; ++i) {
            Measure::Ctx ctx;
            ctx.id = i;
            ctx.armor_num = armor_num;
            ctx.armor_number = armor_number;
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

    inline std::vector<ISO3> get_armors_pose(auto_aim::ArmorClass armor_number) const {
        std::vector<ISO3> r;
        const double armor_pitch = (armor_number == auto_aim::ArmorClass::OUTPOST)
            ? -auto_aim::FIFTTEN_DEGREE_RAD
            : auto_aim::FIFTTEN_DEGREE_RAD;
        int armor_num = armor_num_by_armor_class(armor_number);
        r.reserve(armor_num);
        for (int i = 0; i < armor_num; ++i) {
            Measure::Ctx ctx;
            ctx.id = i;
            ctx.armor_num = armor_num;
            ctx.armor_number = armor_number;
            Measure m {
                .ctx = ctx,
            };
            double ax, ay, az, ayaw;
            m.armor_pose(x.data(), ax, ay, az, ayaw);
            ISO3 pose;
            auto p = Vec3 { ax, ay, az };
            pose.translation() = p;
            Mat3 R = utils::euler2matrix(Vec3 { ayaw, armor_pitch, 0 }, utils::EulerOrder::ZYX);
            pose.linear() = R;

            r.push_back(pose);
        }

        return r;
    }

    inline void predict(const TimePoint& t, auto_aim::ArmorClass armor_number) {
        auto dt = std::chrono::duration<double>(t - timestamp).count();
        predict(dt, armor_number);
    }
    inline void predict(double dt, auto_aim::ArmorClass armor_number) {
        Predict p { .dt = dt, .armor_number = armor_number };
        p.f(x, x);
        timestamp +=
            std::chrono::duration_cast<TimePoint::duration>(std::chrono::duration<double>(dt));
    }
    inline double get_armor_r(int id, auto_aim::ArmorClass armor_number) const {
        Measure::Ctx ctx {
            .armor_num = armor_num_by_armor_class(armor_number),
            .id = id,
        };
        Measure m {
            .ctx = ctx,
        };
        return m.get_armor_r(x.data());
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

    inline double yaw() const noexcept {
        return x[idx::YAW];
    }
    inline double vyaw() const noexcept {
        return x[idx::VYAW];
    }

    inline double r() const noexcept {
        return x[idx::R];
    }
    inline double l() const noexcept {
        return x[idx::L];
    }
    inline double h() const noexcept {
        return x[idx::H];
    }
    inline double outpost01DZ() const noexcept {
        return x[idx::OUTPOST01DZ];
    }
    inline double outpost02DZ() const noexcept {
        return x[idx::OUTPOST02DZ];
    }
};

using RobotStateEKF = kalman_hybird_lib::ExtendedKalmanFilter<X_N, Z_N, Predict, Measure>;

using RobotStateESEKF = kalman_hybird_lib::ErrorStateEKF<X_N, Z_N, Predict, Measure>;

} // namespace awakening::armor_point_motion_model