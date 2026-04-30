#include "armor_target.hpp"
#include "angles.h"
#include "tasks/auto_aim/armor_tracker/motion_model_point.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "utils/common/type_common.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
namespace awakening::auto_aim {
void ArmorTarget::reset(
    Armor& a,
    const ArmorTrackerCfg& c,
    const TimePoint& timestamp,
    int frame_id,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) {
    cfg = c;
    measure_ctx.armor_num = armor_num_by_armor_class(a.number);
    measure_ctx.id = 0;
    measure_ctx.armor_number = a.number;
    measure_ctx.camera_cv_in_odom = camera_cv_in_odom;
    measure_ctx.camera_info = camera_info;
    target_number = a.number;
    double r_pre;
    Eigen::DiagonalMatrix<double, X_N> p0;
    if (a.number == ArmorClass::OUTPOST) {
        p0.diagonal() << 1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0.1, 0.1;
        r_pre = 0.2765;
    } else if (a.number == ArmorClass::BASE) {
        p0.diagonal() << 1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0;
        r_pre = 0.3205;
    } else {
        p0.diagonal() << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
        r_pre = 0.26;
    }
    const auto u_q = [this]() {
        Eigen::Matrix<double, X_N, X_N> q;
        return q;
    };
    const auto u_r = [this](const Eigen::Matrix<double, Z_N, 1>& z) {
        Eigen::Matrix<double, Z_N, Z_N> r;
        return r;
    };
    esekf = RobotStateESEKF(
        Predict { .dt = 0.005, .armor_number = target_number },
        Measure { .ctx = measure_ctx },
        u_q,
        u_r,
        p0
    );
    esekf.value().setResidualFunc([this](
                                      const Eigen::Matrix<double, Z_N, 1>& z_pred,
                                      const Eigen::Matrix<double, Z_N, 1>& z
                                  ) {
        Eigen::Matrix<double, Z_N, 1> r = z - z_pred;
        return r;
    });
    esekf.value().setIterationNum(cfg.esekf_iter_num);
    esekf.value().setInjectFunc(
        [this](const Eigen::Matrix<double, X_N, 1>& delta, Eigen::Matrix<double, X_N, 1>& nominal) {
            for (int i = 0; i < X_N; i++) {
                if (i == idx::YAW)
                    continue;
                nominal[i] += delta[i];
            }
            nominal[idx::YAW] = angles::normalize_angle(nominal[idx::YAW] + delta[idx::YAW]);
        }
    );
    armor_pnp(a, camera_info, camera_cv_in_odom);
    auto pos = a.pose.translation();
    const double xa = pos.x();
    const double ya = pos.y();
    const double za = pos.z();
    auto ypr = utils::matrix2euler(a.pose.linear(), utils::EulerOrder::ZYX);
    const double yaw = ypr[0];

    target_state.x = Eigen::VectorXd::Zero(X_N);
    const double r = r_pre;
    const double xc = xa + r * cos(yaw);
    const double yc = ya + r * sin(yaw);
    const double zc = za;
    double l = 0.0;
    double h = 0.0;
    target_state.x << xc, 0, yc, 0, zc, 0, yaw, 0, r, l, h;
    target_state.timestamp = timestamp;
    target_state.frame_id = frame_id;
    esekf.value().setState(target_state.x);

    last_update = timestamp;
    is_inited = true;
    jumped = false;
    last_match_id = -1;
    if (target_number == ArmorClass::OUTPOST) {
        outpost_has_all_and_has_set_ids =
            std::make_pair(false, std::vector<bool>(armor_num(), false));
        outpost_has_all_and_has_set_ids.value().second[0] = true;
    } else {
        outpost_has_all_and_has_set_ids = std::nullopt;
    }
    track_state.reset();
    this_id = GOBAL_ID++;
    update_count++;
}

void ArmorTarget::armor_pnp(Armor& a, const CameraInfo& camera_info, const ISO3& camera_cv_in_odom)
    const noexcept {
    cv::Mat rvec, tvec;
    if (!cv::solvePnP(
            getArmorKeyPoints3D<cv::Point3f>(a.number),
            a.key_points.landmarks(),
            camera_info.camera_matrix,
            camera_info.distortion_coefficients,
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_IPPE
        ))
    {
        // continue;
    }
    // auto rvec = rvecs[0];
    // auto tvec = tvecs[0];
    cv::Mat R_cv_armor_in_camera_cv;
    cv::Rodrigues(rvec, R_cv_armor_in_camera_cv);
    Mat3 R_eigen_armor_in_camera_cv = Mat3::Zero();
    cv::cv2eigen(R_cv_armor_in_camera_cv, R_eigen_armor_in_camera_cv);

    Vec3 t_eigen_armor_in_camera_cv = Vec3::Zero();
    cv::cv2eigen(tvec, t_eigen_armor_in_camera_cv);
    a.pose.translation() = t_eigen_armor_in_camera_cv;
    a.pose.linear() = R_eigen_armor_in_camera_cv;
    auto armor_in_odom = camera_cv_in_odom * a.pose;
    a.pose = armor_in_odom;
}
Eigen::Matrix<double, Z_N, Z_N>
ArmorTarget::measurement_covariance(const Eigen::Matrix<double, Z_N, 1>& z) const noexcept {
    Eigen::Matrix<double, Z_N, Z_N> r;

    double u_r = cfg.r_uv_at_1m * log((1.0 / target_state.pos().norm()) + 1);

    r << u_r, 0, 0, 0, 0, 0, 0, 0, 0, u_r, 0, 0, 0, 0, 0, 0, 0, 0, u_r, 0, 0, 0, 0, 0, 0, 0, 0, u_r,
        0, 0, 0, 0, 0, 0, 0, 0, u_r, 0, 0, 0, 0, 0, 0, 0, 0, u_r, 0, 0, 0, 0, 0, 0, 0, 0, u_r, 0, 0,
        0, 0, 0, 0, 0, 0, u_r;
    return r;
}
Eigen::Matrix<double, X_N, X_N> ArmorTarget::process_noise(double dt) const noexcept {
    Eigen::Matrix<double, X_N, X_N> q;
    Vec3 q_xyz;
    double q_yaw;
    double q_l, q_h;
    if (target_number == ArmorClass::OUTPOST) {
        q_xyz = cfg.qxyz_output; // 前哨站加速度方差
        q_yaw = cfg.qyaw_output; // 前哨站角加速度方差
        q_l = cfg.q_outpost_dz;
        q_h = cfg.q_outpost_dz;
    } else {
        q_xyz = cfg.qxyz_common; // 加速度方差
        q_yaw = cfg.qyaw_common; // 角加速度方差
        q_l = cfg.q_l;
        q_h = cfg.q_h;
    }
    const double t = dt;
    const double q_x_x = pow(t, 4) / 4 * q_xyz.x(), q_x_vx = pow(t, 3) / 2 * q_xyz.x(),
                 q_vx_vx = pow(t, 2) * q_xyz.x();
    const double q_y_y = pow(t, 4) / 4 * q_xyz.y(), q_y_vy = pow(t, 3) / 2 * q_xyz.y(),
                 q_vy_vy = pow(t, 2) * q_xyz.y();
    const double q_z_z = pow(t, 4) / 4 * q_xyz.z(), q_z_vz = pow(t, 3) / 2 * q_xyz.z(),
                 q_vz_vz = pow(t, 2) * q_xyz.z();
    const double q_yaw_yaw = pow(t, 4) / 4 * q_yaw, q_yaw_vyaw = pow(t, 3) / 2 * q_yaw,
                 q_vyaw_vyaw = pow(t, 2) * q_yaw;
    const double q_r = cfg.q_r;

    // clang-format off
            //      xc      v_xc    yc      v_yc    zc      v_zc    yaw         v_yaw       r       l   h
            q <<    q_x_x,  q_x_vx, 0,      0,      0,      0,      0,          0,          0,      0,  0,
                    q_x_vx, q_vx_vx,0,      0,      0,      0,      0,          0,          0,      0,  0,
                    0,      0,      q_y_y,  q_y_vy, 0,      0,      0,          0,          0,      0,  0,
                    0,      0,      q_y_vy, q_vy_vy,0,      0,      0,          0,          0,      0,  0,
                    0,      0,      0,      0,      q_z_z,  q_z_vz, 0,          0,          0,      0,  0,
                    0,      0,      0,      0,      q_z_vz, q_vz_vz,0,          0,          0,      0,  0,
                    0,      0,      0,      0,      0,      0,      q_yaw_yaw,  q_yaw_vyaw, 0,      0,  0,
                    0,      0,      0,      0,      0,      0,      q_yaw_vyaw, q_vyaw_vyaw,0,      0,  0,
                    0,      0,      0,      0,      0,      0,      0,          0,          q_r,      0,  0,
                    0,      0,      0,      0,      0,      0,      0,          0,          0,      q_l,  0,
                    0,      0,      0,      0,      0,      0,      0,          0,          0,      0,  q_h;
    // clang-format on
    return q;
}

[[nodiscard]] Eigen::Matrix<double, Z_N, 1> ArmorTarget::get_measurement(Armor& a) const noexcept {
    Eigen::Matrix<double, Z_N, 1> z;
    auto key_points = a.key_points.landmarks();
    z[idx::LEFT_TOP_X] = key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_TOP)].x;
    z[idx::LEFT_TOP_Y] = key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_TOP)].y;
    z[idx::LEFT_BOTTOM_X] = key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_BOTTOM)].x;
    z[idx::LEFT_BOTTOM_Y] = key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_BOTTOM)].y;
    z[idx::RIGHT_BOTTOM_X] = key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_BOTTOM)].x;
    z[idx::RIGHT_BOTTOM_Y] = key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_BOTTOM)].y;
    z[idx::RIGHT_TOP_X] = key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_TOP)].x;
    z[idx::RIGHT_TOP_Y] = key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_TOP)].y;
    return z;
}
[[nodiscard]] Eigen::Matrix<double, Z_N, 1>
ArmorTarget::get_measurement(Armor& a, const VecZ& z_pred, MeasureType mt) const noexcept {
    Eigen::Matrix<double, Z_N, 1> z;
    switch (mt) {
        case ARMOR: {
            z = get_measurement(a);
            break;
        }

        case L_LIGHT: {
            auto key_points = a.key_points.landmarks();
            z[idx::LEFT_TOP_X] = key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_TOP)].x;
            z[idx::LEFT_TOP_Y] = key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_TOP)].y;
            z[idx::LEFT_BOTTOM_X] =
                key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_BOTTOM)].x;
            z[idx::LEFT_BOTTOM_Y] =
                key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_BOTTOM)].y;
            z[idx::RIGHT_BOTTOM_X] = z_pred[idx::RIGHT_BOTTOM_X];
            z[idx::RIGHT_BOTTOM_Y] = z_pred[idx::RIGHT_BOTTOM_Y];
            z[idx::RIGHT_TOP_X] = z_pred[idx::RIGHT_TOP_X];
            z[idx::RIGHT_TOP_Y] = z_pred[idx::RIGHT_TOP_Y];
            break;
        }

        case R_LIGHT: {
            auto key_points = a.key_points.landmarks();
            z[idx::RIGHT_TOP_X] = key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_TOP)].x;
            z[idx::RIGHT_TOP_Y] = key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_TOP)].y;
            z[idx::RIGHT_BOTTOM_X] =
                key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_BOTTOM)].x;
            z[idx::RIGHT_BOTTOM_Y] =
                key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_BOTTOM)].y;
            z[idx::LEFT_BOTTOM_X] = z_pred[idx::LEFT_BOTTOM_X];
            z[idx::LEFT_BOTTOM_Y] = z_pred[idx::LEFT_BOTTOM_Y];
            z[idx::LEFT_TOP_X] = z_pred[idx::LEFT_TOP_X];
            z[idx::LEFT_TOP_Y] = z_pred[idx::LEFT_TOP_Y];
        }

        break;
    }
    return z;
}
void ArmorTarget::predict_ekf(const TimePoint& timestamp) {
    if (!esekf) {
        throw std::runtime_error("ESEKF is not initialized");
    }
    auto dt = std::chrono::duration<double>(timestamp - target_state.timestamp).count();
    esekf.value().setPredictFunc(Predict { .dt = dt, .armor_number = target_number });
    esekf.value().setUpdateQ([&]() { return process_noise(dt); });
    target_state.x = esekf.value().predict();
    target_state.timestamp = timestamp;
    this_id = GOBAL_ID++;
}
bool ArmorTarget::update(
    std::pair<int, Armor>& a,
    const TimePoint& timestamp,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) {
    if (!esekf) {
        throw std::runtime_error("ESEKF is not initialized");
    }
    auto armor = a.second;
    const auto id = a.first;
    if (id != 0) {
        jumped = true;
    }
    if (outpost_has_all_and_has_set_ids) {
        if (!outpost_has_all_and_has_set_ids.value().first) {
            outpost_has_all_and_has_set_ids.value().second[id] = true;
            int count = 0;
            for (auto has_id: outpost_has_all_and_has_set_ids.value().second) {
                if (has_id) {
                    count++;
                }
            }
            if (count >= outpost_has_all_and_has_set_ids->second.size()) {
                outpost_has_all_and_has_set_ids->first = true;
            }
        }
    }
    last_match_id = id;
    MeasureType mt = MeasureType::ARMOR;
    esekf.value().setUpdateR([&](const Eigen::Matrix<double, Z_N, 1>& z) {
        return measurement_covariance(z);
    });

    measure_ctx.id = id;
    measure_ctx.camera_cv_in_odom = camera_cv_in_odom;

    Measure measure { .ctx = measure_ctx };
    VecZ z_pred;
    measure.h(target_state.x, z_pred);
    auto measurement = get_measurement(armor, z_pred, mt);
    esekf.value().setMeasureFunc(measure);

    target_state.x = esekf.value().update(measurement);
    target_state.timestamp = timestamp;
    last_update = timestamp;
    this_id = GOBAL_ID++;
    update_count++;
    return true;
}
std::vector<std::pair<int, Armor>> ArmorTarget::match(
    std::vector<Armor>& armors,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) const noexcept {
    std::vector<std::pair<int, Armor>> result;
    const int n_obs = static_cast<int>(armors.size());
    const int armors_num = armor_num();
    bool all_init =
        (outpost_has_all_and_has_set_ids.has_value() ? outpost_has_all_and_has_set_ids.value().first
                                                     : jumped);
    const double GATE = (all_init ? cfg.match_gate_at_1m : cfg.match_gate_not_all_init_at_1m);

    const double max_cost = 1e9;
    std::vector<std::vector<double>> cost(n_obs, std::vector<double>(armors_num, max_cost + 1));

    std::vector<VecZ> meas_list(n_obs);
    for (int j = 0; j < n_obs; ++j) {
        meas_list[j] = get_measurement(armors[j]);
    }

    for (int j = 0; j < n_obs; ++j) {
        bool in_gate = false;
        double min_d2 = std::numeric_limits<double>::max();
        for (int id = 0; id < armors_num; ++id) {
            Measure::Ctx tmp_ctx {
                .armor_num = armors_num,
                .id = id,
                .camera_cv_in_odom = camera_cv_in_odom,
                .camera_info = camera_info,
                .armor_number = target_number,

            };
            Measure measure { .ctx = tmp_ctx };
            VecZ z_pred;
            measure.h(target_state.x, z_pred);

            VecZ nu = meas_list[j] - z_pred;
            auto R = measurement_covariance(z_pred);
            double d2 = nu.transpose() * R.ldlt().solve(nu);

            if (std::isfinite(d2) && d2 < GATE) {
                cost[j][id] = d2;
                in_gate = true;
            }
            if (d2 < min_d2) {
                min_d2 = d2;
            }
        }
        if (!in_gate) {
            AWAKENING_WARN("match out of gate min d2: {}", min_d2);
        }
    }

    std::vector<bool> used_obs(n_obs, false);
    std::vector<bool> used_id(armors_num, false);

    while (true) {
        double best = max_cost;
        int best_j = -1;
        int best_id = -1;

        for (int j = 0; j < n_obs; ++j) {
            if (used_obs[j])
                continue;
            for (int id = 0; id < armors_num; ++id) {
                if (used_id[id])
                    continue;
                if (cost[j][id] < best) {
                    best = cost[j][id];
                    best_j = j;
                    best_id = id;
                }
            }
        }

        if (best_j < 0 || best_id < 0) {
            break;
        }

        used_obs[best_j] = true;
        used_id[best_id] = true;
        result.push_back(std::make_pair(best_id, armors[best_j]));
    }
    return result;
}
[[nodiscard]] cv::Rect ArmorTarget::expanded_one_one(
    const TimePoint& timestamp,
    const ISO3& camera_cv_in_odom,
    const CameraInfo& camera_info,
    const cv::Size& image_size
) const noexcept {
    const double dt = std::chrono::duration<double>(timestamp - last_update).count();

    if (!is_inited || dt > cfg.lost_time_thres) {
        return cv::Rect(0, 0, image_size.width, image_size.height);
    }

    float car_box_half = std::max(target_state.r(), target_state.r() + target_state.l()) + 0.15f;
    if (target_number == ArmorClass::OUTPOST) {
        car_box_half = target_state.r() + 0.15f;
    }
    static std::vector<cv::Point3f> CAR_BOX;
    CAR_BOX = { { 0, car_box_half, -car_box_half },
                { 0, -car_box_half, -car_box_half },
                { 0, -car_box_half, car_box_half },
                { 0, car_box_half, car_box_half } };

    auto target_pos_in_odom = target_state.pos();
    if (target_number == ArmorClass::OUTPOST) {
        target_pos_in_odom.z() += (target_state.outpost01DZ() + target_state.outpost02DZ()) / 2.0;
    }
    auto target_pos_in_camera_cv = camera_cv_in_odom.inverse() * target_pos_in_odom;

    if (target_pos_in_camera_cv.z() <= 0.2) {
        return cv::Rect(0, 0, image_size.width, image_size.height);
    }

    const cv::Mat tvec =
        (cv::Mat_<double>(3, 1) << target_pos_in_camera_cv.x(),
         target_pos_in_camera_cv.y(),
         target_pos_in_camera_cv.z());

    auto target_R_in_odom = utils::euler2matrix(
        Vec3(std::atan2(target_pos_in_odom.y(), target_pos_in_odom.x()), 0, 0),
        utils::EulerOrder::ZYX
    );

    auto target_R_in_camera_cv = camera_cv_in_odom.inverse() * target_R_in_odom;

    const cv::Mat rot_mat =
        (cv::Mat_<double>(3, 3) << target_R_in_camera_cv(0, 0),
         target_R_in_camera_cv(0, 1),
         target_R_in_camera_cv(0, 2),
         target_R_in_camera_cv(1, 0),
         target_R_in_camera_cv(1, 1),
         target_R_in_camera_cv(1, 2),
         target_R_in_camera_cv(2, 0),
         target_R_in_camera_cv(2, 1),
         target_R_in_camera_cv(2, 2));

    cv::Mat rvec;
    cv::Rodrigues(rot_mat, rvec);

    std::vector<cv::Point2f> pts_2d;
    cv::projectPoints(
        CAR_BOX,
        rvec,
        tvec,
        camera_info.camera_matrix,
        camera_info.distortion_coefficients,
        pts_2d
    );

    const cv::Rect rect = cv::boundingRect(pts_2d);

    const cv::Rect img_rect(0, 0, image_size.width, image_size.height);

    if ((rect & img_rect).area() <= 0) {
        return cv::Rect(0, 0, image_size.width, image_size.height);
    }

    const double lost_dt = cfg.lost_time_thres;

    double alpha = std::clamp(dt / lost_dt, 0.0, 1.0);

    double x1 = rect.x;
    double y1 = rect.y;
    double x2 = rect.x + rect.width;
    double y2 = rect.y + rect.height;

    const double img_x1 = 0.0;
    const double img_y1 = 0.0;
    const double img_x2 = image_size.width;
    const double img_y2 = image_size.height;

    x1 = std::clamp(x1, img_x1, img_x2);
    x2 = std::clamp(x2, img_x1, img_x2);
    y1 = std::clamp(y1, img_y1, img_y2);
    y2 = std::clamp(y2, img_y1, img_y2);

    x1 = x1 + (img_x1 - x1) * alpha;
    y1 = y1 + (img_y1 - y1) * alpha;
    x2 = x2 + (img_x2 - x2) * alpha;
    y2 = y2 + (img_y2 - y2) * alpha;

    cv::Rect expanded_rect(
        static_cast<int>(x1),
        static_cast<int>(y1),
        static_cast<int>(x2 - x1),
        static_cast<int>(y2 - y1)
    );

    int cx = expanded_rect.x + expanded_rect.width / 2;
    int cy = expanded_rect.y + expanded_rect.height / 2;

    int side = std::max(expanded_rect.width, expanded_rect.height);

    cv::Rect square(cx - side / 2, cy - side / 2, side, side);

    square &= img_rect;

    return square;
}
[[nodiscard]] cv::Rect ArmorTarget::expanded(
    const TimePoint& timestamp,
    const ISO3& camera_cv_in_odom,
    const CameraInfo& camera_info,
    const cv::Size& image_size
) const noexcept {
    std::vector<cv::Point2f> pts;
    auto tmp_target_state = target_state;
    tmp_target_state.predict(timestamp, target_number);
    const int armors_num = armor_num();
    for (int id = 0; id < armors_num; ++id) {
        Measure::Ctx tmp_ctx {
            .armor_num = armors_num,
            .id = id,
            .camera_cv_in_odom = camera_cv_in_odom,
            .camera_info = camera_info,
            .armor_number = target_number,

        };
        Measure measure { .ctx = tmp_ctx };
        VecZ z_pred;
        measure.h(target_state.x, z_pred);
        pts.push_back(cv::Point2f(z_pred[idx::LEFT_TOP_X], z_pred[idx::LEFT_TOP_Y]));
        pts.push_back(cv::Point2f(z_pred[idx::RIGHT_TOP_X], z_pred[idx::RIGHT_TOP_Y]));
        pts.push_back(cv::Point2f(z_pred[idx::RIGHT_BOTTOM_X], z_pred[idx::RIGHT_BOTTOM_Y]));
        pts.push_back(cv::Point2f(z_pred[idx::LEFT_BOTTOM_X], z_pred[idx::LEFT_BOTTOM_Y]));
    }
    cv::Rect rect = cv::boundingRect(pts);
    const cv::Rect img_rect(0, 0, image_size.width, image_size.height);
    if ((rect & img_rect).area() <= 0) {
        return cv::Rect(0, 0, image_size.width, image_size.height);
    }
    return rect;
}
} // namespace awakening::auto_aim