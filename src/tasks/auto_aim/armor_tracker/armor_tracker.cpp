#include "armor_tracker.hpp"
#include "angles.h"
#include "motion_model.hpp"
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/auto_aim/type.hpp"
#include "utils/common/type_common.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"
#include <array>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <utility>
namespace awakening::auto_aim {
struct ArmorTracker::Impl {
    Impl(const YAML::Node& config) {
        cfg_.load(config);
    }
    ArmorTarget
    track(Armors& armors, const CameraInfo& camera_info, const ISO3& camera_cv_in_odom) {
        static TimePoint last_track = Clock::now();
        double dt = std::chrono::duration<double>(armors.timestamp - last_track).count();
        dt = std::clamp(dt, 1e-3, 0.1);
        last_track = armors.timestamp;
        lost_thres_ = std::abs(static_cast<int>(cfg_.lost_time_thres / dt));
        pose_solve(armors, camera_info, camera_cv_in_odom);
        auto process = [&](int idx) {
            bool found =
                (target_buf_[idx].track_state.tracker_state == ArmorTarget::TrackState::LOST)
                ? init_target(idx, armors)
                : update_target(idx, armors);
            update_fsm(found, idx);
            return found;
        };

        process(cur_target_idx_);
        if (target_buf_[cur_target_idx_].track_state.tracker_state
            == ArmorTarget::TrackState::TEMP_LOST) {
            process(pre_target_idx_);

            if (target_buf_[pre_target_idx_].track_state.tracker_state
                == ArmorTarget::TrackState::TRACKING) {
                std::swap(target_buf_[pre_target_idx_], target_buf_[cur_target_idx_]);
                target_buf_[pre_target_idx_] = ArmorTarget();
            }
        } else if (target_buf_[cur_target_idx_].track_state.tracker_state == ArmorTarget::TrackState::TRACKING)
        {
            target_buf_[pre_target_idx_].track_state.tracker_state = ArmorTarget::TrackState::LOST;
        }
        return target_buf_[cur_target_idx_];
    }
    bool init_target(size_t i, const Armors& armors) noexcept {
        if (armors.armors.empty()) {
            return false;
        }
        auto& target = target_buf_[i];
        bool found = false;
        Armor init_target;
        Armors others = armors;
        others.armors.clear();
        for (auto& a: armors.armors) {
            if (!(a.color == ArmorColor::NONE || a.color == ArmorColor::PURPLE) && !found) {
                init_target = a;
                found = true;
                continue;
            }
            others.armors.push_back(a);
        }
        if (!found) {
            return false;
        }
        AWAKENING_INFO("init target: {}", getStringByArmorClass(init_target.number));
        target = ArmorTarget(init_target, cfg_, armors.timestamp);
        target.track_state.tracker_state = ArmorTarget::TrackState::DETECTING;
        return true;
    }
    bool update_target(size_t i, const Armors& armors) noexcept {
        if (armors.armors.empty())
            return false;
        auto& target = target_buf_[i];
        target.predict_ekf(armors.timestamp);
        std::vector<Armor> candidates;
        candidates.reserve(armors.armors.size());
        auto target_state = target.get_target_state();
        double center_yaw =
            angles::normalize_angle(std::atan2(target_state.pos().y(), target_state.pos().x()));
        for (const auto& a: armors.armors) {
            if (a.number == target.target_number) {
                if (target.check()) {
                    // if (std::abs(angles::normalize_angle(
                    //         utils::R2yaw<ArmorTarget>(a.pose.linear()) - center_yaw
                    //     )) > (cfg_.max_yaw_diff_deg * M_PI / 180.0)
                    //     || std::abs((a.pose.translation() - target_.target_state.pos()).norm())
                    //         > cfg_.max_dis_diff)
                    // {
                    //     AWAKENING_WARN("Armor too far from the target");
                    //     continue;
                    // }
                }
                candidates.emplace_back(a);
            }
        }

        if (candidates.empty())
            return false;

        int updated = 0;
        const auto matches = target.match(candidates);

        for (const auto& m: matches) {
            if (m.second.color == ArmorColor::NONE || m.second.color == ArmorColor::PURPLE) {
                if (++is_none_purple_count_ > 100)
                    continue;
            } else {
                is_none_purple_count_ = 0;
            }

            if (target.update(m, armors.timestamp))
                ++updated;
        }

        return updated > 0;
    }
    void update_fsm(bool found, size_t i) noexcept {
        auto& target = target_buf_[i];
        switch (target.track_state.tracker_state) {
            case ArmorTarget::TrackState::DETECTING:
                if (found) {
                    if (++target.track_state.detect_count > cfg_.tracking_thres) {
                        target.track_state.detect_count = 0;
                        target.track_state.tracker_state = ArmorTarget::TrackState::TRACKING;
                    }
                } else {
                    target.track_state.detect_count = 0;
                    target.track_state.tracker_state = ArmorTarget::TrackState::LOST;
                }
                break;

            case ArmorTarget::TrackState::TRACKING:
                if (!found) {
                    target.track_state.tracker_state = ArmorTarget::TrackState::TEMP_LOST;
                    target.track_state.lost_count = 1;
                }
                break;

            case ArmorTarget::TrackState::TEMP_LOST:
                if (!found) {
                    if (++target.track_state.lost_count > lost_thres_) {
                        target.track_state.lost_count = 0;
                        target.track_state.tracker_state = ArmorTarget::TrackState::LOST;
                    }
                } else {
                    target.track_state.lost_count = 0;
                    target.track_state.tracker_state = ArmorTarget::TrackState::TRACKING;
                }
                break;

            default:
                break;
        }

        if (found)
            ++found_count_;
    }
    void pose_solve(Armors& armors, const CameraInfo& camera_info, const ISO3& camera_cv_in_odom) {
        for (auto& armor: armors.armors) {
            cv::Mat rvec, tvec;
            if (!cv::solvePnP(
                    getArmorKeyPoints3D<cv::Point3f>(armor.number),
                    armor.key_points.landmarks(),
                    camera_info.camera_matrix,
                    camera_info.distortion_coefficients,
                    rvec,
                    tvec,
                    false,
                    cv::SOLVEPNP_IPPE
                ))
            {
                continue;
            }

            cv::Mat R_cv_armor_in_camera_cv;
            cv::Rodrigues(rvec, R_cv_armor_in_camera_cv);
            Mat3 R_eigen_armor_in_camera_cv = Mat3::Zero();
            cv::cv2eigen(R_cv_armor_in_camera_cv, R_eigen_armor_in_camera_cv);
            Vec3 t_eigen_armor_in_camera_cv = Vec3::Zero();
            cv::cv2eigen(tvec, t_eigen_armor_in_camera_cv);
            armor.pose.translation() = t_eigen_armor_in_camera_cv;
            armor.pose.linear() = R_eigen_armor_in_camera_cv;
            auto armor_R_in_odom = opt_R(armor, camera_info, camera_cv_in_odom);
            auto armor_in_odom = camera_cv_in_odom * armor.pose;
            armor_in_odom.linear() = armor_R_in_odom;
            armor.pose = armor_in_odom;
        }
    }

    double reprojection_error_yaw(
        const Mat3& armor_R_in_odom,
        const CameraInfo& camera_info,
        const std::vector<cv::Point3f>& object_points,
        const std::array<cv::Point2f, std::to_underlying(ArmorKeyPointsIndex::N)>& landmarks,
        const ISO3& camera_cv_in_odom,
        const ISO3& armor_pose_in_camera_cv_raw
    ) const noexcept {
        auto armor_pose_in_odom = camera_cv_in_odom * armor_pose_in_camera_cv_raw;
        armor_pose_in_odom.linear() = armor_R_in_odom;
        auto armor_pose_in_camera_cv = camera_cv_in_odom.inverse() * armor_pose_in_odom;
        const auto image_points = utils::reprojection(
            camera_info.camera_matrix,
            camera_info.distortion_coefficients,
            object_points,
            armor_pose_in_camera_cv
        );
        double cost = 0.0;
        for (int i = 0; i < std::to_underlying(ArmorKeyPointsIndex::N); i++) {
             cost += cv::norm(image_points[i] - landmarks[i]);
        }
        // for (auto& p: armor_keypoints::sys_pairs) {
        //     const auto mid = 0.5 * (image_points[p.first] + image_points[p.second]);
        //     const auto meas = 0.5 * (landmarks[p.first] + landmarks[p.second]);
        //     auto d = mid - meas;
        //     double __cost = d.x * d.x + d.y * d.y;
        //     // cost += cv::norm(mid - meas);
        //     cost += __cost;
        // }
        return cost;
    }
    double search_yaw(
        double init,
        double pitch,
        double roll,
        const CameraInfo& camera_info,
        const std::vector<cv::Point3f>& object_points,
        const std::array<cv::Point2f, std::to_underlying(ArmorKeyPointsIndex::N)>& landmarks,
        const ISO3& camera_cv_in_odom,
        const ISO3& armor_pose_in_camera_cv_raw
    ) const noexcept {
        double best_yaw = init;
        double min_error = std::numeric_limits<double>::max();
        constexpr double SEARCH_RANGE = 140;
        constexpr double step = 1.0 * M_PI / 180.0;
        auto opt_yaw_start = angles::normalize_angle(init - ((SEARCH_RANGE / 2.0) * M_PI / 180.0));
        for (int i = 0; i < SEARCH_RANGE; i++) {
            auto yaw = angles::normalize_angle(opt_yaw_start + i * step);
            auto armor_R_in_odom =
                utils::euler2matrix(Vec3(yaw, pitch, roll), utils::EulerOrder::ZYX);
            double error = reprojection_error_yaw(
                armor_R_in_odom,
                camera_info,
                object_points,
                landmarks,
                camera_cv_in_odom,
                armor_pose_in_camera_cv_raw
            );
            if (error < min_error) {
                min_error = error;
                best_yaw = yaw;
            }
            // std::cout<<"yaw: "<<yaw*180.0/M_PI<<" error: "<<error<<std::endl;
        }
        // std::cout << "init_yaw: " << init * 180.0 / M_PI << " best_yaw: " << best_yaw * 180.0 / M_PI
        //           << std::endl;
        return best_yaw;
    }
    Mat3 opt_R(Armor& armor, const CameraInfo& camera_info, const ISO3& camera_cv_in_odom) {
        const auto& lm = armor.key_points.landmarks();
        const auto obj_pts = getArmorKeyPoints3D<cv::Point3f>(armor.number);
        auto tmp_armor_pose = armor.pose;
        auto tmp_armor_pose_in_odom = camera_cv_in_odom * tmp_armor_pose;
        auto armor_R_in_odom_init = tmp_armor_pose_in_odom.linear();
        auto raw_ypr = utils::matrix2euler(armor_R_in_odom_init, utils::EulerOrder::ZYX);
        const double armor_pitch =
            (armor.number == ArmorClass::OUTPOST) ? -FIFTTEN_DEGREE_RAD : FIFTTEN_DEGREE_RAD;
        double init_yaw = raw_ypr[0];
        double roll = raw_ypr[2];
        // double roll = 0.0;
        auto fin_yaw = search_yaw(
            init_yaw,
            armor_pitch,
            roll,
            camera_info,
            obj_pts,
            lm,
            camera_cv_in_odom,
            armor.pose
        );
        auto fin_R_in_odom =
            utils::euler2matrix(Vec3(fin_yaw, armor_pitch, roll), utils::EulerOrder::ZYX);
        return fin_R_in_odom;
    }
    int lost_thres_;
    int is_none_purple_count_ = 0;
    int found_count_ = 0;

    size_t cur_target_idx_ = 0;
    size_t pre_target_idx_ = 1;
    std::array<ArmorTarget, 2> target_buf_;
    ArmorTrackerCfg cfg_;
};
ArmorTracker::ArmorTracker(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}
ArmorTracker::~ArmorTracker() noexcept {
    _impl.reset();
}
void ArmorTracker::pose_solve(
    Armors& armors,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) {
    _impl->pose_solve(armors, camera_info, camera_cv_in_odom);
}
ArmorTarget
ArmorTracker::track(Armors& armors, const CameraInfo& camera_info, const ISO3& camera_cv_in_odom) {
    return _impl->track(armors, camera_info, camera_cv_in_odom);
}
int ArmorTracker::get_count() {
    return _impl->found_count_;
}
void ArmorTracker::reset_count() {
    _impl->found_count_ = 0;
}
} // namespace awakening::auto_aim