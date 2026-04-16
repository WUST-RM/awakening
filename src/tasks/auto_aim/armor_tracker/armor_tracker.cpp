#include "armor_tracker.hpp"
#include "angles.h"
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
    ArmorTarget track(
        Armors& armors,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom,
        int frame_id
    ) {
        static TimePoint last_track = Clock::now();
        double dt = std::chrono::duration<double>(armors.timestamp - last_track).count();
        dt = std::clamp(dt, 1e-3, 0.1);
        last_track = armors.timestamp;
        lost_thres_ = std::abs(static_cast<int>(cfg_.lost_time_thres / dt));
        auto process = [&](int idx) {
            auto& t = target_buf_[idx];
            bool found = (t.track_state.tracker_state == ArmorTarget::TrackState::LOST)
                ? init_target(t, armors, frame_id, camera_info, camera_cv_in_odom)
                : update_target(t, armors, camera_info, camera_cv_in_odom);
            update_fsm(found, idx);
            if (found) {
                found_count_++;
            }
            return found;
        };

        process(cur_target_idx_);
        auto& cur = target_buf_[cur_target_idx_];
        auto& pre = target_buf_[pre_target_idx_];

        if (cur.track_state.tracker_state == ArmorTarget::TrackState::TEMP_LOST) {
            process(pre_target_idx_);

            if (pre.track_state.tracker_state == ArmorTarget::TrackState::TRACKING) {
                std::swap(cur, pre);
                pre.track_state.tracker_state = ArmorTarget::TrackState::LOST;
            }
        } else if (cur.track_state.tracker_state == ArmorTarget::TrackState::TRACKING) {
            pre.track_state.tracker_state = ArmorTarget::TrackState::LOST;
        }

        return target_buf_[cur_target_idx_].fast_copy_without_ekf();
    }
    bool init_target(
        ArmorTarget& target,
        const Armors& armors,
        int frame_id,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom
    ) noexcept {
        if (armors.armors.empty()) {
            return false;
        }
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
        AWAKENING_INFO("init target: {}", string_by_armor_class(init_target.number));
        target.reset(init_target, cfg_, armors.timestamp, frame_id, camera_info, camera_cv_in_odom);
        target.track_state.tracker_state = ArmorTarget::TrackState::DETECTING;
        update_target(target, others, camera_info, camera_cv_in_odom);
        return true;
    }
    bool update_target(
        ArmorTarget& target,
        const Armors& armors,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom
    ) noexcept {
        if (armors.armors.empty())
            return false;
        target.predict_ekf(armors.timestamp);
        std::vector<Armor> candidates;
        candidates.reserve(armors.armors.size());
        auto target_state = target.get_target_state();
        double center_yaw =
            angles::normalize_angle(std::atan2(target_state.pos().y(), target_state.pos().x()));
        for (const auto& a: armors.armors) {
            if (a.number == target.target_number) {
                candidates.emplace_back(a);
            }
        }

        if (candidates.empty())
            return false;

        int updated = 0;
        const auto matches = target.match(candidates, camera_info, camera_cv_in_odom);

        for (const auto& m: matches) {
            if (m.second.color == ArmorColor::NONE || m.second.color == ArmorColor::PURPLE) {
                // if (++is_none_purple_count_ > 100)
                continue;
            } else {
                is_none_purple_count_ = 0;
            }

            if (target.update(m, armors.timestamp, camera_info, camera_cv_in_odom))
                ++updated;
        }

        return updated > 0;
    }
    void update_fsm(bool found, size_t i) noexcept {
        auto& s = target_buf_[i].track_state;

        switch (s.tracker_state) {
            case ArmorTarget::TrackState::DETECTING:
                if (!found) {
                    s.detect_count = 0;
                    s.tracker_state = ArmorTarget::TrackState::LOST;
                    return;
                }
                if (++s.detect_count > cfg_.tracking_thres) {
                    s.detect_count = 0;
                    s.tracker_state = ArmorTarget::TrackState::TRACKING;
                }
                return;

            case ArmorTarget::TrackState::TRACKING:
                if (!found) {
                    s.tracker_state = ArmorTarget::TrackState::TEMP_LOST;
                    s.lost_count = 1;
                }
                return;

            case ArmorTarget::TrackState::TEMP_LOST:
                if (found) {
                    s.lost_count = 0;
                    s.tracker_state = ArmorTarget::TrackState::TRACKING;
                    return;
                }
                if (++s.lost_count > lost_thres_) {
                    s.lost_count = 0;
                    s.tracker_state = ArmorTarget::TrackState::LOST;
                }
                return;

            default:
                return;
        }

        if (found)
            ++found_count_;
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

ArmorTarget ArmorTracker::track(
    Armors& armors,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom,
    int frame_id
) {
    return _impl->track(armors, camera_info, camera_cv_in_odom, frame_id);
}
int ArmorTracker::get_count() {
    return _impl->found_count_;
}
void ArmorTracker::reset_count() {
    _impl->found_count_ = 0;
}
} // namespace awakening::auto_aim