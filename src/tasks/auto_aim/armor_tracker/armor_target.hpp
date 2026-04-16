#pragma once
#include "angles.h"
#include "motion_model.hpp"
#include "motion_model_point.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/web.hpp"
#include "utils/common/type_common.hpp"
#include <chrono>
#include <optional>
#include <string>
#include <utility>
#include <vector>
namespace awakening::auto_aim {
using namespace armor_point_motion_model;
struct ArmorTrackerCfg {
    int esekf_iter_num;
    double lost_time_thres;
    int tracking_thres;
    double max_yaw_diff_deg;
    double max_dis_diff;
    double match_gate;
    double qyaw_common;
    double qyaw_output;
    Vec3 qxyz_common;
    Vec3 qxyz_output;
    double q_r;
    double q_l;
    double q_h;
    double q_outpost_dz;
    double r_uv;
    void load(const YAML::Node& config) {
        esekf_iter_num = config["esekf_iter_num"].as<int>();
        lost_time_thres = config["lost_time_thres"].as<double>();
        tracking_thres = config["tracking_thres"].as<int>();
        max_yaw_diff_deg = config["max_yaw_diff_deg"].as<double>();
        max_dis_diff = config["max_dis_diff"].as<double>();
        match_gate = config["match_gate"].as<double>();
        qyaw_common = config["qyaw_common"].as<double>();
        qyaw_output = config["qyaw_output"].as<double>();
        auto qxyz_common_vec = config["qxyz_common"].as<std::vector<double>>();
        qxyz_common << qxyz_common_vec[0], qxyz_common_vec[1], qxyz_common_vec[2];
        auto qxyz_output_vec = config["qxyz_output"].as<std::vector<double>>();
        qxyz_output << qxyz_output_vec[0], qxyz_output_vec[1], qxyz_output_vec[2];
        q_r = config["q_r"].as<double>();
        q_l = config["q_l"].as<double>();
        q_h = config["q_h"].as<double>();
        q_outpost_dz = config["q_outpost_dz"].as<double>();
        r_uv = config["r_uv"].as<double>();
    }
};
static inline int GOBAL_ID = 0;
class ArmorTarget {
public:
    struct TrackState {
        enum State {
            LOST,
            DETECTING,
            TRACKING,
            TEMP_LOST,
        };
        State tracker_state = LOST;
        int detect_count = 0;
        int lost_count = 0;
        static inline std::string string_by_state(State state) {
            constexpr const char* details[] = { "LOST", "DETECTING", "TRACKING", "TEMP_LOST" };
            return std::string(details[state]);
        }
        bool is_tracking() const noexcept {
            return tracker_state == TRACKING || tracker_state == TEMP_LOST;
        }
        void reset() {
            tracker_state = LOST;
            detect_count = 0;
            lost_count = 0;
        }
    };
    enum MeasureType { ARMOR, R_LIGHT, L_LIGHT };
    ArmorTarget() = default;
    void armor_pnp(Armor& a, const CameraInfo& camera_info, const ISO3& camera_cv_in_odom)
        const noexcept;
    void reset(
        const Armor& a,
        const ArmorTrackerCfg& c,
        const TimePoint& timestamp,
        int frame_id,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom
    );
    [[nodiscard]] cv::Rect expanded_one_one(
        const TimePoint& timestamp,
        const ISO3& camera_cv_in_odom,
        const CameraInfo& camera_info,
        const cv::Size& image_size
    ) const noexcept;
    [[nodiscard]] cv::Rect expanded(
        const TimePoint& timestamp,
        const ISO3& camera_cv_in_odom,
        const CameraInfo& camera_info,
        const cv::Size& image_size
    ) const noexcept;
    [[nodiscard]] Eigen::Matrix<double, Z_N, Z_N>
    measurement_covariance(const Eigen::Matrix<double, Z_N, 1>& z) const noexcept;
    [[nodiscard]] Eigen::Matrix<double, X_N, X_N> process_noise(double dt) const noexcept;
    [[nodiscard]] Eigen::Matrix<double, Z_N, 1> get_measurement(Armor& a) const noexcept;
    [[nodiscard]] Eigen::Matrix<double, Z_N, 1>
    get_measurement(Armor& a, const VecZ& z_pred, MeasureType mt) const noexcept;
    void predict_ekf(const TimePoint& timestamp);
    bool update(
        const std::pair<int, Armor>& a,
        const TimePoint& timestamp,
        const CameraInfo& camera_info,
        const ISO3& camera_cv_in_odom
    );
    std::vector<std::pair<int, Armor>>
    match(std::vector<Armor>& armors, const CameraInfo& camera_info, const ISO3& camera_cv_in_odom)
        const noexcept;
    Measure::Ctx measure_ctx;
    std::optional<RobotStateESEKF> esekf;
    ArmorTrackerCfg cfg;
    State get_target_state() const {
        return target_state;
    }
    template<typename F>
    void set_target_state(F&& f) {
        this_id = GOBAL_ID++;
        f(target_state);
    }
    bool is_inited = false;
    bool jumped = false;
    int last_match_id = -1;
    std::optional<std::pair<bool, std::vector<bool>>> outpost_has_all_and_has_set_ids;
    TrackState track_state;
    TimePoint last_update;
    ArmorClass target_number = ArmorClass::UNKNOWN;
    int this_id = -1;
    [[nodiscard]] inline ArmorTarget fast_copy_without_ekf() const noexcept {
        ArmorTarget target;
        target.target_number = this->target_number;
        target.target_state = this->target_state;
        target.last_update = this->last_update;
        target.cfg = this->cfg;
        target.track_state = this->track_state;
        target.is_inited = this->is_inited;
        target.jumped = this->jumped;
        target.last_match_id = this->last_match_id;
        target.outpost_has_all_and_has_set_ids = this->outpost_has_all_and_has_set_ids;
        target.this_id = this->this_id;
        return target;
    }
    [[nodiscard]] inline bool check() const noexcept {
        auto v = track_state.is_tracking()
            && std::chrono::duration<double>(Clock::now() - last_update).count()
                < cfg.lost_time_thres;
        return v;
    }
    [[nodiscard]] int armor_num() const noexcept {
        return measure_ctx.armor_num;
    }
    void write_log() {
        web::write_log("armor_target", [&](auto& j) {
            j["timestamp"] = static_cast<int>(
                std::chrono::duration<double>(last_update.time_since_epoch()).count()
            );
            j["target_number"] = string_by_armor_class(target_number);
            j["track_state"] = TrackState::string_by_state(track_state.tracker_state);
            auto& j_target_state = j["target_state"];
            j_target_state["cx"] = web::val(target_state.pos().x());
            j_target_state["cy"] = web::val(target_state.pos().y());
            j_target_state["cz"] = web::val(target_state.pos().z());
            j_target_state["vx"] = web::val(target_state.vel().x());
            j_target_state["vy"] = web::val(target_state.vel().y());
            j_target_state["vz"] = web::val(target_state.vel().z());
            j_target_state["yaw"] = web::val(target_state.yaw());
            j_target_state["vyaw"] = web::val(target_state.vyaw());
            j_target_state["r"] = web::val(target_state.r());
            j_target_state["l"] = web::val(target_state.l());
            j_target_state["h"] = web::val(target_state.h());
        });
    }

private:
    State target_state;
};
} // namespace awakening::auto_aim