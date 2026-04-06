#pragma once
#include "angles.h"
#include "motion_model.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/web.hpp"
#include "utils/common/type_common.hpp"
#include <chrono>
#include <string>
#include <vector>
namespace awakening::auto_aim {
using namespace armor_motion_model;
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
    double yp_r;
    double dis_r_front;
    double dis_r_side;
    double dis2_r_ratio;
    double yaw_r_base_front;
    double yaw_r_base_side;
    double yaw_r_log_ratio;
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
        yp_r = config["yp_r"].as<double>();
        dis_r_front = config["dis_r_front"].as<double>();
        dis_r_side = config["dis_r_side"].as<double>();
        dis2_r_ratio = config["dis2_r_ratio"].as<double>();
        yaw_r_base_front = config["yaw_r_base_front"].as<double>();
        yaw_r_base_side = config["yaw_r_base_side"].as<double>();
        yaw_r_log_ratio = config["yaw_r_log_ratio"].as<double>();
    }
};
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
    };

    ArmorTarget() = default;
    ArmorTarget(const Armor& a, const ArmorTrackerCfg& c, const TimePoint& timestamp, int frame_id);
    [[nodiscard]] cv::Rect expanded(
        const TimePoint& timestamp,
        const ISO3& camera_cv_in_odom,
        const CameraInfo& camera_info,
        const cv::Size& image_size
    ) const noexcept;
    [[nodiscard]] Eigen::Matrix<double, Z_N, Z_N>
    measurement_covariance(const Eigen::Matrix<double, Z_N, 1>& z) const noexcept;
    [[nodiscard]] Eigen::Matrix<double, X_N, X_N> process_noise(double dt) const noexcept;
    [[nodiscard]] Eigen::Matrix<double, Z_N, 1> get_measurement(const Armor& a) noexcept;
    void predict_ekf(const TimePoint& timestamp);
    bool update(const std::pair<int, Armor>& a, const TimePoint& timestamp) noexcept;
    std::vector<std::pair<int, Armor>> match(const std::vector<Armor>& armors) noexcept;
    Measure::Ctx measure_ctx;
    RobotStateESEKF esekf;
    ArmorTrackerCfg cfg;
    State get_target_state() const {
        return target_state;
    }
    template<typename F>
    void set_target_state(F&& f) {
        f(target_state);
    }
    bool is_inited = false;
    bool jumped = false;
    TrackState track_state;
    TimePoint last_update;
    ArmorClass target_number = ArmorClass::UNKNOWN;
    [[nodiscard]] inline bool check() const noexcept {
        auto v = track_state.is_tracking()
            && std::chrono::duration<double>(Clock::now() - last_update).count()
                < cfg.lost_time_thres;
        return v;
    }
    [[nodiscard]] int armor_num() {
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