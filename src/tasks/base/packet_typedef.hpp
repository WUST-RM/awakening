#pragma once

#include "tasks/base/web.hpp"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <vector>
#include <wust_vl/common/drivers/serial_driver.hpp>

namespace awakening {

constexpr const char* TARGET_TOPIC = "vision_target";
constexpr const char* NAV_STATE_TOPIC = "rose_state";
constexpr const char* MODE_TOPIC = "sentry_mode";
constexpr const char* ROBO_STATE_TOPIC = "robo_state";
constexpr const char* GOAL_TOPIC = "rose_goal";

struct ReceiveRobotData {
    static constexpr uint8_t ID = 0x02;

    uint8_t cmd_ID;
    uint32_t time_stamp;

    float yaw, pitch,
        roll; //坐标系定义： +x:前，+y:左，+z：上，旋转角绕轴逆时针正，顺时针负，旋转顺序ZYX
    float yaw_vel, pitch_vel, roll_vel;
    float v_x, v_y, v_z;

    float bullet_speed;
    uint8_t detect_color; //0 r 1 b
    uint32_t bullet_count; //发出弹+1

    static std::optional<ReceiveRobotData> create(const std::vector<uint8_t>& data) {
        if (data.size() != sizeof(ReceiveRobotData) || data[0] != ID)
            return std::nullopt;

        ReceiveRobotData out;
        std::memcpy(&out, data.data(), sizeof(out));
        return out;
    }

    void update_log() {
        using namespace web;
        write_log("robo", [&](auto& j) {
            j["timestamp"] = val(time_stamp);

            j["yaw"] = val(yaw);
            j["pitch"] = val(pitch);
            j["roll"] = val(roll);

            j["yaw_vel"] = val(yaw_vel);
            j["pitch_vel"] = val(pitch_vel);
            j["roll_vel"] = val(roll_vel);

            j["v_x"] = val(v_x);
            j["v_y"] = val(v_y);
            j["v_z"] = val(v_z);

            j["bullet_speed"] = val(bullet_speed);
            j["detect_color"] = (detect_color == 0 ? "Red" : "Blue");
            j["bullet_count"] = val(bullet_count);
        });
    }

} __attribute__((packed));

struct ReceiveSentryData {
    static constexpr uint8_t ID = 0x03;

    uint8_t cmd_ID;
    uint32_t time_stamp;

    float big_yaw_in_world;
    int game_time;
    int max_health;
    int cur_health;
    int cur_bullet;
    uint8_t center_state;

    static std::optional<ReceiveSentryData> create(const uint8_t* data, std::size_t len) {
        if (len != sizeof(ReceiveSentryData) || data[0] != ID)
            return std::nullopt;

        ReceiveSentryData out;
        std::memcpy(&out, data, sizeof(out));
        return out;
    }

    void update_log() {
        using namespace web;
        write_log("sentry", [&](auto& j) {
            j["timestamp"] = val(time_stamp);
            j["big_yaw_in_world"] = val(big_yaw_in_world);

            j["game_time"] = val(game_time);
            j["max_health"] = val(max_health);
            j["cur_health"] = val(cur_health);
            j["cur_bullet"] = val(cur_bullet);

            j["center_state"] = val(center_state);
        });
    }

} __attribute__((packed));

struct SendRobotCmdData {
    static constexpr uint8_t ID = 0x01;

    uint8_t cmd_ID = ID;
    uint32_t time_stamp;

    uint8_t appear;

    float pitch, yaw;
    float target_yaw, target_pitch;

    float enable_yaw_diff, enable_pitch_diff;
    float v_yaw, v_pitch;
    float a_yaw, a_pitch;

    uint8_t detect_color;

} __attribute__((packed));

struct SendNavCmdData {
    static constexpr uint8_t ID = 0x02;

    uint8_t cmd_ID = ID;
    uint32_t time_stamp;

    float vx, vy, wz;

} __attribute__((packed));

} // namespace awakening