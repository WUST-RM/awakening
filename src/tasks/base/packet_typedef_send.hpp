#pragma once

#include "tasks/base/web.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <utility>
#include <vector>
#include <wust_vl/common/drivers/serial_driver.hpp>

namespace awakening {

constexpr const char* TARGET_TOPIC = "vision_target";
constexpr const char* NAV_STATE_TOPIC = "rose_state";
constexpr const char* MODE_TOPIC = "sentry_mode";
constexpr const char* ROBO_STATE_TOPIC = "robo_state";
constexpr const char* GOAL_TOPIC = "rose_goal";

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
    float vx, vy, wz;

} __attribute__((packed));

struct SentryyRefereeSend {
    static constexpr uint8_t ID = 0x03;
    uint8_t set_current_pose; //1 为进攻姿态，2 为防御姿态，3 为移动姿态
} __attribute__((packed));
struct SentryySendYUNTAISHOU {
    static constexpr uint8_t ID = 0x04;
    uint8_t user_data[30];
} __attribute__((packed));

} // namespace awakening