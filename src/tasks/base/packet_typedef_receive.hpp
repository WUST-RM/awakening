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

struct ReceiveRobotData {
    static constexpr uint8_t ID = 0x02;

    uint8_t cmd_ID;
    uint32_t time_stamp_pc; //收到的上一包的PC时间戳
    uint32_t time_stamp_receive_micro; //收到的上一包时STM32的时间戳
    uint32_t time_stamp_send_micro; //发送此包时STM32的时间戳

    float yaw, pitch,
        roll; //坐标系定义： +x:前，+y:左，+z：上，+yaw 左，+pitch 下。+roll 右倾，旋转顺序ZYX
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
            j["timestamp_pc"] = val(time_stamp_pc);
            j["timestamp_receive_micro"] = val(time_stamp_receive_micro);
            j["timestamp_send_micro"] = val(time_stamp_send_micro);

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

struct SentryJointState {
    static constexpr uint8_t ID = 0x04;

    uint8_t cmd_ID;
    float big_yaw_in_world;
    static std::optional<SentryJointState> create(const std::vector<uint8_t>& data) {
        if (data.size() != sizeof(SentryJointState) || data[0] != ID)
            return std::nullopt;

        SentryJointState out;
        std::memcpy(&out, data.data(), sizeof(out));
        return out;
    }
};
struct SentryRefereeReceive {
    static constexpr uint8_t ID = 0x05;
    uint8_t cmd_ID;
    uint8_t robo_id;
    uint16_t current_hp;
    uint16_t max_hp;
    uint16_t allowance_bullets;
    uint16_t fort_allowance_bullets;
    uint8_t current_pose; //1 为进攻姿态，2 为防御姿态，3 为移动姿态
    int32_t game_time; //没开比赛发-1 ，记得把裁判系统包的remain_time转换
    uint16_t ally_outpost_hp;
    uint16_t ally_base_hp;
    uint8_t ally_outpost_occ_state; //0 为未被占领，1 为被己方占领，2 为被对方占领，3 为被双方占领
    uint8_t ally_fort_occ_state; //0 为未被占领，1 为被己方占领，2 为被对方占领，3 为被双方占领
    static std::optional<SentryRefereeReceive> create(const std::vector<uint8_t>& data) {
        if (data.size() != sizeof(SentryRefereeReceive) || data[0] != ID)
            return std::nullopt;

        SentryRefereeReceive out;
        std::memcpy(&out, data.data(), sizeof(out));
        return out;
    }
    void update_log() {
        using namespace web;
        write_log("referee", [&](auto& j) {
            j["robo_id"] = val(robo_id);
            j["current_hp"] = val(current_hp);
            j["max_hp"] = val(max_hp);
            j["allowance_bullets"] = val(allowance_bullets);
            j["fort_allowance_bullets"] = val(fort_allowance_bullets);
            j["current_pose"] = val(current_pose);
            j["game_time"] = val(game_time);
            j["ally_outpost_hp"] = val(ally_outpost_hp);
            j["ally_base_hp"] = val(ally_base_hp);
            j["ally_outpost_occ_state"] = val(ally_outpost_occ_state);
            j["ally_fort_occ_state"] = val(ally_fort_occ_state);
        });
    }
};

} // namespace awakening