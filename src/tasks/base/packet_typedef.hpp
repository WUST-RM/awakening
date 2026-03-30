#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <wust_vl/common/drivers/serial_driver.hpp>

namespace awakening {

// ===================== 常量 =====================
constexpr const char* TARGET_TOPIC = "vision_target";
constexpr const char* NAV_STATE_TOPIC = "rose_state";
constexpr const char* MODE_TOPIC = "sentry_mode";
constexpr const char* ROBO_STATE_TOPIC = "robo_state";
constexpr const char* GOAL_TOPIC = "rose_goal";

// ===================== packed 安全访问 =====================
template<typename T>
inline auto val(const T& v) {
    return +v; // 强制值语义（避免 packed 引用问题）
}

// ===================== 日志缓冲 =====================
struct SerialLogBuffer {
    std::mutex mtx;
    nlohmann::json j;
    bool dirty = false;

    std::ofstream file { "/dev/shm/serial_log.json" };
};

inline SerialLogBuffer& getLogBuffer() {
    static SerialLogBuffer buf;
    return buf;
}

// ===================== FPS =====================
template<typename Tag>
inline void updateFPS(nlohmann::json& j) {
    static int frame_count = 0;
    static double fps = 0.0;
    static auto last_time = std::chrono::steady_clock::now();

    ++frame_count;

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - last_time).count();

    if (elapsed >= 1.0) {
        fps = frame_count / elapsed;
        frame_count = 0;
        last_time = now;
    }

    j["fps"] = fps;
}

// ===================== flush =====================
inline void flushSerialLog() {
    static auto last_flush = std::chrono::steady_clock::now();
    auto& buf = getLogBuffer();

    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - last_flush).count() < 0.05)
        return;

    std::lock_guard<std::mutex> lock(buf.mtx);

    if (!buf.dirty || !buf.file.is_open())
        return;

    buf.file.seekp(0);
    buf.file << buf.j.dump(2);
    buf.file.flush();

    buf.dirty = false;
    last_flush = now;
}

template<typename Func>
inline void writeLog(const char* key, Func&& f) {
    auto& buf = getLogBuffer();

    {
        std::lock_guard<std::mutex> lock(buf.mtx);
        auto& j = buf.j[key];
        f(j);
        buf.dirty = true;
    }

    flushSerialLog();
}

struct ReceiveRobotData {
    static constexpr uint8_t ID = 0x02;

    uint8_t cmd_ID;
    uint32_t time_stamp;

    float yaw, pitch, roll;
    float yaw_vel, pitch_vel, roll_vel;
    float v_x, v_y, v_z;

    float bullet_speed;
    uint8_t detect_color;

    static std::optional<ReceiveRobotData> create(const uint8_t* data, std::size_t len) {
        if (len != sizeof(ReceiveRobotData) || data[0] != ID)
            return std::nullopt;

        ReceiveRobotData out;
        std::memcpy(&out, data, sizeof(out));
        return out;
    }

    void updateSerialLog() {
        writeLog("robo", [&](auto& j) {
            updateFPS<ReceiveRobotData>(j);

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
        });
    }

} __attribute__((packed));

// ----------- 接收：哨兵 -----------
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

    void updateSerialLog() {
        writeLog("sentry", [&](auto& j) {
            updateFPS<ReceiveSentryData>(j);

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

// ----------- 发送：机器人 -----------
struct SendRobotCmdData {
    static constexpr uint8_t ID = 0x01;

    uint8_t cmd_ID = ID;
    uint32_t time_stamp;

    uint8_t appear;
    uint8_t shoot_rate = 3;

    float pitch, yaw;
    float target_yaw, target_pitch;

    float enable_yaw_diff, enable_pitch_diff;
    float v_yaw, v_pitch;
    float a_yaw, a_pitch;

    uint8_t detect_color;

    void updateSerialLog() {
        writeLog("cmd", [&](auto& j) {
            j["appear"] = val(appear);
            j["shoot_rate"] = val(shoot_rate);

            j["yaw"] = val(yaw);
            j["pitch"] = val(pitch);

            j["target_yaw"] = val(target_yaw);
            j["target_pitch"] = val(target_pitch);

            j["v_yaw"] = val(v_yaw);
            j["v_pitch"] = val(v_pitch);

            j["a_yaw"] = val(a_yaw);
            j["a_pitch"] = val(a_pitch);

            j["enable_yaw_diff"] = val(enable_yaw_diff);
            j["enable_pitch_diff"] = val(enable_pitch_diff);

            j["detect_color"] = (detect_color == 0 ? "Red" : "Blue");
        });
    }

} __attribute__((packed));

// ----------- 发送：导航 -----------
struct SendNavCmdData {
    static constexpr uint8_t ID = 0x02;

    uint8_t cmd_ID = ID;
    uint32_t time_stamp;

    float vx, vy, wz;

    void updateSerialLog() {
        writeLog("nav_cmd", [&](auto& j) {
            j["vx"] = val(vx);
            j["vy"] = val(vy);
            j["wz"] = val(wz);
        });
    }

} __attribute__((packed));

} // namespace awakening