#pragma once
#include "utils/common/image.hpp" // 自定义 ImageFrame
#include "utils/logger.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>

namespace awakening {

using Clock = std::chrono::steady_clock;

inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
        .count();
}

template<typename Tag>
struct RecordTagTraits;

struct RecordHeader {
    uint64_t timestamp;
    uint32_t tag_id;
    uint32_t payload_size;
};

class Recorder {
public:
    explicit Recorder(const std::string& path) {
        ofs_.open(path, std::ios::binary | std::ios::out);
        if (!ofs_.is_open()) {
            throw std::runtime_error("Failed to open file for recording: " + path);
        }
    }

    ~Recorder() {
        ofs_.close();
    }

    template<typename Tag>
    void record(const typename RecordTagTraits<Tag>::Type& data) {
        auto payload = RecordTagTraits<Tag>::serialize(data);

        RecordHeader h;
        h.timestamp = now_ns();
        h.tag_id = RecordTagTraits<Tag>::id;
        h.payload_size = static_cast<uint32_t>(payload.size());

        ofs_.write(reinterpret_cast<char*>(&h), sizeof(h));
        ofs_.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    }

private:
    std::ofstream ofs_;
};

class Player {
public:
    using RawCallback = std::function<void(const std::vector<uint8_t>&)>;

    explicit Player(const std::string& path) {
        ifs_.open(path, std::ios::binary);
        if (!ifs_.is_open()) {
            throw std::runtime_error("Failed to open file for playback: " + path);
        }
    }

    template<typename Tag>
    void subscribe(std::function<void(typename RecordTagTraits<Tag>::Type&&)> cb) {
        subs_[RecordTagTraits<Tag>::id] = [cb](const std::vector<uint8_t>& data) {
            auto obj = RecordTagTraits<Tag>::deserialize(data);
            cb(std::move(obj));
        };
    }

    void play(double speed = 1.0) {
        uint64_t first_ts = 0;
        uint64_t start_wall = now_ns();

        while (true) {
            RecordHeader h;
            // 读取 header
            if (!ifs_.read(reinterpret_cast<char*>(&h), sizeof(h)))
                break;

            // 读取 payload
            std::vector<uint8_t> payload(h.payload_size);
            if (!ifs_.read(reinterpret_cast<char*>(payload.data()), h.payload_size)) {
                AWAKENING_WARN(" [Player] Failed to read payload, maybe file truncated");
                break;
            }

            // 计算时间等待
            if (first_ts == 0)
                first_ts = h.timestamp;
            uint64_t dt = h.timestamp - first_ts;
            uint64_t target = start_wall + static_cast<uint64_t>(dt / speed);
            while (now_ns() < target)
                std::this_thread::sleep_for(std::chrono::microseconds(100));

            dispatch(h.tag_id, payload);
        }
    }

private:
    void dispatch(uint32_t tag_id, const std::vector<uint8_t>& payload) {
        auto it = subs_.find(tag_id);
        if (it != subs_.end()) {
            it->second(payload);
        }
    }

private:
    std::ifstream ifs_;
    std::unordered_map<uint32_t, RawCallback> subs_;
};

} // namespace awakening