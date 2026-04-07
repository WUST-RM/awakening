#pragma once

#include "utils/logger.hpp"
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
namespace awakening::web {
inline void write_shm(const cv::Mat& img) {
    constexpr size_t shm_max_size = 2 * 1024 * 1024;
    struct writer {
        writer(const char* name, mode_t mode = 0666) {
            fd_ = shm_open(name, O_CREAT | O_RDWR, mode);
            if (fd_ == -1) {
                // std::cerr << "[SHM] shm_open failed\n";
                AWAKENING_ERROR("shm: {} open failed", name);
                return;
            }

            if (ftruncate(fd_, shm_max_size) == -1) {
                AWAKENING_ERROR("shm: {} ftruncate failed", name);
                close(fd_);
                fd_ = -1;
                return;
            }

            ptr_ = mmap(nullptr, shm_max_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);

            if (ptr_ == MAP_FAILED) {
                AWAKENING_ERROR("shm: {} mmap failed", name);
                close(fd_);
                fd_ = -1;
                ptr_ = nullptr;
            }
        }
        ~writer() {
            if (ptr_)
                munmap(ptr_, shm_max_size);
            if (fd_ != -1)
                close(fd_);
        }
        void write(const cv::Mat& img) {
            if (!ptr_)
                return;

            static const std::vector<int> jpeg_params = { cv::IMWRITE_JPEG_QUALITY, 75 };

            std::vector<uchar> buf;
            cv::imencode(".jpg", img, buf, jpeg_params);

            if (buf.size() + 4 > shm_max_size)
                return;

            uint32_t size = static_cast<uint32_t>(buf.size());
            std::memcpy(ptr_, &size, 4);
            std::memcpy(static_cast<char*>(ptr_) + 4, buf.data(), size);
        }
        int fd_ { -1 };
        void* ptr_ { nullptr };
    };
    static writer w("/awaking_frame");
    w.write(img);
}
struct LogBuffer {
    std::mutex mtx;
    nlohmann::json j;
    bool dirty = false;

    std::ofstream file { "/dev/shm/awakening_log.json" };
    void flush() {
        static auto last_flush = std::chrono::steady_clock::now();

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_flush).count() < 0.01)
            return;

        std::lock_guard<std::mutex> lock(mtx);

        if (!dirty)
            return;

        try {
            const std::string path = "/dev/shm/awakening_log.json";
            const std::string tmp_path = path + ".tmp";

            {
                std::ofstream tmp_file(tmp_path, std::ios::out | std::ios::trunc);
                if (!tmp_file.is_open())
                    return;

                tmp_file << j.dump(2);
                tmp_file.flush();
            }

            std::rename(tmp_path.c_str(), path.c_str());

            dirty = false;
            last_flush = now;

        } catch (...) {
        }
    }
};
template<typename T>
inline auto val(const T& v) {
    return +v;
}
inline LogBuffer& get_log_buffer() {
    static LogBuffer buf;
    return buf;
}
template<typename Func>
inline void write_log(const char* key, Func&& f) {
    auto& buf = get_log_buffer();
    // static std::mutex mtx;
    {
        std::lock_guard<std::mutex> lock(buf.mtx);
        auto& j = buf.j[key];
        f(j);
        buf.dirty = true;
    }

    buf.flush();
}
template<typename T, int MAX_N>
class DatasStream {
public:
    DatasStream(const std::string& n, nlohmann::json& _j): j(_j) {
        name = n;
    }
    void handle_once(const T& t) {
        log_data.push_back(t);
        trim();
        insert_data(j);
    }
    void push_back(const T& t) {
        log_data.push_back(t);
    }
    void trim() {
        while (log_data.size() > MAX_N) {
            log_data.erase(log_data.begin());
        }
    }
    void insert_data(nlohmann::json& _j) {
        _j[name] = log_data;
    }
    void clear() {
        log_data.clear();
    }

private:
    std::string name;
    std::vector<T> log_data;
    nlohmann::json& j;
};
struct DebugDatas {
    nlohmann::json j;
#define DEBUG_LOG_LIST(X) \
    X(double, 100, time) \
    X(double, 100, yaw) \
    X(double, 100, pitch) \
    X(double, 100, target_yaw) \
    X(double, 100, target_pitch) \
    X(double, 100, gimbal_yaw) \
    X(double, 100, gimbal_pitch) \
    X(double, 100, control_v_yaw) \
    X(double, 100, control_v_pitch) \
    X(double, 100, control_a_yaw) \
    X(double, 100, control_a_pitch) \
    X(double, 100, fly_time) \
    X(double, 100, target_v_yaw)

#define GEN_LOG(TYPE, SIZE, NAME) DatasStream<TYPE, SIZE> NAME##_log { #NAME, j };

#define X(TYPE, SIZE, NAME) GEN_LOG(TYPE, SIZE, NAME)
    DEBUG_LOG_LIST(X)
#undef X

    void clear() {
#define X(TYPE, SIZE, NAME) NAME##_log.clear();
        DEBUG_LOG_LIST(X)
#undef X
    }
    void write() {
        const std::string path = "/dev/shm/awakening_data.json";
        const std::string tmp_path = path + ".tmp";

        {
            std::ofstream tmp_file(tmp_path, std::ios::out | std::ios::trunc);
            if (!tmp_file.is_open())
                return;

            tmp_file << j.dump(2);
            tmp_file.flush();
        }

        std::rename(tmp_path.c_str(), path.c_str());
    }
};
} // namespace awakening::web