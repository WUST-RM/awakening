#pragma once
#include "utils/common/image.hpp"
#include "utils/logger.hpp"
#include <stdexcept>
#include <thread>
#include <yaml-cpp/node/node.h>
namespace awakening {
class UVCCamera {
public:
    UVCCamera(const YAML::Node& config) {
        config_ = config;
        AWAKENING_INFO("try get device_name by ls -l /dev/v4l/by-id/  ls -l /dev/v4l/by-path/");
        try {
            device_name_ = config["device_name"].as<std::string>();

            fps_ = config["fps"].as<int>();

            width_ = config["width"].as<int>();

            height_ = config["height"].as<int>();

            exposure_ = config["exposure"].as<double>();

            gain_ = config["gain"].as<double>();

            gamma_ = config["gamma"].as<double>();

        } catch (const std::exception& e) {
            AWAKENING_ERROR("Failed to load config: {}", e.what());
            throw std::runtime_error("Failed to load UVC config");
        }
    }
    ~UVCCamera() {
        stop();
    }
    void stop() {
        if (!running_)
            return;

        running_ = false;

        if (daemon_thread_.joinable())
            daemon_thread_.join();

        cap_.release();

        AWAKENING_INFO("uvc: {} stopped", device_name_);
    }
    bool set_and_check(
        cv::VideoCapture& cap,
        int prop,
        double value,
        const std::string& name,
        double tol = 1e-3
    ) {
        if (!cap.set(prop, value)) {
            AWAKENING_WARN("{} set() failed", name);
            return false;
        }

        double actual = cap.get(prop);

        if (std::abs(actual - value) > tol) {
            AWAKENING_WARN("{} mismatch. requested={} actual={}", name, value, actual);
            return false;
        }

        AWAKENING_INFO("{} OK = {}", name, actual);
        return true;
    }
    void start() {
        if (running_) {
            AWAKENING_WARN("Camera already running.");
            return;
        }

        AWAKENING_INFO("Starting camera: {}", device_name_);
        if (!cap_.open(device_name_, cv::CAP_V4L2)) {
            AWAKENING_ERROR("Failed to open camera.");
            return;
        }

        set_and_check(cap_, cv::CAP_PROP_AUTO_EXPOSURE, 1, "AUTO_EXPOSURE");
        set_and_check(
            cap_,
            cv::CAP_PROP_FOURCC,
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            "FOURCC"
        );

        set_and_check(cap_, cv::CAP_PROP_FRAME_WIDTH, width_, "WIDTH");
        set_and_check(cap_, cv::CAP_PROP_FRAME_HEIGHT, height_, "HEIGHT");
        set_and_check(cap_, cv::CAP_PROP_EXPOSURE, exposure_, "EXPOSURE");
        set_and_check(cap_, cv::CAP_PROP_GAIN, gain_, "GAIN");
        set_and_check(cap_, cv::CAP_PROP_GAMMA, gamma_, "GAMMA");
        set_and_check(cap_, cv::CAP_PROP_FPS, fps_, "FPS");
        running_ = true;
    }
    ImageFrame read() {
        ImageFrame f;
        if (!cap_.isOpened()) {
            AWAKENING_ERROR("Camera is not opened.");
        }
        cv::Mat mat;

        if (cap_.read(mat)) {
            f.src_img = std::move(mat);
            f.timestamp = std::chrono::steady_clock::now();
            f.format = PixelFormat::BGR;
        } else {
            AWAKENING_WARN("uvc: {} read_image failed.", device_name_);
        }
        return f;
    }
    void restart() {
        AWAKENING_WARN("Restarting camera.");

        cap_.release();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        if (!cap_.open(device_name_)) {
            AWAKENING_ERROR("Camera reopen failed.");
            return;
        }

        AWAKENING_INFO("Camera reopened successfully.");
    }
    std::string device_name_;
    int fps_;
    int width_;
    int height_;
    double exposure_;
    double gain_;
    double gamma_;
    YAML::Node config_;
    bool running_ = false;
    cv::VideoCapture cap_;
    std::thread daemon_thread_;
    std::chrono::steady_clock::time_point last_frame_time_;
};
} // namespace awakening