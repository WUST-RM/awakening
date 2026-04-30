#include "mv_camera.hpp"
namespace awakening {
void MvCamera::stop() {
    if (!running_) {
        return;
    }
    running_ = false;
    if (daemon_thread_.joinable()) {
        daemon_thread_.join();
    }

    CameraUnInit(h_camera_);

    AWAKENING_INFO("mv_camera has stop ");
}
void MvCamera::restart() {
    AWAKENING_WARN("Restarting camera");
    CameraUnInit(h_camera_);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    load(config_);

    CameraPlay(h_camera_);

    AWAKENING_INFO("Camera restarted successfully!");
    return;
}
void MvCamera::init() {
    load(config_);
}
void MvCamera::load(const YAML::Node& config) {
    initialize_camera();
    CameraGetCapability(h_camera_, &t_capability_);
    CameraSetAeState(h_camera_, false);
    CameraSetIspOutFormat(h_camera_, CAMERA_MEDIA_TYPE_BGR8);
    CameraSetTriggerMode(h_camera_, 0);
    auto exposure_time = config["exposure_time"].as<double>();
    set_ExposureTime(exposure_time);
    if(config["r_gain"]&&config["b_gain"]&&config["g_gain"]) {
        auto r_gain = config["r_gain"].as<double>();
        auto b_gain = config["b_gain"].as<double>();
        auto g_gain = config["g_gain"].as<double>();
        set_rgb_gain(r_gain, b_gain, g_gain);
    }
    if(config["analog_gain"]) {
        auto analog_gain = config["analog_gain"].as<double>();
        set_analog_gain(analog_gain);
    }
    if(config["gamma"]) {
        auto gamma = config["gamma"].as<double>();
        set_gamma(gamma);
    }
}
void MvCamera::set_ExposureTime(double exposure_time) {
    double exposure_line_time;
    CameraGetExposureLineTime(h_camera_, &exposure_line_time);
    exposure_time = std::clamp(
        exposure_time,
        t_capability_.sExposeDesc.uiExposeTimeMin * exposure_line_time,
        t_capability_.sExposeDesc.uiExposeTimeMax * exposure_line_time
    );
    CameraSetExposureTime(h_camera_, exposure_time);
    AWAKENING_INFO(
        "MvCamera: set exposure_time {}, min {} max {}",
        exposure_time,
        t_capability_.sExposeDesc.uiExposeTimeMin * exposure_line_time,
        t_capability_.sExposeDesc.uiExposeTimeMax * exposure_line_time
    );
}
double MvCamera::get_ExposureTime() {
    double exposure_time;
    CameraGetExposureTime(h_camera_, &exposure_time);
    return exposure_time;
}
void MvCamera::set_rgb_gain(int r_gain, int b_gain, int g_gain) {
    r_gain = std::clamp(
        r_gain,
        t_capability_.sRgbGainRange.iRGainMin,
        t_capability_.sRgbGainRange.iRGainMax
    );
    b_gain = std::clamp(
        b_gain,
        t_capability_.sRgbGainRange.iBGainMin,
        t_capability_.sRgbGainRange.iBGainMax
    );
    g_gain = std::clamp(
        g_gain,
        t_capability_.sRgbGainRange.iGGainMin,
        t_capability_.sRgbGainRange.iGGainMax
    );
    CameraSetGain(h_camera_, r_gain, g_gain, b_gain);
    AWAKENING_INFO(
        "MvCamera: set rgb_gain = {}, {}, {}, r: min {} max {} , g: min {} max {} , b: min {} max {}",
        r_gain,
        g_gain,
        b_gain,
        t_capability_.sRgbGainRange.iRGainMin,
        t_capability_.sRgbGainRange.iRGainMax,
        t_capability_.sRgbGainRange.iGGainMin,
        t_capability_.sRgbGainRange.iGGainMax,
        t_capability_.sRgbGainRange.iBGainMin,
        t_capability_.sRgbGainRange.iBGainMax
    );
}
void MvCamera::set_analog_gain(double analog_gain) {
    analog_gain = std::clamp(
        analog_gain,
        (double)t_capability_.sExposeDesc.uiAnalogGainMin,
        (double)t_capability_.sExposeDesc.uiAnalogGainMax
    );
    CameraSetAnalogGain(h_camera_, analog_gain);
    AWAKENING_INFO(
        "MvCamera: set analog_gain {}, min {} max {}",
        analog_gain,
        t_capability_.sExposeDesc.uiAnalogGainMin,
        t_capability_.sExposeDesc.uiAnalogGainMax
    );
}
void MvCamera::set_gamma(double gamma) {
    gamma = std::clamp(
        gamma,
        (double)t_capability_.sGammaRange.iMin,
        (double)t_capability_.sGammaRange.iMax
    );
    CameraSetGamma(h_camera_, gamma);
    AWAKENING_INFO(
        "MvCamera: set gamma {}, min {} max {}",
        gamma,
        t_capability_.sGammaRange.iMin,
        t_capability_.sGammaRange.iMax
    );
}
bool MvCamera::initialize_camera() {
    // if (h_camera_ != nullptr) {
    //     CameraUnInit(*h_camera_);
    //     h_camera_ = nullptr;
    // }
    while (running_) {
        int i_camera_counts = 1;
        int i_status = -1;
        tSdkCameraDevInfo t_camera_enum_list;
        i_status = CameraEnumerateDevice(&t_camera_enum_list, &i_camera_counts);
        AWAKENING_INFO("Enumerate state = {}", i_status);
        AWAKENING_INFO("Found camera count = {}", i_camera_counts);

        // 没有连接设备
        if (i_camera_counts == 0) {
            AWAKENING_ERROR("No camera found!");
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        i_status = CameraInit(&t_camera_enum_list, -1, -1, &h_camera_);

        // 初始化失败
        AWAKENING_INFO("MvCamera: Init state = {}", i_status);
        if (i_status != CAMERA_STATUS_SUCCESS) {
            AWAKENING_ERROR("MvCamera: Init failed!");
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        AWAKENING_INFO("Camera initialized successfully");
        return true;
    }

    return false;
}
} // namespace awakening