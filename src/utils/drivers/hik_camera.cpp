#include "hik_camera.hpp"

namespace awakening {
void HikCamera::load(const YAML::Node& config) {
    std::string target_sn = config["target_sn"].as<std::string>();
    initializeCamera(target_sn);
    auto acquisition_frame_rate = config["acquisition_frame_rate"].as<double>();
    setAcquisitionFrameRate(acquisition_frame_rate);
    auto exposure_time = config["exposure_time"].as<double>();
    setExposureTime(exposure_time);
    auto gain = config["gain"].as<double>();
    setGain(gain);
    auto gamma = config["gamma"].as<double>();
    setGamma(gamma);
    auto adc_bit_depth = config["adc_bit_depth"].as<std::string>();
    setADCBitDepth(adc_bit_depth);
    auto pixel_format = config["pixel_format"].as<std::string>();
    setPixelFormat(pixel_format);
    auto acquisition_frame_rate_enable = config["acquisition_frame_rate_enable"].as<bool>();
    setAcquisitionFrameRateEnable(acquisition_frame_rate_enable);
    auto width = config["width"].as<int>();
    setWidth(width);
    auto height = config["height"].as<int>();
    setHeight(height);
    auto offset_x = config["offset_x"].as<int>();
    setOffsetX(offset_x);
    auto offset_y = config["offset_y"].as<int>();
    setOffsetY(offset_y);
    auto reverse_x = config["reverse_x"].as<bool>();
    setReverseX(reverse_x);
    auto reverse_y = config["reverse_y"].as<bool>();
    setReverseY(reverse_y);
    auto trigger_mode = config["trigger_mode"].as<std::string>();
    setTriggerMode(trigger_mode);
    auto trigger_source = config["trigger_source"].as<std::string>();
    setTriggerSource(trigger_source);
    auto trigger_activation = config["trigger_activation"].as<std::string>();
    setTriggerActivation(trigger_activation);

    auto format_str = config["format"].as<std::string>();
    target_format_ = string2PixelFormat(format_str);
    AWAKENING_INFO("Camera parameters set successfully!");
}
void HikCamera::stop() {
    if (!running_) {
        return;
    }
    running_ = false;
    if (daemon_thread_.joinable()) {
        daemon_thread_.join();
    }
    if (camera_handle_) {
        MV_CC_StopGrabbing(camera_handle_);
        MV_CC_CloseDevice(camera_handle_);
        MV_CC_DestroyHandle(&camera_handle_);
    }
    AWAKENING_INFO("hik_camera has stop ");
}
void HikCamera::restart() {
    AWAKENING_WARN("Restarting camera");
    MV_CC_StopGrabbing(camera_handle_);
    MV_CC_CloseDevice(camera_handle_);
    MV_CC_DestroyHandle(&camera_handle_);
    camera_handle_ = nullptr;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    load(config_);

    int n_ret = MV_CC_StartGrabbing(camera_handle_);
    if (n_ret != MV_OK) {
        AWAKENING_ERROR("Failed to start grabbing after restart");
        return;
    }

    AWAKENING_INFO("Camera restarted successfully!");
    return;
}
ImageFrame HikCamera::toImageFrame(Frame& f) {
    ImageFrame img_frame {
        .format = target_format_,
        .timestamp = f.timestamp,
    };
    const auto& info = f.out_frame.stFrameInfo;
    cv::Mat src(cv::Size(info.nWidth, info.nHeight), CV_8U, f.out_frame.pBufAddr);
    const auto pixel_type = info.enPixelType;
    const auto& ref_cvt = getCvtMap();
    int cvt_code = ref_cvt.at(pixel_type);
    if (cvt_code != -1) {
        cv::cvtColor(src, img_frame.src_img, cvt_code);
    } else {
        img_frame.src_img = src.clone();
    }
    MV_CC_FreeImageBuffer(camera_handle_, &f.out_frame);
    return img_frame;
}

bool HikCamera::initializeCamera(const std::string& target_sn) {
    target_sn_ = target_sn;

    if (camera_handle_ != nullptr) {
        AWAKENING_INFO("Closing previously opened camera");
        MV_CC_CloseDevice(camera_handle_);
        MV_CC_DestroyHandle(camera_handle_);
        camera_handle_ = nullptr;
    }

    while (running_) {
        MV_CC_DEVICE_INFO_LIST device_list = { 0 };
        int n_ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
        if (n_ret != MV_OK) {
            AWAKENING_ERROR("MV_CC_EnumDevices failed, error code: {}", n_ret);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        if (device_list.nDeviceNum == 0) {
            AWAKENING_ERROR("No USB cameras found");
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        AWAKENING_INFO("Found {} USB camera(s):", device_list.nDeviceNum);
        for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
            auto info = device_list.pDeviceInfo[i];
            const char* sn =
                reinterpret_cast<const char*>(info->SpecialInfo.stUsb3VInfo.chSerialNumber);
            AWAKENING_INFO("[ {} ] SN = {}", i, sn);
        }

        int sel = -1;
        for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
            auto info = device_list.pDeviceInfo[i];
            const char* sn =
                reinterpret_cast<const char*>(info->SpecialInfo.stUsb3VInfo.chSerialNumber);
            if (target_sn == sn) {
                sel = i;
                break;
            }
        }

        if (sel < 0) {
            AWAKENING_ERROR("Camera with serial {} not found", target_sn);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        AWAKENING_INFO("Selecting camera at index {} (SN={})", sel, target_sn);

        n_ret = MV_CC_CreateHandle(&camera_handle_, device_list.pDeviceInfo[sel]);
        if (n_ret != MV_OK) {
            AWAKENING_ERROR("MV_CC_CreateHandle failed: {}", n_ret);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        n_ret = MV_CC_OpenDevice(camera_handle_);
        if (n_ret != MV_OK) {
            AWAKENING_ERROR("MV_CC_OpenDevice failed: {}", n_ret);
            MV_CC_DestroyHandle(camera_handle_);
            camera_handle_ = nullptr;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        AWAKENING_INFO("Camera initialized successfully");
        return true;
    }
    return false;
}
} // namespace awakening