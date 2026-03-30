#pragma once
#include "MvCameraControl.h"
#include "utils/common/image.hpp"
#include "utils/logger.hpp"
#include "utils/scheduler/node.hpp"
#include "utils/scheduler/scheduler.hpp"
#include "utils/utils.hpp"
#include <array>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <utility>
namespace awakening {

class HikCamera {
public:
    struct Frame {
        MV_FRAME_OUT out_frame;
        std::chrono::steady_clock::time_point timestamp;
    };
    HikCamera(const YAML::Node& config, Scheduler& scheduler): scheduler_(scheduler) {
        config_ = config;
        running_ = true;
        load(config_);
    }
    ~HikCamera() {
        stop();
    }
    void load(const YAML::Node& config);
    void stop();
    void restart();
    template<typename Tag>
    void start(std::string source_name) {
        using IO = IOPair<Tag, ImageFrame>;
        source_snapshot_id_ = scheduler_.register_source<IO>(source_name);
        int n_ret = MV_CC_StartGrabbing(camera_handle_);
        if (n_ret != MV_OK) {
            AWAKENING_ERROR("Failed to start camera grabbing!");
        }
        running_ = true;
        daemon_thread_ = std::thread(&HikCamera::runLoop<IO>, this);
    }
    template<typename IO>
    void runLoop() {
        while (running_) {
            hikCaptureLoop<IO>();
            if (!running_) {
                break;
            }
            restart();
        }
    }
    template<typename IO>
    void hikCaptureLoop() {
        AWAKENING_INFO("Starting image capture loop!");
        Frame frame;
        while (running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            int n_ret = MV_CC_GetImageBuffer(camera_handle_, &frame.out_frame, 100);
            if (n_ret == MV_OK) {
                const auto current_time = std::chrono::steady_clock::now();

                const auto half_exposure =
                    std::chrono::microseconds(static_cast<long>(getExposureTime() / 2));

                frame.timestamp = current_time - half_exposure;
                auto img_frame = toImageFrame(frame);
                scheduler_.runtime_push_source<IO>(
                    source_snapshot_id_,
                    [f = std::move(img_frame)]() {
                        return std::make_tuple(std::optional<typename IO::second_type>(std::move(f))
                        );
                    }
                );
            } else {
                AWAKENING_ERROR("Failed to get image buffer!");
                break;
            }
        }
        AWAKENING_INFO("Exiting image capture loop.");
    }

    const std::unordered_map<MvGvspPixelType, int> CVT_MAP_BGR = {
        { PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB },
        { PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB },
        { PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB },
        { PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB },
        { PixelType_Gvsp_RGB8_Packed, -1 },
        { PixelType_Gvsp_Mono8, cv::COLOR_GRAY2RGB },
    };

    const std::unordered_map<MvGvspPixelType, int> CVT_MAP_RGB = {
        { PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2BGR },
        { PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2BGR },
        { PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2BGR },
        { PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2BGR },
        { PixelType_Gvsp_RGB8_Packed, cv::COLOR_RGB2BGR },
        { PixelType_Gvsp_Mono8, cv::COLOR_GRAY2BGR },
    };

    const std::unordered_map<MvGvspPixelType, int> CVT_MAP_GRAY = {
        { PixelType_Gvsp_BayerGR8, -1 },
        { PixelType_Gvsp_BayerRG8, -1 },
        { PixelType_Gvsp_BayerGB8, -1 },
        { PixelType_Gvsp_BayerBG8, -1 },
        { PixelType_Gvsp_RGB8_Packed, cv::COLOR_RGB2GRAY },
        { PixelType_Gvsp_Mono8, -1 },
    };
    const std::unordered_map<MvGvspPixelType, int>& getCvtMap() {
        static const std::array details { CVT_MAP_BGR, CVT_MAP_GRAY, CVT_MAP_RGB };
        return details[std::to_underlying(target_format_)];
    }
    ImageFrame toImageFrame(Frame& f);

    bool initializeCamera(const std::string& target_sn);
    inline void HikSetFloatRange(void* camera_handle, const char* param, double val) {
        MVCC_FLOATVALUE fv {};
        int s = MV_CC_GetFloatValue(camera_handle, param, &fv);
        if (s != MV_OK) {
            AWAKENING_ERROR("hik_camera: Failed to get {} range, status={}", param, s);
            return;
        }

        double c = std::clamp(val, (double)fv.fMin, (double)fv.fMax);
        int r = MV_CC_SetFloatValue(camera_handle, param, c);
        if (r == MV_OK) {
            AWAKENING_INFO("hik_camera: {} set to {}", param, c);
        } else {
            AWAKENING_ERROR("hik_camera: Failed to set {}, status={}", param, r);
        }
    }
    inline void HikSetIntRange(void* camera_handle, const char* param, int64_t val) {
        MVCC_INTVALUE iv {};
        int s = MV_CC_GetIntValue(camera_handle, param, &iv);
        if (s != MV_OK) {
            AWAKENING_ERROR("hik_camera: Failed to get {} range, status={}", param, s);
            return;
        }

        int64_t c = std::clamp(val, static_cast<int64_t>(iv.nMin), static_cast<int64_t>(iv.nMax));
        int r = MV_CC_SetIntValue(camera_handle, param, c);
        if (r == MV_OK) {
            AWAKENING_INFO("hik_camera: {} set to {}", param, c);
        } else {
            AWAKENING_ERROR("hik_camera: Failed to set {}, status={}", param, r);
        }
    }
    inline void HikSetBool(void* camera_handle, const char* param, bool val) {
        int r = MV_CC_SetBoolValue(camera_handle, param, val);
        if (r == MV_OK) {
            AWAKENING_INFO("hik_camera: {} set to {}", param, (val ? 1 : 0));
        } else {
            AWAKENING_ERROR("hik_camera: Failed to set {}, status={}", param, r);
        }
    }
    inline void HikSetEnumStr(void* camera_handle, const char* param, const std::string& val) {
        int r = MV_CC_SetEnumValueByString(camera_handle, param, val.c_str());
        if (r == MV_OK) {
            AWAKENING_INFO("hik_camera: {} set to {}", param, val);
        } else {
            AWAKENING_ERROR("hik_camera: Failed to set {}, status={}", param, r);
        }
    }

    template<typename T>
    inline void HikSetRangeDispatch(void* camera_handle, const char* param, const T& val) {
        if constexpr (std::is_same_v<T, bool>) {
            HikSetBool(camera_handle, param, val);

        } else if constexpr (std::is_integral_v<T>) {
            HikSetIntRange(camera_handle, param, static_cast<int64_t>(val));

        } else if constexpr (std::is_floating_point_v<T>) {
            HikSetFloatRange(camera_handle, param, static_cast<double>(val));

        } else if constexpr (std::is_same_v<T, std::string>) {
            HikSetEnumStr(camera_handle, param, val);

        } else {
            static_assert(!sizeof(T), "Unsupported HikCamera param type");
        }
    }

#define HIK_GEN_MEMBER_GET_SET(type, camera_handle, param) \
    type param##_val {}; \
    inline void set##param(const type& v) { \
        param##_val = v; \
        HikSetRangeDispatch(camera_handle, #param, param##_val); \
    } \
    inline type get##param() const { \
        return param##_val; \
    }
    HIK_GEN_MEMBER_GET_SET(std::string, camera_handle_, PixelFormat)
    HIK_GEN_MEMBER_GET_SET(std::string, camera_handle_, ADCBitDepth)
    HIK_GEN_MEMBER_GET_SET(std::string, camera_handle_, TriggerMode)
    HIK_GEN_MEMBER_GET_SET(std::string, camera_handle_, TriggerSource)
    HIK_GEN_MEMBER_GET_SET(std::string, camera_handle_, TriggerActivation)
    HIK_GEN_MEMBER_GET_SET(bool, camera_handle_, ReverseX)
    HIK_GEN_MEMBER_GET_SET(bool, camera_handle_, ReverseY)
    HIK_GEN_MEMBER_GET_SET(int, camera_handle_, Width)
    HIK_GEN_MEMBER_GET_SET(int, camera_handle_, Height)
    HIK_GEN_MEMBER_GET_SET(int, camera_handle_, OffsetX)
    HIK_GEN_MEMBER_GET_SET(int, camera_handle_, OffsetY)
    HIK_GEN_MEMBER_GET_SET(bool, camera_handle_, AcquisitionFrameRateEnable)
    HIK_GEN_MEMBER_GET_SET(double, camera_handle_, AcquisitionFrameRate)
    HIK_GEN_MEMBER_GET_SET(double, camera_handle_, Gain)
    HIK_GEN_MEMBER_GET_SET(double, camera_handle_, Gamma)
    HIK_GEN_MEMBER_GET_SET(double, camera_handle_, ExposureTime)

    void* camera_handle_ = nullptr;
    std::string target_sn_;
    std::atomic<bool> running_ { false };
    YAML::Node config_;
    std::string trigger_source_;
    int64_t trigger_activation_;
    PixelFormat target_format_ = PixelFormat::BGR;
    Scheduler& scheduler_;
    size_t source_snapshot_id_;
    std::thread daemon_thread_;
};
} // namespace awakening