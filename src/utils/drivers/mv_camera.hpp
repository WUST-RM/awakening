#pragma once
#include "CameraApi.h"
#include "utils/common/image.hpp"
#include "utils/logger.hpp"
#include "utils/scheduler/scheduler.hpp"
#include <algorithm>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <utility>
#ifdef USE_TRT
    #include "utils/cuda/cvtcolor.hpp"
#endif
namespace awakening {
class MvCamera {
public:
    MvCamera(const YAML::Node& config, Scheduler& scheduler): scheduler_(scheduler) {
        config_ = config;
        running_ = true;
        CameraSdkInit(0);
    }
    ~MvCamera() {
        stop();
    }
    void stop();
    void restart();
    template<typename Tag>
    void start(std::string source_name) {
        using IO = IOPair<Tag, ImageFrame>;
        source_snapshot_id_ = scheduler_.register_source<IO>(source_name);
        CameraPlay(h_camera_);
        running_ = true;
        daemon_thread_ = std::thread(&MvCamera::run_loop<IO>, this);
    }
    template<typename IO>
    void run_loop() {
        while (running_) {
            mv_capture_loop<IO>();
            if (!running_) {
                break;
            }
            restart();
        }
    }
    template<typename IO>
    void mv_capture_loop() {
        AWAKENING_INFO("Starting image capture loop!");
        tSdkFrameHead head;
        BYTE* raw;

        while (running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

            auto status = CameraGetImageBuffer(h_camera_, &head, &raw, 100);
            if (status == CAMERA_STATUS_SUCCESS) {
                const auto current_time = std::chrono::steady_clock::now();

                const auto half_exposure =
                    std::chrono::microseconds(static_cast<long>(get_ExposureTime() / 2));
                auto img = cv::Mat(cv::Size(head.iWidth, head.iHeight), CV_8UC1, raw);
                ImageFrame frame;
#ifdef USE_TRT
                if (use_cuda_cvt_) {
                    static utils::__cuda::CvtColor cvt;
                    cvt.process(img, frame.src_img, cv::COLOR_BayerRG2BGR);
                } else {
#endif
                    cv::cvtColor(img, frame.src_img, cv::COLOR_BayerRG2BGR);
#ifdef USE_TRT
                }
#endif
                CameraReleaseImageBuffer(h_camera_, raw);

                frame.timestamp = current_time - half_exposure;

                frame.format = PixelFormat::BGR;
                scheduler_.runtime_push_source<IO>(source_snapshot_id_, [f = std::move(frame)]() {
                    return std::make_tuple(std::optional<typename IO::second_type>(std::move(f)));
                });
            } else {
                AWAKENING_ERROR("Failed to get image buffer!");
                break;
            }
        }
        AWAKENING_INFO("Exiting image capture loop.");
    }
    void init();
    void load(const YAML::Node& config);
    void set_ExposureTime(double exposure_time);
    double get_ExposureTime();
    void set_rgb_gain(int r_gain, int b_gain, int g_gain);
    void set_analog_gain(double analog_gain);
    void set_gamma(double gamma);
    bool initialize_camera();
    std::atomic<bool> running_ { false };
    YAML::Node config_;
    CameraHandle h_camera_;
    tSdkCameraCapbility t_capability_; // 设备描述信息
    size_t source_snapshot_id_;
    std::thread daemon_thread_;
    Scheduler& scheduler_;
    bool use_cuda_cvt_ = false;
};
} // namespace awakening