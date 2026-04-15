#pragma once
#include "common.hpp"
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
namespace awakening::utils::__cuda {

class CvtColor {
public:
    CvtColor();
    ~CvtColor();

    // cv::Mat (bayer, CV_8UC1) -> cv::Mat (rgb, CV_8UC3)
    void process(const cv::Mat& bayer, cv::Mat& rgb, int cv_bayer_code);

private:
    void alloc_if_needed(int width, int height);
    void release();

private:
    static constexpr int BUF_NUM = 2; // 双缓冲

    // device pointers
    uint8_t* d_bayer_[BUF_NUM] { nullptr, nullptr };
    uint8_t* d_target_[BUF_NUM] { nullptr, nullptr };
    size_t d_bayer_pitch_[BUF_NUM] { 0, 0 };
    size_t d_target_pitch_[BUF_NUM] { 0, 0 };

    // host pinned buffers
    uint8_t* h_bayer_pinned_[BUF_NUM] { nullptr, nullptr };
    uint8_t* h_target_pinned_[BUF_NUM] { nullptr, nullptr };

    cudaEvent_t events_[BUF_NUM] { nullptr, nullptr };
    int curBuf_ = 0; // 当前 buffer 索引

    cudaStream_t stream_ = nullptr;

    int width_ = 0;
    int height_ = 0;

    size_t bayer_capacity_ = 0; // bytes (host)
    size_t rgb_capacity_ = 0; // bytes (host)
};

} // namespace awakening::utils::__cuda