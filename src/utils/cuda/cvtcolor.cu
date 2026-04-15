#include "cvtcolor.hpp"
#include <iostream>
#include <npp.h>
#include <nppcore.h>
#include <nppdefs.h>
#include <nppi_color_conversion.h>

namespace awakening::utils::__cuda {

CvtColor::CvtColor() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

CvtColor::~CvtColor() {
    try {
        release();
    } catch (...) {
    }
    CUDA_CHECK(cudaStreamDestroy(stream_));
    stream_ = nullptr;
}

void CvtColor::release() {
    if (stream_)
        CUDA_CHECK(cudaStreamSynchronize(stream_));

    for (int i = 0; i < BUF_NUM; i++) {
        if (d_bayer_[i]) {
            CUDA_CHECK(cudaFree(d_bayer_[i]));
            d_bayer_[i] = nullptr;
        }
        if (d_target_[i]) {
            CUDA_CHECK(cudaFree(d_target_[i]));
            d_target_[i] = nullptr;
        }
        if (h_bayer_pinned_[i]) {
            CUDA_CHECK(cudaFreeHost(h_bayer_pinned_[i]));
            h_bayer_pinned_[i] = nullptr;
        }
        if (h_target_pinned_[i]) {
            CUDA_CHECK(cudaFreeHost(h_target_pinned_[i]));
            h_target_pinned_[i] = nullptr;
        }
        if (events_[i]) {
            CUDA_CHECK(cudaEventDestroy(events_[i]));
            events_[i] = nullptr;
        }
        d_bayer_pitch_[i] = 0;
        d_target_pitch_[i] = 0;
    }

    width_ = 0;
    height_ = 0;
    bayer_capacity_ = 0;
    rgb_capacity_ = 0;
}

void CvtColor::alloc_if_needed(int width, int height) {
    if (width <= 0 || height <= 0)
        throw std::runtime_error("alloc_if_needed: invalid dims");

    size_t needed_bayer = static_cast<size_t>(width) * height;
    size_t needed_rgb = needed_bayer * 3;

    if (needed_bayer <= bayer_capacity_ && needed_rgb <= rgb_capacity_) {
        width_ = width;
        height_ = height;
        return;
    }

    release();

    for (int i = 0; i < BUF_NUM; i++) {
        CUDA_CHECK(cudaMallocPitch(
            reinterpret_cast<void**>(&d_bayer_[i]),
            &d_bayer_pitch_[i],
            width,
            height
        ));
        CUDA_CHECK(cudaMallocPitch(
            reinterpret_cast<void**>(&d_target_[i]),
            &d_target_pitch_[i],
            width * 3,
            height
        ));
        // pinned host buffers
        CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&h_bayer_pinned_[i]),
            needed_bayer,
            cudaHostAllocDefault
        ));
        CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&h_target_pinned_[i]),
            needed_rgb,
            cudaHostAllocDefault
        ));
        // events
        CUDA_CHECK(cudaEventCreate(&events_[i]));
    }

    width_ = width;
    height_ = height;
    bayer_capacity_ = needed_bayer;
    rgb_capacity_ = needed_rgb;
}

void CvtColor::process(const cv::Mat& bayer, cv::Mat& target, int cv_bayer_code) {
    CV_Assert(bayer.type() == CV_8UC1 && bayer.isContinuous());
    int w = bayer.cols;
    int h = bayer.rows;
    if (w <= 0 || h <= 0)
        throw std::runtime_error("process: invalid image size");

    alloc_if_needed(w, h);
    size_t host_bayer_bytes = static_cast<size_t>(w) * h;
    size_t host_target_bytes = host_bayer_bytes * 3;

    int prevBuf = 1 - curBuf_;

    // 上一帧完成后拷贝到 target
    if (events_[prevBuf]) {
        CUDA_CHECK(cudaEventSynchronize(events_[prevBuf]));
        if (target.rows != h || target.cols != w || target.type() != CV_8UC3)
            target.create(h, w, CV_8UC3);
        std::memcpy(target.data, h_target_pinned_[prevBuf], host_target_bytes);
    }

    // copy Bayer to pinned host
    std::memcpy(h_bayer_pinned_[curBuf_], bayer.data, host_bayer_bytes);

    // map cv codes to NPP
    NppiBayerGridPosition grid;
    switch (cv_bayer_code) {
        case cv::COLOR_BayerRG2BGR:
            grid = NPPI_BAYER_RGGB;
            break;
        case cv::COLOR_BayerBG2BGR:
            grid = NPPI_BAYER_BGGR;
            break;
        case cv::COLOR_BayerGB2BGR:
            grid = NPPI_BAYER_GBRG;
            break;
        case cv::COLOR_BayerGR2BGR:
            grid = NPPI_BAYER_GRBG;
            break;
        case cv::COLOR_BayerRG2BGR_EA:
            grid = NPPI_BAYER_RGGB;
            break;
        case cv::COLOR_BayerBG2BGR_EA:
            grid = NPPI_BAYER_BGGR;
            break;
        case cv::COLOR_BayerGB2BGR_EA:
            grid = NPPI_BAYER_GBRG;
            break;
        case cv::COLOR_BayerGR2BGR_EA:
            grid = NPPI_BAYER_GRBG;
            break;
        default:
            throw std::runtime_error("Unsupported Bayer code");
    }

    // async H2D
    CUDA_CHECK(cudaMemcpy2DAsync(
        d_bayer_[curBuf_],
        d_bayer_pitch_[curBuf_],
        h_bayer_pinned_[curBuf_],
        static_cast<size_t>(w),
        w,
        h,
        cudaMemcpyHostToDevice,
        stream_
    ));

    // NPP context
    NppStreamContext ctx {};
    ctx.hStream = stream_;
    NppiSize roiSize { w, h };
    NppiRect roiRect { 0, 0, w, h };

    if (d_bayer_pitch_[curBuf_] > static_cast<size_t>(std::numeric_limits<int>::max())
        || d_target_pitch_[curBuf_] > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("pitch too large for NPP");

    NppStatus nppSt = nppiCFAToRGB_8u_C1C3R_Ctx(
        d_bayer_[curBuf_],
        static_cast<int>(d_bayer_pitch_[curBuf_]),
        roiSize,
        roiRect,
        d_target_[curBuf_],
        static_cast<int>(d_target_pitch_[curBuf_]),
        grid,
        NPPI_INTER_UNDEFINED,
        ctx
    );

    // async D2H
    CUDA_CHECK(cudaMemcpy2DAsync(
        h_target_pinned_[curBuf_],
        static_cast<size_t>(w * 3),
        d_target_[curBuf_],
        d_target_pitch_[curBuf_],
        static_cast<size_t>(w * 3),
        h,
        cudaMemcpyDeviceToHost,
        stream_
    ));

    // record event for下一帧同步
    CUDA_CHECK(cudaEventRecord(events_[curBuf_], stream_));

    // switch buffer
    curBuf_ = prevBuf;
}
} // namespace awakening::utils::__cuda