#pragma once
#include <cuda_runtime.h>
#include <utils/net_detector/net_detector_base.hpp>
namespace awakening::utils::__cuda {

class LetterBox {
public:
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int PIX_PER_THREAD = 4;
    using Ptr = std::shared_ptr<LetterBox>;
    LetterBox(NetDetectorBase::Config config);
    ~LetterBox() noexcept;
    float* letterbox_pitched(
        const unsigned char* input_bgr_host,
        PixelFormat pixel_format,
        int img_w,
        int img_h,
        int host_step,
        Eigen::Matrix3f& tf_matrix,
        cudaStream_t stream
    );
    cv::Mat tensorToMat(float* d_nchw, cudaStream_t stream) const;
    void release();
    bool isInitialized() const {
        return d_input_bgr_ && d_nchw_ && d_input_bgr_pitched_;
    }
    void getOutEnoughMem(int img_w, int img_h);
    void rellocMem();
    unsigned char* d_input_bgr_ = nullptr;
    float* d_nchw_ = nullptr;
    unsigned char* d_input_bgr_pitched_ = nullptr;
    size_t input_pitch_bytes_ = 0;
    NetDetectorBase::Config config_;
    int max_src_w_, max_src_h_;
};
} // namespace awakening::utils::__cuda