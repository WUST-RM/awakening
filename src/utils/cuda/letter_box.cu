#include "letter_box.hpp"
#include "utils/common/image.hpp"

namespace awakening::utils::__cuda {

constexpr int PRE_MAX_SRC_W = 1920;
constexpr int PRE_MAX_SRC_H = 1080;

LetterBox::LetterBox(NetDetectorBase::Config config) {
    config_ = config;
    max_src_h_ = PRE_MAX_SRC_H;
    max_src_w_ = PRE_MAX_SRC_W;
    d_input_bgr_ = nullptr;
    d_input_bgr_pitched_ = nullptr;
    d_nchw_ = nullptr;
    input_pitch_bytes_ = 0;
    relloc_mem();
}

LetterBox::~LetterBox() noexcept {
    release();
}

void LetterBox::relloc_mem() {
    release();

    CUDA_CHECK(cudaMalloc(&d_input_bgr_, max_src_w_ * max_src_h_ * 3 * sizeof(unsigned char)));
    CUDA_CHECK(cudaMallocPitch(
        &d_input_bgr_pitched_,
        &input_pitch_bytes_,
        max_src_w_ * 3 * sizeof(unsigned char),
        max_src_h_
    ));
    CUDA_CHECK(cudaMalloc(&d_nchw_, config_.target_w * config_.target_h * 3 * sizeof(float)));

    printf("Realloc memory for CudaInfer\n");
}
template<bool SwapRB, int PIX_PER_THREAD = 4>
__global__ void nchw_float_to_hwc_uchar4(
    const float* __restrict__ src,
    uchar4* __restrict__ dst,
    int W,
    int H,
    float norm
) {
    const int plane = W * H;
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    for (int idx = tidx; idx < W * H; idx += total_threads * PIX_PER_THREAD) {
#pragma unroll
        for (int i = 0; i < PIX_PER_THREAD && (idx + i) < W * H; i++) {
            int pidx = idx + i;
            float r = __ldg(src + pidx + 0 * plane) / norm;
            float g = __ldg(src + pidx + 1 * plane) / norm;
            float b = __ldg(src + pidx + 2 * plane) / norm;

            // clamp 0-255
            r = r < 0.f ? 0.f : (r > 255.f ? 255.f : r);
            g = g < 0.f ? 0.f : (g > 255.f ? 255.f : g);
            b = b < 0.f ? 0.f : (b > 255.f ? 255.f : b);
            if constexpr (SwapRB) {
                dst[pidx] = make_uchar4((unsigned char)b, (unsigned char)g, (unsigned char)r, 255);
            } else {
                dst[pidx] = make_uchar4((unsigned char)r, (unsigned char)g, (unsigned char)b, 255);
            }
        }
    }
}

cv::Mat LetterBox::tensor_to_mat(float* d_nchw, cudaStream_t stream, bool swap_rb) const {
    static uchar4* d_hwc = nullptr;
    static size_t cap = 0;
    const size_t need = config_.target_w * config_.target_h * sizeof(uchar4);

    if (cap < need) {
        if (d_hwc)
            cudaFree(d_hwc);
        cudaMalloc(&d_hwc, need);
        cap = need;
    }

    const int GRID_SIZE = (config_.target_w * config_.target_h + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (swap_rb) {
        nchw_float_to_hwc_uchar4<true, PIX_PER_THREAD><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
            d_nchw,
            d_hwc,
            config_.target_w,
            config_.target_h,
            config_.preprocess_scale
        );
    } else {
        nchw_float_to_hwc_uchar4<false, PIX_PER_THREAD><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
            d_nchw,
            d_hwc,
            config_.target_w,
            config_.target_h,
            config_.preprocess_scale
        );
    }

    cv::Mat img(config_.target_h, config_.target_w, CV_8UC4);
    cudaMemcpyAsync(img.data, d_hwc, need, cudaMemcpyDeviceToHost, stream);
    return img;
}
void LetterBox::check_out_enough_mem(int img_w, int img_h) {
    if (img_w > max_src_w_ || img_h > max_src_h_) {
        max_src_w_ = std::max(max_src_w_, img_w);
        max_src_h_ = std::max(max_src_h_, img_h);
        relloc_mem();
    }
}

void LetterBox::release() {
    if (d_input_bgr_) {
        cudaFree(d_input_bgr_);
        d_input_bgr_ = nullptr;
    }
    if (d_input_bgr_pitched_) {
        cudaFree(d_input_bgr_pitched_);
        d_input_bgr_pitched_ = nullptr;
    }
    if (d_nchw_) {
        cudaFree(d_nchw_);
        d_nchw_ = nullptr;
    }
    input_pitch_bytes_ = 0;
}

template<bool SwapRB, int PIX_PER_THREAD = 4>
__global__ void letterbox_kernel_pitched(
    const unsigned char* __restrict__ d_input_bgr,
    size_t pitch,
    int src_w,
    int src_h,
    float* __restrict__ d_nchw,
    int out_w,
    int out_h,
    float inv_scale,
    int pad_t,
    int pad_l,
    float norm
) {
    const int plane = out_w * out_h;
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    for (int idx = tidx * PIX_PER_THREAD; idx < out_w * out_h;
         idx += total_threads * PIX_PER_THREAD) {
#pragma unroll
        for (int i = 0; i < PIX_PER_THREAD && (idx + i) < out_w * out_h; ++i) {
            int out_idx = idx + i;
            int oy = out_idx / out_w;
            int ox = out_idx % out_w;

            float fx = ((float)ox - pad_l) * inv_scale;
            float fy = ((float)oy - pad_t) * inv_scale;

            fx = fminf(fmaxf(fx, 0.f), src_w - 2.f);
            fy = fminf(fmaxf(fy, 0.f), src_h - 2.f);

            const int x0 = __float2int_rd(fx);
            const int y0 = __float2int_rd(fy);
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;

            const float dx = fx - x0;
            const float dy = fy - y0;

            const float w00 = (1.f - dx) * (1.f - dy);
            const float w01 = dx * (1.f - dy);
            const float w10 = (1.f - dx) * dy;
            const float w11 = dx * dy;

            const unsigned char* row0 = d_input_bgr + (size_t)y0 * pitch;
            const unsigned char* row1 = d_input_bgr + (size_t)y1 * pitch;

            const unsigned char* p00 = row0 + (size_t)x0 * 3;
            const unsigned char* p01 = row0 + (size_t)x1 * 3;
            const unsigned char* p10 = row1 + (size_t)x0 * 3;
            const unsigned char* p11 = row1 + (size_t)x1 * 3;

            const float b = w00 * __ldg(p00 + 0) + w01 * __ldg(p01 + 0) + w10 * __ldg(p10 + 0)
                + w11 * __ldg(p11 + 0);
            const float g = w00 * __ldg(p00 + 1) + w01 * __ldg(p01 + 1) + w10 * __ldg(p10 + 1)
                + w11 * __ldg(p11 + 1);
            const float r = w00 * __ldg(p00 + 2) + w01 * __ldg(p01 + 2) + w10 * __ldg(p10 + 2)
                + w11 * __ldg(p11 + 2);

            if constexpr (SwapRB) {
                d_nchw[out_idx + 0 * plane] = r * norm;
                d_nchw[out_idx + 1 * plane] = g * norm;
                d_nchw[out_idx + 2 * plane] = b * norm;
            } else {
                d_nchw[out_idx + 0 * plane] = b * norm;
                d_nchw[out_idx + 1 * plane] = g * norm;
                d_nchw[out_idx + 2 * plane] = r * norm;
            }
        }
    }
}

float* LetterBox::letterbox_pitched(
    const unsigned char* input_bgr_host,
    PixelFormat pixel_format,
    int img_w,
    int img_h,
    int host_step,
    Eigen::Matrix3f& tf_matrix,
    cudaStream_t stream
) {
    if (!is_initialized())
        throw std::runtime_error("CudaInfer not initialized properly.");

    check_out_enough_mem(img_w, img_h);

    const float scale = fminf((float)config_.target_w / img_w, (float)config_.target_h / img_h);
    const float inv_scale = 1.f / scale;
    const int rw = (int)(img_w * scale + 0.5f);
    const int rh = (int)(img_h * scale + 0.5f);
    const int pad_l = (config_.target_w - rw) / 2;
    const int pad_t = (config_.target_h - rh) / 2;

    tf_matrix << inv_scale, 0, -pad_l * inv_scale, 0, inv_scale, -pad_t * inv_scale, 0, 0, 1;

    CUDA_CHECK(cudaMemcpy2DAsync(
        d_input_bgr_pitched_,
        input_pitch_bytes_,
        input_bgr_host,
        host_step,
        (size_t)img_w * 3,
        img_h,
        cudaMemcpyHostToDevice,
        stream
    ));

    const int GRID_SIZE = (config_.target_w * config_.target_h + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (pixel_format != config_.target_format) {
        letterbox_kernel_pitched<true, PIX_PER_THREAD><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
            d_input_bgr_pitched_,
            input_pitch_bytes_,
            img_w,
            img_h,
            d_nchw_,
            config_.target_w,
            config_.target_h,
            inv_scale,
            pad_t,
            pad_l,
            config_.preprocess_scale
        );
    } else {
        letterbox_kernel_pitched<false, PIX_PER_THREAD><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
            d_input_bgr_pitched_,
            input_pitch_bytes_,
            img_w,
            img_h,
            d_nchw_,
            config_.target_w,
            config_.target_h,
            inv_scale,
            pad_t,
            pad_l,
            config_.preprocess_scale
        );
    }

    CUDA_CHECK(cudaGetLastError());
    return d_nchw_;
}

} // namespace awakening::utils::__cuda