#include "encoder.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>

#include <mutex>
#include <queue>
#include <stdexcept>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h> // 必须在这里！
#include <libavutil/imgutils.h> // 必须在这里！
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

namespace awakening::eyes_of_blind {

struct Encoder::Impl {
    int in_w { 300 }, in_h { 256 };
    int enc_w { 300 }, enc_h { 256 }; // 目标编码尺寸
    int fps { 30 }, bitrate { 8000 }; // 修正比特率单位，通常为 bps

    const AVCodec* codec { nullptr };
    AVCodecContext* ctx { nullptr };
    AVFrame* frame { nullptr };
    AVPacket* pkt { nullptr };
    SwsContext* sws { nullptr };

    std::mutex mtx;
    std::queue<BlindSend> out_q;

    uint64_t seq { 0 };
    int64_t pts { 0 };
    bool stopped { false };

    Impl() {
        codec = avcodec_find_encoder_by_name("libx265");
        if (!codec)
            throw std::runtime_error("HEVC encoder not found");

        ctx = avcodec_alloc_context3(codec);
        if (!ctx)
            throw std::runtime_error("Failed to allocate codec context");

        ctx->width = enc_w;
        ctx->height = enc_h;
        ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        ctx->time_base = { 1, fps };
        ctx->framerate = { fps, 1 };
        ctx->bit_rate = bitrate;
        ctx->gop_size = fps;
        ctx->max_b_frames = 0;
        ctx->thread_count = 1;

        av_opt_set(ctx->priv_data, "preset", "ultrafast", 0);
        av_opt_set(ctx->priv_data, "tune", "zerolatency", 0);

        if (avcodec_open2(ctx, codec, nullptr) < 0)
            throw std::runtime_error("Failed to open encoder");

        frame = av_frame_alloc();
        pkt = av_packet_alloc();

        frame->format = ctx->pix_fmt;
        frame->width = ctx->width;
        frame->height = ctx->height;

        if (av_frame_get_buffer(frame, 32) < 0)
            throw std::runtime_error("Failed to allocate frame buffer");
    }

    ~Impl() {
        flush();
        if (sws)
            sws_freeContext(sws);
        avcodec_free_context(&ctx);
        av_frame_free(&frame);
        av_packet_free(&pkt);
    }

    void drain() {
        while (avcodec_receive_packet(ctx, pkt) >= 0) {
            int offset = 0;
            while (offset < pkt->size) {
                int chunk = std::min((int)PAYLOAD_SIZE, pkt->size - offset);

                BlindSend out {};
                out.header.sequence_id = seq++;
                out.header.payload_size = chunk;
                out.header.frame_end = (offset + chunk == pkt->size);

                std::memcpy(out.data.data(), pkt->data + offset, chunk);

                out_q.push(std::move(out));
                offset += chunk;
            }
            av_packet_unref(pkt);
        }
    }

    void encode(const cv::Mat& img) {
        if (stopped || img.empty())
            return;

        int x = (img.cols - in_w) / 2;
        int y = (img.rows - in_h) / 2;
        auto roi = img(cv::Rect(x, y, in_w, in_h));
        if (!sws || roi.cols != in_w || roi.rows != in_h) {
            in_w = roi.cols;
            in_h = roi.rows;
            sws = sws_getCachedContext(
                sws,
                in_w,
                in_h,
                AV_PIX_FMT_BGR24,
                enc_w,
                enc_h,
                AV_PIX_FMT_YUV420P,
                SWS_BILINEAR,
                nullptr,
                nullptr,
                nullptr
            );
        }

        std::lock_guard<std::mutex> lk(mtx);

        if (av_frame_make_writable(frame) < 0)
            return;

        const uint8_t* in_data[1] = { roi.data };
        int in_linesize[1] = { (int)roi.step };
        sws_scale(sws, in_data, in_linesize, 0, in_h, frame->data, frame->linesize);

        frame->pts = pts++;

        if (avcodec_send_frame(ctx, frame) >= 0) {
            drain();
        }
    }

    bool pop(BlindSend& out) {
        std::lock_guard<std::mutex> lk(mtx);
        if (out_q.empty())
            return false;
        out = std::move(out_q.front());
        out_q.pop();
        return true;
    }

    void flush() {
        std::lock_guard<std::mutex> lk(mtx);
        if (stopped)
            return;
        stopped = true;
        avcodec_send_frame(ctx, nullptr);
        drain();
    }
};

Encoder::Encoder(const YAML::Node&) {
    _impl = std::make_unique<Impl>();
}
Encoder::~Encoder() = default;
void Encoder::push_frame(const cv::Mat& frame) {
    _impl->encode(frame);
}
bool Encoder::try_pop_packet(BlindSend& out) {
    return _impl->pop(out);
}

} // namespace awakening::eyes_of_blind