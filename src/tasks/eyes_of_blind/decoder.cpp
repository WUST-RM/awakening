#include "decoder.hpp"
#include "utils/utils.hpp"

#include <cstring>

#include <mutex>
#include <queue>
#include <stdexcept>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace awakening::eyes_of_blind {

struct Decoder::Impl {
    const AVCodec* codec { nullptr };
    AVCodecContext* ctx { nullptr };
    AVFrame* frame { nullptr };
    AVPacket* pkt_ref { nullptr }; // 复用 Packet 对象
    SwsContext* sws { nullptr };

    std::mutex mtx;
    std::queue<cv::Mat> frame_q;
    std::vector<uint8_t> reasm_buffer;

    Impl() {
        codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
        if (!codec)
            throw std::runtime_error("HEVC decoder not found");

        ctx = avcodec_alloc_context3(codec);
        if (!ctx)
            throw std::runtime_error("Failed to allocate decoder context");

        if (avcodec_open2(ctx, codec, nullptr) < 0)
            throw std::runtime_error("Failed to open decoder");

        frame = av_frame_alloc();
        pkt_ref = av_packet_alloc();

        reasm_buffer.reserve(1024 * 256);
    }

    ~Impl() {
        if (sws)
            sws_freeContext(sws);
        avcodec_free_context(&ctx);
        av_frame_free(&frame);
        av_packet_free(&pkt_ref);
    }

    void push_packet(const BlindSend& pkt) {
        static int count = 0;
        count++;
        utils::dt_once(
            [&]() {
                std::cout << count << std::endl;
                count = 0;
            },
            std::chrono::duration<double>(1.0)
        );
        uint16_t safe_size = std::min(pkt.header.payload_size, (uint16_t)PAYLOAD_SIZE);

        if (safe_size == 0)
            return;

        reasm_buffer.insert(reasm_buffer.end(), pkt.data.begin(), pkt.data.begin() + safe_size);

        if (!pkt.header.frame_end)
            return;

        pkt_ref->data = reasm_buffer.data();
        pkt_ref->size = static_cast<int>(reasm_buffer.size());

        if (avcodec_send_packet(ctx, pkt_ref) >= 0) {
            while (avcodec_receive_frame(ctx, frame) >= 0) {
                convert_and_push(frame);
            }
        }

        reasm_buffer.clear();
        av_packet_unref(pkt_ref);
    }

    void convert_and_push(AVFrame* f) {
        sws = sws_getCachedContext(
            sws,
            f->width,
            f->height,
            (AVPixelFormat)f->format,
            f->width,
            f->height,
            AV_PIX_FMT_BGR24,
            SWS_BILINEAR,
            nullptr,
            nullptr,
            nullptr
        );

        // 2. 使用 FFmpeg 的对齐规则计算需要的 buffer 大小
        // 这里的 32 是为了匹配 av_frame_get_buffer 的对齐要求
        int size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, f->width, f->height, 32);
        uint8_t* buffer = (uint8_t*)av_malloc(size); // 分配对齐内存

        uint8_t* dst_data[4];
        int dst_linesize[4];
        // 将 buffer 映射到数组，获取符合对齐要求的 linesize
        av_image_fill_arrays(
            dst_data,
            dst_linesize,
            buffer,
            AV_PIX_FMT_BGR24,
            f->width,
            f->height,
            32
        );

        // 3. 安全转换
        sws_scale(sws, f->data, f->linesize, 0, f->height, dst_data, dst_linesize);

        // 4. 将对齐的 buffer 包装成 cv::Mat (注意：cv::Mat 现在不拥有 buffer 权)
        // 必须使用支持自定义 step 的构造函数
        cv::Mat img(f->height, f->width, CV_8UC3, dst_data[0], dst_linesize[0]);

        // 5. 深度拷贝一份给队列，并释放 buffer (或者用自定义 deleter 管理)
        cv::Mat out_img = img.clone();
        av_free(buffer);

        std::lock_guard<std::mutex> lk(mtx);
        if (frame_q.size() > 10)
            frame_q.pop();
        frame_q.push(std::move(out_img));
    }

    bool pop(cv::Mat& out) {
        std::lock_guard<std::mutex> lk(mtx);
        if (frame_q.empty())
            return false;
        out = std::move(frame_q.front());
        frame_q.pop();
        return true;
    }
};

Decoder::Decoder() {
    _impl = std::make_unique<Impl>();
}
Decoder::~Decoder() = default;
void Decoder::push_packet(const BlindSend& pkt) {
    _impl->push_packet(pkt);
}
bool Decoder::try_pop_frame(cv::Mat& out) {
    return _impl->pop(out);
}
void Decoder::reset() {
    _impl = std::make_unique<Impl>();
}

} // namespace awakening::eyes_of_blind