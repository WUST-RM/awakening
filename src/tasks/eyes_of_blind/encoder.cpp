#include "encoder.hpp"
#include "utils/logger.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <stdexcept>
#include <vector>

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <yaml-cpp/node/node.h>

namespace awakening::eyes_of_blind {

struct Encoder::Impl {
    struct Params {
        int out_w {}, out_h {}, fps {};
        int target_bitrate {};
        int roi_w {}, roi_h {};
        int max_packets_per_sec = 30;

        void load(const YAML::Node& config) {
            out_w = config["output_w"].as<int>();
            out_h = config["output_h"].as<int>();
            fps = config["fps"].as<int>();
            target_bitrate = config["target_bitrate"].as<int>();
            roi_w = config["roi_w"].as<int>();
            roi_h = config["roi_h"].as<int>();

            if (config["max_packets_per_sec"])
                max_packets_per_sec = config["max_packets_per_sec"].as<int>();

            if (max_packets_per_sec <= 0)
                max_packets_per_sec = 30;
        }
    } params_;

    struct TokenBucket {
        double tokens = 0.0;
        double rate = 30.0;
        double capacity = 60.0;
        int64_t last_ns = 0;

        static int64_t now_ns() {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch()
            )
                .count();
        }

        void init(double r, double cap) {
            rate = r;
            capacity = cap;
            tokens = cap;
            last_ns = 0;
        }

        bool consume(double n = 1.0) {
            int64_t now = now_ns();

            if (last_ns == 0)
                last_ns = now;

            double dt = (now - last_ns) / 1e9;
            tokens = std::min(capacity, tokens + dt * rate);
            last_ns = now;

            if (tokens < n)
                return false;

            tokens -= n;
            return true;
        }
    };

    TokenBucket bucket_;

    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GstElement* appsink_ = nullptr;
    GstBus* bus_ = nullptr;

    std::mutex pkg_mutex_;
    std::condition_variable pkg_cv_;
    std::deque<BlindSend> pkg_queue_;
    size_t max_queue_packets_ = 0;

    uint64_t packet_sequence_id_ = 0;

    std::mutex buffer_mutex_;
    std::vector<uint8_t> stream_buffer_;

    Impl(const YAML::Node& config) {
        params_.load(config);

        bucket_.init(params_.max_packets_per_sec, params_.max_packets_per_sec * 2);

        max_queue_packets_ = params_.max_packets_per_sec * 4;

        initialize_gstreamer();
    }

    ~Impl() {
        shutdown_gstreamer();
    }

    void initialize_gstreamer() {
        gst_init(nullptr, nullptr);

        pipeline_ = gst_pipeline_new("encoder_pipe");
        appsrc_ = gst_element_factory_make("appsrc", "source");
        appsink_ = gst_element_factory_make("appsink", "sink");

        GstElement* convert = gst_element_factory_make("videoconvert", "convert");
        GstElement* encoder = gst_element_factory_make("x264enc", "encoder");
        GstElement* parser = gst_element_factory_make("h264parse", "parser");

        if (!pipeline_ || !appsrc_ || !appsink_ || !convert || !encoder || !parser) {
            AWAKENING_ERROR("GStreamer element creation failed");
            return;
        }

        GstCaps* caps = gst_caps_new_simple(
            "video/x-raw",
            "format",
            G_TYPE_STRING,
            "BGR",
            "width",
            G_TYPE_INT,
            params_.out_w,
            "height",
            G_TYPE_INT,
            params_.out_h,
            "framerate",
            GST_TYPE_FRACTION,
            params_.fps,
            1,
            nullptr
        );

        g_object_set(
            appsrc_,
            "caps",
            caps,
            "stream-type",
            0,
            "format",
            GST_FORMAT_TIME,
            "is-live",
            TRUE,
            "block",
            TRUE,
            "do-timestamp",
            TRUE,
            nullptr
        );

        gst_caps_unref(caps);

        g_object_set(
            encoder,
            "bitrate",
            params_.target_bitrate,
            "speed-preset",
            10,
            "tune",
            0x00000004,
            "byte-stream",
            TRUE,
            "key-int-max",
            30,
            "bframes",
            0,
            "rc-lookahead",
            0,
            "sync-lookahead",
            0,
            "sliced-threads",
            TRUE,
            "ref",
            1,
            "aud",
            TRUE,
            "option-string",
            "repeat-headers=1:scenecut=0:force-cfr=1",
            nullptr
        );

        g_object_set(parser, "config-interval", -1, "disable-passthrough", TRUE, nullptr);

        GstCaps* h264_caps = gst_caps_new_simple(
            "video/x-h264",
            "stream-format",
            G_TYPE_STRING,
            "byte-stream",
            "alignment",
            G_TYPE_STRING,
            "au",
            nullptr
        );

        g_object_set(
            appsink_,
            "caps",
            h264_caps,
            "max-buffers",
            5,
            "drop",
            FALSE,
            "emit-signals",
            FALSE,
            "sync",
            FALSE,
            nullptr
        );

        gst_caps_unref(h264_caps);

        gst_bin_add_many(GST_BIN(pipeline_), appsrc_, convert, encoder, parser, appsink_, nullptr);

        if (!gst_element_link_many(appsrc_, convert, encoder, parser, appsink_, nullptr)) {
            AWAKENING_ERROR("GStreamer pipeline link failed");
            return;
        }

        gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        bus_ = gst_element_get_bus(pipeline_);
    }

    void shutdown_gstreamer() {
        if (!pipeline_)
            return;

        gst_element_set_state(pipeline_, GST_STATE_NULL);

        if (bus_) {
            gst_object_unref(bus_);
            bus_ = nullptr;
        }

        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        appsrc_ = nullptr;
        appsink_ = nullptr;
    }

    void push_frame_to_gstreamer(const cv::Mat& frame) {
        if (!appsrc_ || frame.empty())
            return;

        cv::Mat cont = frame.isContinuous() ? frame : frame.clone();
        size_t size = cont.total() * cont.elemSize();

        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
        if (!buffer)
            return;

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
            std::memcpy(map.data, cont.data, size);
            gst_buffer_unmap(buffer, &map);

            gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
        } else {
            gst_buffer_unref(buffer);
        }
    }

    void pull_stream_and_packetize() {
        if (!appsink_)
            return;

        constexpr size_t packet_bytes = PAYLOAD_SIZE;

        while (true) {
            GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink_), 0);

            if (!sample)
                break;

            GstBuffer* buffer = gst_sample_get_buffer(sample);
            if (!buffer) {
                gst_sample_unref(sample);
                continue;
            }

            GstMapInfo map;
            if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
                std::lock_guard<std::mutex> lock(buffer_mutex_);

                size_t old = stream_buffer_.size();
                stream_buffer_.resize(old + map.size);
                std::memcpy(stream_buffer_.data() + old, map.data, map.size);

                while (stream_buffer_.size() >= packet_bytes) {
                    if (!bucket_.consume()) {
                        break;
                    }

                    BlindSend pkt {};
                    pkt.header.sequence_id = packet_sequence_id_++;

                    std::memcpy(pkt.data.data(), stream_buffer_.data(), packet_bytes);

                    {
                        std::lock_guard<std::mutex> qlock(pkg_mutex_);
                        if (pkg_queue_.size() < max_queue_packets_) {
                            pkg_queue_.push_back(pkt);
                        }
                    }

                    std::memmove(
                        stream_buffer_.data(),
                        stream_buffer_.data() + packet_bytes,
                        stream_buffer_.size() - packet_bytes
                    );

                    stream_buffer_.resize(stream_buffer_.size() - packet_bytes);
                }
            }

            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
        }
    }

    cv::Mat preprocess(const cv::Mat& frame) {
        if (frame.empty())
            return {};

        int roi_w = std::min(params_.roi_w, frame.cols);
        int roi_h = std::min(params_.roi_h, frame.rows);

        int x = (frame.cols - roi_w) / 2;
        int y = (frame.rows - roi_h) / 2;

        cv::Mat roi = frame(cv::Rect(x, y, roi_w, roi_h));

        cv::Mat out;
        cv::resize(roi, out, cv::Size(params_.out_w, params_.out_h));
        return out;
    }

    void push_frame(const cv::Mat& frame) {
        if (frame.empty())
            return;

        auto img = preprocess(frame);
        push_frame_to_gstreamer(img);
        pull_stream_and_packetize();
    }

    bool try_pop_packet(BlindSend& out) {
        std::lock_guard<std::mutex> lock(pkg_mutex_);
        if (pkg_queue_.empty())
            return false;

        out = pkg_queue_.front();
        pkg_queue_.pop_front();
        pkg_cv_.notify_one();
        return true;
    }
};

Encoder::Encoder(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}

Encoder::~Encoder() = default;

void Encoder::push_frame(const cv::Mat& frame) {
    _impl->push_frame(frame);
}

bool Encoder::try_pop_packet(BlindSend& out) {
    return _impl->try_pop_packet(out);
}

} // namespace awakening::eyes_of_blind