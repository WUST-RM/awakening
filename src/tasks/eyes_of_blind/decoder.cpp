#include "decoder.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"

#include <cstring>
#include <mutex>
#include <queue>
#include <vector>

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

namespace awakening::eyes_of_blind {

struct Decoder::Impl {
    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GstElement* appsink_ = nullptr;
    GstBus* bus_ = nullptr;

    std::mutex frame_mutex_;
    std::deque<cv::Mat> frame_queue_;

    struct Params {
        int out_w = 300;
        int out_h = 300;
    } params_;

    Impl() {
        initialize_gstreamer();
    }

    ~Impl() {
        shutdown_gstreamer();
    }

    static GstFlowReturn on_new_sample(GstAppSink* sink, gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        self->pull_frame_from_gstreamer(sink);
        return GST_FLOW_OK;
    }

    void initialize_gstreamer() {
        gst_init(nullptr, nullptr);

        pipeline_ = gst_pipeline_new("decoder_pipe");
        appsrc_ = gst_element_factory_make("appsrc", "source");
        GstElement* parser = gst_element_factory_make("h264parse", "parser");
        GstElement* decoder = gst_element_factory_make("avdec_h264", "decoder");
        GstElement* convert = gst_element_factory_make("videoconvert", "convert");
        appsink_ = gst_element_factory_make("appsink", "sink");

        if (!pipeline_ || !appsrc_ || !parser || !decoder || !convert || !appsink_) {
            AWAKENING_ERROR("GStreamer element creation failed");
            return;
        }

        GstCaps* src_caps = gst_caps_new_simple(
            "video/x-h264",
            "stream-format",
            G_TYPE_STRING,
            "byte-stream",
            nullptr
        );

        g_object_set(
            G_OBJECT(appsrc_),
            "caps",
            src_caps,
            "is-live",
            TRUE,
            "format",
            GST_FORMAT_TIME,
            "block",
            TRUE,
            "do-timestamp",
            TRUE,
            nullptr
        );
        gst_caps_unref(src_caps);

        GstCaps* sink_caps = gst_caps_new_simple(
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
            nullptr
        );

        g_object_set(
            G_OBJECT(appsink_),
            "emit-signals",
            TRUE,
            "sync",
            FALSE,
            "caps",
            sink_caps,
            nullptr
        );
        gst_caps_unref(sink_caps);

        g_signal_connect(appsink_, "new-sample", G_CALLBACK(on_new_sample), this);

        gst_bin_add_many(GST_BIN(pipeline_), appsrc_, parser, decoder, convert, appsink_, nullptr);
        if (!gst_element_link_many(appsrc_, parser, decoder, convert, appsink_, nullptr)) {
            AWAKENING_ERROR("GStreamer pipeline link failed");
            return;
        }

        if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            AWAKENING_ERROR("GStreamer pipeline start failed");
            return;
        }

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

    void push_packet_to_gstreamer(const BlindSend& pkt) {
        static int count = 0;
        count++;
        utils::dt_once(
            [&]() {
                AWAKENING_INFO("receive {} pkt", count);
                count = 0;
            },
            std::chrono::duration<double>(1.0)
        );
        if (!appsrc_)
            return;

        const size_t size = pkt.data.size();
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
        if (!buffer)
            return;

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
            std::memcpy(map.data, pkt.data.data(), size);
            gst_buffer_unmap(buffer, &map);

            GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
            if (ret != GST_FLOW_OK) {
                AWAKENING_WARN("push packet failed: {}", static_cast<int>(ret));
            }
        } else {
            gst_buffer_unref(buffer);
        }
    }

    void pull_frame_from_gstreamer(GstAppSink* sink) {
        if (!sink)
            return;

        GstSample* sample = gst_app_sink_pull_sample(sink);
        if (!sample)
            return;

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        if (!buffer) {
            gst_sample_unref(sample);
            return;
        }

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            cv::Mat frame(params_.out_h, params_.out_w, CV_8UC3, map.data);
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                frame_queue_.push_back(frame.clone());
                if (frame_queue_.size() > 10) {
                    frame_queue_.pop_front();
                }
            }
            gst_buffer_unmap(buffer, &map);
        }

        gst_sample_unref(sample);
    }

    bool try_pop_frame(cv::Mat& out) {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (frame_queue_.empty())
            return false;
        out = frame_queue_.front();
        frame_queue_.pop_front();
        return true;
    }
};

Decoder::Decoder() {
    _impl = std::make_unique<Impl>();
}

Decoder::~Decoder() = default;

void Decoder::push_packet(const BlindSend& pkt) {
    _impl->push_packet_to_gstreamer(pkt);
}

bool Decoder::try_pop_frame(cv::Mat& out) {
    return _impl->try_pop_frame(out);
}

} // namespace awakening::eyes_of_blind