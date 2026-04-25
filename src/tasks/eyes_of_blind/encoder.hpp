#pragma once

#include "utils/logger.hpp"

#include <memory>
#include <yaml-cpp/node/node.h>

#include "common.hpp"
#include <opencv2/opencv.hpp>
namespace awakening::eyes_of_blind {

class Encoder {
public:
    explicit Encoder(const YAML::Node& config);

    ~Encoder();

    void push_frame(const cv::Mat& frame);
    bool try_pop_packet(BlindSend& out);
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace awakening::eyes_of_blind