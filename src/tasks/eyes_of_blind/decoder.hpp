#pragma once

#include "tasks/eyes_of_blind/common.hpp"
#include <cstring>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <vector>

namespace awakening::eyes_of_blind {

class Decoder {
public:
    Decoder();
    ~Decoder();

    void push_packet(const BlindSend& pkt);

    bool try_pop_frame(cv::Mat& out);

    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace awakening::eyes_of_blind