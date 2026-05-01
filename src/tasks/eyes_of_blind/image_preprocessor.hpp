#pragma once

#include "utils/impl.hpp"          
#include <opencv2/core/mat.hpp>
#include <yaml-cpp/node/node.h>

namespace awakening::eyes_of_blind {

class ImagePreprocessor {
public:
    explicit ImagePreprocessor(const YAML::Node& config);
    AWAKENING_IMPL_DEFINITION(ImagePreprocessor)
    
    cv::Mat process(const cv::Mat& input,
                    cv::Mat* roi_out = nullptr,
                    cv::Mat* static_removed_out = nullptr);
};

} // namespace awakening::eyes_of_blind