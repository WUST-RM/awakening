#pragma once
#include "utils/common/image.hpp"
#include <opencv2/core/mat.hpp>
#include <optional>
namespace awakening::utils {

class NetDetectorBase {
public:
    using Ptr = std::unique_ptr<NetDetectorBase>;

    virtual cv::Mat detect(const cv::Mat& img, PixelFormat format) = 0;
    virtual ~NetDetectorBase() = default;
};
} // namespace awakening::utils