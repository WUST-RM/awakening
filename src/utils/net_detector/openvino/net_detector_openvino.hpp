#pragma once
#include "../net_detector_base.hpp"
#include "utils/impl.hpp"
namespace awakening::utils {
class NetDetectorOpenVINO: public NetDetectorBase {
public:
    NetDetectorOpenVINO(const YAML::Node& config, Config c);
    OutPut detect(const cv::Mat& img, PixelFormat format) noexcept override;
    AWAKENING_IMPL_DEFINITION(NetDetectorOpenVINO)
};
} // namespace awakening::utils