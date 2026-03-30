#pragma once
#include "type_common.hpp"
#include "utils/utils.hpp"
namespace awakening {
enum class PixelFormat : int { BGR = 0, GRAY, RGB };
inline PixelFormat string2PixelFormat(const std::string& str) {
    auto key = utils::to_upper(str);
    if (key == "BGR")
        return PixelFormat::BGR;
    if (key == "GRAY")
        return PixelFormat::GRAY;
    if (key == "RGB")
        return PixelFormat::RGB;

    return PixelFormat::BGR; // Default
}
struct ImageFrame {
    cv::Mat src_img;
    PixelFormat format;
    TimePoint timestamp;
};
} // namespace awakening