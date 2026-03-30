#pragma once
#include "utils/common/image.hpp"
#include "utils/common/type_common.hpp"
namespace awakening {
struct CommonFrame {
    ImageFrame img_frame;
    int id;
    int frame_id;
    cv::Rect expanded;
    cv::Point2f offset = cv::Point2f(0, 0);
};
enum class EnemyColor : bool {
    RED = 0,
    BLUE = 1,
};
struct CameraInfo {
    cv::Mat camera_matrix;
    cv::Mat distortion_coefficients;
};
} // namespace awakening