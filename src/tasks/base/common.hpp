#pragma once
#include "utils/common/image.hpp"
#include "utils/common/type_common.hpp"
namespace awakening {
static const Mat3 R_CV2PHYSICS =
    (Mat3() << 0.0, 0.0, 1.0, -1.0, -0.0, 0.0, 0.0, -1.0, 0.0).finished();
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
    void load(const YAML::Node& config) {
        std::vector<double> camera_k = config["camera_matrix"]["data"].as<std::vector<double>>();
        std::vector<double> camera_d =
            config["distortion_coefficients"]["data"].as<std::vector<double>>();

        assert(camera_k.size() == 9);
        assert(camera_d.size() == 5);

        cv::Mat K(3, 3, CV_64F);
        std::memcpy(K.data, camera_k.data(), 9 * sizeof(double));

        cv::Mat D(1, 5, CV_64F);
        std::memcpy(D.data, camera_d.data(), 5 * sizeof(double));

        camera_matrix = K;
        distortion_coefficients = D;
    }
};
} // namespace awakening