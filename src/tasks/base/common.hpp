#pragma once
#include "utils/common/image.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
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
inline EnemyColor enemy_color_from_string(const std::string& str) {
    auto key = utils::to_upper(str);
    if (key == "RED")
        return EnemyColor::RED;
    else if (key == "BLUE") {
        return EnemyColor::BLUE;
    } else {
        throw std::runtime_error("Invalid enemy color: " + key);
    }
}
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

        camera_matrix = K.clone();
        distortion_coefficients = D.clone();
    }
};
struct AimPoint {
    ISO3 pose;
    double d_angle;
    int frame_id;
    static AimPoint lerp(const AimPoint& a, const AimPoint& b, double t) {
        AimPoint p;

        Vec3 trans = (1.0 - t) * a.pose.translation() + t * b.pose.translation();
        Quaternion qa(a.pose.rotation());
        Quaternion qb(b.pose.rotation());
        Quaternion q = qa.slerp(t, qb);

        p.pose = ISO3::Identity();
        p.pose.linear() = q.toRotationMatrix();
        p.pose.translation() = trans;
        p.d_angle = utils::lerp_angle(a.d_angle, b.d_angle, t);
        return p;
    }
    void transform(const ISO3& old_in_new, int new_frame_id) {
        pose = old_in_new * pose;
        frame_id = new_frame_id;
    }
};
struct GimbalCmd {
    std::chrono::steady_clock::time_point timestamp;
    double pitch = 0;
    double yaw = 0;
    double enable_yaw_diff = 0;
    double enable_pitch_diff = 0;
    double target_yaw = 0;
    double target_pitch = 0;
    double v_yaw = 0;
    double v_pitch = 0;
    double a_yaw = 0;
    double a_pitch = 0;
    bool fire_advice = false;
    double fly_time = 0;
    bool appear = false;
    AimPoint aim_point;
    int select_id = 0;
    inline bool is_valid() const noexcept {
        auto bad = [](double x) { return std::isnan(x) || std::isinf(x); };

        if (bad(pitch) || bad(yaw) || bad(target_yaw) || bad(target_pitch) || bad(target_pitch)
            || bad(v_yaw) || bad(v_pitch) || bad(enable_yaw_diff) || bad(enable_pitch_diff))
            return false;

        return true;
    }
    inline void no_shoot() {
        fire_advice = false;
        enable_pitch_diff = 0;
        enable_pitch_diff = 0;
    }
};
} // namespace awakening