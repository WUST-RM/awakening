#pragma once
#include "angles.h"
#include "utils/common/type_common.hpp"
#include <opencv2/core/eigen.hpp>
#include <pwd.h>
#include <regex>
namespace awakening::utils {
enum class EulerOrder { XYZ, XZY, YXZ, YZX, ZXY, ZYX };
inline Vec3 axis_vec(int axis) {
    switch (axis) {
        case 0:
            return Vec3::UnitX();
        case 1:
            return Vec3::UnitY();
        case 2:
            return Vec3::UnitZ();
        default:
            throw std::invalid_argument("Invalid axis");
    }
}

inline std::array<int, 3> get_axes(EulerOrder order) {
    switch (order) {
        case EulerOrder::XYZ:
            return { 0, 1, 2 };
        case EulerOrder::XZY:
            return { 0, 2, 1 };
        case EulerOrder::YXZ:
            return { 1, 0, 2 };
        case EulerOrder::YZX:
            return { 1, 2, 0 };
        case EulerOrder::ZXY:
            return { 2, 0, 1 };
        case EulerOrder::ZYX:
            return { 2, 1, 0 };
        default:
            throw std::invalid_argument("Unsupported EulerOrder");
    }
}

inline Quaternion euler2quat(const Vec3& angles, EulerOrder order) {
    auto axes = get_axes(order);

    Quaternion q = Quaternion::Identity();

    for (int i = 0; i < 3; ++i) {
        q = q * Quaternion(AngleAxis(angles[i], axis_vec(axes[i])));
    }

    return q;
}

inline Mat3 euler2matrix(const Vec3& angles, EulerOrder order) {
    return euler2quat(angles, order).toRotationMatrix();
}
inline Vec3 matrix2euler(const Mat3& R, EulerOrder order) {
    Vec3 angles;

    switch (order) {
        case EulerOrder::XYZ: {
            double sy = -R(2, 0);
            if (std::abs(sy) < 1.0 - 1e-6) {
                angles[1] = std::asin(sy);
                angles[0] = std::atan2(R(2, 1), R(2, 2));
                angles[2] = std::atan2(R(1, 0), R(0, 0));
            } else {
                angles[1] = std::asin(sy);
                angles[0] = std::atan2(-R(1, 2), R(1, 1));
                angles[2] = 0;
            }
            break;
        }

        case EulerOrder::ZYX: {
            double sy = -R(2, 0);
            if (std::abs(sy) < 1.0 - 1e-6) {
                angles[1] = std::asin(sy);
                angles[0] = std::atan2(R(1, 0), R(0, 0));
                angles[2] = std::atan2(R(2, 1), R(2, 2));
            } else {
                angles[1] = std::asin(sy);
                angles[0] = std::atan2(-R(0, 1), R(1, 1));
                angles[2] = 0;
            }
            break;
        }

        case EulerOrder::XZY: {
            double sz = R(1, 0);
            if (std::abs(sz) < 1.0 - 1e-6) {
                angles[2] = std::asin(sz);
                angles[0] = std::atan2(-R(1, 2), R(1, 1));
                angles[1] = std::atan2(-R(2, 0), R(0, 0));
            } else {
                angles[2] = std::asin(sz);
                angles[0] = std::atan2(R(2, 1), R(2, 2));
                angles[1] = 0;
            }
            break;
        }

        case EulerOrder::YXZ: {
            double sx = -R(1, 2);
            if (std::abs(sx) < 1.0 - 1e-6) {
                angles[0] = std::asin(sx);
                angles[1] = std::atan2(R(0, 2), R(2, 2));
                angles[2] = std::atan2(R(1, 0), R(1, 1));
            } else {
                angles[0] = std::asin(sx);
                angles[1] = std::atan2(-R(2, 0), R(0, 0));
                angles[2] = 0;
            }
            break;
        }

        case EulerOrder::YZX: {
            double sz = -R(0, 1);
            if (std::abs(sz) < 1.0 - 1e-6) {
                angles[2] = std::asin(sz);
                angles[1] = std::atan2(R(0, 2), R(0, 0));
                angles[0] = std::atan2(R(2, 1), R(1, 1));
            } else {
                angles[2] = std::asin(sz);
                angles[1] = std::atan2(-R(2, 0), R(2, 2));
                angles[0] = 0;
            }
            break;
        }

        case EulerOrder::ZXY: {
            double sx = R(2, 1);
            if (std::abs(sx) < 1.0 - 1e-6) {
                angles[0] = std::asin(sx);
                angles[2] = std::atan2(-R(0, 1), R(1, 1));
                angles[1] = std::atan2(-R(2, 0), R(2, 2));
            } else {
                angles[0] = std::asin(sx);
                angles[2] = std::atan2(R(1, 0), R(0, 0));
                angles[1] = 0;
            }
            break;
        }

        default:
            throw std::invalid_argument("Unsupported EulerOrder");
    }

    return angles;
}

inline Vec3 quat2euler(const Quaternion& q, EulerOrder order) {
    return matrix2euler(q.toRotationMatrix(), order);
}

inline std::string expand_env(const std::string& s) {
    std::regex env_re(R"(\$\{([^}]+)\})");
    std::smatch match;
    std::string result = s;
    while (std::regex_search(result, match, env_re)) {
        const char* env = std::getenv(match[1].str().c_str());
        std::string val = env ? env : "";
        result.replace(match.position(0), match.length(0), val);
    }
    return result;
}
template<typename Func>
void dt_once(Func&& func, std::chrono::duration<double> dt) noexcept {
    static auto last_call = std::chrono::steady_clock::now();

    auto now = std::chrono::steady_clock::now();
    if (now - last_call >= dt) {
        last_call = now;
        func();
    }
}
template<typename T>
concept Point2DLike = requires(T p) {
    {
        p.x
        } -> std::convertible_to<float>;
    {
        p.y
        } -> std::convertible_to<float>;
    T { 0.f, 0.f };
};
template<Point2DLike T>
[[nodiscard]] inline T transform_point2D(const Eigen::Matrix3f& H, const T& p) noexcept {
    const Eigen::Vector3f hp { p.x, p.y, 1.f };
    const Eigen::Vector3f tp = H * hp;
    return { tp.x(), tp.y() };
}
inline cv::Rect2f transformRect(const Eigen::Matrix3f& H, const cv::Rect2f& rect) {
    cv::Point2f p1(rect.x, rect.y);
    cv::Point2f p2(rect.x + rect.width, rect.y);
    cv::Point2f p3(rect.x, rect.y + rect.height);
    cv::Point2f p4(rect.x + rect.width, rect.y + rect.height);

    auto tp1 = utils::transform_point2D(H, p1);
    auto tp2 = utils::transform_point2D(H, p2);
    auto tp3 = utils::transform_point2D(H, p3);
    auto tp4 = utils::transform_point2D(H, p4);

    float min_x = std::min({ tp1.x, tp2.x, tp3.x, tp4.x });
    float min_y = std::min({ tp1.y, tp2.y, tp3.y, tp4.y });
    float max_x = std::max({ tp1.x, tp2.x, tp3.x, tp4.x });
    float max_y = std::max({ tp1.y, tp2.y, tp3.y, tp4.y });

    return cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
}
inline cv::Mat letterbox(
    const cv::Mat& img,
    Eigen::Matrix3f& transform_matrix,
    const int new_shape_w,
    const int new_shape_h
) noexcept {
    const int img_h = img.rows;
    const int img_w = img.cols;

    const float scale = std::min((float)new_shape_h / img_h, (float)new_shape_w / img_w);
    const int resize_h = int(img_h * scale + 0.5f);
    const int resize_w = int(img_w * scale + 0.5f);

    const int pad_h = new_shape_h - resize_h;
    const int pad_w = new_shape_w - resize_w;
    const int top = pad_h / 2;
    const int left = pad_w / 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat out;
    cv::copyMakeBorder(
        resized,
        out,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    const float inv_scale = 1.0f / scale;

    transform_matrix << inv_scale, 0, -left * inv_scale, 0, inv_scale, -top * inv_scale, 0, 0, 1;

    return out;
}
inline std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
    return s;
}
template<std::size_t N1, std::size_t N2>
consteval auto concat(const char (&a)[N1], const char (&b)[N2]) {
    std::array<char, N1 + N2 - 1> result {}; // -1 是因为两个字面量都有 '\0'
    for (std::size_t i = 0; i < N1 - 1; ++i)
        result[i] = a[i];
    for (std::size_t i = 0; i < N2; ++i)
        result[i + N1 - 1] = b[i]; // 包含 '\0'
    return result;
}
template<typename T>
inline T from_vector(const std::vector<uint8_t>& data) {
    T packet {};
    std::memcpy(&packet, data.data(), sizeof(T));
    return packet;
}

template<typename T>
inline std::vector<uint8_t> to_vector(const T& data) {
    std::vector<uint8_t> packet(sizeof(T));
    std::memcpy(packet.data(), &data, sizeof(T));
    return packet;
}
template<class Tag>
[[nodiscard]] double R2yaw(const Mat3& R) noexcept {
    static double last_yaw = 0;
    double roll, pitch, yaw;
    const auto euler = utils::matrix2euler(R, utils::EulerOrder::ZYX);
    yaw = euler[0];
    yaw = last_yaw + angles::shortest_angular_distance(last_yaw, yaw);
    last_yaw = yaw;
    return yaw;
}
inline std::vector<cv::Point2f> reprojection(
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const std::vector<cv::Point3f>& object_points,
    const ISO3& pose_in_camera_cv
) noexcept {
    cv::Mat rvec, R_cv;
    Mat3 R = pose_in_camera_cv.linear();
    cv::eigen2cv(R, R_cv);
    cv::Rodrigues(R_cv, rvec);
    auto t = pose_in_camera_cv.translation();
    const cv::Mat tvec = (cv::Mat_<double>(3, 1) << t.x(), t.y(), t.z());

    std::vector<cv::Point2f> pts_2d;
    pts_2d.reserve(object_points.size());
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, pts_2d);
    return pts_2d;
}
[[nodiscard]] inline double lerp_angle(double a0, double a1, double t) noexcept {
    double d = std::remainder(a1 - a0, 2.0 * M_PI);
    return a0 + t * d;
}
[[nodiscard]] inline Vec3 load_vec3(const YAML::Node& node) {
    auto vec = node.as<std::vector<double>>();
    return Vec3(vec[0], vec[1], vec[2]);
}
[[nodiscard]] inline Mat3 load_mat3(const YAML::Node& node) {
    Mat3 result;

    if (node.IsSequence() && node.size() == 9) {
        // 一维数组
        auto vec = node.as<std::vector<double>>();
        for (int i = 0; i < 9; ++i) {
            result(i / 3, i % 3) = vec[i];
        }
    } else {
        // 二维数组
        auto mat = node.as<std::vector<std::vector<double>>>();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result(i, j) = mat[i][j];
            }
        }
    }

    return result;
}
[[nodiscard]] inline ISO3 load_isometry3(const YAML::Node& node) {
    auto trans = load_vec3(node["t"]);
    auto rot = load_mat3(node["R"]);
    ISO3 result = ISO3::Identity();
    result.translation() = trans;
    result.linear() = rot;
    return result;
}
} // namespace awakening::utils