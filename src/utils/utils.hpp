#pragma once
#include "utils/common/type_common.hpp"
#include <pwd.h>
#include <regex>
namespace awakening::utils {
inline std::string expandEnv(const std::string& s) {
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
void XSecOnce(Func&& func, double dt) noexcept {
    static auto last_call = std::chrono::steady_clock::now();

    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - last_call).count();

    if (elapsed >= dt) {
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
[[nodiscard]] inline T transformPoint2D(const Eigen::Matrix3f& H, const T& p) noexcept {
    const Eigen::Vector3f hp { p.x, p.y, 1.f };
    const Eigen::Vector3f tp = H * hp;
    return { tp.x(), tp.y() };
}
inline cv::Rect2f transformRect(const Eigen::Matrix3f& H, const cv::Rect2f& rect) {
    cv::Point2f p1(rect.x, rect.y);
    cv::Point2f p2(rect.x + rect.width, rect.y);
    cv::Point2f p3(rect.x, rect.y + rect.height);
    cv::Point2f p4(rect.x + rect.width, rect.y + rect.height);

    auto tp1 = utils::transformPoint2D(H, p1);
    auto tp2 = utils::transformPoint2D(H, p2);
    auto tp3 = utils::transformPoint2D(H, p3);
    auto tp4 = utils::transformPoint2D(H, p4);

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
} // namespace awakening::utils