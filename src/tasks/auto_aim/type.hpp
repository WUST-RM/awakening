#pragma once
#include "tasks/base/common.hpp"
#include "utils/utils.hpp"
#include <array>
#include <cstddef>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>
namespace awakening::auto_aim {
constexpr double SIMPLE_SMALL_ARMOR_WIDTH = 133.0 / 1000.0; // 135
constexpr double SIMPLE_SMALL_ARMOR_HEIGHT = 50.0 / 1000.0; // 55
constexpr double LARGE_ARMOR_WIDTH = 225.0 / 1000.0;
constexpr double LARGE_ARMOR_HEIGHT = 50.0 / 1000.0; // 55
enum class ArmorColor : int { BLUE = 0, RED, NONE, PURPLE };
enum class ArmorClass : int { SENTRY = 0, NO1, NO2, NO3, NO4, NO5, OUTPOST, BASE, UNKNOWN };
constexpr int getArmorNumByArmorClass(const ArmorClass& armor_class) {
    constexpr std::array details { 4, 4, 4, 4, 4, 4, 3, 4, 4 };
    return details[std::to_underlying(armor_class)];
}
inline std::string getStringByArmorColor(ArmorColor armor_class) {
    constexpr const char* details[] = { "blue", "red", "none", "purple" };
    return std::string(details[std::to_underlying(armor_class)]);
}

inline std::string getStringByArmorClass(ArmorClass armor_class) {
    constexpr const char* details[] = { "sentry", "no1",     "no2",  "no3",    "no4",
                                        "no5",    "outpost", "base", "unknown" };
    return std::string(details[std::to_underlying(armor_class)]);
}

enum class ArmorKeyPointsIndex : int {
    RIGHT_BOTTOM,
    RIGHT_TOP,
    LEFT_TOP,
    LEFT_BOTTOM,
    LEFT_MID,
    RIGHT_MID,
    N
};
inline std::string getStringByArmorKeyPointsIndex(int index) {
    constexpr const char* details[] = { "right_bottom", "right_top", "left_top",
                                        "left_bottom",  "left_mid",  "right_mid" };
    return std::string(details[index]);
}
namespace armor_keypoints {
    using I = ArmorKeyPointsIndex;
    constexpr std::array sys_pairs = {
        std::pair { std::to_underlying(I::RIGHT_BOTTOM), std::to_underlying(I::LEFT_BOTTOM) },
        std::pair { std::to_underlying(I::RIGHT_MID), std::to_underlying(I::LEFT_MID) },
        std::pair { std::to_underlying(I::RIGHT_MID), std::to_underlying(I::LEFT_TOP) }
    };
} // namespace armor_keypoints
struct ArmorKeyPoints2D {
    using PointT = cv::Point2f;
    using I = ArmorKeyPointsIndex;

    void addOffset(const PointT& offset) noexcept {
        for (auto& p_opt: points) {
            if (p_opt) {
                *p_opt += offset;
            }
        }
        full_points.reset();
        bbox.reset();
    }

    void transform(const Eigen::Matrix<float, 3, 3>& transform_matrix) noexcept {
        for (auto& p_opt: points) {
            if (p_opt) {
                *p_opt = utils::transformPoint2D(transform_matrix, *p_opt);
            }
        }
        full_points.reset();
        bbox.reset();
    }

    std::array<PointT, 6>& landmarks() {
        auto computeMid = [&](I mid, I top, I bottom) {
            auto& mid_opt = points[std::to_underlying(mid)];
            if (!mid_opt) {
                mid_opt =
                    (*points[std::to_underlying(top)] + *points[std::to_underlying(bottom)]) / 2.f;
            }
        };

        computeMid(I::RIGHT_MID, I::RIGHT_TOP, I::RIGHT_BOTTOM);
        computeMid(I::LEFT_MID, I::LEFT_TOP, I::LEFT_BOTTOM);

        if (!full_points) {
            std::array<PointT, 6> tmp {};
            for (size_t i = 0; i < points.size(); ++i) {
                if (!points[i]) {
                    throw std::runtime_error("ArmorKeyPoints2D::points(): point not set");
                }
                tmp[i] = *points[i];
            }
            full_points = tmp;
        }

        return *full_points;
    }
    cv::Rect2f boundingBox() {
        if (!bbox.has_value()) {
            float min_x = std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            float max_x = std::numeric_limits<float>::lowest();
            float max_y = std::numeric_limits<float>::lowest();

            bool has_point = false;

            for (const auto& p: points) {
                if (!p.has_value()) {
                    continue;
                }
                const auto& pt = p.value();
                min_x = std::min(min_x, pt.x);
                min_y = std::min(min_y, pt.y);
                max_x = std::max(max_x, pt.x);
                max_y = std::max(max_y, pt.y);
                has_point = true;
            }
            if (!has_point) {
                throw std::runtime_error("No point in the contour");
            }
            bbox = cv::Rect2f { min_x, min_y, max_x - min_x, max_y - min_y };
        }

        return bbox.value();
    }

    std::array<std::optional<PointT>, 6> points {};

private:
    std::optional<std::array<PointT, 6>> full_points;
    std::optional<cv::Rect2f> bbox;
};

template<typename PointT, double W, double H>
struct ArmorKeyPoint3D {
    using I = ArmorKeyPointsIndex;
    static constexpr std::array points = {

        PointT(0, -W / 2, H / 2), // 左上

        PointT(0, -W / 2, -H / 2), // 左下

        PointT(0, W / 2, -H / 2), // 右下

        PointT(0, W / 2, H / 2), // 右上

        PointT(0, -W / 2, 0.0), // 左中

        PointT(0, W / 2, 0.0), // 右中

    };
};
struct Armor {
    ArmorColor color = ArmorColor::NONE;
    ArmorClass number = ArmorClass::UNKNOWN;
    ArmorKeyPoints2D key_points;

    ISO3 pose;
    bool has_tidy = false;
    struct NetCtx {
        double confidence = 0;
        ArmorColor color = ArmorColor::NONE;
        ArmorClass number = ArmorClass::UNKNOWN;
        ArmorKeyPoints2D key_points;
        std::vector<std::array<std::optional<cv::Point2f>, 6>> tmp_points;
    };
    NetCtx net;
    struct NumberClassifierCtx {
        ArmorClass number = ArmorClass::UNKNOWN;
        cv::Mat number_img;
        double confidence = 0;
    };
    std::optional<NumberClassifierCtx> number_classifier;
    struct ColorClassifierCtx {
        static constexpr size_t LEFT = 0;
        static constexpr size_t RIGHT = 1;
        std::array<cv::RotatedRect, 2> lights_box;
        std::array<ArmorColor, 2> light_colors = { ArmorColor::NONE, ArmorColor::NONE };
    };
    std::optional<ColorClassifierCtx> color_classifier;
    void tidy() {
        color = net.color;
        number = net.number;
        key_points = net.key_points;

        if (number_classifier) {
            if (number_classifier->number != ArmorClass::UNKNOWN) {
                number = number_classifier->number;
            }
        }
        if (color_classifier) {
            auto l = color_classifier->light_colors[ColorClassifierCtx::LEFT];
            auto r = color_classifier->light_colors[ColorClassifierCtx::RIGHT];
            if (l == r) {
                if (l == ArmorColor::NONE) {
                    if (color != ArmorColor::NONE || color != ArmorColor::PURPLE) {
                        color = ArmorColor::NONE;
                    }
                } else {
                    color = l;
                }
            } else if (l == ArmorColor::NONE && r != ArmorColor::NONE) {
                color = r;
            } else if (r == ArmorColor::NONE && l != ArmorColor::NONE) {
                color = l;
            }
        }
        has_tidy = true;
    }
    void addOffset(const cv::Point2f& offset) {
        if (!has_tidy) {
            throw std::runtime_error("addOffset called before tidy");
        }
        key_points.addOffset(offset);
    }
    void transform(const Eigen::Matrix<float, 3, 3>& transform_matrix) {
        if (!has_tidy) {
            throw std::runtime_error("transform called before tidy");
        }
        key_points.transform(transform_matrix);
    }
    void draw(cv::Mat& img) {
        if (!has_tidy)
            return;

        auto& pts = key_points.landmarks();

        using I = ArmorKeyPointsIndex;

        auto get = [&](I idx) -> cv::Point { return pts[std::to_underlying(idx)]; };

        cv::Point lt = get(I::LEFT_TOP);
        cv::Point rt = get(I::RIGHT_TOP);
        cv::Point rb = get(I::RIGHT_BOTTOM);
        cv::Point lb = get(I::LEFT_BOTTOM);

        // 顺序：左上 → 右下 → 右上 → 左下 → 左上
        cv::line(img, lt, rb, cv::Scalar(0, 255, 0), 2);
        cv::line(img, rb, rt, cv::Scalar(0, 255, 0), 2);
        cv::line(img, rt, lb, cv::Scalar(0, 255, 0), 2);
        cv::line(img, lb, lt, cv::Scalar(0, 255, 0), 2);

        cv::Point bottom_center = (lb + rb) * 0.5;

        bottom_center.y += 20;

        std::string text = getStringByArmorColor(color) + " " + getStringByArmorClass(number);

        int font = cv::FONT_HERSHEY_SIMPLEX;
        double scale = 1.5;
        int thickness = 1;

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, font, scale, thickness, &baseline);

        cv::Point text_org(
            bottom_center.x - text_size.width / 2,
            bottom_center.y + text_size.height / 2
        );

        cv::putText(img, text, text_org, font, scale, cv::Scalar(255, 255, 255), thickness);
    }
    Armor() = default;
};
struct Armors {
    std::chrono::steady_clock::time_point timestamp;
    int id = -1;
    int frame_id = -1;
    std::vector<Armor> armors;
    void draw(cv::Mat& img) {
        for (auto& armor: armors) {
            armor.draw(img);
        }
    }
};
} // namespace awakening::auto_aim