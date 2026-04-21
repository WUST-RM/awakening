#pragma once
#include "tasks/base/common.hpp"
#include "utils/utils.hpp"
#include <array>
#include <cstddef>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>
namespace awakening::auto_aim {
constexpr double FIFTTEN_DEGREE_RAD = 15 * CV_PI / 180;
enum class ArmorClass : int { SENTRY = 0, NO1, NO2, NO3, NO4, NO5, OUTPOST, BASE, UNKNOWN };
enum class ArmorType : int { SimpleSmall, Large, BuildingSmall };
template<ArmorType T>
struct ArmorTypeTraits; // declare
template<>
struct ArmorTypeTraits<ArmorType::SimpleSmall> {
    static constexpr double WIDTH = 133.0 / 1000.0;
    static constexpr double HEIGHT = 50.0 / 1000.0;
};
template<>
struct ArmorTypeTraits<ArmorType::Large> {
    static constexpr double WIDTH = 225.0 / 1000.0;
    static constexpr double HEIGHT = 50.0 / 1000.0;
};
template<>
struct ArmorTypeTraits<ArmorType::BuildingSmall> {
    static constexpr double WIDTH = 129.0 / 1000.0;
    static constexpr double HEIGHT = 50.0 / 1000.0;
};
template<typename PointT, ArmorType T>
struct ArmorKeyPoint3D {
    static constexpr double W = ArmorTypeTraits<T>::WIDTH;
    static constexpr double H = ArmorTypeTraits<T>::HEIGHT;
    inline static std::vector<PointT> build() {
        return {

            PointT(0, W / 2, H / 2), // 左上
            PointT(0, W / 2, -H / 2), // 左下
            PointT(0, -W / 2, -H / 2), // 右下
            PointT(0, -W / 2, H / 2), // 右上
            // PointT(0, W / 2, 0),
            // PointT(0, -W / 2, 0),
        };
    }
};
enum class ArmorKeyPointsIndex : int {
    LEFT_TOP,
    LEFT_BOTTOM,
    RIGHT_BOTTOM,
    RIGHT_TOP,
    // LEFT_MID,
    // RIGHT_MID,
    N
};
inline std::string string_by_armor_key_points_index(int index) {
    constexpr const char* details[] = { "left_top", "left_bottom", "right_bottom", "right_top" };
    return std::string(details[index]);
}

namespace armor_keypoints {
    using I = ArmorKeyPointsIndex;
    constexpr std::array sys_pairs = {
        std::pair { std::to_underlying(I::RIGHT_BOTTOM), std::to_underlying(I::LEFT_BOTTOM) },
        // std::pair { std::to_underlying(I::RIGHT_MID), std::to_underlying(I::LEFT_MID) },
        std::pair { std::to_underlying(I::RIGHT_TOP), std::to_underlying(I::LEFT_TOP) }
    };
} // namespace armor_keypoints
struct ArmorKeyPoints2D {
    using PointT = cv::Point2f;
    using I = ArmorKeyPointsIndex;

    inline void add_offset(const PointT& offset) noexcept {
        for (auto& p_opt: points) {
            if (p_opt) {
                *p_opt += offset;
            }
        }
        // compute_mid(I::LEFT_MID, I::LEFT_TOP, I::LEFT_BOTTOM);
        // compute_mid(I::RIGHT_MID, I::RIGHT_TOP, I::RIGHT_BOTTOM);
        full_points.reset();
        bbox.reset();
    }

    inline void transform(const Eigen::Matrix<float, 3, 3>& transform_matrix) noexcept {
        for (auto& p_opt: points) {
            if (p_opt) {
                *p_opt = utils::transform_point2D(transform_matrix, *p_opt);
            }
        }
        // compute_mid(I::LEFT_MID, I::LEFT_TOP, I::LEFT_BOTTOM);
        // compute_mid(I::RIGHT_MID, I::RIGHT_TOP, I::RIGHT_BOTTOM);
        full_points.reset();
        bbox.reset();
    }
    inline void compute_mid(I mid, I top, I bottom) {
        auto& mid_opt = points[std::to_underlying(mid)];
        mid_opt = (*points[std::to_underlying(top)] + *points[std::to_underlying(bottom)]) / 2.f;
    };

    inline std::array<PointT, std::to_underlying(I::N)>& landmarks() {
        if (!full_points) {
            // compute_mid(I::LEFT_MID, I::LEFT_TOP, I::LEFT_BOTTOM);
            // compute_mid(I::RIGHT_MID, I::RIGHT_TOP, I::RIGHT_BOTTOM);
            std::array<PointT, std::to_underlying(I::N)> tmp {};
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
    inline cv::Rect2f bounding_box() {
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

    std::array<std::optional<PointT>, std::to_underlying(I::N)> points {};

private:
    std::optional<std::array<PointT, std::to_underlying(I::N)>> full_points;
    std::optional<cv::Rect2f> bbox;
};
inline ArmorType armor_type_by_armor_class(ArmorClass armor_class) {
    if (armor_class == ArmorClass::NO1) {
        return ArmorType::Large;
    } else if (armor_class == ArmorClass::BASE || armor_class == ArmorClass::OUTPOST) {
        return ArmorType::BuildingSmall;
    } else {
        return ArmorType::SimpleSmall;
    }
}
template<typename PointT>
inline std::vector<PointT> getArmorKeyPoints3D(ArmorClass armor_class) {
    auto armor_type = armor_type_by_armor_class(armor_class);
    if (armor_type == ArmorType::Large) {
        return ArmorKeyPoint3D<PointT, ArmorType::Large>::build();
    } else if (armor_type == ArmorType::BuildingSmall) {
        return ArmorKeyPoint3D<PointT, ArmorType::BuildingSmall>::build();
    } else {
        return ArmorKeyPoint3D<PointT, ArmorType::SimpleSmall>::build();
    }
}
enum class ArmorColor : int { BLUE = 0, RED, NONE, PURPLE };

constexpr int armor_num_by_armor_class(const ArmorClass& armor_class) {
    constexpr std::array details { 4, 4, 4, 4, 4, 4, 3, 4, 4 };
    return details[std::to_underlying(armor_class)];
}
inline std::string string_by_armor_color(ArmorColor armor_color) {
    constexpr const char* details[] = { "blue", "red", "none", "purple" };
    return std::string(details[std::to_underlying(armor_color)]);
}
inline cv::Scalar CV_color_by_armor_class(ArmorColor armor_color) {
    static cv::Scalar details[] = { cv::Scalar(255, 0, 0),
                                    cv::Scalar(0, 0, 255),
                                    cv::Scalar(255, 255, 255),
                                    cv::Scalar(255, 0, 255) };
    return details[std::to_underlying(armor_color)];
}
inline std::string string_by_armor_class(ArmorClass armor_class) {
    constexpr const char* details[] = { "sentry", "no1",     "no2",  "no3",    "no4",
                                        "no5",    "outpost", "base", "unknown" };
    return std::string(details[std::to_underlying(armor_class)]);
}

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
        std::vector<
            std::array<std::optional<cv::Point2f>, std::to_underlying(ArmorKeyPointsIndex::N)>>
            tmp_points;
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
    void add_offset(const cv::Point2f& offset) {
        if (!has_tidy) {
            throw std::runtime_error("addOffset called before tidy");
        }
        key_points.add_offset(offset);
    }
    inline void transform(const Eigen::Matrix<float, 3, 3>& transform_matrix) {
        if (!has_tidy) {
            throw std::runtime_error("transform called before tidy");
        }
        key_points.transform(transform_matrix);
    }
    inline void draw(cv::Mat& img) {
        if (!has_tidy)
            return;

        auto& pts = key_points.landmarks();

        using I = ArmorKeyPointsIndex;

        auto get = [&](I idx) -> cv::Point {
            auto p = pts[std::to_underlying(idx)];
            // cv::circle(img, p, 5,cv::Scalar(0, 255, 0));
            // cv::putText(img, string_by_armor_key_points_index(std::to_underlying(idx)),p, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));
            return p;
        };

        cv::Point lt = get(I::LEFT_TOP);
        cv::Point rt = get(I::RIGHT_TOP);
        cv::Point rb = get(I::RIGHT_BOTTOM);
        cv::Point lb = get(I::LEFT_BOTTOM);

        cv::line(img, lt, rb, cv::Scalar(0, 255, 0), 2);
        cv::line(img, rb, rt, cv::Scalar(0, 255, 0), 2);
        cv::line(img, rt, lb, cv::Scalar(0, 255, 0), 2);
        cv::line(img, lb, lt, cv::Scalar(0, 255, 0), 2);

        cv::Point bottom_center = (lb + rb) * 0.5;

        bottom_center.y += 20;

        std::string text = get_str();

        int font = cv::FONT_HERSHEY_COMPLEX;
        double scale = 0.5;
        int thickness = 1;

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, font, scale, thickness, &baseline);

        cv::Point text_org(
            bottom_center.x - text_size.width / 2,
            bottom_center.y + text_size.height / 2
        );

        cv::putText(img, text, text_org, font, scale, CV_color_by_armor_class(color), thickness);
    }
    inline std::string get_str() const noexcept {
        return string_by_armor_color(color) + "_" + string_by_armor_class(number);
    }
    Armor() = default;
};
struct Armors {
    std::chrono::steady_clock::time_point timestamp;
    int id = -1;
    int frame_id = -1;
    std::vector<Armor> armors;

    inline void draw(cv::Mat& img) {
        for (auto& armor: armors) {
            armor.draw(img);
        }
    }
};
} // namespace awakening::auto_aim