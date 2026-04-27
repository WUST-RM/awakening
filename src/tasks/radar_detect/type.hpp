#pragma once
#include "utils/utils.hpp"
#include <array>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>
namespace awakening::radar_detect {
enum class ArmorClass : int { NO1, NO2, NO3, NO4, NO5, SENTRY, OUTPOST, UNKNOWN, N };
enum class ArmorColor : int { RED, BLUE, NONE, N };
inline std::string armor_class_to_str(ArmorClass cls) {
    constexpr const char* details[] = { "NO1", "NO2",    "NO3",     "NO4",
                                        "NO5", "SENTRY", "OUTPOST", "UNKNOWN" };
    return details[static_cast<size_t>(cls)];
}
inline std::string armor_color_to_str(ArmorColor color) {
    constexpr const char* details[] = { "RED", "BLUE", "NONE" };
    return details[static_cast<size_t>(color)];
}
inline cv::Scalar armor_color_to_cv_scalar(ArmorColor color) {
    switch (color) {
        case ArmorColor::RED:
            return cv::Scalar(0, 0, 255);
        case ArmorColor::BLUE:
            return cv::Scalar(255, 0, 0);
        case ArmorColor::NONE:
        default:
            return cv::Scalar(0, 255, 0);
    }
}
struct Armor {
    ArmorClass number;
    cv::Rect2f bbox;
    float confidence;
    ArmorColor color;
    void draw(cv::Mat& img) const {
        cv::rectangle(img, bbox, armor_color_to_cv_scalar(color), 5);
        cv::putText(
            img,
            armor_class_to_str(number) + "_" + armor_color_to_str(color),
            bbox.tl(),
            cv::FONT_HERSHEY_SIMPLEX,
            1.5,
            armor_color_to_cv_scalar(color),
            2
        );
    }
};
struct Car {
    cv::Rect2f bbox;
    float confidence;
    std::vector<Armor> armors;
    ArmorClass number = ArmorClass::UNKNOWN;
    ArmorColor color = ArmorColor::NONE;
    void tidy() {
        if (armors.empty())
            return;
        std::array<double, std::to_underlying(ArmorColor::N)> color_scores {};
        std::array<double, std::to_underlying(ArmorClass::N)> class_scores {};

        for (const auto& armor: armors) {
            color_scores[std::to_underlying(armor.color)] += armor.confidence;
            class_scores[std::to_underlying(armor.number)] += armor.confidence;
        }

        auto max_color_it = std::max_element(color_scores.begin(), color_scores.end());
        color = static_cast<ArmorColor>(std::distance(color_scores.begin(), max_color_it));

        auto max_class_it = std::max_element(class_scores.begin(), class_scores.end());
        number = static_cast<ArmorClass>(std::distance(class_scores.begin(), max_class_it));
    }
    void draw(cv::Mat& img) const {
        cv::rectangle(img, bbox, armor_color_to_cv_scalar(color), 5);
        cv::putText(
            img,
            armor_class_to_str(number) + "_" + armor_color_to_str(color),
            bbox.tl(),
            cv::FONT_HERSHEY_SIMPLEX,
            1.5,
            armor_color_to_cv_scalar(color),
            2
        );
        for (const auto& armor: armors) {
            armor.draw(img);
        }
    }
};
} // namespace awakening::radar_detect