#pragma once
#include "rclcpp/rclcpp.hpp"
#include "tasks/radar_detect/target.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <array>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <rclcpp/duration.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <visualization_msgs/msg/marker_array.hpp>
namespace awakening::radar_detect {
enum class ArmorClass : int { NO1, NO2, NO3, NO4, NO5, SENTRY, OUTPOST, UNKNOWN, N };
enum class ArmorColor : int { RED, BLUE, NONE, N };
enum class SelfColor : bool { RED, BLUE };
inline SelfColor SelfColor_from_str(const std::string& _str) {
    auto str = utils::to_upper(_str);
    if (str == "RED") {
        return SelfColor::RED;
    } else if (str == "BLUE") {
        return SelfColor::BLUE;
    } else {
        throw std::invalid_argument("Invalid self color string");
    }
}
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
enum class CarState : int { ACTIVE, NEEDGUESS, GUESSING };
enum class CarClass : int {
    R1 = 1,
    R2 = 2,
    R3 = 3,
    R4 = 4,
    R7 = 7,
    B1 = 101,
    B2 = 102,
    B3 = 103,
    B4 = 104,
    B7 = 107,
    UNKNOWN = -10,
};
inline std::string CarClass_to_str(CarClass c) {
    switch (c) {
        case CarClass::R1:
            return "R1";
        case CarClass::R2:
            return "R2";
        case CarClass::R3:
            return "R3";
        case CarClass::R4:
            return "R4";
        case CarClass::R7:
            return "R7";
        case CarClass::B1:
            return "B1";
        case CarClass::B2:
            return "B2";
        case CarClass::B3:
            return "B3";
        case CarClass::B4:
            return "B4";
        case CarClass::B7:
            return "B7";
        case CarClass::UNKNOWN:
        default:
            return "UNKNOWN";
    }
}
struct Car {
    cv::Rect2f bbox;
    float confidence;
    std::vector<Armor> armors;
    ArmorClass number = ArmorClass::UNKNOWN;
    ArmorColor color = ArmorColor::NONE;
    Eigen::Vector3d point_in_uwb;
    TimePoint timestamp;
    CarClass get_car_class() const {
        if (color == ArmorColor::NONE) {
            return CarClass::UNKNOWN;
        } else if (color == ArmorColor::BLUE) {
            switch (number) {
                case ArmorClass::NO1: {
                    return CarClass::B1;
                }
                case ArmorClass::NO2: {
                    return CarClass::B2;
                }
                case ArmorClass::NO3: {
                    return CarClass::B3;
                }
                case ArmorClass::NO4: {
                    return CarClass::B4;
                }
                case ArmorClass::SENTRY: {
                    return CarClass::B7;
                }
                default: {
                    return CarClass::UNKNOWN;
                }
            }
        } else {
            switch (number) {
                case ArmorClass::NO1: {
                    return CarClass::R1;
                }
                case ArmorClass::NO2: {
                    return CarClass::R2;
                }
                case ArmorClass::NO3: {
                    return CarClass::R3;
                }
                case ArmorClass::NO4: {
                    return CarClass::R4;
                }
                case ArmorClass::SENTRY: {
                    return CarClass::R7;
                }
                default: {
                    return CarClass::UNKNOWN;
                }
            }
        }
        return CarClass::UNKNOWN;
    }
    cv::Point2f get_key_point() const {
        auto car_center = bbox.tl() + cv::Point2f(bbox.width / 2, bbox.height / 2);
        cv::Point2f key_p = car_center;
        double min_dis = std::numeric_limits<double>::max();
        for (const auto& armor: armors) {
            auto armor_center =
                armor.bbox.tl() + cv::Point2f(armor.bbox.width / 2, armor.bbox.height / 2);
            double dis = cv::norm(car_center - armor_center);
            if (dis < min_dis) {
                min_dis = dis;
                key_p = armor_center;
            }
        }
        return key_p;
    }
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
struct Cars {
    std::vector<Car> cars;
    TimePoint t;
    int id;
};
struct AABB {
    Eigen::Vector3f min_pt;
    Eigen::Vector3f max_pt;
    AABB() = default;
    AABB(Eigen::Vector3f min_pt_, Eigen::Vector3f max_pt_): min_pt(min_pt_), max_pt(max_pt_) {}
    double volume() const {
        return (max_pt - min_pt).prod();
    }

    std::array<Eigen::Vector3f, 8> get_corners() const {
        const auto& min = min_pt;
        const auto& max = max_pt;

        return {
            Eigen::Vector3f(min.x(), min.y(), min.z()), Eigen::Vector3f(min.x(), min.y(), max.z()),
            Eigen::Vector3f(min.x(), max.y(), min.z()), Eigen::Vector3f(min.x(), max.y(), max.z()),
            Eigen::Vector3f(max.x(), min.y(), min.z()), Eigen::Vector3f(max.x(), min.y(), max.z()),
            Eigen::Vector3f(max.x(), max.y(), min.z()), Eigen::Vector3f(max.x(), max.y(), max.z())
        };
    }
};
struct ClusterResult {
    std::vector<Eigen::Vector4f> cluster;
    AABB aabb;
    Eigen::Vector3f grav;
};
static constexpr double FIELD_WIDTH = 15.0;
static constexpr double FIELD_LONGTH = 28.0;
struct Image {
    cv::Mat image;
    SelfColor self_color;
    Image clone() const {
        return { .image = image.clone(), .self_color = self_color };
    }
};
inline Eigen::Vector2d image_to_uwb(const Image& image, const cv::Point2f& img_point) {
    Eigen::Vector2d uwb;
    double x_scale = FIELD_LONGTH / image.image.cols;
    double y_scale = FIELD_WIDTH / image.image.rows;
    if (image.self_color == SelfColor::RED) {
        uwb.x() = img_point.x * x_scale;
        uwb.y() = (image.image.rows - img_point.y) * y_scale;
    } else {
        uwb.x() = (image.image.cols - img_point.x) * x_scale;
        uwb.y() = img_point.y * y_scale;
    }
    return uwb;
};
inline cv::Point2f uwb_to_image(const Image& image, const Eigen::Vector2d& uwb_point) {
    cv::Point2f img_point;
    double x_scale = FIELD_LONGTH / image.image.cols;
    double y_scale = FIELD_WIDTH / image.image.rows;
    if (image.self_color == SelfColor::RED) {
        img_point.x = uwb_point.x() / x_scale;
        img_point.y = (FIELD_WIDTH - uwb_point.y()) / y_scale;
    } else {
        img_point.x = (FIELD_LONGTH - uwb_point.x()) / x_scale;
        img_point.y = uwb_point.y() / y_scale;
    }
    return img_point;
}
} // namespace awakening::radar_detect