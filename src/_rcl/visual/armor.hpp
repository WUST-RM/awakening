#pragma once
#include "../node.hpp"
#include "tasks/auto_aim/type.hpp"
#include <memory>
#include <optional>
#include <string>
#include <visualization_msgs/msg/marker_array.hpp>
namespace awakening::rcl {
inline void
pub_armor_marker(RclcppNode& node, std::string frame_id, const auto_aim::Armors& armors) {
    static rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub;
    if (!marker_pub) {
        marker_pub =
            node.make_pub<visualization_msgs::msg::MarkerArray>("armor_markers", rclcpp::QoS(10));
    }
    static std::optional<visualization_msgs::msg::Marker> armor_marker;
    if (!armor_marker) {
        armor_marker = visualization_msgs::msg::Marker();
        armor_marker->action = visualization_msgs::msg::Marker::ADD;
        armor_marker->type = visualization_msgs::msg::Marker::CUBE;
        armor_marker->scale.x = 0.05;
        armor_marker->color.a = 1.0;
        armor_marker->color.g = 0.2;
        armor_marker->color.b = 0.2;
        armor_marker->color.r = 1.0;
        armor_marker->lifetime = rclcpp::Duration::from_seconds(0.1);
    }
    auto getWH = [](auto_aim::ArmorType type) {
        switch (type) {
            case auto_aim::ArmorType::SimpleSmall:
                return std::make_pair(
                    auto_aim::ArmorTypeTraits<auto_aim::ArmorType::SimpleSmall>::WIDTH,
                    auto_aim::ArmorTypeTraits<auto_aim::ArmorType::SimpleSmall>::HEIGHT
                );
            case auto_aim::ArmorType::Large:
                return std::make_pair(
                    auto_aim::ArmorTypeTraits<auto_aim::ArmorType::Large>::WIDTH,
                    auto_aim::ArmorTypeTraits<auto_aim::ArmorType::Large>::HEIGHT
                );
        }
        return std::make_pair(
            auto_aim::ArmorTypeTraits<auto_aim::ArmorType::SimpleSmall>::WIDTH,
            auto_aim::ArmorTypeTraits<auto_aim::ArmorType::SimpleSmall>::HEIGHT
        );
    };
    armor_marker.value().id = 0;
    visualization_msgs::msg::MarkerArray marker_array;
    for (const auto& armor: armors.armors) {
        visualization_msgs::msg::Marker& marker = armor_marker.value();
        marker.id++;
        marker.header.frame_id = frame_id;
        marker.header.stamp = rclcpp::Clock().now();
        auto wh = getWH(auto_aim::getArmorTypebyArmorClass(armor.number));
        marker.scale.y = wh.first;
        marker.scale.z = 0.135;
        const Eigen::Vector3d& t = armor.pose.translation();
        const Eigen::Matrix3d& R = armor.pose.rotation();
        Eigen::Quaterniond q(R);
        marker.pose.position.x = t.x();
        marker.pose.position.y = t.y();
        marker.pose.position.z = t.z();
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();
        marker.pose.orientation.w = q.w();
        marker_array.markers.push_back(marker);
    }

    marker_pub->publish(marker_array);
}
} // namespace awakening::rcl