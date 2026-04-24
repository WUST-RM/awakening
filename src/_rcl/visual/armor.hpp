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
    static bool once = true;
    static visualization_msgs::msg::Marker armor_marker;
    static visualization_msgs::msg::Marker text_marker;
    if (once) {
        armor_marker = visualization_msgs::msg::Marker();
        armor_marker.action = visualization_msgs::msg::Marker::ADD;
        armor_marker.type = visualization_msgs::msg::Marker::CUBE;
        armor_marker.scale.x = 0.05;
        armor_marker.color.a = 1.0;
        armor_marker.color.g = 0.2;
        armor_marker.color.b = 0.2;
        armor_marker.color.r = 1.0;
        armor_marker.lifetime = rclcpp::Duration::from_seconds(0.1);

        text_marker.ns = "classification";
        text_marker.action = visualization_msgs::msg::Marker::ADD;
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.scale.z = 0.1;
        text_marker.color.a = 1.0;
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
    }
    auto getWH = [](auto_aim::ArmorType type) {
        switch (type) {
            case auto_aim::ArmorType::SimpleSmall:
                return std::make_pair(
                    auto_aim::ArmorTypeTraits<auto_aim::ArmorType::SimpleSmall>::WIDTH,
                    auto_aim::ArmorTypeTraits<auto_aim::ArmorType::SimpleSmall>::HEIGHT
                );
            case auto_aim::ArmorType::BuildingSmall:
                return std::make_pair(
                    auto_aim::ArmorTypeTraits<auto_aim::ArmorType::BuildingSmall>::WIDTH,
                    auto_aim::ArmorTypeTraits<auto_aim::ArmorType::BuildingSmall>::HEIGHT
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
    armor_marker.id = 0;
    text_marker.id = 0;
    visualization_msgs::msg::MarkerArray marker_array;
    for (const auto& armor: armors.armors) {
        armor_marker.id++;
        text_marker.id++;
        armor_marker.header.frame_id = frame_id;
        armor_marker.header.stamp = rclcpp::Clock().now();
        text_marker.header = armor_marker.header;
        auto wh = getWH(auto_aim::armor_type_by_armor_class(armor.number));
        armor_marker.scale.y = wh.first;
        armor_marker.scale.z = 0.135;
        const Eigen::Vector3d& t = armor.pose.translation();
        const Eigen::Matrix3d& R = armor.pose.rotation();
        Eigen::Quaterniond q(R);
        armor_marker.pose.position.x = t.x();
        armor_marker.pose.position.y = t.y();
        armor_marker.pose.position.z = t.z();
        armor_marker.pose.orientation.x = q.x();
        armor_marker.pose.orientation.y = q.y();
        armor_marker.pose.orientation.z = q.z();
        armor_marker.pose.orientation.w = q.w();
        text_marker.pose.position = armor_marker.pose.position;
        text_marker.pose.position.z += armor_marker.scale.z + text_marker.scale.z / 2.0;
        std::string text = armor.get_str();
        text_marker.text = text;
        marker_array.markers.push_back(armor_marker);
        marker_array.markers.push_back(text_marker);
    }

    marker_pub->publish(marker_array);
}
} // namespace awakening::rcl