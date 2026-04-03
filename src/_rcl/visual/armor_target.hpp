#pragma once
#include "../node.hpp"
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "utils/common/type_common.hpp"
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <visualization_msgs/msg/marker_array.hpp>

namespace awakening::rcl {
inline void pub_armor_target_marker(
    RclcppNode& node,
    std::string frame_id,
    const auto_aim::ArmorTarget& target
) {
    static rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub;
    if (!marker_pub) {
        marker_pub = node.make_pub<visualization_msgs::msg::MarkerArray>(
            "armor_target_markers",
            rclcpp::QoS(10)
        );
    }
    static bool once = true;
    static visualization_msgs::msg::Marker position_marker;
    static visualization_msgs::msg::Marker linear_v_marker;
    static visualization_msgs::msg::Marker angular_v_marker;
    static visualization_msgs::msg::Marker armors_marker;
    if (once) {
        once = false;
        position_marker.ns = "position";
        position_marker.type = visualization_msgs::msg::Marker::SPHERE;
        position_marker.scale.x = position_marker.scale.y = position_marker.scale.z = 0.1;
        position_marker.color.a = 1.0;
        position_marker.color.g = 1.0;
        linear_v_marker.type = visualization_msgs::msg::Marker::ARROW;
        linear_v_marker.ns = "linear_v";
        linear_v_marker.scale.x = 0.03;
        linear_v_marker.scale.y = 0.05;
        linear_v_marker.color.a = 1.0;
        linear_v_marker.color.r = 1.0;
        linear_v_marker.color.g = 1.0;
        angular_v_marker.type = visualization_msgs::msg::Marker::ARROW;
        angular_v_marker.ns = "angular_v";
        angular_v_marker.scale.x = 0.03;
        angular_v_marker.scale.y = 0.05;
        angular_v_marker.color.a = 1.0;
        angular_v_marker.color.b = 1.0;
        angular_v_marker.color.g = 1.0;
        armors_marker.ns = "filtered_armors";
        armors_marker.type = visualization_msgs::msg::Marker::CUBE;
        armors_marker.scale.x = 0.03;
        armors_marker.scale.z = 0.125;
        armors_marker.color.a = 1.0;
        armors_marker.color.b = 1.0;
    }
    visualization_msgs::msg::MarkerArray marker_array;
    if (target.check()) {
        auto target_state = target.get_target_state();
        target_state.predict(Clock::now());
        position_marker.header.frame_id = frame_id;
        position_marker.header.stamp = rclcpp::Clock().now();
        position_marker.action = visualization_msgs::msg::Marker::ADD;
        position_marker.id = 1;
        position_marker.pose.position.x = target_state.pos().x();
        position_marker.pose.position.y = target_state.pos().y();
        position_marker.pose.position.z = target_state.pos().z();
        linear_v_marker.header = position_marker.header;
        linear_v_marker.action = visualization_msgs::msg::Marker::ADD;
        linear_v_marker.id = 1;
        linear_v_marker.points.clear();
        linear_v_marker.points.emplace_back(position_marker.pose.position);
        geometry_msgs::msg::Point arrow_end = position_marker.pose.position;
        arrow_end.x += target_state.vel().x();
        arrow_end.y += target_state.vel().y();
        arrow_end.z += target_state.vel().z();
        linear_v_marker.points.emplace_back(arrow_end);
        angular_v_marker.header = position_marker.header;
        angular_v_marker.action = visualization_msgs::msg::Marker::ADD;
        angular_v_marker.id = 1;
        angular_v_marker.points.clear();
        angular_v_marker.points.emplace_back(position_marker.pose.position);
        arrow_end = position_marker.pose.position;
        arrow_end.z += target_state.vyaw() / M_PI;
        angular_v_marker.points.emplace_back(arrow_end);

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
        armors_marker.header = position_marker.header;
        armors_marker.action = visualization_msgs::msg::Marker::ADD;

        auto wh = getWH(auto_aim::getArmorTypebyArmorClass(target.target_number));
        armors_marker.scale.y = wh.first;
        armors_marker.scale.z = 0.135;
        auto armors_pose = target_state.get_armors_pose(target.target_number);
        armors_marker.id = 0;
        for (auto& armor_pose: armors_pose) {
            armors_marker.id++;
            const Eigen::Vector3d& t = armor_pose.translation();
            const Eigen::Matrix3d& R = armor_pose.rotation();
            Eigen::Quaterniond q(R);
            armors_marker.pose.position.x = t.x();
            armors_marker.pose.position.y = t.y();
            armors_marker.pose.position.z = t.z();
            armors_marker.pose.orientation.x = q.x();
            armors_marker.pose.orientation.y = q.y();
            armors_marker.pose.orientation.z = q.z();
            armors_marker.pose.orientation.w = q.w();
            marker_array.markers.push_back(armors_marker);
        }
        marker_array.markers.push_back(position_marker);
        marker_array.markers.push_back(linear_v_marker);
        marker_array.markers.push_back(angular_v_marker);
    }

    marker_pub->publish(marker_array);
}
} // namespace awakening::rcl