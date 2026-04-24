#pragma once
#include "../node.hpp"
#include "utils/common/type_common.hpp"
#include <string>
#include <visualization_msgs/msg/marker_array.hpp>
namespace awakening::rcl {
template<class Tag>
inline void pub_arrow(
    std::string topic,
    RclcppNode& node,
    std::string frame_id,
    const Vec3& start,
    const Vec3& end
) {
    static rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub;
    if (!marker_pub) {
        marker_pub = node.make_pub<visualization_msgs::msg::MarkerArray>(topic, rclcpp::QoS(10));
    }
    static visualization_msgs::msg::Marker arrow_marker;
    static visualization_msgs::msg::Marker point_marker;
    static bool once = true;
    if (once) {
        once = false;
        arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
        arrow_marker.ns = "arrow";
        arrow_marker.scale.x = 0.03;
        arrow_marker.scale.y = 0.05;
        arrow_marker.color.a = 1.0;
        arrow_marker.color.r = 1.0;
        arrow_marker.color.g = 1.0;
        arrow_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
        point_marker.ns = "position";
        point_marker.type = visualization_msgs::msg::Marker::SPHERE;
        point_marker.scale.x = point_marker.scale.y = point_marker.scale.z = 0.3;
        point_marker.color.a = 1.0;
        point_marker.color.g = 1.0;
        point_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
    }
    visualization_msgs::msg::MarkerArray marker_array;
    arrow_marker.header.frame_id = frame_id;
    arrow_marker.header.stamp = node.get_node()->get_clock()->now();
    arrow_marker.action = visualization_msgs::msg::Marker::ADD;
    arrow_marker.id = 1;
    arrow_marker.points.clear();
    auto get_point = [](const Vec3& p) {
        geometry_msgs::msg::Point point;
        point.x = p.x();
        point.y = p.y();
        point.z = p.z();
        return point;
    };
    arrow_marker.points.emplace_back(get_point(start));
    arrow_marker.points.emplace_back(get_point(end));
    point_marker.header = arrow_marker.header;
    point_marker.action = visualization_msgs::msg::Marker::ADD;
    point_marker.id = 1;
    point_marker.pose.position.x = start.x();
    point_marker.pose.position.y = start.y();
    point_marker.pose.position.z = start.z();
    marker_array.markers.push_back(arrow_marker);
    marker_array.markers.push_back(point_marker);
    marker_pub->publish(marker_array);
}
} // namespace awakening::rcl