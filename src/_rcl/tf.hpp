#pragma once
#include "node.hpp"
#include "utils/common/type_common.hpp"
#include "utils/runtime_tf.hpp"
#include <cstddef>
#include <functional>
#include <memory>
#include <rclcpp/node.hpp>
#include <string>
#include <tf2_ros/transform_broadcaster.h>
namespace awakening::rcl {
class TF {
public:
    TF(RclcppNode& node) {
        node_ = node.get_node();
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
    }
    template<typename FrameEnum, size_t N, bool Static, typename F>
    void pub_robot_tf(const utils::tf::RobotTF<FrameEnum, N, Static>& r_tf, F&& get_frame_name) {
        auto now = node_->now();
        auto edges = r_tf.get_edges();
        for (const auto& edge: edges) {
            auto parent = edge.parent;
            auto child = edge.child;
            ISO3 pose = r_tf.pose_in(
                static_cast<FrameEnum>(child),
                static_cast<FrameEnum>(parent),
                Clock::now()
            );

            geometry_msgs::msg::TransformStamped t_msg;
            t_msg.header.stamp = now;
            t_msg.header.frame_id = get_frame_name(static_cast<FrameEnum>(parent));
            t_msg.child_frame_id = get_frame_name(static_cast<FrameEnum>(child));

            Eigen::Vector3d trans = pose.translation();
            t_msg.transform.translation.x = trans.x();
            t_msg.transform.translation.y = trans.y();
            t_msg.transform.translation.z = trans.z();

            Eigen::Quaterniond q(pose.linear());
            t_msg.transform.rotation.x = q.x();
            t_msg.transform.rotation.y = q.y();
            t_msg.transform.rotation.z = q.z();
            t_msg.transform.rotation.w = q.w();
            tf_broadcaster_->sendTransform(t_msg);
        }
    }

    std::shared_ptr<rclcpp::Node> node_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
} // namespace awakening::rcl