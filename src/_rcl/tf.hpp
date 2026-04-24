#pragma once
#include "node.hpp"
#include "utils/common/type_common.hpp"
#include "utils/runtime_tf.hpp"
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <rclcpp/clock.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/node.hpp>
#include <string>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
namespace awakening::rcl {
class TF {
public:
    TF(RclcppNode& node) {
        node_ = node.get_node();
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node_->get_clock());
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, node_);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
    }
    std::optional<tf2::Transform>
    get_tf2_transform(const std::string& target, const std::string& source, rclcpp::Time t)
        const noexcept {
        try {
            auto tf_msg = tf_buffer_->lookupTransform(target, source, t);
            tf2::Transform tf;
            tf2::fromMsg(tf_msg.transform, tf);
            return tf;
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(rclcpp::get_logger("tf"), "TF lookup failed: %s", ex.what());
            return std::nullopt;
        }
    }

    std::optional<tf2::Transform> get_tf2_transform(
        const std::string& target,
        const std::string& source,
        rclcpp::Time t,
        const rclcpp::Duration& timeout
    ) const noexcept {
        try {
            auto tf_msg = tf_buffer_->lookupTransform(target, source, t, timeout);
            tf2::Transform tf;
            tf2::fromMsg(tf_msg.transform, tf);
            return tf;
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(rclcpp::get_logger("tf"), "TF lookup failed: %s", ex.what());
            return std::nullopt;
        }
    }

    template<typename Scalar>
    using Isometry3 = Eigen::Transform<Scalar, 3, Eigen::Isometry>;

    template<typename Scalar>
    static Isometry3<Scalar> tf2eigen(const geometry_msgs::msg::TransformStamped& tf) noexcept {
        Isometry3<Scalar> T = Isometry3<Scalar>::Identity();

        T.translation() << static_cast<Scalar>(tf.transform.translation.x),
            static_cast<Scalar>(tf.transform.translation.y),
            static_cast<Scalar>(tf.transform.translation.z);

        const auto& q = tf.transform.rotation;
        Eigen::Quaternion<Scalar> Q(
            static_cast<Scalar>(q.w),
            static_cast<Scalar>(q.x),
            static_cast<Scalar>(q.y),
            static_cast<Scalar>(q.z)
        );

        Q.normalize(); // 防止数值漂移
        T.linear() = Q.toRotationMatrix();

        return T;
    }
    template<typename Scalar>
    static geometry_msgs::msg::Transform eigen2tf(const Isometry3<Scalar>& T) noexcept {
        geometry_msgs::msg::Transform msg;

        const auto& t = T.translation();
        msg.translation.x = static_cast<double>(t.x());
        msg.translation.y = static_cast<double>(t.y());
        msg.translation.z = static_cast<double>(t.z());

        Eigen::Quaternion<Scalar> q(T.rotation());
        q.normalize();

        msg.rotation.x = static_cast<double>(q.x());
        msg.rotation.y = static_cast<double>(q.y());
        msg.rotation.z = static_cast<double>(q.z());
        msg.rotation.w = static_cast<double>(q.w());

        return msg;
    }

    template<typename Scalar>
    std::optional<Isometry3<Scalar>>
    get_transform(const std::string& target, const std::string& source, rclcpp::Time t)
        const noexcept {
        try {
            auto tf_msg = tf_buffer_->lookupTransform(target, source, t);
            return tf2eigen<Scalar>(tf_msg);
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(rclcpp::get_logger("tf"), "TF lookup failed: %s", ex.what());
            return std::nullopt;
        }
    }

    template<typename Scalar>
    std::optional<Isometry3<Scalar>> get_transform(
        const std::string& target,
        const std::string& source,
        rclcpp::Time t,
        const rclcpp::Duration& timeout
    ) const noexcept {
        try {
            auto tf_msg = tf_buffer_->lookupTransform(target, source, t, timeout);
            return tf2eigen<Scalar>(tf_msg);
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(rclcpp::get_logger("tf"), "TF lookup failed: %s", ex.what());
            return std::nullopt;
        }
    }
    template<typename FrameEnum, size_t N, bool Static, typename F>
    void pub_robot_tf(const utils::tf::RobotTF<FrameEnum, N, Static>& r_tf, F&& get_frame_name) {
        auto now = node_->now();
        auto edges = r_tf.get_edges();
        for (const auto& edge: edges) {
            auto parent = edge.parent;
            auto child = edge.child;
            ISO3 pose = r_tf.pose_a_in_b(
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
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
};
} // namespace awakening::rcl