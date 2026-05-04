#pragma once
#include "utils/common/type_common.hpp"
#include "utils/logger.hpp"
#include <rclcpp/executors.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/time.hpp>
#include <vector>
namespace awakening::rcl {
class RclcppNode {
public:
    struct RclcppGuard {
        RclcppGuard() {
            if (rclcpp::ok() == false) {
                rclcpp::init(0, nullptr);
                AWAKENING_INFO("ROS init");
            }
        }
    } guard_;
    RclcppNode(const std::string& name) {
        rclcpp = std::make_shared<rclcpp::Node>(name);
    }
    [[nodiscard]] auto get_node() {
        return rclcpp;
    }
    void spin_once() {
        rclcpp::spin_some(rclcpp);
    }
    void spin() {
        rclcpp::spin(rclcpp);
    }
    void shutdown() {
        rclcpp::shutdown();
    }
    template<class T>
    auto make_pub(const std::string& topic_name, const rclcpp::QoS& qos) noexcept {
        auto pub = rclcpp->create_publisher<T>(topic_name, qos);
        return pub;
    }
    template<class T, typename F>
    auto make_sub(const std::string& topic_name, const rclcpp::QoS& qos, F&& callback) {
        return rclcpp->create_subscription<T>(topic_name, qos, std::forward<F>(callback));
    }
    void push_sub(rclcpp::SubscriptionBase::SharedPtr sub) {
        subs.push_back(sub);
    }
    TimePoint form_ros_time(rclcpp::Time ros_time) {
        auto ros_now = rclcpp->get_clock()->now();
        auto now = Clock::now();
        auto diff = ros_now - ros_time;
        return now - std::chrono::nanoseconds(diff.nanoseconds());
    }

    std::shared_ptr<rclcpp::Node> rclcpp;
    std::vector<rclcpp::SubscriptionBase::SharedPtr> subs;
    std::vector<rclcpp::PublisherBase> pubs;
};
} // namespace awakening::rcl