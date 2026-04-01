#pragma once
#include <rclcpp/executors.hpp>
#include <rclcpp/node.hpp>
namespace awakening::rcl {
class RclcppNode {
public:
    struct RclcppGuard {
        RclcppGuard() {
            if (rclcpp::ok() == false) {
                rclcpp::init(0, nullptr);
            }
        }
    } guard_;
    RclcppNode(const std::string& name, const std::string& namespace_ = "") {
        rclcpp = std::make_shared<rclcpp::Node>(name, namespace_);
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
        return rclcpp->create_publisher<T>(topic_name, qos);
    }
    template<class T, typename F>
    auto make_sub(const std::string& topic_name, const rclcpp::QoS& qos, F&& callback) {
        return rclcpp->create_subscription<T>(topic_name, qos, std::forward<F>(callback));
    }

    std::shared_ptr<rclcpp::Node> rclcpp;
};
} // namespace awakening::rcl