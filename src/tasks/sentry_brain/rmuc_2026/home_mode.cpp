#include "home_mode.hpp"
#include "map.hpp"
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/sentry_brain/rmuc_2026/gobal_state.hpp"
#include "utils/logger.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <cstdlib>
#include <geometry_msgs/msg/detail/pose_stamped__struct.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <nav_msgs/msg/detail/odometry__struct.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <optional>
#include <rclcpp/publisher.hpp>
#include <rclcpp/subscription.hpp>
#include <string>
#include <thread>
#include <yaml-cpp/node/node.h>
namespace awakening::sentry_brain {
struct HomeMode::Impl {
    struct Params {
        double go_home_hp_ratio;
        int home_bullet_num;
        void load(const YAML::Node& config) {
            go_home_hp_ratio = config["go_home_hp_ratio"].as<double>();
            home_bullet_num = config["home_bullet_num"].as<int>();
        }
    } params_;
    Impl(rcl::RclcppNode& rcl_node, rcl::TF& rcl_tf, const YAML::Node& config):
        rcl_node_(rcl_node),
        rcl_tf_(rcl_tf) {
        goal_pub_ =
            rcl_node_.make_pub<geometry_msgs::msg::PoseStamped>("rose_goal", rclcpp::QoS(10));
        odom_sub_ = rcl_node_.make_sub<nav_msgs::msg::Odometry>(
            "Odometry",
            rclcpp::QoS(10),
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                const auto& odom_in = *msg;

                static Eigen::Isometry3d T;
                if (auto opt = rcl_tf_.get_transform<double>(
                        "map",
                        odom_in.header.frame_id,
                        odom_in.header.stamp,
                        rclcpp::Duration::from_seconds(0.1)
                    ))
                {
                    T = *opt;
                } else {
                }
                Eigen::Vector3d p(
                    odom_in.pose.pose.position.x,
                    odom_in.pose.pose.position.y,
                    odom_in.pose.pose.position.z
                );
                p = T * p;
                current_pos_ = p.head<3>();
            }
        );
    }
    ~Impl() {
        stop();
    }
    void stop() {
        running_ = false;
        if (pub_goal_thread_.joinable()) {
            pub_goal_thread_.join();
        }
        if (tick_thread_.joinable()) {
            tick_thread_.join();
        }
    }
    void start() {
        running_ = true;
        pub_goal_thread_ = std::thread([&]() {
            auto next_tp = std::chrono::steady_clock::now();
            while (rclcpp::ok()) {
                next_tp += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(3.0)
                );
                pub_goal_callback();
                std::this_thread::sleep_until(next_tp);
            }
        });
        tick_thread_ = std::thread([&]() {
            auto next_tp = std::chrono::steady_clock::now();
            while (rclcpp::ok()) {
                next_tp += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(1 / 2.0)
                );
                tick_callback();
                std::this_thread::sleep_until(next_tp);
            }
        });
    }
    void update_gobal_state(const SentryRefereeReceive& packet) noexcept {
        state_.update(packet);
    }
    void update_armor_target(const auto_aim::ArmorTarget& t) noexcept {
        target_in_big_yaw_ = t;
    }
    template<typename Func>
    void wait_until(Func&& func, std::chrono::duration<double> check_dt) const noexcept {
        while (running_) {
            if (func())
                break;
            std::this_thread::sleep_for(check_dt);
        }
    }
    void tick_callback() noexcept {
        auto& map = RMUC2026Map::instance();
        if (state_.current_game_time_ < 0) {
            AWAKENING_INFO("waiting for game start... current_time: {}", state_.current_game_time_);
            return;
        }
        if (in_home()) {
            state_.home_allowance_bullets_ = 0;
        }
        if (target_in_big_yaw_.check()) {
            sentry_pose = GobalState::Pose::Attack;
        }
        double cur_hp_ratio = double(state_.current_hp) / state_.max_hp;
        if (cur_hp_ratio < params_.go_home_hp_ratio || state_.current_hp < 60) {
            auto tmp_pose = sentry_pose;
            sentry_pose = GobalState::Pose::Defend;
            go<home_t>();
            wait_until(
                [&]() {
                    if (std::abs(state_.current_hp - state_.max_hp) < 50) {
                        AWAKENING_INFO("hp is enough: {}", state_.current_hp);
                        return true;
                    }
                    AWAKENING_INFO("waiting for hp to recover: {}", state_.current_hp);
                    return false;
                },
                std::chrono::duration<double>(1.0)
            );
            sentry_pose = tmp_pose;
            return;
        }
        if (state_.current_bullets_ < params_.home_bullet_num) {
            auto tmp_pose = sentry_pose;
            sentry_pose = GobalState::Pose::Move;
            if (state_.home_allowance_bullets_ > 10) {
                go<home_t>();
            } else {
                go<ally_fort_t>();
            }
            sentry_pose = tmp_pose;
            return;
        }
        if (state_.current_game_time_ < 60) {
            auto tmp_pose = sentry_pose;
            sentry_pose = GobalState::Pose::Move;
            go<enemy_fly_land_t>();
            sentry_pose = tmp_pose;
            return;
        }
        auto tmp_pose = sentry_pose;
        sentry_pose = GobalState::Pose::Move;
        go<ally_second_step_bottom_t>();
        sentry_pose = tmp_pose;
    }
    bool in_home() {
        auto& map = RMUC2026Map::instance();
        return (current_pos_ - map.get<home_t>()).norm() < 0.5;
    }
    template<typename Key>
    bool wait_reached() {
        auto& map = RMUC2026Map::instance();
        if ((current_pos_ - map.get<Key>()).norm() < 0.5) {
            AWAKENING_INFO("{} has reached", Key::name);
            return true;
        }
        return false;
    }
    template<typename Key>
    void go() noexcept {
        auto& map = RMUC2026Map::instance();
        go(map.get<Key>(), Key::name);
    }
    void go(const Vec3& goal, std::string name) noexcept {
        current_goal_ = goal;
        AWAKENING_INFO("go to {}: x: {} y: {} z: {}", name, goal.x(), goal.y(), goal.z());
    }

    void pub_goal_callback() noexcept {
        if (!current_goal_) {
            return;
        }
        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp = rcl_node_.get_node()->now();
        msg.header.frame_id = "map";
        msg.pose.position.x = current_goal_->x();
        msg.pose.position.y = current_goal_->y();
        msg.pose.position.z = current_goal_->z();
        goal_pub_->publish(msg);
    }
    GobalState::Pose sentry_pose = GobalState::Pose::Attack;
    std::optional<Eigen::Vector3d> current_goal_;
    std::thread pub_goal_thread_;
    std::thread tick_thread_;
    GobalState state_;
    Eigen::Vector3d current_pos_;
    auto_aim::ArmorTarget target_in_big_yaw_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    bool running_ = false;
    rcl::RclcppNode& rcl_node_;
    rcl::TF& rcl_tf_;
};
HomeMode::HomeMode(rcl::RclcppNode& rcl_node, rcl::TF& rcl_tf, const YAML::Node& config) {
    _impl = std::make_unique<Impl>(rcl_node, rcl_tf, config);
}
void HomeMode::update_armor_target(const auto_aim::ArmorTarget& target_in_big_yaw) {
    _impl->update_armor_target(target_in_big_yaw);
}
void HomeMode::update_gobal_state(const SentryRefereeReceive& packet) {
    _impl->update_gobal_state(packet);
}
HomeMode::~HomeMode() {
    _impl.reset();
}
} // namespace awakening::sentry_brain