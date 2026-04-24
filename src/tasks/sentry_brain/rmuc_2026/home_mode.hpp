#pragma once
#include "_rcl/node.hpp"
#include "_rcl/tf.hpp"
#include "mode_base.hpp"
#include <memory>
namespace awakening::sentry_brain {

class HomeMode: public ModeBase {
public:
    HomeMode(rcl::RclcppNode& rcl_node, rcl::TF& rcl_tf, const YAML::Node& config);
    void update_gobal_state(const SentryRefereeReceive& packet) override;
    void update_armor_target(const auto_aim::ArmorTarget& target_in_big_yaw) override;
    ~HomeMode();
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace awakening::sentry_brain