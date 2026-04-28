#pragma once
#include "_rcl/tf.hpp"
#include "mode_base.hpp"
#include "utils/impl.hpp"
namespace awakening::sentry_brain {
class DogMode: ModeBase {
public:
    DogMode(rcl::RclcppNode& rcl_node, rcl::TF& rcl_tf, const YAML::Node& config);
    void update_gobal_state(const SentryRefereeReceive& packet) override;
    void update_armor_target(const auto_aim::ArmorTarget& target_in_big_yaw) override;
    AWAKENING_IMPL_DEFINITION(DogMode)
};
} // namespace awakening::sentry_brain