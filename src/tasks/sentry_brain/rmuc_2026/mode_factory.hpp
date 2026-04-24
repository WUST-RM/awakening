#pragma once
#include "home_mode.hpp"
#include "tasks/sentry_brain/rmuc_2026/mode_base.hpp"
#include "utils/utils.hpp"
#include <memory>
namespace awakening::sentry_brain {
inline std::unique_ptr<ModeBase>
create_brain_mode(rcl::RclcppNode& rcl_node, rcl::TF& rcl_tf, const YAML::Node& config) {
    auto mode_str = utils::to_upper(config["mode"].as<std::string>());
    if (mode_str == "HOME") {
        return std::make_unique<HomeMode>(rcl_node, rcl_tf, config["home"]);
    }
    // Add more modes as needed
    throw std::invalid_argument("Unknown mode: " + mode_str);
}
} // namespace awakening::sentry_brain