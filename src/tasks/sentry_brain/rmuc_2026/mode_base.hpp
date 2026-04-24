#pragma once
#include "gobal_state.hpp"
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/base/packet_typedef_receive.hpp"
namespace awakening::sentry_brain {
class ModeBase {
public:
    virtual void update_gobal_state(const SentryRefereeReceive& packet) = 0;
    virtual void update_armor_target(const auto_aim::ArmorTarget& target_in_big_yaw) = 0;
    virtual ~ModeBase() = default;
};
} // namespace awakening::sentry_brain