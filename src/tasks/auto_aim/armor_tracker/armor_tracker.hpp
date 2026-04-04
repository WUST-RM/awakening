#pragma once
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "utils/impl.hpp"
#include <yaml-cpp/node/node.h>
namespace awakening::auto_aim {
class ArmorTracker {
public:
    ArmorTracker(const YAML::Node& config);
    [[nodiscard]] ArmorTarget
    track(Armors& armors, const CameraInfo& camera_info, const ISO3& camera_cv_in_odom);
    void pose_solve(Armors& armors, const CameraInfo& camera_info, const ISO3& camera_cv_in_odom);
    int get_count();
    void reset_count();
    AWAKENING_IMPL_DEFINITION(ArmorTracker)
};
} // namespace awakening::auto_aim