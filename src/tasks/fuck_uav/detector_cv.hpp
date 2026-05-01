#pragma once

#include "tasks/base/common.hpp"
#include "utils/impl.hpp"
#include <yaml-cpp/node/node.h>
namespace awakening::fuck_uav {
class DetectorCV {
public:
    DetectorCV(const YAML::Node& config);
    void detect(const CommonFrame& frame);
    AWAKENING_IMPL_DEFINITION(DetectorCV)
};
} // namespace awakening::fuck_uav