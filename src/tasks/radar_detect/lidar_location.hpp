#pragma once
#include "tasks/radar_detect/type.hpp"
#include "utils/impl.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <utility>
#include <vector>
#include <yaml-cpp/node/node.h>
namespace awakening::radar_detect {
class LidarLocation {
public:
    LidarLocation(const YAML::Node& node);
    // CarPool detect(const std::vector<Eigen::Vector3f>& pts);
    std::pair<Eigen::Vector3f, Eigen::Vector3f> get_target_map_bbox();
    std::vector<Eigen::Vector3f>& get_target_map_pts();
    AWAKENING_IMPL_DEFINITION(LidarLocation)
};
} // namespace awakening::radar_detect