#pragma once
#include "tasks/auto_aim/type.hpp"
#include "utils/impl.hpp"
#include <vector>
namespace awakening::auto_aim {
class ArmorDetector {
public:
    using Ptr = std::unique_ptr<ArmorDetector>;
    ArmorDetector(const YAML::Node& config);
    [[nodiscard]] std::vector<Armor> detect(const CommonFrame& frame);
    AWAKENING_IMPL_DEFINITION(ArmorDetector)
};
} // namespace awakening::auto_aim