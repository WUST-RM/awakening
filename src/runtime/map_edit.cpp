#include "tasks/sentry_brain/rmuc_2026/map.hpp"
#include "utils/logger.hpp"
#include <optional>
#include <string>
using namespace awakening::sentry_brain;
int main(int argc, char** argv) {
    awakening::logger::init(spdlog::level::trace);
    auto get_arg = [&](int i) -> std::optional<std::string> {
        if (i < argc) {
            AWAKENING_INFO("get args {} ", std::string(argv[i]));
            return std::make_optional(std::string(argv[i]));
        }
        return std::nullopt;
    };
    auto& map = RMUC2026Map::instance();
    auto second_arg = get_arg(2);
    if (second_arg) {
        map.load_points_yaml(second_arg.value());
    }
    map.load_ros_map_yaml(get_arg(1).value());
    map.visualize();
    map.dump_yaml(second_arg.value_or("rmuc_2026_map_point.yaml"));
}