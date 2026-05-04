#include "_rcl/node.hpp"
#include "ascii_banner.hpp"
#include "tasks/radar_detect/rmuc_2026_map.hpp"
#include "utils/common/type_common.hpp"
#include "utils/io/pcd_io.h"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

using namespace awakening;

int main(int argc, char** argv) {
    print_banner();
    auto& signal = utils::SignalGuard::instance();
    logger::init(spdlog::level::trace);

    // 获取配置文件路径
    auto get_arg = [&](int i) -> std::optional<std::string> {
        if (i < argc) {
            AWAKENING_INFO("get args {} ", std::string(argv[i]));
            return std::make_optional(std::string(argv[i]));
        }
        return std::nullopt;
    };

    std::string config_path;
    auto first_arg = get_arg(1);
    if (first_arg) {
        config_path = first_arg.value();
    } else {
        return 1;
    }

    auto config = YAML::LoadFile(config_path);
    radar_detect::RMUC2026Map map(config["map"], radar_detect::SelfColor::RED);
    map.edit();
    map.dump_yaml("guess.yaml");
    return 0;
}