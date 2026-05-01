#include "ascii_banner.hpp"
#include "tasks/radar_detect/livox_vi_driver/livox_v1_publisher.hpp"
#include "utils/signal_guard.hpp"
using namespace awakening;

int main(int argc, char** argv) {
    print_banner();
    auto& signal = utils::SignalGuard::instance();
    logger::init(spdlog::level::trace);
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

    rcl::RclcppNode rcl_node("lidar_driver");
    livox_v1_lidar::LidarPublisher lidar(config["lidar_driver"], rcl_node);
    std::thread([&]() { rcl_node.spin(); }).detach();
    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    rcl_node.shutdown();
    return 0;
}