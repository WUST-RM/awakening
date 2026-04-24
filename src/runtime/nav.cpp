#include "_rcl/node.hpp"
#include "ascii_banner.hpp"
#include "backward-cpp/backward.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "tasks/base/packet_typedef_receive.hpp"
#include "tasks/base/packet_typedef_send.hpp"
#include "utils/drivers/serial_driver.hpp"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
using namespace awakening;
namespace backward {
static backward::SignalHandling sh;
}
struct SerialTag {};
using SerialIO = IOPair<SerialTag, std::vector<uint8_t>>;
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
    Scheduler s;
    SerialDriver serial(config["serial"], s);
    rcl::RclcppNode rcl_node("auto_aim");
    s.register_task<SerialIO>("receive_serial", [&](SerialIO::second_type&& data) {

    });
    auto cmd_sub = rcl_node.make_sub<geometry_msgs::msg::Twist>(
        "cmd_vel",
        rclcpp::QoS(10),
        [&](const geometry_msgs::msg::Twist::SharedPtr msg) {
            SendNavCmdData send;

            send.cmd_ID = SendNavCmdData::ID;
            send.vx = msg->linear.x;
            send.vy = msg->linear.y;
            send.wz = msg->angular.z;
            serial.write(utils::to_vector(send));
        }
    );
    rcl_node.push_sub(cmd_sub);
    serial.start<SerialTag>("serial");
    std::thread([&]() { rcl_node.spin(); }).detach();
    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();

    rcl_node.shutdown();

    return 0;
}