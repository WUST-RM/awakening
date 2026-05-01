#include "ascii_banner.hpp"
#include "backward-cpp/backward.hpp"
#include "config.hpp"
#include "param_deliver.h"
#include "tasks/auto_aim/armor_tracker/motion_model_point.hpp"
#include "tasks/base/ballistic_trajectory.hpp"
#include "tasks/base/common.hpp"
#include "tasks/base/packet_typedef_receive.hpp"
#include "tasks/base/packet_typedef_send.hpp"
#include "tasks/base/recorder_player..hpp"
#include "tasks/base/web.hpp"
#include "tasks/base/wheel_odometry.hpp"
#include "utils/buffer.hpp"
#include "utils/common/image.hpp"
#include "utils/common/type_common.hpp"
#include "utils/drivers/hik_camera.hpp"
#include "utils/drivers/mv_camera.hpp"
#include "utils/drivers/serial_driver.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_tf.hpp"
#include "utils/scheduler/scheduler.hpp"
#include "utils/semaphore_guard.hpp"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
namespace backward {
static backward::SignalHandling sh;
}
using namespace awakening;
struct CameraTag {};
struct SerialTag {};
struct FrameTag {};
using CameraIO = IOPair<CameraTag, ImageFrame>;
using SerialIO = IOPair<SerialTag, std::vector<uint8_t>>;
using CommonFrameIo = IOPair<FrameTag, CommonFrame>;
int main(int argc, char* argv[]) {
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
    bool debug = false;
    std::string config_path;
    std::string robot_name;
    auto first_arg = get_arg(1);
    if (first_arg) {
        robot_name = first_arg.value();
        config_path = get_robot_config_path(robot_name).value_or(robot_name);
    } else {
        std::cout << "fuck" << std::endl;
        return 1;
    }
    auto second_arg = get_arg(2);
    if (second_arg) {
        debug = second_arg.value() == "true";
    }
    auto config = YAML::LoadFile(config_path);
    Scheduler s;
    auto camera_config = config["camera"];
    std::unique_ptr<MvCamera> camera;
    utils::SignalGuard::add_callback([&]() {
        if (camera) {
            camera->stop();
        }
    });

    camera = std::make_unique<MvCamera>(camera_config["mv_camera"], s);
    camera->init();
    if (!camera->running_) {
        return 0;
    }
    s.register_task<CameraIO, CommonFrameIo>("push_common_frame", [&](CameraIO::second_type&& f) {
        static int current_id = 0;

        CommonFrame frame {
            .img_frame = std::move(f),
            .id = current_id++,
            .frame_id = 0,
            .expanded = cv::Rect(0, 0, frame.img_frame.src_img.cols, frame.img_frame.src_img.rows),
            .offset = cv::Point2f(0, 0),
        };

        return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::move(frame)));
    });
    s.register_task<CommonFrameIo>("detector", [&](CommonFrameIo::second_type&& frame) {
        static std::unique_ptr<std::counting_semaphore<>> detector_sem;
        if (!detector_sem) {
            detector_sem = std::make_unique<std::counting_semaphore<>>(1);
        }

        return;
    });
    if (camera) {
        camera->start<CameraTag>("hik");
    }

    s.build();
    s.run();

    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();

    for (int i = 0; i < 10; ++i) {
        AWAKENING_CRITICAL("改了东西记得同步其他有关的exe的src");
    }
    return 0;
}