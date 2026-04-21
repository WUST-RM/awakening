#include "ascii_banner.hpp"
#include "backward-cpp/backward.hpp"
#include "tasks/auto_aim/armor_detect/armor_detector.hpp"
#include "tasks/auto_aim/debug.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "tasks/base/web.hpp"
#include "utils/buffer.hpp"
#include "utils/common/image.hpp"
#include "utils/drivers/uvc_camera.hpp"
#include "utils/logger.hpp"
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
struct FrameTag {};

using CameraIO = IOPair<CameraTag, ImageFrame>;
using CommonFrameIo = IOPair<FrameTag, CommonFrame>;

int main(int argc, char** argv) {
    auto start_tp = std::chrono::steady_clock::now();
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
    EnemyColor enemy_color = enemy_color_from_string(config["enemy_color"].as<std::string>());
    auto camera_config = config["uvc_camera_test"];
    std::unique_ptr<UVCCamera> camera;
    utils::SignalGuard::add_callback([&]() {
        if (camera) {
            camera->stop();
        }
    });

    camera = std::make_unique<UVCCamera>(camera_config["uvc_camera"]);
    camera->start();

    CameraInfo camera_info;
    camera_info.load(camera_config["camera_info"]);
    auto_aim::ArmorDetector armor_detector(config["armor_detector"]);

    std::optional<auto_aim::AutoAimDebugCtx> auto_aim_dbg;

    auto_aim_dbg.emplace();
    auto_aim_dbg->camera_info_ = camera_info;

    s.add_rate_source<>("armor_omni", 60.0, [&]() {
        auto img_frame = camera->read();
        if (img_frame.src_img.empty()) {
            AWAKENING_WARN("Failed to read image from camera.");
            return;
        }
        CommonFrame common_frame;
        common_frame.expanded = cv::Rect(0, 0, img_frame.src_img.cols, img_frame.src_img.rows);
        common_frame.offset = cv::Point2f(0, 0);
        common_frame.img_frame = std::move(img_frame);
        common_frame.frame_id = 0;
        auto_aim::Armors armors { .timestamp = common_frame.img_frame.timestamp,
                                  .id = common_frame.frame_id,
                                  .frame_id = common_frame.frame_id };
        auto tmp_armors = armor_detector.detect(common_frame);
        for (auto& a: tmp_armors) {
            if ((enemy_color == EnemyColor::BLUE && a.color == auto_aim::ArmorColor::RED)
                || (enemy_color == EnemyColor::RED && a.color == auto_aim::ArmorColor::BLUE))
            {
                continue;
            }
            armors.armors.push_back(a);
        }
        auto_aim_dbg->img_frame.set(common_frame.img_frame);
        auto_aim_dbg->armors.set(armors);
        auto img = auto_aim_dbg->img_frame.get();
        auto debug_img = img.src_img;
        if (img.format == PixelFormat::RGB) {
            cv::cvtColor(debug_img, debug_img, cv::COLOR_RGB2BGR);
        }
        if (!debug_img.empty()) {
            static cv::Mat last_draw;
            if (debug_img.data != last_draw.data) {
                auto_aim::draw_auto_aim(debug_img, auto_aim_dbg.value());
                web::write_shm(debug_img);
            }
            last_draw = debug_img;
        }
    });

    s.build();
    s.run();

    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();

    for (int i = 0; i < 10; ++i) {
        AWAKENING_CRITICAL("改了东西记得同步其他有关的exe的src");
    }
    return 0;
}
