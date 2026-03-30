#include "param_deliver.h"
#include "tasks/auto_aim/armor_detect/armor_detector.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/robot.hpp"
#include "utils/buffer.hpp"
#include "utils/common/image.hpp"
#include "utils/drivers/hik_camera.hpp"
#include "utils/logger.hpp"
#include "utils/scheduler/scheduler.hpp"
#include "utils/semaphore_guard.hpp"
#include "utils/signal_guard.hpp"
#include <opencv2/highgui.hpp>
#include <vector>
using namespace awakening;

struct HikTag {};
struct DetectTag {};
struct FrameTag {};
using HikIO = IOPair<HikTag, ImageFrame>;
using CommonFrameIo = IOPair<FrameTag, CommonFrame>;
using DetIo = IOPair<DetectTag, std::vector<auto_aim::Armors>>;
struct LogCtx {
    int camera_count = 0;
    int detect_count = 0;
    int track_count = 0;
    int solve_count = 0;
    std::vector<double> latency_ms;
    void reset() {
        camera_count = 0;
        detect_count = 0;
        track_count = 0;
        solve_count = 0;
        latency_ms.clear();
    }
};
static constexpr auto CAMERA_CONFIG_PATH_ARR = utils::concat(ROOT_DIR, "/config/camera.yaml");
static constexpr std::string_view CAMERA_CONFIG_PATH(CAMERA_CONFIG_PATH_ARR.data());
static constexpr auto AUTO_AIM_CONFIG_PATH_ARR = utils::concat(ROOT_DIR, "/config/auto_aim.yaml");
static constexpr std::string_view AUTO_AIM_CONFIG_PATH(AUTO_AIM_CONFIG_PATH_ARR.data());
int main() {
    logger::init(spdlog::level::debug);
    Scheduler s;

    auto camera_config = YAML::LoadFile(std::string(CAMERA_CONFIG_PATH));
    auto auto_aim_config = YAML::LoadFile(std::string(AUTO_AIM_CONFIG_PATH));
    HikCamera camera(camera_config["hik_camera"], s);
    auto_aim::ArmorDetector armor_detector(auto_aim_config["armor_detector"]);
    utils::OrderedQueue<auto_aim::Armors> armors_queue;
    LogCtx log_ctx;
    s.register_task<HikIO, CommonFrameIo>("push_common_frame", [&](HikIO::second_type&& f) {
        static int current_id = 0;
        log_ctx.camera_count++;
        CommonFrame frame {
            .img_frame = std::move(f),
            .id = current_id++,
            .frame_id = std::to_underlying(Frame::CAMERA),
            .expanded = cv::Rect(0, 0, frame.img_frame.src_img.cols, frame.img_frame.src_img.rows),
            .offset = cv::Point2f(0, 0),
        };
        return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::move(frame)));
    });
    s.register_task<CommonFrameIo, DetIo>("detector", [&](CommonFrameIo::second_type&& frame) {
        static std::unique_ptr<std::counting_semaphore<>> detector_sem;
        if (!detector_sem) {
            detector_sem = std::make_unique<std::counting_semaphore<>>(5);
        }
        auto_aim::Armors armors { .timestamp = frame.img_frame.timestamp,
                                  .id = frame.id,
                                  .frame_id = frame.frame_id };
        {
            bool got = detector_sem->try_acquire();
            utils::SemaphoreGuard guard(*detector_sem, got);
            if (got) {
                armors.armors = armor_detector.detect(frame);
            }
        }

        log_ctx.detect_count++;
        // auto& show = frame.img_frame.src_img;
        // armors.draw(show);
        // cv::imshow("detect", show);
        // cv::waitKey(1); 
        armors_queue.enqueue(armors);
        auto batch_armors = armors_queue.dequeue_batch();
        return std::make_tuple(std::optional<DetIo::second_type>(std::move(batch_armors)));
    });

    s.register_task<DetIo>("tracker", [&](DetIo::second_type&& io) {
        log_ctx.track_count++;
        for (const auto& armors: io) {
            auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - armors.timestamp
            )
                                  .count();
            log_ctx.latency_ms.push_back(latency_ms);
        }
    });
    s.add_rate_source<0>("slover", 1000.0, [&]() { log_ctx.solve_count++; });
    s.add_rate_source<1>("logger", 1.0, [&]() {
        double avg_latency_ms =
            std::accumulate(log_ctx.latency_ms.begin(), log_ctx.latency_ms.end(), 0.0)
            / log_ctx.latency_ms.size();
        AWAKENING_INFO(
            "detect: {} track: {} solve: {} camera: {} avg_latency: {:.2} ms",
            log_ctx.detect_count,
            log_ctx.track_count,
            log_ctx.solve_count,
            log_ctx.camera_count,
            avg_latency_ms
        );
        log_ctx.reset();
    });
    camera.start<HikTag>("hik");
    s.build();
    s.run();
    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();
    return 0;
}
