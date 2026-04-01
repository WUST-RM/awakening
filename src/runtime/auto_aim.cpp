#include "_rcl/node.hpp"
#include "_rcl/tf.hpp"
#include "param_deliver.h"
#include "tasks/auto_aim/armor_detect/armor_detector.hpp"
#include "tasks/auto_aim/type.hpp"
#include "utils/buffer.hpp"
#include "utils/common/image.hpp"
#include "utils/common/type_common.hpp"
#include "utils/drivers/hik_camera.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_tf.hpp"
#include "utils/scheduler/scheduler.hpp"
#include "utils/semaphore_guard.hpp"
#include "utils/signal_guard.hpp"
#include <functional>
#include <opencv2/highgui.hpp>
#include <spdlog/common.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>
using namespace awakening;

enum class SimpleFrame : int { ODOM, GIMBAL_ODOM, GIMBAL, CAMERA, CAMERA_CV, SHOOT, N };

using SimpleRobotTF = utils::tf::RobotTF<SimpleFrame, static_cast<size_t>(SimpleFrame::N), true>;
std::string SimpleFrame_to_str(SimpleFrame frame) {
    constexpr const char* details[] = { "odom",   "gimbal_odom", "gimbal",
                                        "camera", "camera_cv",   "shoot" };
    return std::string(details[std::to_underlying(frame)]);
}
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
    double latency_ms_total = 0.0;
    void reset() {
        camera_count = 0;
        detect_count = 0;
        track_count = 0;
        solve_count = 0;
        latency_ms_total = 0.0;
    }
};
static constexpr auto CAMERA_CONFIG_PATH_ARR = utils::concat(ROOT_DIR, "/config/camera.yaml");
static constexpr std::string_view CAMERA_CONFIG_PATH(CAMERA_CONFIG_PATH_ARR.data());
static constexpr auto AUTO_AIM_CONFIG_PATH_ARR = utils::concat(ROOT_DIR, "/config/auto_aim.yaml");
static constexpr std::string_view AUTO_AIM_CONFIG_PATH(AUTO_AIM_CONFIG_PATH_ARR.data());
int main() {
    logger::init(spdlog::level::trace);
    Scheduler s;
    rcl::RclcppNode rcl_node("auto_aim");
    auto camera_config = YAML::LoadFile(std::string(CAMERA_CONFIG_PATH));
    auto auto_aim_config = YAML::LoadFile(std::string(AUTO_AIM_CONFIG_PATH));
    HikCamera camera(camera_config["hik_camera"], s);
    auto_aim::ArmorDetector armor_detector(auto_aim_config["armor_detector"]);
    utils::OrderedQueue<auto_aim::Armors> armors_queue;
    LogCtx log_ctx;
    rcl::TF rcl_tf(rcl_node);
    auto tf = SimpleRobotTF::create();
    tf->add_edge(SimpleFrame::ODOM, SimpleFrame::GIMBAL_ODOM);
    tf->add_edge(SimpleFrame::GIMBAL_ODOM, SimpleFrame::GIMBAL);
    tf->add_edge(SimpleFrame::GIMBAL, SimpleFrame::CAMERA);
    tf->add_edge(SimpleFrame::GIMBAL, SimpleFrame::SHOOT);
    tf->add_edge(SimpleFrame::CAMERA, SimpleFrame::CAMERA_CV);
    ISO3 cv_2_camera = ISO3::Identity();
    cv_2_camera.translation() = Vec3(0, 0, 0);
    cv_2_camera.linear() << 0.0, 0.0, 1.0, -1.0, -0.0, 0.0, 0.0, -1.0, 0.0;
    tf->push(SimpleFrame::CAMERA, SimpleFrame::CAMERA_CV, Clock::now(), cv_2_camera.inverse());
    ISO3 camera_2_gimbal = ISO3::Identity();
    camera_2_gimbal.translation() = Vec3(1.1, 0, 0.2);
    tf->push(SimpleFrame::GIMBAL, SimpleFrame::CAMERA, Clock::now(), camera_2_gimbal.inverse());
    auto a = tf->get(SimpleFrame::CAMERA, SimpleFrame::GIMBAL, Clock::now());
    std::cout << a.translation() << "  " << a.rotation() << std::endl;
    s.register_task<HikIO, CommonFrameIo>("push_common_frame", [&](HikIO::second_type&& f) {
        static int current_id = 0;
        log_ctx.camera_count++;
        CommonFrame frame {
            .img_frame = std::move(f),
            .id = current_id++,
            .frame_id = std::to_underlying(SimpleFrame::CAMERA),
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
                log_ctx.detect_count++;
            }
        }
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

            log_ctx.latency_ms_total += latency_ms;
        }
    });
    s.add_rate_source<0>("slover", 1000.0, [&]() { log_ctx.solve_count++; });
    s.add_rate_source<1>("logger", 1.0, [&]() {
        double avg_latency_ms = log_ctx.latency_ms_total / log_ctx.track_count;
        AWAKENING_INFO(
            "detect: {} track: {} solve: {} camera: {} avg_latency: {:.3} ms",
            log_ctx.detect_count,
            log_ctx.track_count,
            log_ctx.solve_count,
            log_ctx.camera_count,
            avg_latency_ms
        );
        log_ctx.reset();
    });
    s.add_rate_source<1>("tf_pub", 10.0, [&]() {
        rcl_tf.pub_robot_tf(*tf, [](SimpleFrame frame) { return SimpleFrame_to_str(frame); });
    });
    camera.start<HikTag>("hik");
    s.build();
    s.run();
    std::thread([&]() { rcl_node.spin(); }).detach();
    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();
    rcl_node.shutdown();
    return 0;
}
