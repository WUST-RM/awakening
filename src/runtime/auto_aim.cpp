#include "_rcl/node.hpp"
#include "_rcl/tf.hpp"
#include "_rcl/visual/armor.hpp"
#include "_rcl/visual/armor_target.hpp"
#include "param_deliver.h"
#include "tasks/auto_aim/armor_detect/armor_detector.hpp"
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/auto_aim/armor_tracker/armor_tracker.hpp"
#include "tasks/auto_aim/debug.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/packet_typedef.hpp"
#include "utils/buffer.hpp"
#include "utils/common/image.hpp"
#include "utils/common/type_common.hpp"
#include "utils/drivers/hik_camera.hpp"
#include "utils/drivers/serial_driver.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_tf.hpp"
#include "utils/scheduler/scheduler.hpp"
#include "utils/semaphore_guard.hpp"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
#include <angles.h>
#include <functional>
#include <iostream>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
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
struct SerialTag {};
struct DetectTag {};
struct FrameTag {};
using HikIO = IOPair<HikTag, ImageFrame>;
using SerialIO = IOPair<SerialTag, std::vector<uint8_t>>;
using CommonFrameIo = IOPair<FrameTag, CommonFrame>;
using DetIo = IOPair<DetectTag, std::vector<auto_aim::Armors>>;
struct LogCtx {
    int camera_count = 0;
    int detect_count = 0;
    int track_count = 0;
    int solve_count = 0;
    int serial_count = 0;
    int found_count = 0;
    double latency_ms_total = 0.0;
    void reset() {
        camera_count = 0;
        detect_count = 0;
        track_count = 0;
        solve_count = 0;
        serial_count = 0;
        found_count = 0;
        latency_ms_total = 0.0;
    }
};
static constexpr auto CAMERA_CONFIG_PATH_ARR = utils::concat(ROOT_DIR, "/config/camera.yaml");
static constexpr std::string_view CAMERA_CONFIG_PATH(CAMERA_CONFIG_PATH_ARR.data());
static constexpr auto AUTO_AIM_CONFIG_PATH_ARR = utils::concat(ROOT_DIR, "/config/auto_aim.yaml");
static constexpr std::string_view AUTO_AIM_CONFIG_PATH(AUTO_AIM_CONFIG_PATH_ARR.data());
static constexpr auto ROBOT_CONFIG_PATH_ARR = utils::concat(ROOT_DIR, "/config/robot.yaml");
static constexpr std::string_view ROBOT_CONFIG_PATH(ROBOT_CONFIG_PATH_ARR.data());
int main() {
    logger::init(spdlog::level::trace);
    Scheduler s;
    EnemyColor enemy_color = EnemyColor::RED;
    utils::SWMR<auto_aim::ArmorTarget> armor_target;
    rcl::RclcppNode rcl_node("auto_aim");
    auto camera_config = YAML::LoadFile(std::string(CAMERA_CONFIG_PATH));
    auto auto_aim_config = YAML::LoadFile(std::string(AUTO_AIM_CONFIG_PATH));
    auto robot_config = YAML::LoadFile(std::string(ROBOT_CONFIG_PATH));
    std::unique_ptr<SerialDriver> serial;
    if (robot_config["serial"]["enable"].as<bool>()) {
        serial = std::make_unique<SerialDriver>(robot_config["serial"], s);
    }
    HikCamera camera(camera_config["hik_camera"], s);
    CameraInfo camera_info;
    camera_info.load(YAML::LoadFile(camera_config["camera_info_path"].as<std::string>()));
    auto_aim::ArmorDetector armor_detector(auto_aim_config["armor_detector"]);
    auto_aim::ArmorTracker armor_tracker(auto_aim_config["armor_tracker"]);
    utils::OrderedQueue<auto_aim::Armors> armors_queue;
    LogCtx log_ctx;
    auto_aim::AutoAimDebugCtx auto_aim_dbg;
    auto_aim_dbg.camera_info_buffer.write(camera_info);
    rcl::TF rcl_tf(rcl_node);
    auto tf = SimpleRobotTF::create();
    tf->add_edge(SimpleFrame::ODOM, SimpleFrame::GIMBAL_ODOM);
    tf->add_edge(SimpleFrame::GIMBAL_ODOM, SimpleFrame::GIMBAL);
    tf->add_edge(SimpleFrame::GIMBAL, SimpleFrame::CAMERA);
    tf->add_edge(SimpleFrame::GIMBAL, SimpleFrame::SHOOT);
    tf->add_edge(SimpleFrame::CAMERA, SimpleFrame::CAMERA_CV);
    ISO3 cv_in_camera = ISO3::Identity();
    cv_in_camera.translation() = Vec3(0, 0, 0);
    cv_in_camera.linear() = R_CV2PHYSICS;
    tf->push(SimpleFrame::CAMERA, SimpleFrame::CAMERA_CV, Clock::now(), cv_in_camera);
    ISO3 camera_in_gimbal = ISO3::Identity();
    camera_in_gimbal.translation() = Vec3(0.0, 0, 0.0);
    tf->push(SimpleFrame::GIMBAL, SimpleFrame::CAMERA, Clock::now(), camera_in_gimbal);
    s.register_task<HikIO, CommonFrameIo>("push_common_frame", [&](HikIO::second_type&& f) {
        static int current_id = 0;
        log_ctx.camera_count++;
        CommonFrame frame {
            .img_frame = std::move(f),
            .id = current_id++,
            .frame_id = std::to_underlying(SimpleFrame::CAMERA_CV),
            .expanded = cv::Rect(0, 0, frame.img_frame.src_img.cols, frame.img_frame.src_img.rows),
            .offset = cv::Point2f(0, 0),
        };
        return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::move(frame)));
    });
    if (serial) {
        s.register_task<SerialIO>("receive_serial", [&](SerialIO::second_type&& data) {
            auto robo_opt = ReceiveRobotData::create(data);
            log_ctx.serial_count++;
            if (robo_opt.has_value()) {
                auto robo = robo_opt.value();
                double yaw = angles::from_degrees(robo.yaw);
                double pitch = angles::from_degrees(robo.pitch);
                double roll = angles::from_degrees(robo.roll);
                ISO3 gimbal_2_gimbal_odom = ISO3::Identity();
                gimbal_2_gimbal_odom.translation() = Vec3(0, 0, 0);
                gimbal_2_gimbal_odom.linear() =
                    utils::euler2matrix(Vec3(yaw, pitch, roll), utils::EulerOrder::ZYX);
                tf->push(
                    SimpleFrame::GIMBAL_ODOM,
                    SimpleFrame::GIMBAL,
                    Clock::now(),
                    gimbal_2_gimbal_odom
                );
            }
        });
    }

    s.register_task<CommonFrameIo, DetIo>("detector", [&](CommonFrameIo::second_type&& frame) {
        static std::unique_ptr<std::counting_semaphore<>> detector_sem;
        if (!detector_sem) {
            detector_sem = std::make_unique<std::counting_semaphore<>>(5);
        }
        auto target = armor_target.read();
        if (target.check()) {
            auto __target_state = target.get_target_state();
            auto camera_cv_in_odom =
                tf->pose_in(SimpleFrame::CAMERA_CV, SimpleFrame::ODOM, frame.img_frame.timestamp);
            __target_state.predict(frame.img_frame.timestamp);
            auto bbox = target.expanded(
                frame.img_frame.timestamp,
                __target_state,
                camera_cv_in_odom,
                camera_info,
                frame.img_frame.src_img.size()
            );
            if (bbox.area() > 200) {
                frame.expanded = bbox;
                frame.offset = cv::Point2f(bbox.x, bbox.y);
            }
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
        armors_queue.enqueue(armors);
        auto batch_armors = armors_queue.dequeue_batch();
        auto_aim_dbg.expanded_buffer.write(frame.expanded);
        auto_aim_dbg.img_frame_buffer.write(std::move(frame.img_frame));
        return std::make_tuple(std::optional<DetIo::second_type>(std::move(batch_armors)));
    });

    s.register_task<DetIo>("tracker", [&](DetIo::second_type&& io) {
        for (const auto& armors_raw: io) {
            auto armors = armors_raw;
            armors.armors.clear();
            for (auto& a: armors_raw.armors) {
                if ((enemy_color == EnemyColor::BLUE && a.color == auto_aim::ArmorColor::RED)
                    || (enemy_color == EnemyColor::RED && a.color == auto_aim::ArmorColor::BLUE))
                {
                    continue;
                }
                armors.armors.push_back(a);
            }
            auto camera_cv_in_odom =
                tf->pose_in(SimpleFrame::CAMERA_CV, SimpleFrame::ODOM, armors.timestamp);
            auto __armor_target = armor_tracker.track(armors, camera_info, camera_cv_in_odom);
            armor_target.write(__armor_target);
            armors.frame_id = std::to_underlying(SimpleFrame::ODOM);
            rcl::pub_armor_marker(
                rcl_node,
                SimpleFrame_to_str(SimpleFrame(armors.frame_id)),
                armors
            );
            rcl::pub_armor_target_marker(
                rcl_node,
                SimpleFrame_to_str(SimpleFrame(armors.frame_id)),
                __armor_target
            );

            auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - armors.timestamp
            )
                                  .count();
            log_ctx.latency_ms_total += latency_ms;
            log_ctx.found_count += armor_tracker.get_count();
            armor_tracker.reset_count();
            auto_aim_dbg.armors_buffer.write(armors);
            auto_aim_dbg.camera_cv_in_odom_buffer.write(camera_cv_in_odom);
            log_ctx.track_count++;
        }
    });
    s.add_rate_source<0>("slover", 1000.0, [&]() { log_ctx.solve_count++; });
    s.add_rate_source<1>("logger", 1.0, [&]() {
        double avg_latency_ms = log_ctx.latency_ms_total / log_ctx.track_count;
        AWAKENING_INFO(
            "detect: {} track: {} found: {} solve: {} serial: {} camera: {} avg_latency: {:.3} ms",
            log_ctx.detect_count,
            log_ctx.track_count,
            log_ctx.found_count,
            log_ctx.solve_count,
            log_ctx.serial_count,
            log_ctx.camera_count,
            avg_latency_ms
        );
        auto_aim_dbg.avg_latency_ms_buffer.write(avg_latency_ms);
        log_ctx.reset();
    });
    s.add_rate_source<2>("debug", 60.0, [&]() {
        auto target = armor_target.read();
        auto_aim_dbg.armor_target_buffer.write(target);
        auto debug_img = auto_aim_dbg.img_frame().src_img;
        if (!debug_img.empty()) {
            auto_aim::draw_auto_aim(debug_img, auto_aim_dbg);
            cv::imshow("Auto Aim Debug", debug_img);
            cv::waitKey(1);
        }
    });
    s.add_rate_source<1>("tf_pub", 100.0, [&]() {
        rcl_tf.pub_robot_tf(*tf, [](SimpleFrame frame) { return SimpleFrame_to_str(frame); });
    });
    camera.start<HikTag>("hik");
    if (serial) {
        serial->start<SerialTag>("serial");
    }
    s.build();
    s.run();
    std::thread([&]() { rcl_node.spin(); }).detach();
    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();
    rcl_node.shutdown();
    return 0;
}
