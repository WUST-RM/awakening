#include <optional>
#ifdef USE_ROS2
    #include "_rcl/node.hpp"
    #include "_rcl/tf.hpp"
    #include "_rcl/visual/armor.hpp"
    #include "_rcl/visual/armor_target.hpp"
#endif
#include "param_deliver.h"
#include "rerun.hpp"
#include "tasks/auto_aim/armor_control/very_aimer.hpp"
#include "tasks/auto_aim/armor_detect/armor_detector.hpp"
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/auto_aim/armor_tracker/armor_tracker.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/debug.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "tasks/base/packet_typedef.hpp"
#include "tasks/base/web.hpp"
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
using namespace awakening;

enum class SimpleFrame : int { ODOM, GIMBAL_ODOM, GIMBAL, CAMERA, CAMERA_CV, SHOOT, N };

using SimpleRobotTF = utils::tf::RobotTF<SimpleFrame, static_cast<size_t>(SimpleFrame::N), true>;

std::string SimpleFrame_to_str(int frame) {
    constexpr const char* details[] = { "odom",   "gimbal_odom", "gimbal",
                                        "camera", "camera_cv",   "shoot" };
    return std::string(details[frame]);
}
std::string SimpleFrame_to_str(SimpleFrame frame) {
    return SimpleFrame_to_str(std::to_underlying(frame));
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

int main(int argc, char** argv) {
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
    auto first_arg = get_arg(1);
    if (first_arg) {
        config_path = first_arg.value();
    } else {
        return 1;
    }
    auto second_arg = get_arg(2);
    if (second_arg) {
        debug = second_arg.value() == "true";
    }
    auto config = YAML::LoadFile(config_path);

    Scheduler s;
    EnemyColor enemy_color = enemy_color_from_string(config["enemy_color"].as<std::string>());
    double bullet_speed = config["bullet_speed"].as<double>();
#ifdef USE_ROS2
    rcl::RclcppNode rcl_node("auto_aim");
    rcl::TF rcl_tf(rcl_node);
#endif

    std::unique_ptr<SerialDriver> serial;
    if (config["serial"]["enable"].as<bool>()) {
        serial = std::make_unique<SerialDriver>(config["serial"], s);
    }
    auto camera_config = YAML::LoadFile(replace_root_dir(config["camera"].as<std::string>()));
    HikCamera camera(camera_config["hik_camera"], s);
    CameraInfo camera_info;
    camera_info.load(camera_config["camera_info"]);
    auto_aim::ArmorDetector armor_detector(config["armor_detector"]);
    auto_aim::ArmorTracker armor_tracker(config["armor_tracker"]);
    auto_aim::AutoAimFsmController auto_aim_fsm_controller(config["auto_aim_fsm"]);
    auto_aim::VeryAimer very_aimer(config["very_aimer"]);
    utils::OrderedQueue<auto_aim::Armors> armors_queue;
    utils::SWMR<auto_aim::ArmorTarget> armor_target;
    LogCtx log_ctx;
    std::optional<auto_aim::AutoAimDebugCtx> auto_aim_dbg;
    if (debug) {
        auto_aim_dbg.emplace();
        auto_aim_dbg->camera_info_ = camera_info;
    }

    auto tf = SimpleRobotTF::create();
    {
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
        camera_in_gimbal.translation() = Vec3(0.0, 0, 0.1);
        tf->push(SimpleFrame::GIMBAL, SimpleFrame::CAMERA, Clock::now(), camera_in_gimbal);
        ISO3 shoot_in_gimbal = ISO3::Identity();
        shoot_in_gimbal.translation() = Vec3(0.1, 0.0, 0.0);
        tf->push(SimpleFrame::GIMBAL, SimpleFrame::SHOOT, Clock::now(), shoot_in_gimbal);
        ISO3 gimbal_odom_in_odom = ISO3::Identity();
        gimbal_odom_in_odom.translation() = Vec3(0, 0, .0);
        tf->push(SimpleFrame::ODOM, SimpleFrame::GIMBAL_ODOM, Clock::now(), gimbal_odom_in_odom);
    }

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
                robo.update_log();
            }
        });
    }
    s.register_task<CommonFrameIo>("auto_exposure", [&](CommonFrameIo::second_type&& frame) {
        struct AutoExposureCfg {
            bool enable = false;
            double ttarget_brightness;
            double step_gain;
            double decay_step;
            double tolerance;
            double exposure_min;
            double exposure_max;
            double control_interval_ms;
            void load(const YAML::Node& c) {
                ttarget_brightness = c["target_brightness"].as<double>();
                step_gain = c["step_gain"].as<double>();
                decay_step = c["decay_step"].as<double>();
                tolerance = c["tolerance"].as<double>();
                exposure_min = c["exposure_min"].as<double>();
                exposure_max = c["exposure_max"].as<double>();
                control_interval_ms = c["control_interval_ms"].as<double>();
            }
        };
        static std::optional<AutoExposureCfg> auto_exposure_cfg;
        if (config["auto_exposure"]["enable"].as<bool>()) {
            auto_exposure_cfg.emplace();
            auto_exposure_cfg.value().load(config["auto_exposure"]);
        }
        if (auto_exposure_cfg) {
            auto& cfg = auto_exposure_cfg.value();
            utils::dt_once(
                [&]() {
                    cv::Mat img = frame.img_frame.src_img;
                    cv::Mat gray;
                    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                    const double brightness = cv::mean(gray)[0];

                    const double diff = brightness - cfg.ttarget_brightness;
                    const double exposure_min = cfg.exposure_min;
                    const double exposure_max = cfg.exposure_max;
                    double exposure_time = camera.get_ExposureTime();
                    static double last_exposure_time = 0.0;
                    if (std::fabs(diff) > cfg.tolerance && exposure_time > 0.0) {
                        exposure_time -= diff * cfg.step_gain;
                    } else {
                        exposure_time -= cfg.decay_step;
                    }
                    exposure_time = std::clamp(exposure_time, exposure_min, exposure_max);
                    if (std::abs(exposure_time - last_exposure_time) > 10) {
                        camera.set_ExposureTime(exposure_time);
                        last_exposure_time = exposure_time;
                    }
                },
                std::chrono::milliseconds((int)cfg.control_interval_ms)
            );
        }
    });
    s.register_task<CommonFrameIo, DetIo>("detector", [&](CommonFrameIo::second_type&& frame) {
        static std::unique_ptr<std::counting_semaphore<>> detector_sem;
        if (!detector_sem) {
            detector_sem = std::make_unique<std::counting_semaphore<>>(5);
        }
        auto target = armor_target.read();
        if (target.check()) {
            auto camera_cv_in_odom = tf->pose_a_in_b(
                SimpleFrame::CAMERA_CV,
                SimpleFrame::ODOM,
                frame.img_frame.timestamp
            );
            target.set_target_state([&](armor_motion_model::State& state) {
                state.predict(frame.img_frame.timestamp);
            });
            auto bbox = target.expanded(
                frame.img_frame.timestamp,
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
        if (auto_aim_dbg) {
            auto_aim_dbg->expanded_buffer.write(frame.expanded);
            auto_aim_dbg->img_frame_buffer.write(std::move(frame.img_frame.clone()));
        }

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
                tf->pose_a_in_b(SimpleFrame::CAMERA_CV, SimpleFrame::ODOM, armors.timestamp);
            armors.frame_id = std::to_underlying(SimpleFrame::ODOM);
            auto __armor_target =
                armor_tracker.track(armors, camera_info, camera_cv_in_odom, armors.frame_id);
            auto_aim_fsm_controller.update(
                __armor_target.get_target_state().vyaw(),
                __armor_target.jumped
            );
            armor_target.write(__armor_target);

            auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - armors.timestamp
            )
                                  .count();
            log_ctx.latency_ms_total += latency_ms;
            log_ctx.found_count += armor_tracker.get_count();
            armor_tracker.reset_count();
            if (auto_aim_dbg) {
                auto_aim_dbg->armors_buffer.write(armors);
#ifdef USE_ROS2
                rcl::pub_armor_marker(rcl_node, SimpleFrame_to_str(armors.frame_id), armors);
                rcl::pub_armor_target_marker(
                    rcl_node,
                    SimpleFrame_to_str(__armor_target.get_target_state().frame_id),
                    __armor_target
                );
#endif
            }

            log_ctx.track_count++;
        }
    });
    s.add_rate_source<0>("slover", 1000.0, [&]() {
        log_ctx.solve_count++;
        auto target = armor_target.read();
        auto old_in_gimbal_odom = tf->pose_a_in_b(
            SimpleFrame(target.get_target_state().frame_id),
            SimpleFrame::GIMBAL_ODOM,
            target.get_target_state().timestamp
        );
        target.set_target_state([&](armor_motion_model::State& state) {
            state.transform(old_in_gimbal_odom, std::to_underlying(SimpleFrame::GIMBAL_ODOM));
        }); // todo : kill gimbal_odom_vel in odom
        GimbalCmd cmd {
            .appear = false,
        };
        if (target.check()) {
            cmd = very_aimer.very_aim(target, bullet_speed, auto_aim_fsm_controller.get_state());
        }

        if (serial) {
            SendRobotCmdData send;
            send.cmd_ID = SendRobotCmdData::ID;
            send.time_stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch()
            )
                                  .count();
            send.appear = cmd.appear;
            send.detect_color = std::to_underlying(enemy_color);
            send.yaw = cmd.yaw;
            send.pitch = cmd.pitch;
            send.v_yaw = cmd.v_yaw;
            send.v_pitch = cmd.v_pitch;
            send.a_yaw = cmd.a_yaw;
            send.a_pitch = cmd.a_pitch;
            send.enable_yaw_diff = cmd.enable_yaw_diff;
            send.enable_pitch_diff = cmd.enable_pitch_diff;
            serial->write(std::move(utils::to_vector(send)));
        }
        auto old_in_camera_cv = tf->pose_a_in_b(
            SimpleFrame(cmd.aim_point.frame_id),
            SimpleFrame::CAMERA_CV,
            cmd.timestamp
        );
        cmd.aim_point.transform(old_in_camera_cv, std::to_underlying(SimpleFrame::CAMERA_CV));
        if (auto_aim_dbg) {
            auto_aim_dbg->gimbal_cmd_buffer.write(cmd);
        }
    });
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
        if (auto_aim_dbg) {
            auto_aim_dbg->avg_latency_ms_buffer.write(avg_latency_ms);
        }
        log_ctx.reset();
    });
    if (auto_aim_dbg) {
        s.add_rate_source<2>("debug", 60.0, [&]() {
            auto target = armor_target.read();
            auto old_in_camera_cv = tf->pose_a_in_b(
                SimpleFrame(target.get_target_state().frame_id),
                SimpleFrame::CAMERA_CV,
                target.get_target_state().timestamp
            );
            target.set_target_state([&](armor_motion_model::State& state) {
                state.transform(old_in_camera_cv, std::to_underlying(SimpleFrame::CAMERA_CV));
            });
            target.write_log();
            auto_aim_dbg->armor_target_buffer.write(target);
            auto_aim_dbg->fsm_state_buffer.write(auto_aim_fsm_controller.get_state());
            auto gimbal_in_gimbal_odom =
                tf->pose_a_in_b(SimpleFrame::GIMBAL, SimpleFrame::GIMBAL_ODOM, Clock::now());
            auto euler =
                utils::matrix2euler(gimbal_in_gimbal_odom.linear(), utils::EulerOrder::ZYX);
            auto gimbal_yaw_pitch =
                std::make_pair(angles::to_degrees(euler[0]), angles::to_degrees(euler[1]));
            auto_aim_dbg->gimbal_yaw_pitch_buffer.write(gimbal_yaw_pitch);
            write_debug_data(auto_aim_dbg.value());
            auto debug_img = auto_aim_dbg->img_frame().src_img;
            if (!debug_img.empty()) {
                auto_aim::draw_auto_aim(debug_img, auto_aim_dbg.value());
                web::write_shm(debug_img);
            }
        });
#ifdef USE_ROS2
        s.add_rate_source<1>("tf_pub", 100.0, [&]() {
            rcl_tf.pub_robot_tf(*tf, [](SimpleFrame frame) { return SimpleFrame_to_str(frame); });
        });
#endif
    }

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
