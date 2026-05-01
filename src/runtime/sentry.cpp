#include "angles.h"
#include "ascii_banner.hpp"
#include "tasks/auto_aim/armor_omni.hpp"
#include "tasks/base/ballistic_trajectory.hpp"
#include "tasks/base/wheel_odometry.hpp"
#include "tasks/sentry_brain/rmuc_2026/mode_factory.hpp"
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#ifdef USE_ROS2
    #include "_rcl/node.hpp"
    #include "_rcl/tf.hpp"
    #include "_rcl/visual/armor.hpp"
    #include "_rcl/visual/armor_target.hpp"
    #include "_rcl/visual/arrow.hpp"
    #include "sensor_msgs/msg/camera_info.hpp"
    #include "sensor_msgs/msg/image.hpp"
    #include <rclcpp/qos.hpp>
#endif
#include "backward-cpp/backward.hpp"
#include "config.hpp"
#include "param_deliver.h"
#include "tasks/auto_aim/armor_control/very_aimer.hpp"
#include "tasks/auto_aim/armor_detect/armor_detector.hpp"
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/auto_aim/armor_tracker/armor_tracker.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/debug.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "tasks/base/packet_typedef_receive.hpp"
#include "tasks/base/packet_typedef_send.hpp"
#include "tasks/base/recorder_player..hpp"
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
namespace backward {
static backward::SignalHandling sh;
}
using namespace awakening;

enum class SentryFrame : int {
    ODOM,
    GIMBAL_ODOM,
    GIMBAL,
    CAMERA,
    CAMERA_CV,
    SHOOT,
    BIG_YAW,
    OMNI_0,
    OMNI_0_CV,
    OMNI_1,
    OMNI_1_CV,
    N
};

using SimpleRobotTF = utils::tf::RobotTF<SentryFrame, static_cast<size_t>(SentryFrame::N), false>;

std::string SentryFrame_to_str(int frame) {
    constexpr const char* details[] = { "odom",      "gimbal_odom", "gimbal",   "camera",
                                        "camera_cv", "shoot",       "big_yaw",  "omni_0",
                                        "omni_0_cv", "omni_1",      "omni_1_cv" };
    return std::string(details[frame]);
}
std::string SentryFrame_to_str(SentryFrame frame) {
    return SentryFrame_to_str(std::to_underlying(frame));
}
struct CameraTag {};
struct SerialTag {};
struct DetectTag {};
struct FrameTag {};
namespace awakening {
template<>
struct RecordTagTraits<CameraTag> {
    static constexpr uint32_t id = 1;
    using Type = ImageFrame;
    static std::vector<uint8_t> serialize(const Type& img) {
        return img.serialize();
    }
    static Type deserialize(const std::vector<uint8_t>& buf) {
        return Type::deserialize(buf);
    }
};

template<>
struct RecordTagTraits<SerialTag> {
    static constexpr uint32_t id = 2;
    using Type = std::vector<uint8_t>;
    static std::vector<uint8_t> serialize(const Type& obj) {
        return obj;
    }
    static Type deserialize(const std::vector<uint8_t>& buf) {
        return buf;
    }
};
} // namespace awakening
using CameraIO = IOPair<CameraTag, ImageFrame>;
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
    int omni_count = 0;
    double omni_latency_ms_total = 0.0;
    void reset() {
        camera_count = 0;
        detect_count = 0;
        track_count = 0;
        solve_count = 0;
        serial_count = 0;
        found_count = 0;
        latency_ms_total = 0.0;
        omni_count = 0;
        omni_latency_ms_total = 0.0;
    }
};
static constexpr auto RECORD_FOLDER_PATH_ARR = utils::concat(ROOT_DIR, "/record/auto_aim");
static constexpr std::string_view RECORD_FOLDER_PATH(RECORD_FOLDER_PATH_ARR.data());
inline std::string generate_record_filename(const std::string& folder_path) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm {};

#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    std::ostringstream oss;
    oss << folder_path << "/" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << ".bin";
    return oss.str();
}
bool is_web_running() {
    static std::atomic<bool> cached { true };
    utils::dt_once(
        [&]() {
            const int ret = std::system("pgrep -x wust_vision_web > /dev/null 2>&1");
            cached = (ret == 0);
        },
        std::chrono::duration<double>(1.0)
    );
    return cached.load();
}

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
    bool debug = false;
    std::string config_path;
    std::string robot_name;
    auto first_arg = get_arg(1);
    if (first_arg) {
        robot_name = first_arg.value();
        config_path = get_robot_config_path(robot_name).value_or(robot_name);
    } else {
        return 1;
    }
    auto second_arg = get_arg(2);
    if (second_arg) {
        debug = second_arg.value() == "true";
    }
    auto config = YAML::LoadFile(config_path);
    std::unique_ptr<Recorder> recorder;
    if (config["recorder"]["enable"].as<bool>()) {
        recorder =
            std::make_unique<Recorder>(generate_record_filename(std::string(RECORD_FOLDER_PATH)));
    }
    std::unique_ptr<Player> player;
    if (!recorder) {
        if (config["player"]["enable"].as<bool>()) {
            player = std::make_unique<Player>(config["player"]["path"].as<std::string>());
        }
    }
    Scheduler s;
    EnemyColor enemy_color = enemy_color_from_string(config["enemy_color"].as<std::string>());
    double bullet_speed = config["bullet_speed"].as<double>();
#ifdef USE_ROS2
    rcl::RclcppNode rcl_node("auto_aim");
    rcl::TF rcl_tf(rcl_node);
#endif

    std::unique_ptr<SerialDriver> serial;
    if (!player) {
        if (config["serial"]["enable"].as<bool>()) {
            serial = std::make_unique<SerialDriver>(config["serial"], s);
        }
    }
    int serial_send_to_image_microseconds = config["serial_send_to_image_microseconds"].as<int>();

    auto camera_config = config["camera"];
    std::unique_ptr<HikCamera> camera;
    utils::SignalGuard::add_callback([&]() {
        if (camera) {
            camera->stop();
        }
    });
    if (!player) {
        camera = std::make_unique<HikCamera>(camera_config["hik_camera"], s);
        camera->init();
        if (!camera->running_) {
            return 0;
        }
    }

    CameraInfo camera_info;
    camera_info.load(camera_config["camera_info"]);
    auto_aim::ArmorDetector armor_detector(config["armor_detector"]);
    auto_aim::ArmorTracker armor_tracker(config["armor_tracker"]);
    auto_aim::AutoAimFsmController auto_aim_fsm_controller(config["auto_aim_fsm"]);
    auto_aim::ArmorOmni armor_omni(config["armor_omni"]);
    auto_aim::VeryAimer very_aimer(config["very_aimer"]);
    utils::OrderedQueue<auto_aim::Armors> armors_queue;
    utils::SWMR<auto_aim::ArmorTarget> armor_target;
    utils::SWMR<auto_aim::ArmorTarget> omni_armor_target;
    auto brain = sentry_brain::create_brain_mode(rcl_node, rcl_tf, config["brain"]);
    armor_omni.emplace_one(
        config["armor_omni"]["camera0"],
        std::to_underlying(SentryFrame::OMNI_0),
        std::to_underlying(SentryFrame::OMNI_0_CV)
    );
    armor_omni.emplace_one(
        config["armor_omni"]["camera1"],
        std::to_underlying(SentryFrame::OMNI_1),
        std::to_underlying(SentryFrame::OMNI_1_CV)
    );
    BulletPickUp bullet_pick_up(config["bullet_pick_up"]);
    LogCtx log_ctx;
    std::optional<auto_aim::AutoAimDebugCtx> auto_aim_dbg;
    if (debug) {
        auto_aim_dbg.emplace();
        auto_aim_dbg->camera_info_ = camera_info;
    }
    WheelOdometry wheel_odometry(config["wheel_odometry"], Clock::now());
    auto tf = SimpleRobotTF::create();
    {
        tf->add_edge(SentryFrame::ODOM, SentryFrame::GIMBAL_ODOM);
        tf->add_edge(SentryFrame::GIMBAL_ODOM, SentryFrame::GIMBAL);
        tf->add_edge(SentryFrame::GIMBAL, SentryFrame::CAMERA);
        tf->add_edge(SentryFrame::GIMBAL, SentryFrame::SHOOT);
        tf->add_edge(SentryFrame::CAMERA, SentryFrame::CAMERA_CV);
        tf->add_edge(SentryFrame::GIMBAL_ODOM, SentryFrame::BIG_YAW);
        tf->add_edge(SentryFrame::BIG_YAW, SentryFrame::OMNI_0);
        tf->add_edge(SentryFrame::OMNI_0, SentryFrame::OMNI_0_CV);
        tf->add_edge(SentryFrame::BIG_YAW, SentryFrame::OMNI_1);
        tf->add_edge(SentryFrame::OMNI_1, SentryFrame::OMNI_1_CV);
        ISO3 cv_in_camera = ISO3::Identity();
        cv_in_camera.translation() = Vec3(0, 0, 0);
        cv_in_camera.linear() = R_CV2PHYSICS;
        tf->push(SentryFrame::CAMERA, SentryFrame::CAMERA_CV, Clock::now(), cv_in_camera);
        ISO3 camera_in_gimbal = utils::load_isometry3(config["tf"]["camera_in_gimbal"]);
        tf->push(SentryFrame::GIMBAL, SentryFrame::CAMERA, Clock::now(), camera_in_gimbal);
        ISO3 shoot_in_gimbal = utils::load_isometry3(config["tf"]["shoot_in_gimbal"]);
        tf->push(SentryFrame::GIMBAL, SentryFrame::SHOOT, Clock::now(), shoot_in_gimbal);
        ISO3 omin_0_in_big_yaw = utils::load_isometry3(config["tf"]["omin_0_in_big_yaw"]);
        tf->push(SentryFrame::BIG_YAW, SentryFrame::OMNI_0, Clock::now(), omin_0_in_big_yaw);
        ISO3 omin_1_in_big_yaw = utils::load_isometry3(config["tf"]["omin_1_in_big_yaw"]);
        tf->push(SentryFrame::BIG_YAW, SentryFrame::OMNI_1, Clock::now(), omin_1_in_big_yaw);
        tf->push(SentryFrame::OMNI_0, SentryFrame::OMNI_0_CV, Clock::now(), cv_in_camera);
        tf->push(SentryFrame::OMNI_1, SentryFrame::OMNI_1_CV, Clock::now(), cv_in_camera);
    }

    s.register_task<CameraIO, CommonFrameIo>("push_common_frame", [&](CameraIO::second_type&& f) {
        static int current_id = 0;
        log_ctx.camera_count++;
        if (recorder) {
            utils::dt_once(
                [&]() { recorder->record<CameraTag>(f); },
                std::chrono::milliseconds(100)
            );
        }
        CommonFrame frame {
            .img_frame = std::move(f),
            .id = current_id++,
            .frame_id = std::to_underlying(SentryFrame::CAMERA_CV),
            .expanded = cv::Rect(0, 0, frame.img_frame.src_img.cols, frame.img_frame.src_img.rows),
            .offset = cv::Point2f(0, 0),
        };
        auto target = armor_target.read();
        if (target.check()) {
            auto camera_cv_in_old = tf->pose_a_in_b(
                SentryFrame(frame.frame_id),
                SentryFrame(target.get_target_state().frame_id),
                frame.img_frame.timestamp
            );
            target.set_target_state([&](armor_point_motion_model::State& state) {
                state.predict(frame.img_frame.timestamp, target.target_number);
            });
            auto bbox = target.expanded_one_one(
                frame.img_frame.timestamp,
                camera_cv_in_old,
                camera_info,
                frame.img_frame.src_img.size()
            );
            if (bbox.area() > 200) {
                frame.expanded = bbox;
                frame.offset = cv::Point2f(bbox.x, bbox.y);
            }
        }
        return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::move(frame)));
    });
    if (serial || player) {
        s.register_task<SerialIO>("receive_serial", [&](SerialIO::second_type&& data) {
            if (recorder) {
                utils::dt_once(
                    [&]() { recorder->record<SerialTag>(data); },
                    std::chrono::milliseconds(10)
                );
            }
            log_ctx.serial_count++;
            auto robo_opt = ReceiveRobotData::create(data);

            if (robo_opt.has_value()) {
                auto robo = robo_opt.value();
                std::chrono::time_point<std::chrono::steady_clock> packet_time =
                    std::chrono::steady_clock::now()
                    + std::chrono::microseconds(serial_send_to_image_microseconds);
                double yaw = angles::from_degrees(robo.yaw);
                double pitch = angles::from_degrees(robo.pitch);
                double roll = angles::from_degrees(robo.roll);
                ISO3 gimbal_2_gimbal_odom = ISO3::Identity();
                gimbal_2_gimbal_odom.translation() = Vec3(0, 0, 0);
                gimbal_2_gimbal_odom.linear() =
                    utils::euler2matrix(Vec3(yaw, pitch, roll), utils::EulerOrder::ZYX);
                tf->push(
                    SentryFrame::GIMBAL_ODOM,
                    SentryFrame::GIMBAL,
                    packet_time,
                    gimbal_2_gimbal_odom
                );
                double vx = robo.v_x;
                double vy = robo.v_y;
                double vz = robo.v_z;
                wheel_odometry.predict_ekf(packet_time);
                wheel_odometry.update(Vec3(vx, vy, vz), packet_time);
                ISO3 gimbal_odom_in_odom = ISO3::Identity();
                gimbal_odom_in_odom.translation() = wheel_odometry.state.pos();
                tf->push(
                    SentryFrame::ODOM,
                    SentryFrame::GIMBAL_ODOM,
                    packet_time,
                    gimbal_odom_in_odom
                );
                enemy_color = EnemyColor(robo.detect_color);
                robo.update_log();
                static uint32_t last_bullet_count = 0;
                if (robo.bullet_count > last_bullet_count) {
                    auto shoot_in_odom =
                        tf->pose_a_in_b(SentryFrame::SHOOT, SentryFrame::ODOM, Clock::now());
                    Bullet b { .fire_time = Clock::now(),
                               .fire_time_shoot_in_odom = shoot_in_odom,
                               .speed_in_odom = bullet_speed };
                    bullet_pick_up.push_back(std::move(b));
                }
                last_bullet_count = robo.bullet_count;
            }
            auto joint_opt = SentryJointState::create(data);
            if (joint_opt.has_value()) {
                auto sentry = joint_opt.value();
                double big_yaw_in_world = sentry.big_yaw_in_world;
                auto gimbal_2_gimbal_odom =
                    tf->pose_a_in_b(SentryFrame::GIMBAL, SentryFrame::GIMBAL_ODOM, Clock::now());
                auto ypr =
                    utils::matrix2euler(gimbal_2_gimbal_odom.linear(), utils::EulerOrder::ZYX);
                ypr[0] = angles::from_degrees(big_yaw_in_world);
                ypr[1] = 0.0;
                ISO3 big_yaw_2_gimbal_odom = ISO3::Identity();
                big_yaw_2_gimbal_odom.translation() = Vec3(0, 0, 0.0);
                big_yaw_2_gimbal_odom.linear() = utils::euler2matrix(ypr, utils::EulerOrder::ZYX);
                tf->push(
                    SentryFrame::GIMBAL_ODOM,
                    SentryFrame::BIG_YAW,
                    Clock::now(),
                    big_yaw_2_gimbal_odom
                );
            }
            auto referee_opt = SentryRefereeReceive::create(data);
            if (referee_opt && brain) {
                brain->update_gobal_state(referee_opt.value());
            }
        });
    }
    if (camera) {
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
                        double exposure_time = camera->get_ExposureTime();
                        static double last_exposure_time = 0.0;
                        if (std::fabs(diff) > cfg.tolerance && exposure_time > 0.0) {
                            exposure_time -= diff * cfg.step_gain;
                        } else {
                            exposure_time -= cfg.decay_step;
                        }
                        exposure_time = std::clamp(exposure_time, exposure_min, exposure_max);
                        if (std::abs(exposure_time - last_exposure_time) > 10) {
                            camera->set_ExposureTime(exposure_time);
                            last_exposure_time = exposure_time;
                        }
                    },
                    std::chrono::milliseconds((int)cfg.control_interval_ms)
                );
            }
        });
    }

    s.register_task<CommonFrameIo, DetIo>("detector", [&](CommonFrameIo::second_type&& frame) {
        static std::unique_ptr<std::counting_semaphore<>> detector_sem;
        if (!detector_sem) {
            detector_sem =
                std::make_unique<std::counting_semaphore<>>(config["max_infer_num"].as<int>());
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
        if (auto_aim_dbg && is_web_running()) {
            auto_aim_dbg->expanded.set(frame.expanded);
            auto_aim_dbg->img_frame.set(std::move(frame.img_frame.clone()));
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
                tf->pose_a_in_b(SentryFrame(armors.frame_id), SentryFrame::ODOM, armors.timestamp);
            armors.frame_id = std::to_underlying(SentryFrame::ODOM);
            auto __armor_target =
                armor_tracker.track(armors, camera_info, camera_cv_in_odom, armors.frame_id);
            auto_aim_fsm_controller.update(
                __armor_target.get_target_state().vyaw(),
                __armor_target.jumped
            );
            auto target_in_big_yaw = __armor_target;
            auto old_in_big_yaw = tf->pose_a_in_b(
                SentryFrame(target_in_big_yaw.get_target_state().frame_id),
                SentryFrame::BIG_YAW,
                target_in_big_yaw.get_target_state().timestamp
            );
            target_in_big_yaw.set_target_state([&](auto& s) {
                auto pos = s.pos();
                pos = old_in_big_yaw * pos;
                auto vel = s.vel();
                vel = old_in_big_yaw.linear() * vel;
                s.set_pos(pos);
                s.set_vel(vel);
            });
            if (brain) {
                brain->update_armor_target(target_in_big_yaw);
            }

            armor_target.write(__armor_target);

            auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - armors.timestamp
            )
                                  .count();
            log_ctx.latency_ms_total += latency_ms;
            log_ctx.found_count += armor_tracker.get_count();
            armor_tracker.reset_count();
            if (auto_aim_dbg) {
                auto_aim_dbg->armors.set(armors);
#ifdef USE_ROS2
                rcl::pub_armor_marker(rcl_node, SentryFrame_to_str(armors.frame_id), armors);
                rcl::pub_armor_target_marker(
                    rcl_node,
                    SentryFrame_to_str(__armor_target.get_target_state().frame_id),
                    __armor_target
                );
#endif
            }

            log_ctx.track_count++;
        }
    });
    s.add_rate_source<>("solver", 1000.0, [&]() {
        log_ctx.solve_count++;
        bool is_omni = false;
        auto_aim::ArmorTarget target;
        target = armor_target.read();
        if (!target.check()) {
            target = omni_armor_target.read();
            is_omni = true;
        }
        int old_this_id = target.this_id;
        auto gimbal_odom_state_in_odom = wheel_odometry.state; //转为相对gimbal_odom
        gimbal_odom_state_in_odom.predict(Clock::now());
        target.set_target_state([&](auto& s) {
            s.frame_id = std::to_underlying(SentryFrame::GIMBAL_ODOM);
            s.x[armor_point_motion_model::idx::CX] -= gimbal_odom_state_in_odom.pos().x();
            s.x[armor_point_motion_model::idx::CY] -= gimbal_odom_state_in_odom.pos().y();
            s.x[armor_point_motion_model::idx::CZ] -= gimbal_odom_state_in_odom.pos().z();
            s.x[armor_point_motion_model::idx::VCX] -= gimbal_odom_state_in_odom.vel().x();
            s.x[armor_point_motion_model::idx::VCY] -= gimbal_odom_state_in_odom.vel().y();
            s.x[armor_point_motion_model::idx::VCZ] -= gimbal_odom_state_in_odom.vel().z();
        });
        target.this_id = old_this_id;

        GimbalCmd cmd {
            .appear = false,
        };
        if (target.check()) {
            cmd = very_aimer.very_aim(
                target,
                (!is_omni ? bullet_speed : 100.0),
                (!is_omni ? auto_aim_fsm_controller.get_state()
                          : auto_aim::AutoAimFsm::AIM_SINGLE_ARMOR)
            );
        }

        if (serial) {
            SendRobotCmdData send;
            send.cmd_ID = SendRobotCmdData::ID;

            uint32_t t = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - start_tp
            )
                             .count();
            send.time_stamp = t;
            send.appear = cmd.appear;
            send.detect_color = std::to_underlying(enemy_color);
            send.yaw = cmd.yaw;
            send.pitch = cmd.pitch;
            send.v_yaw = cmd.v_yaw;
            send.target_yaw = cmd.target_yaw;
            send.target_pitch = cmd.target_pitch;
            send.v_pitch = cmd.v_pitch;
            send.a_yaw = cmd.a_yaw;
            send.a_pitch = cmd.a_pitch;
            send.enable_yaw_diff = cmd.enable_yaw_diff;
            send.enable_pitch_diff = cmd.enable_pitch_diff;
            serial->write(std::move(utils::to_vector(send)));
        }
        auto old_in_camera_cv = tf->pose_a_in_b(
            SentryFrame(cmd.aim_point.frame_id),
            SentryFrame::CAMERA_CV,
            cmd.timestamp
        );
        cmd.aim_point.transform(old_in_camera_cv, std::to_underlying(SentryFrame::CAMERA_CV));
        if (auto_aim_dbg && is_web_running()) {
            auto_aim_dbg->gimbal_cmd.set(cmd);
        }
    });
    auto armor_omni_task = s.register_source<>("armor_omni");
    s.add_rate_source<>("armor_omni_tragger", armor_omni.fps_, [&]() {
        auto main_target = armor_target.read();
        if (main_target.check()) {
            return;
        }
        auto_aim::ArmorOmni::One& one = armor_omni.get_next();
        auto img_frame = one.camera->read();
        if (img_frame.src_img.empty()) {
            AWAKENING_WARN("Failed to read image from camera.");
            return;
        }

        CommonFrame common_frame;
        common_frame.expanded = cv::Rect(0, 0, img_frame.src_img.cols, img_frame.src_img.rows);
        common_frame.offset = cv::Point2f(0, 0);
        common_frame.img_frame = std::move(img_frame);
        common_frame.frame_id = one.cv_frame_id;
        common_frame.id = one.order_id++;
        auto target = one.target;
        if (target.check()) {
            auto camera_cv_in_old = tf->pose_a_in_b(
                SentryFrame(common_frame.frame_id),
                SentryFrame(target.get_target_state().frame_id),
                common_frame.img_frame.timestamp
            );
            target.set_target_state([&](armor_point_motion_model::State& state) {
                state.predict(common_frame.img_frame.timestamp, target.target_number);
            });
            auto bbox = target.expanded_one_one(
                common_frame.img_frame.timestamp,
                camera_cv_in_old,
                camera_info,
                common_frame.img_frame.src_img.size()
            );
            if (bbox.area() > 200) {
                common_frame.expanded = bbox;
                common_frame.offset = cv::Point2f(bbox.x, bbox.y);
            }
        }
        s.runtime_push_source<>(armor_omni_task, [&, f = std::move(common_frame)]() {
            static std::unique_ptr<std::counting_semaphore<>> detector_sem;
            if (!detector_sem) {
                detector_sem = std::make_unique<std::counting_semaphore<>>(
                    config["armor_omni"]["max_infer_num"].as<int>()
                );
            }
            {
                auto_aim::Armors armors { .timestamp = f.img_frame.timestamp,
                                          .id = f.id,
                                          .frame_id = f.frame_id };
                {
                    bool got = detector_sem->try_acquire();
                    utils::SemaphoreGuard guard(*detector_sem, got);
                    if (got) {
                        auto tmp_armors = armor_omni.detector_->detect(f);
                        for (auto& a: tmp_armors) {
                            if ((enemy_color == EnemyColor::BLUE
                                 && a.color == auto_aim::ArmorColor::RED)
                                || (enemy_color == EnemyColor::RED
                                    && a.color == auto_aim::ArmorColor::BLUE))
                            {
                                continue;
                            }
                            armors.armors.push_back(a);
                        }
                    }
                }

                one.armors_queue->enqueue(armors);
            }

            auto batch_armors = one.armors_queue->dequeue_batch();
            for (auto& _armors: batch_armors) {
                auto camera_cv_in_odom = tf->pose_a_in_b(
                    SentryFrame(_armors.frame_id),
                    SentryFrame::ODOM,
                    _armors.timestamp
                );
                _armors.frame_id = std::to_underlying(SentryFrame::ODOM);
                one.target =
                    one.tracker->track(_armors, camera_info, camera_cv_in_odom, _armors.frame_id);
                if (!one.target.check()) {
                    one.target.update_count = 0;
                }
                one.total_score = one.target.update_count;
                auto best_target = armor_omni.update();
                omni_armor_target.write(best_target);
                auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::steady_clock::now() - _armors.timestamp
                )
                                      .count();
                log_ctx.omni_latency_ms_total += latency_ms;
                log_ctx.omni_count++;
                if (auto_aim_dbg) {
#ifdef USE_ROS2
                    rcl::pub_armor_target_marker(
                        rcl_node,
                        SentryFrame_to_str(best_target.get_target_state().frame_id),
                        best_target
                    );
#endif
                }
            }
        });
    });
    s.add_rate_source<>("logger", 1.0, [&]() {
        double avg_latency_ms = log_ctx.latency_ms_total / log_ctx.track_count;
        double omni_avg_latency_ms = log_ctx.omni_latency_ms_total / log_ctx.omni_count;
        AWAKENING_INFO(
            "detect: {} track: {} found: {} solve: {} serial: {} camera: {} avg_latency: {:.3} ms omni: {} avg_latency: {:.3} ms",
            log_ctx.detect_count,
            log_ctx.track_count,
            log_ctx.found_count,
            log_ctx.solve_count,
            log_ctx.serial_count,
            log_ctx.camera_count,
            avg_latency_ms,
            log_ctx.omni_count,
            omni_avg_latency_ms
        );
        if (auto_aim_dbg) {
            auto_aim_dbg->avg_latency_ms.set(avg_latency_ms);
        }
        log_ctx.reset();
    });
    if (auto_aim_dbg) {
        s.add_rate_source<>("debug", 45.0, [&]() {
            if (!is_web_running()) {
                return;
            }
            auto target = armor_target.read();
            if (!target.check()) {
                omni_armor_target.read().write_log();
            } else {
                target.write_log();
            }
            wheel_odometry.write_log();
            auto img_now = auto_aim_dbg->img_frame.get().timestamp;
            auto_aim_dbg->armor_target.set(target);
            auto_aim_dbg->fsm_state.set(auto_aim_fsm_controller.get_state());
            auto gimbal_in_gimbal_odom =
                tf->pose_a_in_b(SentryFrame::GIMBAL, SentryFrame::GIMBAL_ODOM, Clock::now());
            auto euler =
                utils::matrix2euler(gimbal_in_gimbal_odom.linear(), utils::EulerOrder::ZYX);
            auto gimbal_yaw_pitch =
                std::make_pair(angles::to_degrees(euler[0]), -angles::to_degrees(euler[1]));
            auto_aim_dbg->gimbal_yaw_pitch.set(gimbal_yaw_pitch);
            write_debug_data(auto_aim_dbg.value());
            bullet_pick_up.update(
                Clock::now(),
                auto_aim_dbg->gimbal_cmd.get().appear ? auto_aim_dbg->gimbal_cmd.get().fly_time
                                                      : 0.4
            );
            auto bullet_poss =
                bullet_pick_up.get_bullet_positions(img_now, very_aimer.get_yaw_pitch_offset());
            auto odom_in_camera_cv =
                tf->pose_a_in_b(SentryFrame::ODOM, SentryFrame::CAMERA_CV, img_now);
            for (auto& pos: bullet_poss) {
                pos = odom_in_camera_cv * pos;
            }
            auto_aim_dbg->odom_in_camera_cv.set(odom_in_camera_cv);
            auto_aim_dbg->bullet_positions.set(bullet_poss);
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
#ifdef USE_ROS2
            // auto old_in_big_yaw = tf->pose_a_in_b(
            //     SentryFrame(target.get_target_state().frame_id),
            //     SentryFrame::BIG_YAW,
            //     Clock::now()
            // );
            // auto pos = old_in_big_yaw * target.get_target_state().pos();
            // auto vel = old_in_big_yaw.linear() * target.get_target_state().vel();
            // auto end = pos + vel;
            // struct ARROW_TAG;
            // rcl::pub_arrow<ARROW_TAG>("target_in_big_yaw", rcl_node, "gimbal_yaw", pos, end);

#endif
        });
#ifdef USE_ROS2
        s.add_rate_source<>("tf_pub", 100.0, [&]() {
            rcl_tf.pub_robot_tf(*tf, [](SentryFrame frame) { return SentryFrame_to_str(frame); });
        });
#endif
    }
    auto cmd_sub = rcl_node.make_sub<geometry_msgs::msg::Twist>(
        "cmd_vel",
        rclcpp::QoS(10),
        [&](const geometry_msgs::msg::Twist::SharedPtr msg) {
            SendNavCmdData send;

            send.cmd_ID = SendNavCmdData::ID;
            send.vx = msg->linear.x;
            send.vy = msg->linear.y;
            send.wz = msg->angular.z;
            if (serial) {
                serial->write(utils::to_vector(send));
            }
        }
    );
    rcl_node.push_sub(cmd_sub);

    if (player) {
        auto cam = s.register_source<CameraIO>("hik");
        auto serial = s.register_source<SerialIO>("serial");
        player->subscribe<CameraTag>([&](ImageFrame&& f) {
            s.runtime_push_source<CameraIO>(cam, [&, _f = std::move(f)]() {
                return std::make_tuple(std::optional<CameraIO::second_type>(std::move(_f)));
            });
        });
        player->subscribe<SerialTag>([&](std::vector<uint8_t>&& buf) {
            s.runtime_push_source<SerialIO>(serial, [&, __buf = std::move(buf)]() {
                return std::make_tuple(std::optional<SerialIO::second_type>(std::move(__buf)));
            });
        });

    } else {
        if (camera) {
            camera->start<CameraTag>("hik");
        }

        if (serial) {
            serial->start<SerialTag>("serial");
        }
    }
    s.build();
    s.run();
    if (player) {
        std::thread([&]() { player->play(1.0); }).detach();
    }
#ifdef USE_ROS2
    std::thread([&]() { rcl_node.spin(); }).detach();
#endif
    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();
#ifdef USE_ROS2
    rcl_node.shutdown();
#endif
    for (int i = 0; i < 10; ++i) {
        AWAKENING_CRITICAL("改了东西记得同步其他有关的exe的src");
    }
    return 0;
}
