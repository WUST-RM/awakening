#pragma once
#include "angles.h"
#include "common.hpp"
#include "packet_typedef.hpp"
#include "utils/tf.hpp"
#include "wust_vl/common/drivers/serial_driver.hpp"
#include <functional>
#include <opencv2/core/types.hpp>
#include <yaml-cpp/node/parse.h>
namespace awakening {
enum class Frame : int { ODOM, GIMBAL_ODOM, GIMBAL, CAMERA, SHOOT };
struct LinkBuffer {
    TimePoseBuffer buffer;
    LinkBuffer(size_t size = 1024): buffer(size) {}
};

class RobotTF {
public:
    using Ptr = std::shared_ptr<RobotTF>;
    static Ptr create() {
        return std::make_shared<RobotTF>();
    }

    RobotTF() = default;

    void push(Frame parent, Frame child, const TimePoint& t, const ISO3& pose) {
        LinkBuffer& buf = getBuffer(parent, child);
        buf.buffer.push(t, pose);
    }

    void setStatic(Frame parent, Frame child, const ISO3& pose) {
        LinkBuffer& buf = getBuffer(parent, child);
        buf.buffer.push(TimePoint::min(), pose);
    }

    template<Frame From, Frame To>
    ISO3 get(const TimePoint& t) const {
        return getImpl<From, To>(t);
    }

    template<Frame From, Frame To>
    ISO3 transform(const ISO3& pose, const TimePoint& t) const {
        ISO3 T_A_B = get<From, To>(t);
        return T_A_B * pose;
    }

private:
    template<Frame From, Frame To>
    ISO3 getImpl(const TimePoint& t) const;

    LinkBuffer& getBuffer(Frame parent, Frame child) {
        assert(size_t(parent) < buffers_.size() && size_t(child) < buffers_.size());
        return buffers_[size_t(parent)][size_t(child)];
    }

    const LinkBuffer& getBuffer(Frame parent, Frame child) const {
        assert(size_t(parent) < buffers_.size() && size_t(child) < buffers_.size());
        return buffers_[size_t(parent)][size_t(child)];
    }

private:
    std::array<std::array<LinkBuffer, 5>, 5> buffers_;
};

template<>
inline ISO3 RobotTF::getImpl<Frame::CAMERA, Frame::ODOM>(const TimePoint& t) const {
    return getBuffer(Frame::CAMERA, Frame::GIMBAL).buffer.get(t)
        * getBuffer(Frame::GIMBAL, Frame::GIMBAL_ODOM).buffer.get(t)
        * getBuffer(Frame::GIMBAL_ODOM, Frame::ODOM).buffer.get(t);
}

template<>
inline ISO3 RobotTF::getImpl<Frame::CAMERA, Frame::GIMBAL_ODOM>(const TimePoint& t) const {
    return getBuffer(Frame::CAMERA, Frame::GIMBAL).buffer.get(t)
        * getBuffer(Frame::GIMBAL, Frame::GIMBAL_ODOM).buffer.get(t);
}

template<>
inline ISO3 RobotTF::getImpl<Frame::GIMBAL, Frame::ODOM>(const TimePoint& t) const {
    return getBuffer(Frame::GIMBAL, Frame::GIMBAL_ODOM).buffer.get(t)
        * getBuffer(Frame::GIMBAL_ODOM, Frame::ODOM).buffer.get(t);
}

template<>
inline ISO3 RobotTF::getImpl<Frame::SHOOT, Frame::ODOM>(const TimePoint& t) const {
    return getBuffer(Frame::SHOOT, Frame::GIMBAL).buffer.get(t)
        * getBuffer(Frame::GIMBAL, Frame::GIMBAL_ODOM).buffer.get(t)
        * getBuffer(Frame::GIMBAL_ODOM, Frame::ODOM).buffer.get(t);
}

template<>
inline ISO3 RobotTF::getImpl<Frame::GIMBAL, Frame::GIMBAL_ODOM>(const TimePoint& t) const {
    return getBuffer(Frame::GIMBAL, Frame::GIMBAL_ODOM).buffer.get(t);
}

template<>
inline ISO3 RobotTF::getImpl<Frame::GIMBAL_ODOM, Frame::ODOM>(const TimePoint& t) const {
    return getBuffer(Frame::GIMBAL_ODOM, Frame::ODOM).buffer.get(t);
}
class Robot {
public:
    using Ptr = std::shared_ptr<Robot>;
    Robot() {
        // try {
        //     const auto config = YAML::LoadFile("/home/hy/awakening/config/robot.yaml");

        //     const auto serial_config = config["serial"];
        //     if (serial_config && serial_config["enable"].as<bool>(false)) {
        //         if (!serial_) {
        //             serial_ = std::make_shared<wust_vl::common::drivers::SerialDriver>();
        //         }

        //         wust_vl::common::drivers::SerialDriver::SerialPortConfig cfg {
        //             /*baud*/ 115200,
        //             /*csize*/ 8,
        //             boost::asio::serial_port_base::parity::none,
        //             boost::asio::serial_port_base::stop_bits::one,
        //             boost::asio::serial_port_base::flow_control::none
        //         };

        //         std::string device_name = serial_config["device_name"].as<std::string>();

        //         serial_->init_port(device_name, cfg);

        //         serial_->set_receive_callback(std::bind(
        //             &Robot::serialCallback,
        //             this,
        //             std::placeholders::_1,
        //             std::placeholders::_2
        //         ));

        //         serial_->set_error_callback([this](const boost::system::error_code& ec) {
        //             WUST_ERROR("serial") << "serial error: " << ec.message();
        //         });
        //     }
        //     if (!main_camera_) {
        //         main_camera_ = Camera<MainCameraCtx>::create();
        //     }
        //     if (config["main_camera"]) {
        //         const auto main_camera_path = config["main_camera"].as<std::string>();
        //         const auto main_camera_config = YAML::LoadFile(main_camera_path);
        //         main_camera_->load(
        //             main_camera_config,
        //             std::bind(&Robot::frameCallback, this, std::placeholders::_1)
        //         );
        //     }

        // } catch (const YAML::Exception& e) {
        //     WUST_ERROR("robot") << "YAML error: " << e.what();
        // } catch (const std::exception& e) {
        //     WUST_ERROR("robot") << "std exception: " << e.what();
        // } catch (...) {
        //     WUST_ERROR("robot") << "unknown exception during Robot init";
        // }
    }
    void start() {}
    // void frameCallback(wust_vl::video::ImageFrame& img_frame) {
    //     if (img_frame.src_img.empty()) {
    //         return;
    //     }
    //     CommonFrame frame;
    //     frame.img_frame = std::move(img_frame);
    //     frame.detect_color = enemy_color_;
    //     frame.expanded = cv::Rect(0, 0, frame.img_frame.src_img.cols, frame.img_frame.src_img.rows);
    //     frame.offset = cv::Point2f(0, 0);

    //     if (!main_camera_task_) {
    //         main_camera_task_ = Task<CommonFrame>::create([&]() { return frame; });
    //     } else {
    //         main_camera_task_->fn = [&]() { return frame; };
    //     }

    //     scheduler_.addTask(main_camera_task_);
    // }
    void serialCallback(const uint8_t* data, std::size_t len) {
        auto robo = ReceiveRobotData::create(data, len);
        if (robo) {
            receiveRobo(robo.value());
        }
    }
    void receiveRobo(const ReceiveRobotData& robo) {
        enemy_color_ = EnemyColor(robo.detect_color);
        bullet_speed_ = robo.bullet_speed;

        ISO3 gimbal_in_gimbal_odom;
        gimbal_in_gimbal_odom.translation() = Vec3::Zero();

        const double roll = -angles::from_degrees(robo.roll);
        const double pitch = -angles::from_degrees(robo.pitch);
        const double yaw = angles::from_degrees(robo.yaw);

        Mat3 R =
            (AngleAxis(yaw, Vec3::UnitZ()).toRotationMatrix()
             * AngleAxis(pitch, Vec3::UnitY()).toRotationMatrix()
             * AngleAxis(roll, Vec3::UnitX()).toRotationMatrix());

        gimbal_in_gimbal_odom.linear() = R;

        auto now = Clock::now();
        tf_.push(Frame::GIMBAL_ODOM, Frame::GIMBAL, now, gimbal_in_gimbal_odom);
    }

    // struct MainCameraCtx {
    //     Frame frame_id = Frame::CAMERA;
    // };
    // Task<CommonFrame>::Ptr main_camera_task_;
    // Scheduler scheduler_;
    // Camera<MainCameraCtx>::Ptr main_camera_;
    // std::shared_ptr<wust_vl::common::drivers::SerialDriver> serial_;
    RobotTF tf_;
    EnemyColor enemy_color_;
    double bullet_speed_;
};
} // namespace awakening