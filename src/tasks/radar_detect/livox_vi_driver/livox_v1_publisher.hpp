#pragma once

#include "_rcl/node.hpp"
#include "protocal.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <thread>

#include <boost/asio.hpp>
#include <yaml-cpp/node/node.h>

namespace awakening::livox_v1_lidar {

using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;
constexpr size_t pc_msg_size = 1380;
#pragma pack(push, 1)
typedef struct {
    float x; /**< X axis, Unit:m */
    float y; /**< Y axis, Unit:m */
    float z; /**< Z axis, Unit:m */
    float reflectivity; /**< Reflectivity   */
    uint8_t tag; /**< Livox point tag   */
    uint8_t resv; /**< Reserved   */
    double timestamp; /**< Timestamp of point*/
} LivoxPointXyzrtlt;
#pragma pack(pop)

class LidarPublisher {
private:
    std::string frame_id = "mid_70";
    boost::asio::io_context ctx;
    boost::asio::ip::address_v4 local_ip;
    boost::asio::ip::address_v4 dest_ip;
    uint16_t local_port;
    std::optional<boost::asio::ip::udp::socket> socket;
    std::atomic_bool need_reconnect = true;

    void recv_spin();
    void heartbeat_spin();

    size_t batch_dot_num;
    int timeout_ms;
    rclcpp::Publisher<PointCloud2>::SharedPtr pc_pub;

    rclcpp::TimerBase::SharedPtr timer;
    std::thread recv_thread, heartbeat_thread;
    PointCloud2::SharedPtr pc_msg;

    void process_type2(const protocal::data_header& header, const protocal::type2_span& data) {
        for (size_t i = 0; i < protocal::dot_num; ++i) {
            pc2_write({
                data[i].x.value() / 1000.0f,
                data[i].y.value() / 1000.0f,
                data[i].z.value() / 1000.0f,
                static_cast<float>(data[i].reflectivity.value()),
                data[i].tag.value(),
                0x00,
                static_cast<double>(header.timestamp.value()), // TODO: 需要验证时间戳含义
            });
        }
    }

    void pc2_init() {
        pc_msg = std::make_shared<PointCloud2>();
        sensor_msgs::PointCloud2Modifier modifier(*pc_msg);
        modifier.setPointCloud2Fields(
            7,
            "x",
            1,
            PointField::FLOAT32,
            "y",
            1,
            PointField::FLOAT32,
            "z",
            1,
            PointField::FLOAT32,
            "intensity",
            1,
            PointField::FLOAT32,
            "tag",
            1,
            PointField::UINT8,
            "resv",
            1,
            PointField::UINT8,
            "timestamp",
            1,
            PointField::FLOAT64
        );
        pc_msg->header.frame_id.assign(frame_id);
        pc_msg->height = 1;
        pc_msg->width = batch_dot_num;
        pc_msg->row_step = pc_msg->width * pc_msg->point_step;
        pc_msg->is_bigendian = false;
        pc_msg->is_dense = true;
        pc_msg->data.resize(pc_msg->row_step);
    }

    void pc2_write(const LivoxPointXyzrtlt& pt) {
        auto data = reinterpret_cast<LivoxPointXyzrtlt*>(pc_msg->data.data());
        static size_t it = 0;
        data[it++] = pt;
        if (it == pc_msg->width) {
            pc_msg->header.stamp = node_.get_node()->now();
            pc_pub->publish(*pc_msg);
            it = 0;
        }
    }
    rcl::RclcppNode& node_;

public:
    LidarPublisher(const YAML::Node& config, rcl::RclcppNode& node);
    ~LidarPublisher() {
        if (recv_thread.joinable()) {
            recv_thread.join();
        }
        if (heartbeat_thread.joinable()) {
            heartbeat_thread.join();
        }
    }
};
} // namespace awakening::livox_v1_lidar