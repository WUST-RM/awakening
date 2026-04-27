#include "livox_v1_publisher.hpp"
#include "utils/logger.hpp"
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
using namespace awakening::livox_v1_lidar;

void LidarPublisher::recv_spin() {
    std::array<unsigned char, pc_msg_size> recv_buf;
    boost::system::error_code error;
    std::size_t recv_length = 0;
    while (rclcpp::ok() && socket->is_open()) {
        socket->async_receive(
            boost::asio::buffer(recv_buf),
            [&](const boost::system::error_code& error_, std::size_t length_) {
                error = error_;
                recv_length = length_;
            }
        );
        ctx.restart();
        ctx.run_for(std::chrono::milliseconds(timeout_ms));
        if (!ctx.stopped()) {
            socket->cancel();
            ctx.run();
        }
        if (error && error != boost::asio::error::message_size) {
            AWAKENING_ERROR("Receiver error: {}", error.message().c_str());
            need_reconnect = true;
            continue;
        }

        auto header = reinterpret_cast<protocal::data_header*>(recv_buf.data());
        if (!check_header(*header)) {
            continue;
        }
        auto data = reinterpret_cast<protocal::type2_span*>(
            recv_buf.data() + sizeof(protocal::data_header)
        );
        process_type2(*header, *data);
    }
}

void LidarPublisher::heartbeat_spin() {
    std::array<uint8_t, 256> buf;
    using boost::asio::buffer;

    while (rclcpp::ok() && socket->is_open()) {
        if (need_reconnect) {
            // 发送握手包
            protocal::handshake_data hs;
            std::copy(local_ip.to_bytes().begin(), local_ip.to_bytes().end(), hs.user_ip);
            hs.data_port = local_port;
            hs.cmd_port = local_port;
            hs.imu_port = local_port;
            socket->send_to(
                buffer(
                    buf,
                    protocal::write_frame_buffer(
                        buf.data(),
                        reinterpret_cast<const uint8_t*>(&hs),
                        sizeof(hs),
                        buf.size()
                    )
                ),
                boost::asio::ip::udp::endpoint(dest_ip, protocal::dest_port)
            );

            socket->send_to(
                buffer(
                    buf,
                    protocal::write_frame_buffer(
                        buf.data(),
                        protocal::start_data,
                        sizeof(protocal::start_data),
                        buf.size()
                    )
                ),
                boost::asio::ip::udp::endpoint(dest_ip, protocal::dest_port)
            );
            need_reconnect = false;
        }
        socket->send_to(
            buffer(protocal::heartbeat_raw),
            boost::asio::ip::udp::endpoint(dest_ip, protocal::dest_port)
        );
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
bool is_wired_interface(const char* name) {
    return std::strncmp(name, "eth", 3) == 0 || std::strncmp(name, "enp", 3) == 0
        || std::strncmp(name, "eno", 3) == 0 || std::strncmp(name, "ens", 3) == 0;
}

std::string get_wired_ipv4(std::string default_ip) {
    struct ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == -1) {
        return {};
    }

    std::string result;
    bool found = false;
    for (auto* ifa = ifaddr; ifa; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr)
            continue;
        if (ifa->ifa_addr->sa_family != AF_INET)
            continue;
        if (ifa->ifa_flags & IFF_LOOPBACK)
            continue;
        if (!is_wired_interface(ifa->ifa_name))
            continue;
        found = true;
        char ip[INET_ADDRSTRLEN];
        auto* sa = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr);
        inet_ntop(AF_INET, &sa->sin_addr, ip, sizeof(ip));

        result = ip;
        break; // 取第一个有线 IPv4
    }

    freeifaddrs(ifaddr);
    if (!found) {
        result = default_ip;
    }
    return result;
}
LidarPublisher::LidarPublisher(const YAML::Node& config, rcl::RclcppNode& node): node_(node) {
    batch_dot_num = config["batch_dot_num"].as<size_t>();
    local_ip = boost::asio::ip::make_address_v4(get_wired_ipv4("192.168.10.53"));
    dest_ip = boost::asio::ip::make_address_v4(config["dest_ip"].as<std::string>());

    AWAKENING_INFO(

        "Params: pc_batch_size: {}, local_ip: {}, dest_ip: {}",
        batch_dot_num,
        local_ip.to_string().c_str(),
        dest_ip.to_string().c_str()
    );

    pc_pub =
        node.get_node()->create_publisher<PointCloud2>("lidar", rclcpp::QoS(rclcpp::KeepLast(10)));
    pc2_init();

    local_port = config["udp_port"].as<int>();
    timeout_ms = config["timeout_ms"].as<int>();
    AWAKENING_INFO("Params: udp_port: {}, timeout_ms: {}", local_port, timeout_ms);

    socket.emplace(ctx, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), local_port));
    heartbeat_thread = std::thread(&LidarPublisher::heartbeat_spin, this);
    recv_thread = std::thread(&LidarPublisher::recv_spin, this);
}
