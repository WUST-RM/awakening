#pragma once
#include "utils/logger.hpp"
#include "utils/scheduler/scheduler.hpp"
#include <boost/asio.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <thread>
#include <vector>

namespace awakening {

class SerialDriver {
public:
    struct Params {
        unsigned int baud_rate = 115200;
        unsigned int char_size = 8;
        boost::asio::serial_port_base::parity::type parity =
            boost::asio::serial_port_base::parity::none;
        boost::asio::serial_port_base::stop_bits::type stop_bits =
            boost::asio::serial_port_base::stop_bits::one;
        boost::asio::serial_port_base::flow_control::type flow_control =
            boost::asio::serial_port_base::flow_control::none;
        std::string device_name;
        size_t read_buf_size = 4096;

        void load(const YAML::Node& config) {
            device_name = config["device_name"].as<std::string>();
            baud_rate = config["baud_rate"].as<unsigned int>();
            char_size = config["char_size"].as<unsigned int>();
            read_buf_size = config["read_buf_size"].as<size_t>();
        }
    } params_;

    SerialDriver(const YAML::Node& config, Scheduler& scheduler):
        scheduler_(scheduler),
        io_(),
        port_(io_),
        running_(false) {
        params_.load(config);
        read_buf_.resize(params_.read_buf_size);
    }

    ~SerialDriver() {
        stop();
    }

    void error_handler(const boost::system::error_code& ec) {
        if (ec && ec != boost::asio::error::operation_aborted) {
            AWAKENING_ERROR("serial error: {}", ec.message());
        }
    }

    template<typename Tag>
    void start(std::string source_name) {
        using IO = IOPair<Tag, std::vector<uint8_t>>;
        source_snapshot_id_ = scheduler_.register_source<IO>(source_name);

        running_ = true;

        io_thread_ = std::thread([this]() { run_io<IO>(); });
    }

    void stop() {
        if (!running_)
            return;

        running_ = false;

        boost::system::error_code ec;
        port_.cancel(ec);
        port_.close(ec);

        io_.stop();

        if (io_thread_.joinable())
            io_thread_.join();

        error_handler(ec);
    }

    bool write(const std::vector<uint8_t>& data) {
        uint64_t gen = generation_;

        boost::asio::post(io_, [this, data, gen]() mutable {
            if (gen != generation_)
                return;

            bool idle = write_queue_.empty();
            write_queue_.push_back(std::move(data));

            if (idle) {
                do_write(gen);
            }
        });

        return true;
    }

private:
    template<typename IO>
    void run_io() {
        while (running_) {
            if (!open_port()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            uint64_t gen = ++generation_;

            clear_buffers();

            start_read<IO>(gen);

            io_.run();
            io_.restart();

            close_port();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    bool open_port() {
        boost::system::error_code ec;

        port_.open(params_.device_name, ec);
        if (ec) {
            error_handler(ec);
            return false;
        }

        port_.set_option(boost::asio::serial_port_base::baud_rate(params_.baud_rate), ec);
        port_.set_option(boost::asio::serial_port_base::character_size(params_.char_size), ec);
        port_.set_option(boost::asio::serial_port_base::parity(params_.parity), ec);
        port_.set_option(boost::asio::serial_port_base::stop_bits(params_.stop_bits), ec);
        port_.set_option(boost::asio::serial_port_base::flow_control(params_.flow_control), ec);

        if (ec) {
            error_handler(ec);
            return false;
        }

        AWAKENING_INFO("serial open success: {}", params_.device_name);
        return true;
    }

    void close_port() {
        boost::system::error_code ec;
        port_.cancel(ec);
        port_.close(ec);

        write_queue_.clear();

        error_handler(ec);
    }

    void clear_buffers() {
        std::fill(read_buf_.begin(), read_buf_.end(), 0);
        write_queue_.clear();
    }

    template<typename IO>
    void start_read(uint64_t gen) {
        port_.async_read_some(
            boost::asio::buffer(read_buf_),
            [this, gen](boost::system::error_code ec, size_t n) {
                if (gen != generation_)
                    return;

                if (ec) {
                    if (ec != boost::asio::error::operation_aborted) {
                        error_handler(ec);
                    }
                    io_.stop();
                    return;
                }

                if (n > 0) {
                    std::vector<uint8_t> buf(read_buf_.data(), read_buf_.data() + n);

                    scheduler_.runtime_push_source<IO>(
                        source_snapshot_id_,
                        [__buf = std::move(buf)]() mutable {
                            return std::make_tuple(
                                std::optional<typename IO::second_type>(std::move(__buf))
                            );
                        }
                    );
                }

                start_read<IO>(gen);
            }
        );
    }

    void do_write(uint64_t gen) {
        if (write_queue_.empty())
            return;

        boost::asio::async_write(
            port_,
            boost::asio::buffer(write_queue_.front()),
            [this, gen](boost::system::error_code ec, size_t) {
                if (gen != generation_)
                    return;

                if (ec) {
                    if (ec != boost::asio::error::operation_aborted) {
                        error_handler(ec);
                    }
                    io_.stop();
                    return;
                }

                write_queue_.pop_front();

                if (!write_queue_.empty()) {
                    do_write(gen);
                }
            }
        );
    }

private:
    boost::asio::io_context io_;
    boost::asio::serial_port port_;

    std::vector<uint8_t> read_buf_;
    std::deque<std::vector<uint8_t>> write_queue_;

    std::atomic<bool> running_;
    std::atomic<uint64_t> generation_ { 0 };

    Scheduler& scheduler_;
    size_t source_snapshot_id_ { 0 };

    std::thread io_thread_;
};

} // namespace awakening