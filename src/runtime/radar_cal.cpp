#include "_rcl/node.hpp"
#include "ascii_banner.hpp"
#include "utils/common/type_common.hpp"
#include "utils/io/pcd_io.h"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <chrono>
#include <memory>
#include <optional>
#include <sensor_msgs/msg/detail/point_cloud2__struct.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <small_gicp/ann/kdtree_tbb.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/factors/icp_factor.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/reduction_tbb.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>
#include <small_gicp/util/normal_estimation_tbb.hpp>
#include <thread>
#include <vector>
using namespace awakening;

int main(int argc, char** argv) {
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
    auto cal_config = config["radar_cal"];
    rcl::RclcppNode rcl_node("radar_cal");
    Eigen::Isometry3d source_in_target = utils::load_isometry3(cal_config["source_in_target"]);
    small_gicp::PointCloud::Ptr source_pointcloud = std::make_shared<small_gicp::PointCloud>();
    small_gicp::PointCloud::Ptr target_pointcloud = std::make_shared<small_gicp::PointCloud>();
    std::vector<Eigen::Vector3f> target_points;
    small_gicp::KdTree<small_gicp::PointCloud>::Ptr target_tree;
    auto target_pcd = cal_config["target_pcd"].as<std::string>();
    if (io::pcd::read_pcd(target_pcd, target_points)) {
        target_pointcloud->resize(target_points.size());
        for (size_t i = 0; i < target_points.size(); ++i) {
            target_pointcloud->point(i) << target_points[i].cast<double>(), 1.0;
        }

        small_gicp::estimate_normals_tbb(*target_pointcloud, 20);
        small_gicp::estimate_covariances_tbb(*target_pointcloud, 20);
        target_tree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(
            target_pointcloud,
            small_gicp::KdTreeBuilderTBB()
        );
    } else {
        return 1;
    }
    std::shared_ptr<
        small_gicp::Registration<small_gicp::ICPFactor, small_gicp::ParallelReductionTBB>>
        register_;
    register_ = std::make_shared<
        small_gicp::Registration<small_gicp::ICPFactor, small_gicp::ParallelReductionTBB>>();
    register_->rejector.max_dist_sq = 20.0;
    register_->optimizer.max_iterations = 999;
    std::vector<Eigen::Vector3f> source_points;
    bool has_caled = false;
    auto pc_sub = rcl_node.make_sub<sensor_msgs::msg::PointCloud2>(
        "lidar",
        rclcpp::QoS(10),
        [&](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            static auto start = Clock::now();
            auto now = Clock::now();
            if (now - start < std::chrono::duration<double>(5.0)) {
                sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
                sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
                sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
                const size_t size = msg->width * msg->height;
                source_points.reserve(source_points.size() + size);
                for (size_t i = 0; i < size; ++i) {
                    Eigen::Vector3f p(*iter_x, *iter_y, *iter_z);
                    source_points.push_back(p);
                    ++iter_x;
                    ++iter_y;
                    ++iter_z;
                }
            } else {
                source_pointcloud->resize(source_points.size());
                for (size_t i = 0; i < source_points.size(); ++i) {
                    source_pointcloud->point(i) << source_points[i].cast<double>(), 1.0;
                }
                small_gicp::estimate_normals_tbb(*source_pointcloud, 20);
                small_gicp::estimate_covariances_tbb(*source_pointcloud, 20);

                auto result = register_->align(
                    *target_pointcloud,
                    *source_pointcloud,
                    *target_tree,
                    source_in_target.inverse()
                );
                if (!result.converged) {
                    RCLCPP_ERROR_STREAM(
                        rclcpp::get_logger("rose_nav:lm"),
                        "GICP did not converge, iter_num: " << result.iterations
                    );
                }
                source_in_target = result.T_target_source.inverse();
                std::cout << "t: " << source_in_target.linear() << std::endl;
                std::cout << "r: " << source_in_target.rotation() << std::endl;
            }
        }
    );

    std::thread([&]() { rcl_node.spin(); }).detach();
    // utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    while (!has_caled) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    rcl_node.shutdown();
    return 0;
}