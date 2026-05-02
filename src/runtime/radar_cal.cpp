#include "_rcl/node.hpp"
#include "ascii_banner.hpp"
#include "utils/common/type_common.hpp"
#include "utils/io/pcd_io.h"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <chrono>
#include <memory>
#include <mutex>
#include <open3d/Open3D.h> 
#include <optional>
#include <sensor_msgs/msg/detail/point_cloud2__struct.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <small_gicp/ann/kdtree_tbb.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
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

    // 获取配置文件路径
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

    // 加载配置
    auto config = YAML::LoadFile(config_path);
    auto cal_config = config["radar_cal"];
    rcl::RclcppNode rcl_node("radar_cal");

    // 加载变换矩阵
    Eigen::Isometry3d source_in_target = utils::load_isometry3(cal_config["source_in_target"]);

    // 创建源和目标点云对象
    small_gicp::PointCloud::Ptr source_pointcloud = std::make_shared<small_gicp::PointCloud>();
    small_gicp::PointCloud::Ptr target_pointcloud = std::make_shared<small_gicp::PointCloud>();

    // 目标点云数据
    std::vector<Eigen::Vector3f> target_points;
    small_gicp::KdTree<small_gicp::PointCloud>::Ptr target_tree;
    auto target_pcd = cal_config["target_pcd"].as<std::string>();
    Eigen::Vector3d min_pos = Eigen::Vector3d::Zero();
    auto max_pos = min_pos;

    // 读取目标点云
    if (io::pcd::read_pcd(target_pcd, target_points)) {
        target_pointcloud->resize(target_points.size());
        for (size_t i = 0; i < target_points.size(); ++i) {
            max_pos = max_pos.cwiseMax(target_points[i].cast<double>());
            min_pos = min_pos.cwiseMin(target_points[i].cast<double>());
            target_pointcloud->point(i) << target_points[i].cast<double>(), 1.0;
        }
        // small_gicp::voxelgrid_sampling_tbb(*target_pointcloud, 0.05);
        small_gicp::estimate_normals_tbb(*target_pointcloud, 20);
        small_gicp::estimate_covariances_tbb(*target_pointcloud, 20);
        target_tree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(
            target_pointcloud,
            small_gicp::KdTreeBuilderTBB()
        );
    } else {
        std::cout << "Target PCD load failed!" << std::endl;
        return 1;
    }
    max_pos += Eigen::Vector3d(3, 3, 3);
    min_pos -= Eigen::Vector3d(3, 3, 3);
    // 配准对象
    std::shared_ptr<
        small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionTBB>>
        register_;
    register_ = std::make_shared<
        small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionTBB>>();

    // 配准参数
    register_->rejector.max_dist_sq = 20.0;
    register_->optimizer.max_iterations = 20;

    std::vector<Eigen::Vector3d> source_points;
    std::mutex source_mutex;
    bool new_data = false; // 标记新数据到来
    bool collect_data = false; // 是否开始收集数据
    auto start_time = std::chrono::steady_clock::now(); // 记录开始时间

    // Open3D 可视化相关
    open3d::geometry::PointCloud source_open3d, target_open3d;
    open3d::visualization::Visualizer vis;

    // 订阅点云数据
    auto pc_sub = rcl_node.make_sub<sensor_msgs::msg::PointCloud2>(
        "lidar",
        rclcpp::QoS(10),
        [&](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            static auto start = Clock::now();
            auto now = Clock::now();

            // 如果已经收集了5秒钟数据，则停止收集
            if (collect_data
                && std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= 5)
            {
                collect_data = false; // 停止收集
                return; // 退出数据收集
            }

            if (collect_data) {
                std::lock_guard<std::mutex> lock(source_mutex);
                sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
                sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
                sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
                const size_t size = msg->width * msg->height;

                for (size_t i = 0; i < size; ++i) {
                    Eigen::Vector3d p_src(*iter_x, *iter_y, *iter_z);
                    Eigen::Vector3d p_tgt = source_in_target * p_src;
                    ++iter_x;
                    ++iter_y;
                    ++iter_z;
                    if (p_src.z() > 0||p_src.norm() >25) {
                        continue;
                    }
                    source_points.push_back(p_src); // 仍存储原始源坐标系点，用于后续配准

                    
                }
                new_data = true; // 设置新数据标记
            } else {
                collect_data = true; // 开始收集数据
                start_time = std::chrono::steady_clock::now(); // 重置开始时间
            }
        }
    );

    std::thread([&]() { rcl_node.spin(); }).detach();

    while (utils::SignalGuard::running()) {
        if (new_data && !collect_data) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            std::vector<Eigen::Vector3d> s_pts;
            {
                std::lock_guard<std::mutex> lock(source_mutex);
                s_pts = source_points;
                new_data = false; // 重置数据标记
            }
            std::vector<Eigen::Vector3f> source_points_f;
            for (const auto& p: s_pts) {
                source_points_f.push_back(p.cast<float>());
            }
            io::pcd::write_pcd("source.pcd", source_points_f);
            std::cout << s_pts.size() << std::endl;
            source_pointcloud->resize(s_pts.size());

            // 将 source_points 转换为小GICP所需的点云格式
            for (size_t i = 0; i < s_pts.size(); ++i) {
                source_pointcloud->point(i) << s_pts[i].cast<double>(), 1.0;
            }
            // small_gicp::voxelgrid_sampling_tbb(*source_pointcloud, 0.05);
            small_gicp::estimate_normals_tbb(*source_pointcloud, 20);
            small_gicp::estimate_covariances_tbb(*source_pointcloud, 20);

            auto result =
                register_
                    ->align(*target_pointcloud, *source_pointcloud, *target_tree, source_in_target);

            if (!result.converged) {
                RCLCPP_ERROR_STREAM(
                    rclcpp::get_logger("rose_nav:lm"),
                    "GICP did not converge, iter_num: " << result.iterations
                );
            }

            // 更新 source_in_target 变换矩阵
            source_in_target = result.T_target_source;

            std::cout << "t: " << source_in_target.translation() << std::endl;
            std::cout << "r: " << source_in_target.linear() << std::endl;
            
            // 清空并填充 Open3D 点云
            source_open3d.points_.clear();
            target_open3d.points_.clear();
            source_open3d.colors_.clear(); // Clear the color list as well
            target_open3d.colors_.clear();

            // 定义颜色
            Eigen::Vector3d source_color(1.0, 0.0, 0.0); // Red for the source
            Eigen::Vector3d target_color(0.0, 0.0, 1.0); // Blue for the target

            // 将 source_points 转换为 Open3D 点云格式，并应用颜色
            for (auto& p: s_pts) {
                // Apply transformation to each source point
                p = source_in_target * p;
                source_open3d.points_.emplace_back(p(0), p(1), p(2));
                source_open3d.colors_.emplace_back(
                    source_color(0),
                    source_color(1),
                    source_color(2)
                ); // Apply red color
            }

            // 将 target_points 转换为 Open3D 点云格式，并应用颜色
            for (const auto& p: target_points) {
                target_open3d.points_.emplace_back(p(0), p(1), p(2));
                target_open3d.colors_.emplace_back(
                    target_color(0),
                    target_color(1),
                    target_color(2)
                ); // Apply blue color
            }

            // 创建和更新可视化窗口
            vis.CreateVisualizerWindow("PointCloud Registration", 1600, 900);
            vis.ClearGeometries();
            vis.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(source_open3d));
            vis.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(target_open3d));

            // 运行可视化并销毁窗口
            vis.Run();
            vis.DestroyVisualizerWindow();
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 减少 CPU 占用
    }

    rcl_node.shutdown();
    return 0;
}