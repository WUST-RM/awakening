#include "_rcl/node.hpp"
#include "_rcl/tf.hpp"
#include "ascii_banner.hpp"
#include "tasks/base/common.hpp"
#include "tasks/radar_detect/detector.hpp"
#include "tasks/radar_detect/lidar_location.hpp"
#include "tasks/radar_detect/type.hpp"
#include "utils/buffer.hpp"
#include "utils/common/image.hpp"
#include "utils/common/type_common.hpp"
#include "utils/drivers/hik_camera.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_tf.hpp"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <optional>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/msg/detail/point_cloud2__struct.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <string>
#include <utility>
#include <vector>
#include <yaml-cpp/node/parse.h>
using namespace awakening;
struct CameraTag {};
struct SerialTag {};
struct DetectTag {};
struct FrameTag {};
using CameraIO = IOPair<CameraTag, ImageFrame>;
using SerialIO = IOPair<SerialTag, std::vector<uint8_t>>;
using CommonFrameIo = IOPair<FrameTag, CommonFrame>;
enum class RadarFrame : int { TARGET_MAP, MID70, CAMERA_CV, N };
using RadarTF = utils::tf::RobotTF<RadarFrame, static_cast<size_t>(RadarFrame::N), false>;
std::string RadarFrame_to_str(int f) {
    constexpr const char* details[] = { "target_map", "mid_70", "camera_cv" };
    return std::string(details[f]);
}
std::string RadarFrame_to_str(RadarFrame f) {
    return RadarFrame_to_str(std::to_underlying(f));
}
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
    bool debug = false;
    std::string config_path;
    auto first_arg = get_arg(1);
    if (first_arg) {
        config_path = first_arg.value();
    } else {
        return 1;
    }
    auto config = YAML::LoadFile(config_path);
    auto camera_config = config["camera"];
    CameraInfo camera_info;
    camera_info.load(camera_config["camera_info"]);
    Scheduler s;
    rcl::RclcppNode rcl_node("auto_aim");
    rcl::TF rcl_tf(rcl_node);
    RadarTF tf;
    utils::SWMR<radar_detect::CarPool> car_pool;
    {
        tf.add_edge(RadarFrame::TARGET_MAP, RadarFrame::MID70);
        tf.add_edge(RadarFrame::MID70, RadarFrame::CAMERA_CV);
        ISO3 mid70_in_target_map = utils::load_isometry3(config["tf"]["mid70_in_target_map"]);
        tf.push(RadarFrame::TARGET_MAP, RadarFrame::MID70, Clock::now(), mid70_in_target_map);
        ISO3 camera_cv_in_mid70 = utils::load_isometry3(config["tf"]["camera_cv_in_mid70"]);
        // camera_cv_in_mid70.linear() = R_CV2PHYSICS;
        tf.push(RadarFrame::MID70, RadarFrame::CAMERA_CV, Clock::now(), camera_cv_in_mid70);
    }
    std::unique_ptr<HikCamera> camera;
    utils::SignalGuard::add_callback([&]() {
        if (camera) {
            camera->stop();
        }
    });

    camera = std::make_unique<HikCamera>(camera_config["hik_camera"], s);
    camera->init();
    if (!camera->running_) {
        return 0;
    }
    cv::namedWindow("Video Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Frame", 800, 600);

    radar_detect::Detector detector(config["detector"]);
    radar_detect::LidarLocation lidar_location(config["lidar_location"]);
    s.register_task<CameraIO, CommonFrameIo>("push_common_frame", [&](CameraIO::second_type&& f) {
        static int current_id = 0;
        int x = 0;
        int y = 0;
        int w = f.src_img.cols;
        int h = f.src_img.rows;
        CommonFrame frame {
            .img_frame = std::move(f),
            .id = current_id++,
            .frame_id = 0,
            .expanded = cv::Rect(x, y, w, h),
            .offset = cv::Point2f(x, y),
        };

        return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::move(frame)));
    });
    s.register_task<CommonFrameIo>("detector", [&](CommonFrameIo::second_type&& frame) {
        static std::unique_ptr<std::counting_semaphore<>> detector_sem;
        if (!detector_sem) {
            detector_sem =
                std::make_unique<std::counting_semaphore<>>(config["max_infer_num"].as<int>());
        }
        auto img = frame.img_frame.src_img;
        auto start = Clock::now();
        auto cars = detector.detect(frame);

        std::cout << cars.size() << std::endl;
        auto end = Clock::now();
        std::cout << "cost : "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms" << std::endl;
        cv::rectangle(img, frame.expanded, cv::Scalar(0, 255, 0), 3);
        for (const auto& car: cars) {
            car.draw(img);
        }
        // Show the image with the progress bar
        cv::imshow("Video Frame", img);
        cv::waitKey(1);
        return;
    });

    auto target_map_pub =
        rcl_node.make_pub<sensor_msgs::msg::PointCloud2>("target_map", rclcpp::QoS(10));
    auto target_map_bbox_pub =
        rcl_node.make_pub<visualization_msgs::msg::Marker>("target_map_bbox", rclcpp::QoS(10));
    auto marker_pub =
        rcl_node.make_pub<visualization_msgs::msg::MarkerArray>("radar_marker", rclcpp::QoS(10));
    auto lidar_sub = rcl_node.make_sub<sensor_msgs::msg::PointCloud2>(
        "lidar",
        rclcpp::SensorDataQoS(),
        [&](const sensor_msgs::msg::PointCloud2::SharedPtr pc_msg) {
            // const size_t size = pc_msg->width * pc_msg->height;
            // std::vector<Eigen::Vector3f> pts(size);

            // auto mid70_in_target_map =
            //     tf.pose_a_in_b(RadarFrame::MID70, RadarFrame::TARGET_MAP, Clock::now());
            // sensor_msgs::PointCloud2ConstIterator<float> iter_x(*pc_msg, "x");
            // sensor_msgs::PointCloud2ConstIterator<float> iter_y(*pc_msg, "y");
            // sensor_msgs::PointCloud2ConstIterator<float> iter_z(*pc_msg, "z");

            // for (size_t i = 0; i < size; ++i) {
            //     Eigen::Vector3f p(*iter_x, *iter_y, *iter_z);
            //     pts[i] = mid70_in_target_map.cast<float>() * p;
            //     ++iter_x;
            //     ++iter_y;
            //     ++iter_z;
            // }
            // auto __car_pool = lidar_location.detect(pts);
            // car_pool.write(__car_pool);
            // auto marker = __car_pool.to_marker_array(RadarFrame_to_str(RadarFrame::TARGET_MAP));
            // marker_pub->publish(marker);
        }
    );
    rcl_node.push_sub(lidar_sub);
    s.add_rate_source<>("tf_pub", 100.0, [&]() {
        rcl_tf.pub_robot_tf(tf, [](RadarFrame frame) { return RadarFrame_to_str(frame); });
    });
    s.add_rate_source<>("debug", 10.0, [&]() {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = RadarFrame_to_str(RadarFrame::TARGET_MAP);
        m.header.stamp = rcl_node.get_node()->get_clock()->now();
        m.ns = "box";
        m.id = 0;
        m.type = visualization_msgs::msg::Marker::LINE_LIST;
        m.action = visualization_msgs::msg::Marker::ADD;
        auto boxEdges = [](const radar_detect::AABB& box) {
            const auto& mn = box.min_pt;
            const auto& mx = box.max_pt;

            std::vector<Eigen::Vector3d> p = {
                { mn.x(), mn.y(), mn.z() }, { mx.x(), mn.y(), mn.z() }, { mx.x(), mx.y(), mn.z() },
                { mn.x(), mx.y(), mn.z() }, { mn.x(), mn.y(), mx.z() }, { mx.x(), mn.y(), mx.z() },
                { mx.x(), mx.y(), mx.z() }, { mn.x(), mx.y(), mx.z() },
            };

            int idx[] = { 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7 };

            std::vector<Eigen::Vector3d> edges;
            for (int i: idx)
                edges.push_back(p[i]);
            return edges;
        };
        auto bbox = lidar_location.get_target_map_bbox();
        auto edges = boxEdges(radar_detect::AABB(bbox.first, bbox.second));
        for (auto& p: edges) {
            geometry_msgs::msg::Point pt;
            pt.x = p.x();
            pt.y = p.y();
            pt.z = p.z();
            m.points.push_back(pt);
        }

        m.scale.x = 0.03; // 线宽
        m.color.b = 1.0;
        m.color.a = 1.0;
        m.lifetime = rclcpp::Duration::from_seconds(0.2);

        target_map_bbox_pub->publish(m);
        if (target_map_pub->get_subscription_count() > 0) {
            auto& target_map_pts = lidar_location.get_target_map_pts();
            sensor_msgs::msg::PointCloud2 pc;

            pc.header.stamp = rcl_node.get_node()->get_clock()->now();
            pc.header.frame_id = RadarFrame_to_str(RadarFrame::TARGET_MAP);
            pc.width = target_map_pts.size();
            pc.height = 1;
            pc.fields.reserve(4);
            sensor_msgs::msg::PointField field;
            field.name = "x";
            field.offset = 0;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count = 1;
            pc.fields.push_back(field);
            field.name = "y";
            field.offset = 4;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count = 1;
            pc.fields.push_back(field);
            field.name = "z";
            field.offset = 8;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count = 1;
            pc.fields.push_back(field);
            pc.is_bigendian = false;
            pc.point_step = 12;
            pc.row_step = pc.width * pc.point_step;
            pc.data.resize(pc.row_step * pc.height);
            auto pointer = reinterpret_cast<float*>(pc.data.data());
            for (const auto& point: target_map_pts) {
                *pointer = point.x();
                ++pointer;
                *pointer = point.y();
                ++pointer;
                *pointer = point.z();
                ++pointer;
            }
            target_map_pub->publish(pc);
        }
    });
    if (camera) {
        camera->start<CameraTag>("hik");
    }
    s.build();
    s.run();
    std::thread([&]() { rcl_node.spin(); }).detach();

    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    rcl_node.shutdown();
    s.stop();
    cv::destroyAllWindows();

    for (int i = 0; i < 10; ++i) {
        AWAKENING_CRITICAL("改了东西记得同步其他有关的exe的src");
    }
    return 0;
}