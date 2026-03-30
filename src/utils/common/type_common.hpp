#pragma once
#include <Eigen/Dense>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <param_deliver.h>
#include <yaml-cpp/yaml.h>
namespace awakening {
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using ISO3 = Eigen::Isometry3d;
using Quaternion = Eigen::Quaterniond;
using AngleAxis = Eigen::AngleAxisd;
using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;
} // namespace awakening