#pragma once
#include "traj.hpp"
#include "utils/common/type_common.hpp"
#include <optional>
#include <yaml-cpp/node/node.h>
namespace awakening {
class Bullet {
    TimePoint fire_time;
    ISO3 fire_time_shoot_in_odom;
    double speed_shoot;
};
class BallisticTrajectory {
public:
    using Ptr = std::shared_ptr<BallisticTrajectory>;
    struct Params {
        double gravity = 9.8;
        double resistance = 0.092;
        int max_iter = 10;
        void load(const YAML::Node& config) {
            gravity = config["gravity"].as<double>();
            resistance = config["resistance"].as<double>();
            max_iter = config["max_iter"].as<int>();
        }
    } params_;

    BallisticTrajectory(const YAML::Node& config) {
        params_.load(config);
    }
    static Ptr create(const YAML::Node& config) {
        return std::make_shared<BallisticTrajectory>(config);
    }
    std::optional<double> solve_pitch(const Vec3& target_pos, double v0) const {
        const double target_height = target_pos.z();
        const double distance =
            std::sqrt(target_pos.x() * target_pos.x() + target_pos.y() * target_pos.y());

        if (distance < 1e-6 || v0 < 1e-3) {
            return std::nullopt;
        }

        // 二分法边界 [-45°, 60°]
        double left = -M_PI / 4.0;
        double right = M_PI / 3.0;

        auto f = [&](double angle) -> double {
            double t;
            if (params_.resistance < 1e-6) {
                t = distance / (v0 * std::cos(angle));
            } else {
                double r = std::max(params_.resistance, 1e-6);
                t = (std::exp(r * distance) - 1) / (r * v0 * std::cos(angle));
            }

            return v0 * std::sin(angle) * t - 0.5 * params_.gravity * t * t - target_height;
        };

        double f_left = f(left);
        double f_right = f(right);

        if (f_left * f_right > 0) {
            return std::nullopt; // 没有解
        }

        double mid = 0;
        for (int i = 0; i < params_.max_iter; ++i) {
            mid = 0.5 * (left + right);
            double f_mid = f(mid);

            if (std::abs(f_mid) < 1e-3 || (right - left) < 1e-6) {
                return std::make_optional(mid);
            }

            if (f_left * f_mid < 0) {
                right = mid;
                f_right = f_mid;
            } else {
                left = mid;
                f_left = f_mid;
            }
        }

        return std::make_optional(mid);
    }
    double solve_flytime(const Vec3& target_pos, const double v0) {
        double r = params_.resistance < 1e-4 ? 1e-4 : params_.resistance;
        double distance =
            std::sqrt(target_pos.x() * target_pos.x() + target_pos.y() * target_pos.y());
        double angle = std::atan2(target_pos.z(), distance);
        double t = (std::exp(r * distance) - 1) / (r * v0 * std::cos(angle));

        return t;
    }
    std::pair<double, double> solve_distance_height(double pitch, double v0, double t) const {
        double r = params_.resistance < 1e-4 ? 1e-4 : params_.resistance;
        double g = params_.gravity;

        double cos_theta = std::cos(pitch);
        double sin_theta = std::sin(pitch);

        if (v0 < 1e-6 || std::abs(cos_theta) < 1e-6) {
            return { 0.0, 0.0 };
        }
        double distance = std::log(1 + r * v0 * cos_theta * t) / r;
        double height = v0 * sin_theta * t - 0.5 * g * t * t;

        return { distance, height };
    }
};
} // namespace awakening
