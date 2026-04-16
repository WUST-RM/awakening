#pragma once
#include "traj.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <deque>
#include <optional>
#include <utility>
#include <vector>
#include <yaml-cpp/node/node.h>
namespace awakening {

class BallisticTrajectory {
public:
    using Ptr = std::shared_ptr<BallisticTrajectory>;

    struct Params {
        double gravity = 9.8;
        double resistance = 0.092;

        void load(const YAML::Node& config) {
            if (config["gravity"])
                gravity = config["gravity"].as<double>();
            if (config["resistance"])
                resistance = config["resistance"].as<double>();
            if (gravity <= 0.0) {
                throw std::invalid_argument("gravity must be positive");
            }
        }
    };

public:
    explicit BallisticTrajectory(const YAML::Node& config) {
        params_.load(config);
        k_ = std::max(params_.resistance, kEps);
        inv_k_ = 1.0 / k_;
        g_ = params_.gravity;
    }

    static Ptr create(const YAML::Node& config) {
        return std::make_shared<BallisticTrajectory>(config);
    }

    std::optional<double> solve_pitch(const Vec3& target_pos, double v0) const {
        const double h = target_pos.z();
        const double d = std::hypot(target_pos.x(), target_pos.y());

        if (d < kEps || v0 < kEps) {
            return std::nullopt;
        }

        // 防止 exp 溢出
        const double kd = k_ * d;
        if (kd > 700.0) {
            return std::nullopt;
        }

        // A = (exp(kd) - 1) / k
        const double e = std::expm1(kd);
        const double A = e * inv_k_;

        // C = A / v0
        const double C = A / v0;

        // B = 0.5 * g * C^2
        const double B = 0.5 * g_ * C * C;

        // 二次方程: B*u^2 - A*u + (B + h) = 0
        const double disc = A * A - 4.0 * B * (B + h);
        if (disc < 0.0 || !std::isfinite(disc)) {
            return std::nullopt;
        }

        const double sqrt_disc = std::sqrt(disc);
        const double denom = 2.0 * B;

        if (std::abs(denom) < kEps) {
            return std::nullopt;
        }

        const double u1 = (A + sqrt_disc) / denom;
        const double u2 = (A - sqrt_disc) / denom;

        const double p1 = std::atan(u1);
        const double p2 = std::atan(u2);

        constexpr double left = -M_PI / 4.0;
        constexpr double right = M_PI / 3.0;

        auto valid = [&](double p) {
            return std::isfinite(p) && p >= left && p <= right && std::abs(std::cos(p)) > kEps;
        };

        if (valid(p1))
            return p1;
        if (valid(p2))
            return p2;

        return std::nullopt;
    }

    std::optional<std::pair<double, double>>
    solve_pitch_and_flytime(const Vec3& target_pos, double v0) const {
        const double d = std::hypot(target_pos.x(), target_pos.y());

        auto pitch_opt = solve_pitch(target_pos, v0);
        if (!pitch_opt)
            return std::nullopt;

        const double cos_theta = std::cos(*pitch_opt);
        if (std::abs(cos_theta) < kEps)
            return std::nullopt;

        const double kd = k_ * d;
        if (kd > 700.0)
            return std::nullopt;

        const double t = std::expm1(kd) / (k_ * v0 * cos_theta);

        if (!std::isfinite(t) || t < 0.0) {
            return std::nullopt;
        }

        return std::make_pair(*pitch_opt, t);
    }
    std::pair<double, double> solve_distance_height(double pitch, double v0, double t) const {
        return forward_model(pitch, v0, t);
    }

private:
    static constexpr double kEps = 1e-6;

    Params params_;

    double k_;
    double inv_k_;
    double g_;

private:
    double flight_time(double distance, double v0, double pitch) const {
        const double cos_theta = std::cos(pitch);
        if (std::abs(cos_theta) < kEps) {
            return std::numeric_limits<double>::infinity();
        }

        const double kd = k_ * distance;
        if (kd > 700.0) {
            return std::numeric_limits<double>::infinity();
        }

        return std::expm1(kd) / (k_ * v0 * cos_theta);
    }

    std::pair<double, double> forward_model(double pitch, double v0, double t) const {
        const double cos_theta = std::cos(pitch);

        // x = ln(1 + k*v*cos*t) / k
        const double x = std::log1p(k_ * v0 * cos_theta * t) * inv_k_;

        // y = v*sin*t - 0.5*g*t^2
        const double y = v0 * std::sin(pitch) * t - 0.5 * g_ * t * t;

        return { x, y };
    }
};
struct Bullet {
    TimePoint fire_time;
    ISO3 fire_time_shoot_in_odom;
    double speed_in_odom;
    std::optional<Vec3>
    get_pos_at(TimePoint t, BallisticTrajectory::Ptr b, const std::pair<double, double>& offset)
        const {
        double dt = std::chrono::duration<double>(t - fire_time).count();
        if (dt <= 0) {
            return std::nullopt;
        }
        auto euler = utils::matrix2euler(fire_time_shoot_in_odom.linear(), utils::EulerOrder::ZYX);
        double yaw = euler[0] - offset.first;
        double pitch = -euler[1] - offset.second;
        auto [dis, height] = b->solve_distance_height(pitch, speed_in_odom, dt);
        double x = dis * std::cos(yaw);
        double y = dis * std::sin(yaw);
        double z = height;
        return fire_time_shoot_in_odom.translation() + Vec3(x, y, z);
    }
};
class BulletPickUp {
public:
    mutable std::mutex mtx;
    std::deque<Bullet> bullets;
    BulletPickUp(const YAML::Node& config) {
        b = BallisticTrajectory::create(config["ballistic_trajectory"]);
    }
    void push_back(const Bullet& bullet) {
        std::lock_guard<std::mutex> lock(mtx);
        bullets.push_back(std::move(bullet));
    }
    void update(TimePoint t, double max_fly_time) {
        if (bullets.empty())
            return;
        std::lock_guard<std::mutex> lock(mtx);
        while (!bullets.empty()
               && std::chrono::duration<double>(t - bullets.front().fire_time).count()
                   > max_fly_time)
        {
            bullets.pop_front();
        }
    }
    std::vector<Vec3>
    get_bullet_positions(TimePoint t, const std::pair<double, double>& offset) const {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<Vec3> positions;
        for (const auto& bullet: bullets) {
            auto p_opt = bullet.get_pos_at(t, b, offset);
            if (p_opt) {
                positions.push_back(*p_opt);
            }
        }
        return positions;
    }
    BallisticTrajectory::Ptr b;
};
} // namespace awakening
