#pragma once
#include "traj.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <deque>
#include <optional>
#include <vector>
#include <yaml-cpp/node/node.h>
namespace awakening {

class BallisticTrajectory {
public:
    using Ptr = std::shared_ptr<BallisticTrajectory>;

    enum class Type : int { RESISTANCE };

    struct Params {
        double gravity = 9.8;
        double resistance = 0.092;
        int iter = 20;

        void load(const YAML::Node& config) {
            if (config["gravity"])
                gravity = config["gravity"].as<double>();
            if (config["resistance"])
                resistance = config["resistance"].as<double>();
            if (config["iter"])
                iter = config["iter"].as<int>();

            if (gravity <= 0.0) {
                throw std::invalid_argument("gravity must be positive");
            }
        }
    };

public:
    explicit BallisticTrajectory(const YAML::Node& config) {
        params_.load(config);
        type_ = str2Type(config["type"].as<std::string>());
    }

    static Ptr create(const YAML::Node& config) {
        return std::make_shared<BallisticTrajectory>(config);
    }

    std::optional<double> solve_pitch(const Vec3& target_pos, double v0) const {
        const double target_height = target_pos.z();
        const double distance = std::hypot(target_pos.x(), target_pos.y());

        if (distance < kEps || v0 < kEps) {
            return std::nullopt;
        }

        auto f = [&](double pitch) {
            double t = flight_time(distance, v0, pitch);
            if (!std::isfinite(t))
                return std::numeric_limits<double>::infinity();

            auto [x, y] = forward_model(pitch, v0, t);
            return y - target_height;
        };

        double left = -M_PI / 4.0;
        double right = M_PI / 3.0;

        double f_left = f(left);
        double f_right = f(right);

        if (!std::isfinite(f_left) || !std::isfinite(f_right) || f_left * f_right > 0) {
            return std::nullopt;
        }

        double mid = 0;
        for (int i = 0; i < params_.iter; ++i) {
            mid = 0.5 * (left + right);
            double f_mid = f(mid);

            if (!std::isfinite(f_mid))
                return std::nullopt;

            if (std::abs(f_mid) < 1e-4) {
                return mid;
            }

            if (f_left * f_mid < 0) {
                right = mid;
                f_right = f_mid;
            } else {
                left = mid;
                f_left = f_mid;
            }
        }

        return mid;
    }

    std::optional<double> solve_flytime(const Vec3& target_pos, double v0) const {
        auto pitch_opt = solve_pitch(target_pos, v0);
        if (!pitch_opt)
            return std::nullopt;

        const double distance = std::hypot(target_pos.x(), target_pos.y());
        double t = flight_time(distance, v0, *pitch_opt);

        if (!std::isfinite(t) || t < 0) {
            return std::nullopt;
        }

        return t;
    }

    std::pair<double, double> solve_distance_height(double pitch, double v0, double t) const {
        return forward_model(pitch, v0, t);
    }

private:
    static constexpr double kEps = 1e-6;

    Params params_;
    Type type_;

    static Type str2Type(std::string str) {
        std::transform(str.begin(), str.end(), str.begin(), ::toupper);

        if (str == "RESISTANCE")
            return Type::RESISTANCE;

        throw std::invalid_argument("Invalid type");
    }

    double flight_time(double distance, double v0, double pitch) const {
        const double cos_theta = std::cos(pitch);
        if (std::abs(cos_theta) < kEps) {
            return std::numeric_limits<double>::infinity();
        }

        switch (type_) {
            case Type::RESISTANCE: {
                double k = std::max(params_.resistance, kEps);
                double arg = k * distance;

                if (arg > 700.0)
                    return std::numeric_limits<double>::infinity();

                return std::expm1(arg) / (k * v0 * cos_theta);
            }
        }

        return std::numeric_limits<double>::infinity();
    }

    std::pair<double, double> forward_model(double pitch, double v0, double t) const {
        const double g = params_.gravity;

        switch (type_) {
            case Type::RESISTANCE: {
                double k = std::max(params_.resistance, kEps);

                double x = std::log1p(k * v0 * std::cos(pitch) * t) / k;
                double y = v0 * std::sin(pitch) * t - 0.5 * g * t * t;

                return { x, y };
            }
        }

        return { 0.0, 0.0 };
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
