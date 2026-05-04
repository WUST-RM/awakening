#pragma once
#include "tasks/radar_detect/rmuc_2026_map.hpp"
#include "tasks/radar_detect/target.hpp"
#include "tasks/radar_detect/tracker.hpp"
#include "tasks/radar_detect/type.hpp"
#include "utils/common/type_common.hpp"
#include <chrono>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/node/node.h>
namespace awakening::radar_detect {

struct TheOnlyCar {
    PointTarget uwb_state;
    TimePoint t;
    CarClass car_class;
    CarState car_state;
};
class CarPool {
public:
    struct Params {
        double to_guess_dt;
        void load(const YAML::Node& config) {
            to_guess_dt = config["to_guess_dt"].as<double>();
        }
    } params_;
    CarPool(const YAML::Node& config, RMUC2026Map& map): map_(map) {
        params_.load(config);
        for (const auto& kv: map_.robot_guesses) {
            cars_[kv.first] = TheOnlyCar { .t = Clock::now(),
                                           .car_class = CarClass(kv.first),
                                           .car_state = CarState::NEEDGUESS };
        }
    }
    std::vector<Target> search(CarClass car_class, const std::vector<Target>& targets) {
        std::vector<Target> ret;
        for (auto& target: targets) {
            if (target.fin_class == car_class) {
                ret.push_back(target);
            }
        }
        return ret;
    }
    const Target* match(const std::vector<Target>& targets, TheOnlyCar& car) {
        const Target* best_target = nullptr;
        size_t best_score = 0;

        for (const auto& target: targets) {
            if (target.hit_count > best_score) {
                best_target = &target;
                best_score = target.hit_count;
            }
        }

        if (best_target) {
            car.uwb_state = best_target->uwb_state;

            car.t = best_target->last_update;
            car.car_state = CarState::ACTIVE;
        }

        return best_target;
    }

    void update(const std::vector<Target>& targets) {
        std::vector<const Target*> matched;

        auto record_match = [&](CarClass cc, TheOnlyCar& fc) {
            auto found = search(cc, targets);
            const Target* best = match(found, fc);
            if (best)
                matched.push_back(best);
        };
        for (auto& [key, car]: cars_) {
            record_match(car.car_class, car);
        }
        unknown_cars_.clear();
        for (auto& c: targets) {
            bool is_matched = false;
            for (auto ptr: matched) {
                if (&c == ptr) {
                    is_matched = true;
                    break;
                }
            }
            if (!is_matched) {
                unknown_cars_.push_back(c);
            }
        }
        for (auto& [key, car]: cars_) {
            auto dt = std::chrono::duration<double>(Clock::now() - car.t);
            if (dt.count() > params_.to_guess_dt) {
                car.car_state = CarState::NEEDGUESS;
            }
            if (car.car_state == CarState::NEEDGUESS) {
                car.car_state = CarState::GUESSING;
                auto pos = map_.predict_guess(car.car_class, car.uwb_state.state.pos(), car.uwb_state.state.vel());
                car.uwb_state.state.set_pos(pos);
            }
        }
    }
    std::unordered_map<int, TheOnlyCar> get_fin_cars()const noexcept{
        return cars_;
    }
    RMUC2026Map& map_;
    std::unordered_map<int, TheOnlyCar> cars_;
    std::vector<Target> unknown_cars_;
};
} // namespace awakening::radar_detect