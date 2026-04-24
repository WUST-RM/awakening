#pragma once
#include "tasks/base/packet_typedef_receive.hpp"
#include "utils/logger.hpp"
#include <algorithm>
#include <cstdlib>

namespace awakening::sentry_brain {

class GobalState {
public:
    enum class SelfColor : bool { Red = 0, Blue = 1 } self_color;
    enum class Pose : uint8_t { UNKNOWEN, Attack, Defend, Move } pose;
    void update(const SentryRefereeReceive& d) {
        self_color = (d.robo_id > 100) ? SelfColor::Blue : SelfColor::Red;
        current_hp = d.current_hp;
        max_hp = d.max_hp;
        if (d.game_time < current_game_time_) {
            reset();
        }

        current_game_time_ = d.game_time;

        fort_allowance_bullets_ = d.fort_allowance_bullets;
        current_bullets_ = d.allowance_bullets;
        ally_outpost_hp_ = d.ally_outpost_hp;
        ally_base_hp_ = d.ally_base_hp;

        if (current_game_time_ >= 0) {
            update_pose(Pose(d.current_pose), current_game_time_);
            update_home_bullets(current_game_time_);
            update_remain_build_outpost_chance(ally_outpost_hp_, ally_base_hp_);
        }
    }

private:
    void update_pose(Pose new_pose, int game_time) {
        if (new_pose != pose) {
            pose = new_pose;
            last_change_pose_time_ = game_time;
        }
        if (std::abs(last_change_pose_time_ - game_time) > 170) {
            AWAKENING_INFO("need change pose");
        }
    }

    void update_home_bullets(int game_time) {
        int current_min = game_time / 60;

        if (current_min > last_min_) {
            int delta_min = current_min - last_min_;
            home_allowance_bullets_ += delta_min * 100;
            last_min_ = current_min;
        }
    }

    void update_remain_build_outpost_chance(int ally_outpost_hp, int ally_base_hp) {
        if (!ally_outpost_has_first_dead_) {
            if (ally_outpost_hp < last_ally_outpost_hp_) {
                if (ally_outpost_hp < 1) {
                    ally_outpost_has_first_dead_ = true;
                }
            }
            last_ally_outpost_hp_ = ally_outpost_hp;
            return;
        }

        if (last_base_hp_ - ally_base_hp >= 1000) {
            remain_rebuild_outpost_chance_++;
            last_base_hp_ = ally_base_hp;
        }

        if (ally_outpost_hp > last_ally_outpost_hp_) {
            if (ally_outpost_hp > 1450) {
                remain_rebuild_outpost_chance_--;
            }
        }

        last_ally_outpost_hp_ = ally_outpost_hp;

        remain_rebuild_outpost_chance_ = std::max(0, remain_rebuild_outpost_chance_);
    }

    void reset() {
        last_min_ = 0;
        home_allowance_bullets_ = 0;

        remain_rebuild_outpost_chance_ = 0;
        ally_outpost_has_first_dead_ = false;

        last_ally_outpost_hp_ = 0;
        last_base_hp_ = 5000;
        last_change_pose_time_ = 0;
    }

private:
    int current_hp = 0;
    int max_hp = 400;
    int current_game_time_ = -1;
    int fort_allowance_bullets_ = 0;
    int home_allowance_bullets_ = 0;
    int current_bullets_ = 0;

    int remain_rebuild_outpost_chance_ = 0;

    int ally_base_hp_ = 0;
    int ally_outpost_hp_ = 0;

    int last_min_ = 0;
    int last_ally_outpost_hp_ = 0;
    int last_base_hp_ = 5000;
    int last_change_pose_time_ = 0;
    bool ally_outpost_has_first_dead_ = false;
};

} // namespace awakening::sentry_brain