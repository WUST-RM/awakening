#pragma once
#include "../static_map.hpp"
namespace awakening::sentry_brain {
MAP_KEY(start, 0, 0, 0)
MAP_KEY(home, 0, 0, 0)
MAP_KEY(ally_fort, 0, 0, 0)
MAP_KEY(enemy_fly_land, 0, 0, 0)
MAP_KEY(ally_beijing_tunnel_bottom, 0, 0, 0)
MAP_KEY(ally_beijing_tunnel_top, 0, 0, 0)
MAP_KEY(ally_jiansudai_tunnel_bottom, 0, 0, 0)
MAP_KEY(ally_jiansudai_tunnel_top, 0, 0, 0)
MAP_KEY(ally_second_step_bottom, 0, 0, 0)
MAP_KEY(ally_outpost, 0, 0, 0)
MAP_KEY(ally_highlands_gain, 0, 0, 0)
MAP_KEY(ally_jiansudai_u, 0, 0, 0)
MAP_KEY(enemy_fort, 0, 0, 0)
MAP_KEY(ally_fly_land, 0, 0, 0)
MAP_KEY(enemy_beijing_tunnel_bottom, 0, 0, 0)
MAP_KEY(enemy_beijing_tunnel_top, 0, 0, 0)
MAP_KEY(enemy_jiansudai_tunnel_bottom, 0, 0, 0)
MAP_KEY(enemy_jiansudai_tunnel_top, 0, 0, 0)
MAP_KEY(enemy_second_step_bottom, 0, 0, 0)
MAP_KEY(enemy_outpost, 0, 0, 0)
MAP_KEY(enemy_highlands_gain, 0, 0, 0)
MAP_KEY(enemy_jiansudai_u, 0, 0, 0)
using MapKeys = std::tuple<
    start_t,
    home_t,
    ally_fort_t,
    enemy_fly_land_t,
    ally_beijing_tunnel_bottom_t,
    ally_beijing_tunnel_top_t,
    ally_jiansudai_tunnel_bottom_t,
    ally_jiansudai_tunnel_top_t,
    ally_second_step_bottom_t,
    ally_outpost_t,
    ally_highlands_gain_t,
    ally_jiansudai_u_t,
    enemy_fort_t,
    enemy_fly_land_t,
    enemy_beijing_tunnel_bottom_t,
    enemy_beijing_tunnel_top_t,
    enemy_jiansudai_tunnel_bottom_t,
    enemy_jiansudai_tunnel_top_t,
    enemy_second_step_bottom_t,
    enemy_outpost_t,
    enemy_highlands_gain_t,
    enemy_jiansudai_u_t>;
using RMUC2026Map = StaticMap<MapKeys>;
} // namespace awakening::sentry_brain