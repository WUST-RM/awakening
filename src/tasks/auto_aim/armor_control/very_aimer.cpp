#include "very_aimer.hpp"
#include "angles.h"
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/ballistic_trajectory.hpp"
#include "tasks/base/common.hpp"
#include "tasks/base/traj.hpp"
#include "utils/common/type_common.hpp"
#include "utils/logger.hpp"
#include <cmath>
#include <cstdlib>
#include <deque>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
namespace awakening::auto_aim {
struct VeryAimer::Impl {
    struct ControlPoint {
        double yaw;
        double pitch;
        int aim_id;
        AimPoint aim_point;
        bool valid;
    };
    struct GimbalState {
        struct State {
            double p;
            double v;
            double a;
            bool on_traj;
        };
        State yaw_state;
        State pitch_state;
        int aim_id = 0;
        GimbalState() {}
        GimbalState(const GimbalState::State& y, const GimbalState::State& p):
            yaw_state(y),
            pitch_state(p) {}
        static GimbalState lerp(const GimbalState& s0, const GimbalState& s1, double a) noexcept {
            GimbalState r;
            r.aim_id = (a < 0.5) ? s0.aim_id : s1.aim_id;
            r.yaw_state =
                GimbalState::State { .p = utils::lerp_angle(s0.yaw_state.p, s1.yaw_state.p, a),
                                     .v = std::lerp(s0.yaw_state.v, s1.yaw_state.v, a),
                                     .a = std::lerp(s0.yaw_state.a, s1.yaw_state.a, a) };
            r.pitch_state =
                GimbalState::State { .p = utils::lerp_angle(s0.pitch_state.p, s1.pitch_state.p, a),
                                     .v = std::lerp(s0.pitch_state.v, s1.pitch_state.v, a),
                                     .a = std::lerp(s0.pitch_state.a, s1.pitch_state.a, a) };

            return r;
        }
    };
    struct QuinticSegment {
        double T = 0.0;
        Eigen::Matrix<double, 6, 1> c;
        GimbalState::State head;
        GimbalState::State tail;
        bool on_traj;

        static inline Eigen::Matrix<double, 6, 1> solve1d_closed_form(
            double p0,
            double v0,
            double a0,
            double p1,
            double v1,
            double a1,
            double T
        ) noexcept {
            Eigen::Matrix<double, 6, 1> c;
            double T2 = T * T;
            double T3 = T2 * T;
            double T4 = T3 * T;
            double T5 = T4 * T;

            // known low-order coefficients
            double c0 = p0;
            double c1 = v0;
            double c2 = a0 * 0.5;

            // closed-form for c3, c4, c5 (derived from boundary conditions at t=T)
            double c3 =
                (-3.0 * T2 * a0 + T2 * a1 - 12.0 * T * v0 - 8.0 * T * v1 - 20.0 * p0 + 20.0 * p1)
                / (2.0 * T3);
            double c4 =
                (1.5 * T2 * a0 - T2 * a1 + 8.0 * T * v0 + 7.0 * T * v1 + 15.0 * p0 - 15.0 * p1)
                / T4;
            double c5 = (-T2 * a0 + T2 * a1 - 6.0 * T * v0 - 6.0 * T * v1 - 12.0 * p0 + 12.0 * p1)
                / (2.0 * T5);

            c << c0, c1, c2, c3, c4, c5;
            return c;
        }

        [[nodiscard]] static inline QuinticSegment build(
            const GimbalState::State& s0,
            const GimbalState::State& s1,
            double T,
            bool on_traj
        ) noexcept {
            QuinticSegment seg;
            seg.head = s0;
            seg.tail = s1;
            seg.T = T;
            seg.c = solve1d_closed_form(s0.p, s0.v, s0.a, s1.p, s1.v, s1.a, T);
            seg.on_traj = on_traj;
            return seg;
        }
        static inline double max_abs_acc(const Eigen::Matrix<double, 6, 1>& c, double T) noexcept {
            if (T <= 0.0)
                return 0.0;

            auto acc = [&](double t) {
                double t2 = t * t;
                return 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t2 + 20 * c[5] * t2 * t;
            };

            double max_acc = std::max(std::abs(acc(0.0)), std::abs(acc(T)));

            // jerk = 6c3 + 24c4 t + 60c5 t^2
            double A = 60.0 * c[5];
            double B = 24.0 * c[4];
            double C = 6.0 * c[3];

            const double eps = 1e-9;

            if (std::abs(A) < eps) {
                if (std::abs(B) > eps) {
                    double t = -C / B;
                    if (t > 0.0 && t < T)
                        max_acc = std::max(max_acc, std::abs(acc(t)));
                }
            } else {
                double D = B * B - 4 * A * C;
                if (D >= 0.0) {
                    double sqrtD = std::sqrt(D);
                    double inv2A = 1.0 / (2 * A);

                    double t1 = (-B + sqrtD) * inv2A;
                    double t2 = (-B - sqrtD) * inv2A;

                    if (t1 > 0.0 && t1 < T)
                        max_acc = std::max(max_acc, std::abs(acc(t1)));
                    if (t2 > 0.0 && t2 < T)
                        max_acc = std::max(max_acc, std::abs(acc(t2)));
                }
            }

            return std::isfinite(max_acc) ? max_acc : 0.0;
        }

        [[nodiscard]] double inline duration() const noexcept {
            return T;
        }

        [[nodiscard]] double inline max_acc() const noexcept {
            return QuinticSegment::max_abs_acc(c, T);
        }
        [[nodiscard]] GimbalState::State inline eval(double t) const noexcept {
            GimbalState::State s;
            if (T <= 0.0)
                return s;
            t = std::clamp(t, 0.0, T);
            double t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
            s.p = c[0] + c[1] * t + c[2] * t2 + c[3] * t3 + c[4] * t4 + c[5] * t5;
            s.v = c[1] + 2 * c[2] * t + 3 * c[3] * t2 + 4 * c[4] * t3 + 5 * c[5] * t4;
            s.a = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t2 + 20 * c[5] * t3;
            s.on_traj = on_traj;
            return s;
        }
    };
    class LimitTrajectory: public Trajectory<GimbalState, double> {
    public:
        struct Traj {
            std::vector<QuinticSegment> segs;
            std::vector<double> seg_prefix_time;
            void clear() {
                segs.clear();
                seg_prefix_time.clear();
            }
        };

        Traj yaw_traj;
        Traj pitch_traj;
        static inline double angle_diff(double a, double b) noexcept {
            double d = a - b;
            while (d > M_PI)
                d -= 2 * M_PI;
            while (d < -M_PI)
                d += 2 * M_PI;
            return d;
        }

        static inline double unwrap_angle(double prev, double curr) noexcept {
            return prev + angle_diff(curr, prev);
        }

        void unwrap_states(std::vector<GimbalState>& s) const noexcept {
            for (size_t i = 1; i < s.size(); ++i) {
                s[i].yaw_state.p = unwrap_angle(s[i - 1].yaw_state.p, s[i].yaw_state.p);
                s[i].pitch_state.p = unwrap_angle(s[i - 1].pitch_state.p, s[i].pitch_state.p);
            }
        }
        void clear() {
            Trajectory::clear();
            yaw_traj.clear();
            pitch_traj.clear();
        }

        [[nodiscard]] std::pair<std::vector<GimbalState::State>, std::vector<GimbalState::State>>
        compute_node_states(const std::vector<GimbalState>& gp, const std::vector<double>& prefix)
            const noexcept {
            const size_t N = gp.size();
            std::vector<GimbalState::State> yaw(N), pitch(N);
            for (size_t i = 0; i < N; ++i) {
                yaw[i] = gp[i].yaw_state;
                pitch[i] = gp[i].pitch_state;
            }
            if (N < 2)
                return { yaw, pitch };
            auto compute_va = [&](std::vector<GimbalState::State>& s) {
                // 边界
                s.front().v = s.back().v = 0.0;
                s.front().a = s.back().a = 0.0;

                for (size_t i = 1; i + 1 < N; ++i) {
                    const double dt0 = prefix[i] - prefix[i - 1];
                    const double dt1 = prefix[i + 1] - prefix[i];
                    const double denom = dt0 + dt1;

                    if (denom < 1e-6) {
                        s[i].v = s[i].a = 0.0;
                        continue;
                    }

                    const double w0 = dt1 / denom;
                    const double w1 = dt0 / denom;

                    s[i].v = w0 * (s[i].p - s[i - 1].p) / dt0 + w1 * (s[i + 1].p - s[i].p) / dt1;
                    s[i].a =
                        2.0 * ((s[i + 1].p - s[i].p) / dt1 - (s[i].p - s[i - 1].p) / dt0) / denom;
                }
            };

            compute_va(yaw);
            compute_va(pitch);

            return { yaw, pitch };
        }

        void limit_traj(
            Traj& traj,
            const std::vector<GimbalState::State>& s,
            const std::vector<double>& prefix,
            int near_change_idx,
            double max_acc
        ) const noexcept {
            traj.segs.clear();
            traj.seg_prefix_time.clear();

            const int N = static_cast<int>(s.size());
            if (N <= 1)
                return;

            auto buildSeg = [&](int l, int r) -> QuinticSegment {
                double dur = prefix[r] - prefix[l];
                return QuinticSegment::build(s[l], s[r], dur, false);
            };

            std::optional<std::pair<int, int>> interval;
            if (near_change_idx >= 0) {
                int l = std::clamp(near_change_idx, 0, N - 1);
                int r = std::clamp(near_change_idx + 1, 0, N - 1);
                if (l < r)
                    interval.emplace(l, r);
            }

            if (!interval) {
                traj.segs.reserve(N - 1);
                for (int i = 0; i < N - 1; ++i) {
                    traj.segs.push_back(
                        QuinticSegment::build(s[i], s[i + 1], prefix[i + 1] - prefix[i], true)
                    );
                }

                traj.seg_prefix_time.resize(traj.segs.size() + 1);
                traj.seg_prefix_time[0] = prefix[0];
                for (size_t i = 0; i < traj.segs.size(); ++i)
                    traj.seg_prefix_time[i + 1] = traj.seg_prefix_time[i] + traj.segs[i].duration();
                return;
            }

            {
                int& l = interval->first;
                int& r = interval->second;
                QuinticSegment seg = buildSeg(l, r);

                auto try_candidate = [&](int nl, int nr) -> bool {
                    nl = std::max(0, nl);
                    nr = std::min(N - 1, nr);
                    if (nl == l && nr == r)
                        return false;

                    QuinticSegment cand = buildSeg(nl, nr);
                    if (cand.max_acc() <= seg.max_acc()) {
                        l = nl;
                        r = nr;
                        seg = std::move(cand);
                        return true;
                    }
                    return false;
                };

                while (seg.max_acc() > max_acc) {
                    bool expanded = false;

                    if (l > 0 || r < N - 1)
                        expanded = try_candidate(l - 1, r + 1);

                    if (!expanded && l > 0)
                        expanded = try_candidate(l - 1, r);

                    if (!expanded && r < N - 1)
                        expanded = try_candidate(l, r + 1);

                    if (!expanded && (l > 0 || r < N - 1)) {
                        int nl = std::max(0, l - 1);
                        int nr = std::min(N - 1, r + 1);
                        QuinticSegment forceSeg = buildSeg(nl, nr);

                        if (forceSeg.max_acc() < seg.max_acc() || (nl == 0 && nr == N - 1)) {
                            l = nl;
                            r = nr;
                            seg = std::move(forceSeg);
                            expanded = true;
                        }
                    }

                    if (!expanded)
                        break;
                    if (l == 0 && r == N - 1 && seg.max_acc() > max_acc)
                        break;
                }
            }

            traj.segs.reserve(N - 1);
            for (int i = 0; i < N - 1; ++i) {
                if (interval && i == interval->first) {
                    traj.segs.push_back(buildSeg(interval->first, interval->second));
                    i = interval->second - 1; // skip covered indices
                } else {
                    traj.segs.push_back(
                        QuinticSegment::build(s[i], s[i + 1], prefix[i + 1] - prefix[i], true)
                    );
                }
            }

            traj.seg_prefix_time.resize(traj.segs.size() + 1);
            traj.seg_prefix_time[0] = prefix[0];
            for (size_t i = 0; i < traj.segs.size(); ++i)
                traj.seg_prefix_time[i + 1] = traj.seg_prefix_time[i] + traj.segs[i].duration();
        }

        void build_limit(double max_yaw_acc, double max_pitch_acc, double current_time) noexcept {
            auto& cp_vec = get_cp_vec();
            auto prefix = get_prefix();
            unwrap_states(cp_vec);
            auto [yaw_states, pitch_states] = compute_node_states(cp_vec, prefix);
            int best_idx = -1;
            double best_dist = std::numeric_limits<double>::max();
            const int N = static_cast<int>(cp_vec.size());
            if (N < 2)
                return;

            for (size_t i = 0; i < cp_vec.size(); ++i) {
                if (cp_vec[i].aim_id == cp_vec[i + 1].aim_id)
                    continue;

                const double seg_mid = 0.5 * (prefix[i] + prefix[i + 1]);
                const double dist = std::abs(seg_mid - current_time);

                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = static_cast<int>(i);
                }
            }
            limit_traj(yaw_traj, yaw_states, prefix, best_idx, max_yaw_acc);
            limit_traj(pitch_traj, pitch_states, prefix, best_idx, max_pitch_acc);
        }
        [[nodiscard]] inline GimbalState::State
        state_at(double t, const Traj& traj) const noexcept {
            if (traj.segs.empty())
                return {};
            if (t <= traj.seg_prefix_time[0])
                return traj.segs.front().eval(0.0);

            if (t >= traj.seg_prefix_time.back())
                return traj.segs.back().eval(traj.segs.back().T);

            const auto it =
                std::upper_bound(traj.seg_prefix_time.begin(), traj.seg_prefix_time.end(), t);

            size_t i = std::distance(traj.seg_prefix_time.begin(), it) - 1;
            i = std::min(i, traj.segs.size() - 1);

            const double t0 = traj.seg_prefix_time[i];
            return traj.segs[i].eval(t - t0);
        }
        [[nodiscard]] inline GimbalState state_at(double t) const noexcept {
            GimbalState::State yaw = state_at(t, yaw_traj);
            GimbalState::State pitch = state_at(t, pitch_traj);
            return GimbalState(yaw, pitch);
        }
    };
    struct Params {
        double sample_total_time;
        int sample_horizon;
        double control_delay;
        double max_yaw_acc;
        double max_pitch_acc;
        double prediction_delay;
        double aim_center_more_prediction_time;
        double comming_angle;
        double leaving_angle;
        double shooting_range_h;
        double shooting_range_w_small;
        double shooting_range_w_large;
        double min_enable_pitch_deg;
        double min_enable_yaw_deg;

        void load(const YAML::Node& config) {
            sample_total_time = config["sample_total_time"].as<double>();
            sample_horizon = config["sample_horizon"].as<int>();
            control_delay = config["control_delay"].as<double>();
            max_yaw_acc = config["max_yaw_acc"].as<double>();
            max_pitch_acc = config["max_pitch_acc"].as<double>();
            prediction_delay = config["prediction_delay"].as<double>();
            aim_center_more_prediction_time =
                config["aim_center_more_prediction_time"].as<double>();
            comming_angle = config["comming_angle"].as<double>();
            leaving_angle = config["leaving_angle"].as<double>();
            shooting_range_h = config["shooting_range_h"].as<double>();
            shooting_range_w_small = config["shooting_range_w_small"].as<double>();
            shooting_range_w_large = config["shooting_range_w_large"].as<double>();
            min_enable_pitch_deg = config["min_enable_pitch_deg"].as<double>();
            min_enable_yaw_deg = config["min_enable_yaw_deg"].as<double>();
        }
    } params_;
    Impl(const YAML::Node& config) {
        params_.load(config);
        ballistic_trajectory_ = BallisticTrajectory::create(config["ballistic_trajectory"]);
        base_yaw_offset_rad_ = angles::from_degrees(config["base_yaw_offset"].as<double>());
        base_pitch_offset_rad_ = angles::from_degrees(config["base_pitch_offset"].as<double>());
    }
    [[nodiscard]] int
    select_armor(const ArmorTarget& target, const AutoAimFsm& auto_aim_fsm) const noexcept {
        static int lock_id = -1;
        const auto target_state = target.get_target_state();
        const auto armors_xyza = target_state.get_armors_xyza(target.target_number);
        const int armor_num = static_cast<int>(armors_xyza.size());
        int i_chosen = 0;

        const double center_yaw = std::atan2(target_state.pos().y(), target_state.pos().x());

        std::vector<double> delta_angles;
        delta_angles.reserve(armor_num);
        for (int i = 0; i < armor_num; ++i) {
            delta_angles.push_back(angles::normalize_angle(armors_xyza[i][3] - center_yaw));
        }

        const auto pick_best_by_min_delta = [&](const std::vector<int>& idxs) -> int {
            int best = -1;
            double best_val = std::numeric_limits<double>::infinity();
            for (int i: idxs) {
                const double val = std::abs(delta_angles[i]);
                if (val < best_val) {
                    best_val = val;
                    best = i;
                }
            }
            return best;
        };

        if (auto_aim_fsm == AutoAimFsm::AIM_SINGLE_ARMOR
            && target.target_number != ArmorClass::OUTPOST && armor_num > 0)
        {
            std::vector<int> candidates;
            constexpr double in_first = 60.0 / 57.3;
            for (int i = 0; i < armor_num; ++i)
                if (std::abs(delta_angles[i]) <= in_first)
                    candidates.push_back(i);

            if (!candidates.empty()) {
                if (candidates.size() > 1) {
                    int a = candidates[0], b = candidates[1];
                    if (lock_id != a && lock_id != b) {
                        lock_id = (std::abs(delta_angles[a]) < std::abs(delta_angles[b])) ? a : b;
                    }
                    int pick = (lock_id >= 0 && lock_id < armor_num)
                        ? lock_id
                        : pick_best_by_min_delta(candidates);
                    if (pick >= 0) {
                        i_chosen = pick;
                    }
                } else {
                    lock_id = -1;
                    int pick = candidates[0];
                    i_chosen = pick;
                }
            }

            return i_chosen;
        }
        if (armor_num > 0) {
            int best_idx = -1;

            if (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_ARMOR
                && target.target_number != ArmorClass::OUTPOST) {
                const double coming_angle = params_.comming_angle * M_PI / 180.0;
                const double leaving_angle = params_.leaving_angle * M_PI / 180.0;

                for (int i = 0; i < armor_num; ++i) {
                    if (std::abs(delta_angles[i]) > coming_angle)
                        continue;

                    if (target_state.vyaw() > 0 && delta_angles[i] < leaving_angle)
                        best_idx = i;
                    if (target_state.vyaw() < 0 && delta_angles[i] > -leaving_angle)
                        best_idx = i;
                }
            }

            if (auto_aim_fsm == AutoAimFsm::AIM_WHOLE_CAR_PAIR
                && target.target_number != ArmorClass::OUTPOST) {
                std::vector<int> all;
                if (target_state.h() > 0) {
                    all.push_back(1);
                    all.push_back(3);
                } else {
                    all.push_back(0);
                    all.push_back(2);
                }
                best_idx = pick_best_by_min_delta(all);
            }
            if (best_idx < 0) {
                std::vector<int> all(armor_num);
                std::iota(all.begin(), all.end(), 0);
                best_idx = pick_best_by_min_delta(all);
            }

            i_chosen = best_idx;
        }

        return i_chosen;
    }
    [[nodiscard]] ControlPoint
    get_control_point(const Vec4& armor_xyza, double bullet_speed, int aim_id) const noexcept {
        ControlPoint cp;
        double control_yaw = std::atan2(armor_xyza.y(), armor_xyza.x());
        auto p = armor_xyza.head<3>();
        auto pitch_opt = ballistic_trajectory_->solve_pitch(p, bullet_speed);
        if (!pitch_opt) {
            cp.valid = false;

            AWAKENING_ERROR(
                "very_aimer: get_control_point: Failed to solve pitch armor_pos: [{}, {}, {}], bullet_speed: {}",
                p.x(),
                p.y(),
                p.z(),
                bullet_speed
            );
            return cp;
        }
        cp.valid = true;
        cp.yaw = angles::normalize_angle(control_yaw + base_yaw_offset_rad_);
        cp.pitch = pitch_opt.value() + base_pitch_offset_rad_;
        cp.aim_point.pose = ISO3::Identity();
        cp.aim_point.pose.translation() = p;
        cp.aim_point.d_angle = angles::shortest_angular_distance(control_yaw, armor_xyza[3]);
        cp.aim_id = aim_id;
        return cp;
    };
    [[nodiscard]] ControlPoint select_and_get_control_point(
        const ArmorTarget& target,
        double bullet_speed,
        const AutoAimFsm& fsm
    ) const noexcept {
        const int selected_armor = select_armor(target, fsm);
        auto armors_xyza = target.get_target_state().get_armors_xyza(target.target_number);
        if (fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER) {
            double center_xy_dis = std::hypot(
                target.get_target_state().pos().x(),
                target.get_target_state().pos().y()
            );
            double center_yaw = std::atan2(
                target.get_target_state().pos().y(),
                target.get_target_state().pos().x()
            );
            center_xy_dis -=
                target.get_target_state().get_armor_r(selected_armor, target.target_number);
            armors_xyza[selected_armor].x() = center_xy_dis * std::cos(center_yaw);
            armors_xyza[selected_armor].y() = center_xy_dis * std::sin(center_yaw);
            armors_xyza[selected_armor].z() = target.get_target_state().pos().z();
        }
        return get_control_point(armors_xyza[selected_armor], bullet_speed, selected_armor);
    }
    struct HitCtx {
        ArmorTarget hit_time_target;
        double fly_time;
    };
    std::optional<HitCtx>
    get_hit(const ArmorTarget& target_ready_to_aim, double bullet_speed, const AutoAimFsm& fsm)
        const noexcept {
        auto hit_time_target = target_ready_to_aim;
        const int roughly_select = select_armor(hit_time_target, fsm);
        const auto __armors_xyza =
            hit_time_target.get_target_state().get_armors_xyza(hit_time_target.target_number);
        auto prev_pitch_and_fly_time_opt = ballistic_trajectory_->solve_pitch_and_flytime(
            __armors_xyza[roughly_select].head<3>(),
            bullet_speed
        );
        if (!prev_pitch_and_fly_time_opt) {
            return std::nullopt;
        }
        auto prev_fly_time = prev_pitch_and_fly_time_opt.value().second;

        for (int iter = 0; iter < 10; ++iter) {
            auto i_target = hit_time_target;
            i_target.set_target_state([&](armor_point_motion_model::State& state) {
                state.predict(prev_fly_time);
            });
            auto iter_select = select_armor(i_target, fsm);
            const auto iter_armors_xyza =
                i_target.get_target_state().get_armors_xyza(i_target.target_number);
            auto iter_pitch_and_fly_time_opt = ballistic_trajectory_->solve_pitch_and_flytime(
                iter_armors_xyza[iter_select].head<3>(),
                bullet_speed
            );
            if (!iter_pitch_and_fly_time_opt) {
                return std::nullopt;
            }
            if (std::abs(iter_pitch_and_fly_time_opt.value().second - prev_fly_time) < 1e-3) {
                prev_fly_time = iter_pitch_and_fly_time_opt.value().second;
                break;
            }

            prev_fly_time = iter_pitch_and_fly_time_opt.value().second;
        }
        const double predict_time = prev_fly_time + params_.prediction_delay
            + (fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER ? params_.aim_center_more_prediction_time : 0
            );
        hit_time_target.set_target_state([&](auto& state) { state.predict(predict_time); });
        return HitCtx {
            .hit_time_target = hit_time_target,
            .fly_time = prev_fly_time,
        };
    }
    ArmorTarget last_target_;
    TimePoint last_time_;
    LimitTrajectory limit_traj_;
    ControlPoint limit_traj_cp0_;
    Trajectory<AimPoint, double> aim_traj_;
    Trajectory<GimbalState, double> aim_center_target_traj_;
    ControlPoint aim_center_target_traj_cp0_;
    double last_fly_time_;
    int last_select_;
    [[nodiscard]] GimbalCmd
    very_aim(const ArmorTarget& _target, double bullet_speed, const AutoAimFsm& fsm) noexcept {
        GimbalCmd cmd;
        bool is_same = _target.this_id == last_target_.this_id;
        double time_in_traj = 0.0;
        if (is_same) {
            time_in_traj = std::chrono::duration<double>(Clock::now() - last_time_).count();
        } else {
            last_target_ = _target;
            last_time_ = Clock::now();
        }
        auto target = _target;
        target.set_target_state([&](armor_point_motion_model::State& state) {
            state.predict(Clock::now());
        });

        if (!is_same) {
            auto hit_ctx_opt = get_hit(target, bullet_speed, fsm);
            if (!hit_ctx_opt) {
                cmd.appear = false;
                return cmd;
            }

            auto hit_ctx = hit_ctx_opt.value();
            auto cp0 = select_and_get_control_point(hit_ctx.hit_time_target, bullet_speed, fsm);
            if (!cp0.valid) {
                cmd.appear = false;
                return cmd;
            }
            last_fly_time_ = hit_ctx.fly_time;
            last_select_ = cp0.aim_id;
            auto sample_once = [&](double t,
                                   const ArmorTarget& base_target,
                                   AutoAimFsm fsm_mode,
                                   const ControlPoint& _cp0,
                                   GimbalState& out_gs,
                                   AimPoint& out_ap) -> bool {
                auto tmp_target = base_target;
                tmp_target.set_target_state([&](armor_point_motion_model::State& state) {
                    state.predict(t);
                });

                auto hit_opt = get_hit(tmp_target, bullet_speed, fsm_mode);
                if (!hit_opt)
                    return false;

                auto cp = select_and_get_control_point(hit_opt->hit_time_target, bullet_speed, fsm);
                if (!cp.valid)
                    return false;

                out_gs.aim_id = cp.aim_id;
                out_gs.yaw_state.p = angles::normalize_angle(cp.yaw - _cp0.yaw);
                out_gs.pitch_state.p = angles::normalize_angle(cp.pitch - _cp0.pitch);
                out_ap = cp.aim_point;

                return true;
            };
            auto build_traj = [&](auto& traj_gs,
                                  auto& traj_ap,
                                  const ControlPoint& _cp0,
                                  AutoAimFsm fsm_mode,
                                  int horizon,
                                  double dt) -> bool {
                int half = horizon / 2;
                const int delay_steps = params_.control_delay * 2.0 / dt;
                for (int i = (fsm != AutoAimFsm::AIM_WHOLE_CAR_CENTER ? half : delay_steps); i >= 1;
                     --i) {
                    double t = -i * dt;

                    GimbalState gs;
                    AimPoint ap;

                    if (!sample_once(t, target, fsm_mode, _cp0, gs, ap))
                        return false;

                    traj_gs.push_back(gs, t);
                    traj_ap.push_back(ap, t);
                }

                {
                    GimbalState gs0;
                    gs0.aim_id = _cp0.aim_id;
                    gs0.yaw_state.p = 0.0;
                    gs0.pitch_state.p = 0.0;

                    traj_gs.push_back(gs0, 0.0);
                    traj_ap.push_back(_cp0.aim_point, 0.0);
                }

                for (int i = 1; i <= half; ++i) {
                    double t = i * dt;

                    GimbalState gs;
                    AimPoint ap;

                    if (!sample_once(t, target, fsm_mode, _cp0, gs, ap))
                        return false;

                    traj_gs.push_back(gs, t);
                    traj_ap.push_back(ap, t);
                }

                return true;
            };
            limit_traj_cp0_ = cp0;
            limit_traj_.clear();
            aim_traj_.clear();

            auto make_even = [](int x) { return x % 2 == 0 ? x : x + 1; };

            const int horizon = make_even(params_.sample_horizon);
            const double dt = params_.sample_total_time / horizon;

            limit_traj_.reserve(horizon + 1);
            aim_traj_.reserve(horizon + 1);

            if (!build_traj(limit_traj_, aim_traj_, limit_traj_cp0_, fsm, horizon, dt)) {
                cmd.appear = false;
                return cmd;
            }

            limit_traj_.build_limit(params_.max_yaw_acc, params_.max_pitch_acc, time_in_traj);

            if (fsm == AutoAimFsm::AIM_WHOLE_CAR_CENTER) {
                aim_traj_.clear();
                aim_center_target_traj_.clear();

                aim_center_target_traj_.reserve(horizon + 1);
                aim_traj_.reserve(horizon + 1);

                aim_center_target_traj_cp0_ = select_and_get_control_point(
                    target,
                    bullet_speed,
                    AutoAimFsm::AIM_WHOLE_CAR_ARMOR
                );

                const int delay_steps = params_.control_delay * 2.0 / dt;

                if (!build_traj(
                        aim_center_target_traj_,
                        aim_traj_,
                        aim_center_target_traj_cp0_,
                        AutoAimFsm::AIM_WHOLE_CAR_ARMOR,
                        horizon,
                        dt
                    ))
                {
                    cmd.appear = false;
                    return cmd;
                }
            }
        }

        ControlPoint target_traj_cp0 =
            fsm != AutoAimFsm::AIM_WHOLE_CAR_CENTER ? limit_traj_cp0_ : aim_center_target_traj_cp0_;
        Trajectory<GimbalState, double> target_traj =
            fsm != AutoAimFsm::AIM_WHOLE_CAR_CENTER ? limit_traj_ : aim_center_target_traj_;
        auto target_gimbal_state = target_traj.Trajectory::state_at(time_in_traj);
        auto control = limit_traj_.LimitTrajectory::state_at(time_in_traj);
        double control_yaw = angles::normalize_angle(control.yaw_state.p + limit_traj_cp0_.yaw);
        double control_pitch =
            angles::normalize_angle(control.pitch_state.p + limit_traj_cp0_.pitch);
        double target_yaw =
            angles::normalize_angle(target_gimbal_state.yaw_state.p + target_traj_cp0.yaw);
        double target_pitch =
            angles::normalize_angle(target_gimbal_state.pitch_state.p + target_traj_cp0.pitch);
        cmd.timestamp = Clock::now();
        cmd.yaw = angles::to_degrees(control_yaw);
        cmd.v_yaw = angles::to_degrees(control.yaw_state.v);
        cmd.a_yaw = angles::to_degrees(control.yaw_state.a);
        cmd.pitch = angles::to_degrees(control_pitch);
        cmd.v_pitch = angles::to_degrees(control.pitch_state.v);
        cmd.a_pitch = angles::to_degrees(control.pitch_state.a);
        cmd.target_yaw = angles::to_degrees(target_yaw);
        cmd.target_pitch = angles::to_degrees(target_pitch);
        cmd.fly_time = last_fly_time_;
        cmd.appear = true;
        cmd.aim_point = aim_traj_.state_at(time_in_traj);
        cmd.aim_point.frame_id = target.get_target_state().frame_id;
        cmd.select_id = last_select_;
        bool is_big = target.target_number == ArmorClass::NO1;
        auto cal_enbale_diff = [&](double _t) {
            auto aim_point = aim_traj_.state_at(_t);
            const double distance = aim_point.pose.translation().norm();
            double shooting_range_yaw;
            if (!is_big) {
                shooting_range_yaw =
                    std::abs(std::atan2(params_.shooting_range_w_small / 2, distance));
            } else {
                shooting_range_yaw =
                    std::abs(std::atan2(params_.shooting_range_w_large / 2, distance));
            }
            double shooting_range_pitch =
                std::abs(std::atan2(params_.shooting_range_h / 2, distance));
            const double yaw_factor = std::cos(aim_point.d_angle);
            const double pitch_factor = std::cos(FIFTTEN_DEGREE_RAD);
            // shooting_range_yaw =
            //     std::max(shooting_range_yaw, angles::from_degrees(params_.min_enable_yaw_deg));
            // shooting_range_pitch =
            //     std::max(shooting_range_pitch, angles::from_degrees(params_.min_enable_pitch_deg));
            shooting_range_yaw *= yaw_factor;
            shooting_range_pitch *= pitch_factor;
            shooting_range_yaw =
                std::max(shooting_range_yaw, angles::from_degrees(params_.min_enable_yaw_deg));
            shooting_range_pitch =
                std::max(shooting_range_pitch, angles::from_degrees(params_.min_enable_pitch_deg));
            return std::make_pair(std::abs(shooting_range_yaw), std::abs(shooting_range_pitch));
        };
        auto enable_diff = cal_enbale_diff(time_in_traj);
        cmd.enable_yaw_diff = angles::to_degrees(enable_diff.first);
        cmd.enable_pitch_diff = angles::to_degrees(enable_diff.second);
        cmd.fire_advice =
            std::abs(angles::shortest_angular_distance(
                angles::from_degrees(cmd.target_yaw),
                angles::from_degrees(cmd.yaw)
            )) < cmd.enable_yaw_diff
            && std::abs(angles::shortest_angular_distance(
                   angles::from_degrees(cmd.target_pitch),
                   angles::from_degrees(cmd.pitch)
               ))
                < cmd.enable_pitch_diff;
        if (fsm != AutoAimFsm::AIM_WHOLE_CAR_CENTER) {
            auto delay_fire = [&](double delay) {
                auto delay_control = limit_traj_.LimitTrajectory::state_at(time_in_traj + delay);
                auto delay_target = target_traj.Trajectory::state_at(time_in_traj + delay);
                auto delay_enable = cal_enbale_diff(time_in_traj + delay);
                return std::abs(angles::shortest_angular_distance(
                           angles::normalize_angle(delay_control.yaw_state.p + limit_traj_cp0_.yaw),
                           angles::normalize_angle(delay_target.yaw_state.p + target_traj_cp0.yaw)
                       ))
                    < delay_enable.first
                    && std::abs(angles::shortest_angular_distance(
                           angles::normalize_angle(
                               delay_control.pitch_state.p + limit_traj_cp0_.pitch
                           ),
                           angles::normalize_angle(
                               delay_target.pitch_state.p + target_traj_cp0.pitch
                           )
                       ))
                    < delay_enable.second;
            };

            if (!delay_fire(+params_.control_delay)) {
                cmd.no_shoot();
            } else if (!cmd.fire_advice && delay_fire(-params_.control_delay)) {
                cmd.fire_advice = true;
                cmd.enable_yaw_diff = angles::from_degrees(params_.min_enable_yaw_deg);
                cmd.enable_pitch_diff = angles::from_degrees(params_.min_enable_pitch_deg);
            }
        }

        return cmd;
    }
    BallisticTrajectory::Ptr ballistic_trajectory_;
    double base_yaw_offset_rad_;
    double base_pitch_offset_rad_;
};
VeryAimer::VeryAimer(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}
VeryAimer::~VeryAimer() noexcept {
    _impl.reset();
}
GimbalCmd
VeryAimer::very_aim(const ArmorTarget& target, double bullet_speed, const AutoAimFsm& fsm) {
    return _impl->very_aim(target, bullet_speed, fsm);
}
std::pair<double, double> VeryAimer::get_yaw_pitch_offset() {
    return std::make_pair(_impl->base_yaw_offset_rad_, _impl->base_pitch_offset_rad_);
}
} // namespace awakening::auto_aim