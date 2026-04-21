#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <vector>

namespace awakening {
template<typename T>
concept HasStaticLerp = requires(const T& a, const T& b, double t) {
    {
        T::lerp(a, b, t)
        } -> std::same_as<T>;
};

template<typename T>
struct TimeTraits;

template<>
struct TimeTraits<double> {
    using time_type = double;
    using duration_type = double;

    static duration_type diff(time_type a, time_type b) {
        return a - b;
    }

    static time_type add(time_type t, duration_type d) {
        return t + d;
    }

    static double to_double(duration_type d) {
        return d;
    }

    static time_type zero() {
        return 0.0;
    }
};
template<typename Clock, typename Duration>
struct TimeTraits<std::chrono::time_point<Clock, Duration>> {
    using time_type = std::chrono::time_point<Clock, Duration>;
    using duration_type = typename time_type::duration;

    static duration_type diff(time_type a, time_type b) {
        return a - b;
    }

    static time_type add(time_type t, duration_type d) {
        return t + d;
    }

    static double to_double(duration_type d) {
        return std::chrono::duration<double>(d).count();
    }

    static time_type zero() {
        return time_type {};
    }
};

template<HasStaticLerp PointT, typename TimeT = double>
class Trajectory {
public:
    using size_type = std::size_t;
    using Traits = TimeTraits<TimeT>;
    using duration_type = typename Traits::duration_type;

    inline void reserve(size_type n) {
        cp_.reserve(n);
        prefix_.reserve(n);
    }

    inline void clear() noexcept {
        cp_.clear();
        prefix_.clear();
    }
    inline void push_back(const PointT& p, TimeT t) {
        cp_.emplace_back(p);
        prefix_.emplace_back(t);
    }

    [[nodiscard]] inline TimeT time_at(size_type i) const noexcept {
        return prefix_[i];
    }

    [[nodiscard]] inline const PointT& state_at(size_type i) const noexcept {
        return cp_[i];
    }

    [[nodiscard]] inline PointT state_at(TimeT t) const {
        if (cp_.empty()) [[unlikely]]
            return {};

        if (t <= prefix_.front()) [[likely]]
            return cp_.front();

        if (t >= prefix_.back()) [[likely]]
            return cp_.back();

        const auto it = std::lower_bound(prefix_.begin(), prefix_.end(), t);
        const size_type i1 = static_cast<size_type>(it - prefix_.begin());
        const size_type i0 = i1 - 1;

        const auto dt = Traits::diff(prefix_[i1], prefix_[i0]);
        const double dt_d = Traits::to_double(dt);

        if (dt_d <= 1e-9) [[unlikely]]
            return cp_[i0];

        const double t_rel = Traits::to_double(Traits::diff(t, prefix_[i0]));

        const double a = std::clamp(t_rel / dt_d, 0.0, 1.0);

        return PointT::lerp(cp_[i0], cp_[i1], a);
    }

    [[nodiscard]] inline duration_type duration() const noexcept {
        return prefix_.back() - prefix_[0];
    }

    [[nodiscard]] inline size_type size() const noexcept {
        return cp_.size();
    }

    [[nodiscard]] inline bool empty() const noexcept {
        return cp_.empty();
    }
    inline std::vector<PointT>& get_cp_vec() noexcept {
        return cp_;
    }
    inline const std::vector<TimeT>& get_prefix() const noexcept {
        return prefix_;
    }

private:
    std::vector<PointT> cp_;
    std::vector<TimeT> prefix_;
};

} // namespace awakening