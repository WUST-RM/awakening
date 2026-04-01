#pragma once
#include "utils/common/type_common.hpp"
#include <algorithm>
#include <deque>
#include <mutex>
#include <shared_mutex>

namespace awakening::utils {

struct TimedPose {
    TimePoint stamp;
    ISO3 pose;
};

class TimePoseBuffer {
public:
    explicit TimePoseBuffer(size_t max_size = 1024): max_size_(max_size) {}

    void push(const TimePoint& t, const ISO3& pose) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (!buffer_.empty() && t < buffer_.back().stamp)
            return;
        buffer_.push_back({ t, pose });
        while (buffer_.size() > max_size_)
            buffer_.pop_front();
    }

    ISO3 get(const TimePoint& t) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);

        if (buffer_.empty())
            return ISO3::Identity();
        if (buffer_.size() == 1)
            return buffer_.front().pose;

        auto it = std::lower_bound(
            buffer_.begin(),
            buffer_.end(),
            t,
            [](const TimedPose& p, const TimePoint& t) { return p.stamp < t; }
        );

        if (it == buffer_.begin())
            return it->pose;
        if (it == buffer_.end())
            return extrapolate(t);

        const auto& p1 = *(it - 1);
        const auto& p2 = *it;

        double r = std::clamp(
            std::chrono::duration<double>(t - p1.stamp).count()
                / std::chrono::duration<double>(p2.stamp - p1.stamp).count(),
            0.0,
            1.0
        );

        Vec3 trans = (1 - r) * p1.pose.translation() + r * p2.pose.translation();
        Quaternion q(p1.pose.rotation());
        q = q.slerp(r, Quaternion(p2.pose.rotation()));

        ISO3 T = ISO3::Identity();
        T.linear() = q.toRotationMatrix();
        T.translation() = trans;
        return T;
    }

private:
    ISO3 extrapolate(const TimePoint& t) const {
        if (buffer_.size() < 2)
            return buffer_.back().pose;

        const auto& p1 = buffer_[buffer_.size() - 2];
        const auto& p2 = buffer_.back();

        double dt = std::chrono::duration<double>(p2.stamp - p1.stamp).count();
        if (dt < 1e-6)
            return p2.pose;

        double dt_future = std::chrono::duration<double>(t - p2.stamp).count();

        Vec3 v = (p2.pose.translation() - p1.pose.translation()) / dt;
        Vec3 trans = p2.pose.translation() + v * dt_future;

        Quaternion q1(p1.pose.rotation());
        Quaternion q2(p2.pose.rotation());
        Quaternion dq = q1.inverse() * q2;

        AngleAxis aa(dq);
        Vec3 omega = aa.axis() * aa.angle() / dt;

        double angle = omega.norm() * dt_future;
        Vec3 axis = omega.norm() > 1e-6 ? omega.normalized() : Vec3::UnitX();

        Quaternion q_future = q2 * Quaternion(AngleAxis(angle, axis));

        ISO3 T = ISO3::Identity();
        T.linear() = q_future.toRotationMatrix();
        T.translation() = trans;
        return T;
    }

private:
    mutable std::shared_mutex mutex_;
    std::deque<TimedPose> buffer_;
    size_t max_size_;
};

} // namespace awakening::utils