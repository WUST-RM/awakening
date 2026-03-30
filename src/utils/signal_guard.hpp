#pragma once
#include "utils/logger.hpp"
#include <atomic>
#include <csignal>
#include <functional>
#include <iostream>
#include <mutex>
#include <spdlog/spdlog.h>
#include <thread>
#include <vector>
namespace awakening::utils {
class SignalGuard {
public:
    static SignalGuard& instance() {
        static SignalGuard inst;
        return inst;
    }

    static bool running() noexcept {
        return instance().running_.load(std::memory_order_relaxed);
    }

    static void addCallback(std::function<void()> cb) {
        std::lock_guard<std::mutex> lock(instance().mtx_);
        instance().callbacks_.push_back(std::move(cb));
    }
    template<typename Duration, typename Fn = std::function<void()>>
    static void spin(
        Duration dur,
        Fn&& fn = []() {}
    ) {
        while (running()) {
            fn();
            std::this_thread::sleep_for(dur);
        }
    }

private:
    SignalGuard() {
        std::signal(SIGINT, handleSignal);
        std::signal(SIGTERM, handleSignal);
    }

    static void handleSignal(int signum) {
        auto& inst = instance();

        if (!inst.running_)
            return;

        inst.running_ = false;

        AWAKENING_INFO("Signal {} received, stopping...", signum);

        std::lock_guard<std::mutex> lock(inst.mtx_);
        for (auto& cb: inst.callbacks_) {
            if (cb)
                cb();
        }
    }

private:
    std::atomic<bool> running_ { true };
    std::vector<std::function<void()>> callbacks_;
    std::mutex mtx_;
};
} // namespace awakening::utils
