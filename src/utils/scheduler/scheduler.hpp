#pragma once
#include "node.hpp"
#include "utils/logger.hpp"

#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <tbb/task_arena.h>
#include <tbb/task_group.h>

namespace awakening {

class Scheduler {
public:
    using clock = std::chrono::steady_clock;
    inline static unsigned int __hardware_concurrency = (std::thread::hardware_concurrency());
    explicit Scheduler(size_t threads = __hardware_concurrency):
        worker_count(threads ? threads : 1),
        arena(worker_count) {}

    ~Scheduler() {
        stop();
    }

    template<typename InputPair, typename... OutputPairs, typename Fn>
    void register_task(std::string n, Fn&& fn) {
        auto node = TaskNode<InputPair, OutputPairs...>::create(std::forward<Fn>(fn));
        node->name = std::move(n);

        auto inputs = node->input_tags();
        if (inputs.empty()) {
            throw std::runtime_error("Task must have input");
        }

        static_tasks_snapshot[inputs.front()].push_back(node);
        built = false;
    }

    template<typename... OutputPairs>
    [[nodiscard]] size_t register_source(std::string n) {
        auto node = SourceNode<OutputPairs...>::create();
        node->name = std::move(n);

        source_snapshot.push_back(node);
        built = false;
        return source_snapshot.size() - 1;
    }

    template<typename... OutputPairs, typename Fn>
    void runtime_push_source(size_t snap_id, Fn&& fn) {
        if (!built)
            build();

        if (!is_running())
            return;

        if (snap_id >= source_snapshot.size()) {
            throw std::out_of_range("Invalid source snapshot id");
        }

        using NodeT = SourceNode<OutputPairs...>;
        using FuncT = typename NodeT::Func;

        static_assert(
            std::is_convertible_v<Fn, FuncT>,
            "Fn must be convertible to SourceNode::Func"
        );

        auto& base = source_snapshot[snap_id];

        auto local = base->clone();
        auto source = std::static_pointer_cast<NodeT>(local);

        source->fn = FuncT(std::forward<Fn>(fn));

        schedule(local);
    }

    template<int CoreId, typename... OutputPairs, typename Fn>
    void add_rate_source(std::string n, double rate, Fn&& fn) {
        static_assert(CoreId >= 0, "CoreId must be >= 0");

        if (CoreId >= static_cast<int>(__hardware_concurrency)) {
            throw std::runtime_error("CoreId exceeds hardware concurrency");
        }

        if (rate <= 0.0) {
            throw std::invalid_argument("rate must be > 0");
        }

        using NodeT = SourceNode<OutputPairs...>;
        using FuncT = typename NodeT::Func;

        static_assert(std::is_convertible_v<Fn, FuncT>, "Fn must match SourceNode::Func");

        auto node = NodeT::create(FuncT(std::forward<Fn>(fn)));
        node->name = std::move(n);

        connect(node);

        rate_workers.emplace_back(RateWorker { node, rate, CoreId });
    }

    void run() {
        if (running.exchange(true, std::memory_order_acq_rel))
            return;

        for (auto& w: rate_workers) {
            rate_threads.emplace_back([this, w](std::stop_token st) {
                bind_cpu(w.core_id);

                const auto period = std::chrono::duration_cast<clock::duration>(
                    std::chrono::duration<double>(1.0 / w.rate)
                );

                auto next_time = clock::now();

                while (!st.stop_requested() && running.load(std::memory_order_acquire)) {
                    next_time += period;

                    execute_node(w.node);

                    auto now = clock::now();

                    if (now > next_time) {
                        next_time = now;
                        continue;
                    }

                    std::this_thread::sleep_until(next_time);
                }
            });
        }
    }

    void stop() {
        if (!running.exchange(false, std::memory_order_acq_rel))
            return;

        for (auto& t: rate_threads) {
            if (t.joinable())
                t.request_stop();
        }

        rate_threads.clear();
        arena.execute([this] { tg.wait(); });
        AWAKENING_INFO("awakening Scheduler stopped");
    }
    void build() {
        if (built)
            return;

        for (auto& [_, nodes]: static_tasks_snapshot) {
            for (auto& node: nodes) {
                connect(node);
            }
        }

        for (auto& node: source_snapshot) {
            connect(node);
        }

        for (auto& [_, nodes]: static_tasks_snapshot) {
            for (auto& node: nodes) {
                if (node->connected_count == 0) {
                    throw std::runtime_error("Node '" + node->name + "' is isolated");
                }
            }
        }

        for (auto& node: source_snapshot) {
            if (node->connected_count == 0) {
                throw std::runtime_error("Source '" + node->name + "' has no downstream");
            }
        }

        built = true;
    }

    bool is_running() const {
        return running.load(std::memory_order_acquire);
    }

private:
    struct RateWorker {
        NodeBase::Ptr node;
        double rate;
        int core_id;
    };

    void bind_cpu(int core_id) {
#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
    }

    void schedule(NodeBase::Ptr node) {
        if (!node || !running.load(std::memory_order_acquire))
            return;

        arena.execute([this, node] { tg.run([this, node] { execute_node(node); }); });
    }

    void execute_node(NodeBase::Ptr node) {
        const auto& next = node->execute();

        for (auto& n: next) {
            schedule(n);
        }
    }

    void connect(NodeBase::Ptr node) {
        for (auto& tag: node->output_tags()) {
            auto it = static_tasks_snapshot.find(tag);
            if (it == static_tasks_snapshot.end())
                continue;

            for (auto& d: it->second) {
                node->connected_count++;
                d->connected_count++;

                node->add_downstream(tag, d->clone());
            }
        }
    }

private:
    size_t worker_count;

    std::unordered_map<std::type_index, std::vector<NodeBase::Ptr>> static_tasks_snapshot;

    std::vector<NodeBase::Ptr> source_snapshot;

    std::vector<RateWorker> rate_workers;
    std::vector<std::jthread> rate_threads;

    tbb::task_arena arena;
    tbb::task_group tg;

    std::atomic<bool> running { false };
    bool built { false };
};

} // namespace awakening