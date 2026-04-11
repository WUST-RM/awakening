#pragma once
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>
namespace awakening::utils {

template<typename T>
class SWMR {
public:
    SWMR() = default;

    template<typename Fn>
    void write_fn(Fn&& fn) {
        const size_t w = write_index_.load(std::memory_order_relaxed);

        fn(buffers_[w]);

        ready_index_.store(w, std::memory_order_release);
        const size_t next = back_index_.load(std::memory_order_relaxed);
        back_index_.store(w, std::memory_order_relaxed);
        write_index_.store(next, std::memory_order_relaxed);
    }

    void write(const T& data) {
        write_fn([&](T& buf) { buf = data; });
    }

    T read() const {
        const size_t r = ready_index_.load(std::memory_order_acquire);
        return buffers_[r];
    }

    const T& read_ref() const {
        const size_t r = ready_index_.load(std::memory_order_acquire);
        return buffers_[r];
    }

private:
    std::array<T, 3> buffers_;
    alignas(64) std::atomic<size_t> write_index_ { 0 };
    alignas(64) std::atomic<size_t> back_index_ { 1 };
    alignas(64) std::atomic<size_t> ready_index_ { 2 };
};

template<typename T>
class ResourcePool {
public:
    struct MovableAtomicBool {
        std::atomic<bool> v;

        explicit MovableAtomicBool(bool b = false) noexcept: v(b) {}
        bool load(std::memory_order m = std::memory_order_seq_cst) const noexcept {
            return v.load(m);
        }

        void store(bool b, std::memory_order m = std::memory_order_seq_cst) noexcept {
            v.store(b, m);
        }

        bool compare_exchange_strong(
            bool& expected,
            bool desired,
            std::memory_order success = std::memory_order_seq_cst,
            std::memory_order failure = std::memory_order_seq_cst
        ) noexcept {
            return v.compare_exchange_strong(expected, desired, success, failure);
        }

        bool exchange(bool b, std::memory_order m = std::memory_order_seq_cst) noexcept {
            return v.exchange(b, m);
        }
        MovableAtomicBool(MovableAtomicBool&& o) noexcept: v(o.v.load(std::memory_order_relaxed)) {}
        MovableAtomicBool& operator=(MovableAtomicBool&& o) noexcept {
            v.store(o.v.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }

        MovableAtomicBool(const MovableAtomicBool&) = delete;
        MovableAtomicBool& operator=(const MovableAtomicBool&) = delete;
    };
    struct Resource {
        T value;
        MovableAtomicBool busy;
    };

    class Handle {
    public:
        Handle(Resource* r = nullptr): res_(r) {}

        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;

        Handle(Handle&& other) noexcept: res_(other.res_) {
            other.res_ = nullptr;
        }

        Handle& operator=(Handle&& other) noexcept {
            if (this != &other) {
                release();
                res_ = other.res_;
                other.res_ = nullptr;
            }
            return *this;
        }

        ~Handle() {
            release();
        }

        T* operator->() {
            return &res_->value;
        }
        T& operator*() {
            return res_->value;
        }

        explicit operator bool() const {
            return res_ != nullptr;
        }

    private:
        void release() {
            if (res_) {
                res_->busy.store(false, std::memory_order_release);
                res_ = nullptr;
            }
        }

        Resource* res_;
    };

public:
    ResourcePool() = default;

    void addResource(T&& resource) {
        resources_.emplace_back(Resource { std::move(resource), MovableAtomicBool(false) });
    }

    void addResource(const T& resource) {
        resources_.emplace_back(Resource { resource, MovableAtomicBool(false) });
    }

    Handle acquire() {
        for (auto& r: resources_) {
            bool expected = false;
            if (r.busy.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
                return Handle(&r);
            }
        }
        return Handle(nullptr);
    }

private:
    std::vector<Resource> resources_;
};
template<typename T>
concept HasFrameID = requires(T a) {
    {
        a.id
        } -> std::convertible_to<int>;
};

template<HasFrameID T>
class OrderedQueue {
public:
    OrderedQueue(): current_id_(1) {}

    void enqueue(T item) {
        std::lock_guard<std::mutex> lk(mutex_);

        if (item.id < current_id_) {
            return;
        }

        if (item.id == current_id_) {
            main_queue_.emplace_back(std::move(item));
            current_id_++;

            auto it = buffer_.find(current_id_);
            while (it != buffer_.end()) {
                main_queue_.emplace_back(std::move(it->second));
                buffer_.erase(it);
                current_id_++;
                it = buffer_.find(current_id_);
            }
        } else {
            buffer_.emplace(item.id, std::move(item));
        }
    }

    std::vector<T> dequeue_batch() {
        std::vector<T> out;
        dequeue_batch(out);
        return out;
    }

    bool dequeue_batch(std::vector<T>& out) {
        std::lock_guard<std::mutex> lk(mutex_);

        if (main_queue_.empty())
            return false;

        out.clear();
        out.reserve(main_queue_.size());

        while (!main_queue_.empty()) {
            out.emplace_back(std::move(main_queue_.front()));
            main_queue_.pop_front();
        }

        return true;
    }

    bool try_dequeue(T& item) {
        std::lock_guard<std::mutex> lk(mutex_);

        if (main_queue_.empty())
            return false;

        item = std::move(main_queue_.front());
        main_queue_.pop_front();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return main_queue_.size() + buffer_.size();
    }

private:
    std::deque<T> main_queue_;
    std::unordered_map<int, T> buffer_;
    int current_id_;
    mutable std::mutex mutex_;
};
template<typename T>
class Locked {
public:
    Locked() = default;
    Locked(const T& v): value_(v) {}

    void set(const T& v) {
        std::lock_guard<std::mutex> lock(mtx_);
        value_ = v;
    }

    T get() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return value_;
    }

private:
    mutable std::mutex mtx_;
    T value_ {};
};
} // namespace awakening::utils