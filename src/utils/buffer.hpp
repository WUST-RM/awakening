#pragma once
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>
namespace awakening::utils {

template<typename T>
struct OneBuffer {
public:
    OneBuffer() {
        write_index.store(0, std::memory_order_relaxed);
        read_index.store(1, std::memory_order_relaxed);
        back_index.store(2, std::memory_order_relaxed);
    }
    template<typename Fn>
    void write(Fn&& fn) {
        T& buf = buffers[write_index.load(std::memory_order_relaxed)];
        fn(buf);
        rotate_write();
    }
    void write(const T& value) {
        buffers[write_index.load(std::memory_order_relaxed)] = value;
        rotate_write();
    }

    void write(T&& value) {
        buffers[write_index.load(std::memory_order_relaxed)] = std::move(value);
        rotate_write();
    }
    T read() const {
        size_t idx = read_index.load(std::memory_order_acquire);
        T val = buffers[idx];
        read_index.store(write_index.load(std::memory_order_acquire), std::memory_order_release);
        return val;
    }

private:
    std::array<T, 3> buffers;
    std::atomic<size_t> write_index;
    mutable std::atomic<size_t> read_index;
    std::atomic<size_t> back_index;

    void rotate_write() {
        size_t prevWrite = write_index.load(std::memory_order_relaxed);
        write_index.store(back_index.load(std::memory_order_relaxed), std::memory_order_release);
        back_index.store(prevWrite, std::memory_order_relaxed);
    }
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

        // Move constructor & assignment: 拷贝内部值
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
concept HasFrameIDAndTimestamp = requires(T a) {
    {
        a.id
        } -> std::convertible_to<int>;
};

template<HasFrameIDAndTimestamp T>
class OrderedQueue {
public:
    OrderedQueue(): current_id_(1) {}
    void enqueue(T item) {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (item.id < current_id_)
                return;

            buffer_[item.id] = std::move(item);
        }
    }
    std::vector<T> dequeue_batch() {
        std::vector<T> out;
        dequeue_batch(out);
        return out;
    }
    bool dequeue_batch(std::vector<T>& out) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (buffer_.empty())
            return false;
        auto it = buffer_.begin();
        if (it->first > current_id_ + 1)
            return false;
        int expected = current_id_ + 1;
        out.clear();
        while (it != buffer_.end()) {
            if (it->first != expected)
                break;
            out.emplace_back(std::move(it->second));
            expected = it->first + 1;
            current_id_ = it->first;

            it = buffer_.erase(it);
        }

        return !out.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return buffer_.size();
    }

private:
    std::map<int, T> buffer_;
    int current_id_;
    mutable std::mutex mutex_;
};

} // namespace awakening::utils