#pragma once
#include "ankerl/unordered_dense.h"
#include "utils/tf.hpp"
#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace awakening::utils::tf {

struct LinkBuffer {
    TimePoseBuffer buffer;
    LinkBuffer(size_t size = 1024): buffer(size) {}
};

// ---------------- Edge ----------------
template<typename FrameEnum>
struct Edge {
    FrameEnum parent;
    FrameEnum child;
};

// ---------------- RobotTF ----------------
template<typename FrameEnum, size_t N, bool Static>
class RobotTF {
public:
    using Ptr = std::shared_ptr<RobotTF>;
    static Ptr create() {
        return std::make_shared<RobotTF>();
    }

    void add_edge(FrameEnum parent, FrameEnum child) {
        edges_.push_back({ parent, child });
        adjacency_[parent].push_back(child);
        adjacency_[child].push_back(parent);
        directed_edge_[size_t(parent)][size_t(child)] = true;
    }

    void push(FrameEnum parent, FrameEnum child, const TimePoint& t, const ISO3& pose) {
        buffers_[size_t(parent)][size_t(child)].buffer.push(t, pose);
    }

    ISO3 get(FrameEnum from, FrameEnum to, const TimePoint& t) const {
        auto path = find_path(from, to);
        if (!path) {
            if constexpr (Static) {
                throw std::runtime_error(
                    "No path found from " + std::to_string(size_t(from)) + " to "
                    + std::to_string(size_t(to))
                );
            } else {
                return ISO3::Identity();
            }
        }

        ISO3 T = ISO3::Identity();
        for (size_t i = 0; i + 1 < path->size(); ++i) {
            auto A = (*path)[i];
            auto B = (*path)[i + 1];

            ISO3 t_ab = ISO3::Identity();
            if (has_direct_edge(A, B)) {
                t_ab = buffers_[size_t(A)][size_t(B)].buffer.get(t);
            } else if (has_direct_edge(B, A)) {
                t_ab = buffers_[size_t(B)][size_t(A)].buffer.get(t).inverse();
            } else {
                if constexpr (Static) {
                    throw std::runtime_error(
                        "Broken path edge between " + std::to_string(size_t(A)) + " and "
                        + std::to_string(size_t(B))
                    );
                } else {
                    t_ab = ISO3::Identity();
                }
            }

            T = T * t_ab;
        }
        return T;
    }

    std::vector<Edge<FrameEnum>> get_edges() const {
        return edges_;
    }

    bool has_path(FrameEnum from, FrameEnum to) const {
        return find_path(from, to).has_value();
    }

private:
    std::array<std::array<LinkBuffer, N>, N> buffers_;
    std::vector<Edge<FrameEnum>> edges_;
    ankerl::unordered_dense::map<FrameEnum, std::vector<FrameEnum>> adjacency_;
    std::array<std::array<bool, N>, N> directed_edge_ {};

    bool has_direct_edge(FrameEnum A, FrameEnum B) const {
        return directed_edge_[size_t(A)][size_t(B)];
    }

    std::optional<std::vector<FrameEnum>> find_path(FrameEnum start, FrameEnum goal) const {
        std::queue<std::vector<FrameEnum>> q;
        std::unordered_map<FrameEnum, bool> visited;

        q.push({ start });
        visited[start] = true;

        while (!q.empty()) {
            auto path = q.front();
            q.pop();
            FrameEnum node = path.back();
            if (node == goal)
                return path;

            auto it = adjacency_.find(node);
            if (it == adjacency_.end())
                continue;

            for (auto neighbor: it->second) {
                if (visited[neighbor])
                    continue;
                visited[neighbor] = true;
                auto new_path = path;
                new_path.push_back(neighbor);
                q.push(new_path);
            }
        }

        return std::nullopt;
    }
};

} // namespace awakening::utils::tf