#include "lidar_location.hpp"
#include "Eigen/Dense"
#include "target.hpp"
#include "tasks/radar_detect/type.hpp"
#include "utils/common/type_common.hpp"
#include "utils/io/pcd_io.h"
#include "utils/logger.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <cmath>
#include <deque>
#include <memory>
#include <oneapi/tbb/parallel_for.h>
#include <optional>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/ann/kdtree_tbb.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling_tbb.hpp>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <yaml-cpp/node/node.h>
namespace awakening::radar_detect {
struct CarTarget {
    struct CarTargetConfig {
        PointTarget::PointTargetConfig grav_cfg;
        PointTarget::PointTargetConfig aabb_cfg;
        void load(const YAML::Node& config) {
            grav_cfg.load(config["grav_target"]);
            aabb_cfg.load(config["aabb_target"]);
        }
    };
    CarTarget() {}
    CarTarget(const CarTargetConfig& cfg, const ClusterResult& c, const TimePoint& time) {
        target_config = cfg;
        grav_point_target = PointTarget(cfg.grav_cfg, c.grav, time, true);
        min_point_target = PointTarget(cfg.aabb_cfg, c.aabb.min_pt - c.grav, time, false);
        max_point_target = PointTarget(cfg.aabb_cfg, c.aabb.max_pt - c.grav, time, false);
        last_update = time;
    }
    void predict_ekf(const TimePoint& t) {
        grav_point_target.predict_ekf(t);
        min_point_target.predict_ekf(t);
        max_point_target.predict_ekf(t);
    }
    void update(const ClusterResult& c, const TimePoint& t) {
        grav_point_target.update(c.grav, t);
        min_point_target.update(c.aabb.min_pt - c.grav, t);
        max_point_target.update(c.aabb.max_pt - c.grav, t);

        last_update = t;
    }
    double get_match_score(const ClusterResult& other) const {
        double score = 0.0;
        score += get_coincident_volume(other);

        return score;
    }
    AABB aabb() const {
        return AABB(
            (min_point_target.state.pos() + grav_point_target.state.pos()).cast<float>(),
            (max_point_target.state.pos() + grav_point_target.state.pos()).cast<float>()
        );
    }
    double get_coincident_volume(const ClusterResult& other) const {
        AABB me = aabb();
        AABB he = other.aabb;
        Eigen::Vector3f overlap;

        overlap.x() = std::max(
            0.0,
            (double
            )(std::min(me.max_pt.x(), he.max_pt.x()) - std::max(me.min_pt.x(), he.min_pt.x()))
        );

        overlap.y() = std::max(
            0.0,
            (double
            )(std::min(me.max_pt.y(), he.max_pt.y()) - std::max(me.min_pt.y(), he.min_pt.y()))
        );

        overlap.z() = std::max(
            0.0,
            (double
            )(std::min(me.max_pt.z(), he.max_pt.z()) - std::max(me.min_pt.z(), he.min_pt.z()))
        );

        return overlap.x() * overlap.y() * overlap.z();
    }
    void fill_marker(
        visualization_msgs::msg::MarkerArray& arr,
        int& id,
        int _id,
        std_msgs::msg::Header header
    ) const noexcept {
        const auto& g = grav_point_target.state.pos();
        std_msgs::msg::ColorRGBA color;
        color.r = 1.0f;
        color.a = 1.0;
        {
            visualization_msgs::msg::Marker m;
            m.header = header;
            m.ns = "center";
            m.id = id++;
            m.type = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;

            m.pose.position.x = g.x();
            m.pose.position.y = g.y();
            m.pose.position.z = g.z();
            m.pose.orientation.w = 1.0;

            m.scale.x = m.scale.y = m.scale.z = 0.2;
            m.color = color;
            m.lifetime = rclcpp::Duration::from_seconds(0.2);

            arr.markers.push_back(m);
        }
        {
            visualization_msgs::msg::Marker m;
            m.header = header;
            m.ns = "box";
            m.id = id++;
            m.type = visualization_msgs::msg::Marker::LINE_LIST;
            m.action = visualization_msgs::msg::Marker::ADD;
            auto boxEdges = [](const AABB& box) {
                const auto& mn = box.min_pt;
                const auto& mx = box.max_pt;

                std::vector<Eigen::Vector3d> p = {
                    { mn.x(), mn.y(), mn.z() }, { mx.x(), mn.y(), mn.z() },
                    { mx.x(), mx.y(), mn.z() }, { mn.x(), mx.y(), mn.z() },
                    { mn.x(), mn.y(), mx.z() }, { mx.x(), mn.y(), mx.z() },
                    { mx.x(), mx.y(), mx.z() }, { mn.x(), mx.y(), mx.z() },
                };

                int idx[] = {
                    0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7
                };

                std::vector<Eigen::Vector3d> edges;
                for (int i: idx)
                    edges.push_back(p[i]);
                return edges;
            };
            auto edges = boxEdges(aabb());
            for (auto& p: edges) {
                geometry_msgs::msg::Point pt;
                pt.x = p.x();
                pt.y = p.y();
                pt.z = p.z();
                m.points.push_back(pt);
            }

            m.scale.x = 0.03; // 线宽
            m.color = color;
            m.lifetime = rclcpp::Duration::from_seconds(0.2);

            arr.markers.push_back(m);
        }
        {
            visualization_msgs::msg::Marker m;
            m.header = header;
            m.ns = "label";
            m.id = id++;
            m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            m.action = visualization_msgs::msg::Marker::ADD;

            const auto& box = aabb();
            m.pose.position.x = box.min_pt.x() - 0.05;
            m.pose.position.y = box.min_pt.y() - 0.05;
            m.pose.position.z = box.max_pt.z() + 0.1;

            m.scale.z = 0.2;

            m.text = std::to_string(_id);
            m.color = color;
            m.lifetime = rclcpp::Duration::from_seconds(0.2);

            arr.markers.push_back(m);
        }
        {
            do {
                visualization_msgs::msg::Marker m;
                m.header = header;
                m.ns = "velocity";
                m.id = id++;
                m.type = visualization_msgs::msg::Marker::ARROW;
                m.action = visualization_msgs::msg::Marker::ADD;
                auto vel = grav_point_target.state.vel();
                if (vel.norm() < 0.5)
                    break;
                auto start = grav_point_target.state.pos();
                auto end = start + vel.normalized() * 0.5f;
                geometry_msgs::msg::Point pt_start;
                pt_start.x = start.x();
                pt_start.y = start.y();
                pt_start.z = start.z();
                m.points.push_back(pt_start);

                geometry_msgs::msg::Point pt_end;
                pt_end.x = end.x();
                pt_end.y = end.y();
                pt_end.z = end.z();
                m.points.push_back(pt_end);

                m.scale.x = 0.02; // 箭头线宽
                m.scale.y = 0.04; // 箭头头部宽
                m.scale.z = 0.06; // 箭头头部长

                m.color = color;
                m.lifetime = rclcpp::Duration::from_seconds(0.2);

                arr.markers.push_back(m);
            } while (0);
        }
    }
    CarTargetConfig target_config;
    PointTarget grav_point_target;
    PointTarget min_point_target;
    PointTarget max_point_target;
    TimePoint last_update;
};
class CarPool {
public:
    std::unordered_map<int, CarTarget> car_pool;
    TimePoint time;
    std::vector<std::pair<CarTarget&, const CarTarget&>> get_matched(const CarPool& ref) {
        std::vector<std::pair<CarTarget&, const CarTarget&>> matched_cars;
        for (auto& [id, car]: car_pool) {
            auto it = ref.car_pool.find(id);
            if (it != ref.car_pool.end()) {
                matched_cars.emplace_back(car, it->second);
            }
        }
        return matched_cars;
    }
    visualization_msgs::msg::MarkerArray to_marker_array(const std::string& frame_id) const {
        visualization_msgs::msg::MarkerArray arr;

        visualization_msgs::msg::Marker clear;
        clear.action = visualization_msgs::msg::Marker::DELETEALL;
        arr.markers.push_back(clear);

        int id = 0;
        visualization_msgs::msg::Marker m;
        m.header.frame_id = frame_id;
        rclcpp::Clock clock(RCL_SYSTEM_TIME); // Use system time (or ROS time)
        m.header.stamp = clock.now();
        for (const auto& [_id, obj]: car_pool) {
            obj.fill_marker(arr, id, _id, m.header);
        }
        return arr;
    }
};
struct LidarLocation::Impl {
    struct VoxelKey3D {
        int x, y, z;
        VoxelKey3D(): x(0), y(0), z(0) {}
        VoxelKey3D(int x_, int y_, int z_): x(x_), y(y_), z(z_) {}
        bool operator==(const VoxelKey3D& rhs) const {
            return x == rhs.x && y == rhs.y && z == rhs.z;
        }
    };
    template<typename Cell>
    class VoxelMap {
    public:
        using Ptr = std::shared_ptr<VoxelMap>;
        VoxelMap(const Eigen::Vector3f& min_p, const Eigen::Vector3f& max_p, float voxel_size) {
            this->voxel_size = voxel_size;

            min_key = worldToKey3D(min_p);
            max_key = worldToKey3D(max_p);
            nx = max_key.x - min_key.x + 1;
            ny = max_key.y - min_key.y + 1;
            nz = max_key.z - min_key.z + 1;
            const size_t N = static_cast<size_t>(nx) * ny * nz;
            std::cout << min_p.transpose() << "  " << max_p.transpose() << "  " << voxel_size
                      << std::endl;
            std::cout << "nx: " << nx << ", ny: " << ny << ", nz: " << nz << std::endl;
            std::cout << "grid size: " << N << std::endl;
            grid.resize(N);
        }
        static Ptr
        create(const Eigen::Vector3f& min_p, const Eigen::Vector3f& max_p, float voxel_size) {
            return std::make_shared<VoxelMap>(min_p, max_p, voxel_size);
        }
        size_t size() const noexcept {
            return grid.size();
        }
        inline VoxelKey3D worldToKey3D(const Eigen::Vector3f& p) const noexcept {
            Eigen::Vector3f q = p / voxel_size;
            return { int(std::floor(q.x())), int(std::floor(q.y())), int(std::floor(q.z())) };
        }
        inline Eigen::Vector3f key3DToWorld(const VoxelKey3D& k) const noexcept {
            return Eigen::Vector3f(k.x, k.y, k.z) * voxel_size;
        }

        inline int key3DToIndex3D(const VoxelKey3D& k) const noexcept {
            if (k.x < min_key.x || k.x > max_key.x || k.y < min_key.y || k.y > max_key.y
                || k.z < min_key.z || k.z > max_key.z)
                return -1;
            int dx = k.x - min_key.x;
            int dy = k.y - min_key.y;
            int dz = k.z - min_key.z;
            return (dx * ny + dy) * nz + dz;
        }

        inline VoxelKey3D index3DToKey3D(int idx) const noexcept {
            int dz = idx % nz;
            idx /= nz;
            int dy = idx % ny;
            int dx = idx / ny;
            return { min_key.x + dx, min_key.y + dy, min_key.z + dz };
        }
        double voxel_size;
        int nx, ny, nz;
        std::vector<Cell> grid;
        VoxelKey3D min_key, max_key;
    };

    struct Params {
        std::string target_map_path;
        double voxel_size;
        double diff_th;
        double diff_accumulation_time;
        double lost_dt = 1.0;
        double match_vol_ratio = 0.3;
        double cluster_eps = 0.01;
        int cluster_min_pts = 5;
        double cluster_leaf_size = 0.05;
        void load(const YAML::Node& config) {
            target_map_path = config["target_map_path"].as<std::string>();
            voxel_size = config["voxel_size"].as<double>();
            diff_th = config["diff_th"].as<double>();
            diff_accumulation_time = config["diff_accumulation_time"].as<double>();
            cluster_eps = config["cluster_eps"].as<double>();
            cluster_min_pts = config["cluster_min_pts"].as<int>();
            cluster_leaf_size = config["cluster_leaf_size"].as<double>();
        }
    } params_;
    Impl(const YAML::Node& config) {
        params_.load(config);
        target_map_pts_.clear();
        if (!io::pcd::read_pcd(params_.target_map_path, target_map_pts_)) {
            throw std::runtime_error("Failed to read target map PCD file");
        }
        auto min_p_vec = config["min_p"].as<std::vector<double>>();
        auto max_p_vec = config["max_p"].as<std::vector<double>>();
        Eigen::Vector3f min_p(min_p_vec[0], min_p_vec[1], min_p_vec[2]);
        Eigen::Vector3f max_p(max_p_vec[0], max_p_vec[1], max_p_vec[2]);
        // Eigen::Vector3f min_p, max_p;
        // for (const auto& point: target_map) {
        //     min_p = min_p.cwiseMin(point);
        //     max_p = max_p.cwiseMax(point);
        // }
        target_map_ = VoxelMap<Ceil>::create(min_p, max_p, params_.voxel_size);
        for (const auto& point: target_map_pts_) {
            auto key = target_map_->worldToKey3D(point);
            int index = target_map_->key3DToIndex3D(key);
            if (index > 0) {
                target_map_->grid[index].active = true;
            }
        }
        car_target_config_.load(config["car_target"]);
    }
    CarPool detect(const std::vector<Eigen::Vector3f>& pts) {
        auto now = Clock::now(); // Get the current time once for efficiency.

        // Add the new frame to the queue.
        frame_queue_.emplace_back(Frame { pts, now });

        // Remove old frames outside of the time window.
        while (frame_queue_.size() > 1
               && std::chrono::duration<double>(now - frame_queue_.front().t)
                   > std::chrono::duration<double>(params_.diff_accumulation_time))
        {
            frame_queue_.pop_front();
        }

        // Efficiently accumulate all points.
        std::vector<Eigen::Vector3f> all_pts;
        size_t total_size = 0;
        for (const auto& frame: frame_queue_) {
            total_size += frame.pts.size();
        }
        all_pts.reserve(total_size); // Reserve space to avoid reallocations.

        for (const auto& frame: frame_queue_) {
            all_pts.insert(all_pts.end(), frame.pts.begin(), frame.pts.end());
        }

        // Perform diff computation and clustering.
        auto diff = get_diffs(all_pts);
        auto c = clusterDBSCAN(diff);

        // Track the clusters and update car_pool_.
        track(c, now);

        return car_pool_;
    }
    bool is_diff(const Eigen::Vector3f& point) const noexcept {
        auto key = target_map_->worldToKey3D(point);
        int index = target_map_->key3DToIndex3D(key);
        const int radius = std::max(0, (int)(params_.diff_th / target_map_->voxel_size));
        if (index > 0) {
            for (int kx = key.x - radius; kx <= key.x + radius; ++kx) {
                for (int ky = key.y - radius; ky <= key.y + radius; ++ky) {
                    for (int kz = key.z - radius; kz <= key.z + radius; ++kz) {
                        int nidx = target_map_->key3DToIndex3D({ kx, ky, kz });
                        if (nidx > 0 && target_map_->grid[nidx].active) {
                            return false;
                        }
                    }
                }
            }
        } else {
            return false;
        }

        return true;
    }
    std::vector<Eigen::Vector3f> get_diffs(const std::vector<Eigen::Vector3f>& pointcloud
    ) const noexcept {
        std::vector<Eigen::Vector3f> diffs;
        for (const auto& point: pointcloud) {
            auto key = target_map_->worldToKey3D(point);
            int index = target_map_->key3DToIndex3D(key);
            if (index >= 0 && is_diff(point)) {
                diffs.push_back(point);
            }
        }
        return diffs;
    }
    std::vector<ClusterResult> clusterDBSCAN(const std::vector<Eigen::Vector3f>& input_points
    ) const noexcept {
        std::vector<ClusterResult> objects;
        const double user_eps = params_.cluster_eps;
        const int minPts = params_.cluster_min_pts;
        const double leaf_size = params_.cluster_leaf_size;

        if (input_points.empty())
            return objects;

        auto cloud = std::make_shared<small_gicp::PointCloud>();
        cloud->resize(input_points.size());
        tbb::parallel_for(size_t(0), input_points.size(), [&](size_t i) {
            cloud->point(i).head<3>() = input_points[i].cast<double>();
            cloud->point(i)[3] = 1.0;
        });

        if (cloud->empty())
            return objects;

        if (leaf_size > 0.0) {
            cloud = small_gicp::voxelgrid_sampling_tbb<small_gicp::PointCloud>(*cloud, leaf_size);
        }
        const size_t n = cloud->size();
        if (n == 0)
            return objects;

        auto tree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(
            cloud,
            small_gicp::KdTreeBuilderTBB()
        );

        const size_t max_neighbors = std::min<size_t>(n, 1024);
        std::vector<std::vector<size_t>> nbs(n);
        std::vector<std::vector<double>> nbs_dists(n);

        tbb::parallel_for(size_t(0), n, [&](size_t idx) {
            std::vector<size_t> knn_idx(max_neighbors);
            std::vector<double> knn_dist(max_neighbors);
            size_t found =
                tree->knn_search(cloud->point(idx), max_neighbors, knn_idx.data(), knn_dist.data());

            for (size_t j = 0; j < found; ++j) {
                if (knn_dist[j] <= user_eps * user_eps)
                    nbs[idx].push_back(knn_idx[j]);
                else
                    break;
            }
        });

        std::vector<int> labels(n, -2); // -2: undefined, -1: noise
        std::vector<char> visited(n, 0);
        int cluster_id = 0;

        for (size_t i = 0; i < n; ++i) {
            if (visited[i])
                continue;
            visited[i] = 1;

            if (nbs[i].size() < static_cast<size_t>(minPts)) {
                labels[i] = -1; // 噪声
                continue;
            }

            labels[i] = cluster_id;
            std::vector<size_t> stack = nbs[i];

            while (!stack.empty()) {
                size_t idx = stack.back();
                stack.pop_back();
                if (!visited[idx]) {
                    visited[idx] = 1;
                    if (nbs[idx].size() >= static_cast<size_t>(minPts)) {
                        for (auto nb: nbs[idx])
                            if (!visited[nb])
                                stack.push_back(nb);
                    }
                }
                if (labels[idx] < 0)
                    labels[idx] = cluster_id;
            }

            cluster_id++;
        }

        if (cluster_id == 0)
            return objects;

        std::vector<std::vector<Eigen::Vector4f>> temp_clusters(cluster_id);
        for (size_t i = 0; i < n; ++i) {
            int lab = labels[i];
            if (lab >= 0)
                temp_clusters[lab].push_back(cloud->point(i).cast<float>());
        }

        objects.resize(cluster_id);
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, cluster_id),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t cid = r.begin(); cid != r.end(); ++cid) {
                    auto& pts = temp_clusters[cid];
                    if (pts.empty())
                        continue;

                    Eigen::Vector3f min_pt = pts.front().head<3>();
                    Eigen::Vector3f max_pt = pts.front().head<3>();
                    Eigen::Vector3f center = Eigen::Vector3f::Zero();

                    for (auto& p: pts) {
                        auto xyz = p.head<3>();
                        min_pt = min_pt.cwiseMin(xyz);
                        max_pt = max_pt.cwiseMax(xyz);
                        center += xyz;
                    }
                    center /= static_cast<double>(pts.size());
                    objects[cid] = ClusterResult { .cluster = std::move(pts),
                                                   .aabb = { min_pt, max_pt },
                                                   .grav = center };
                }
            }
        );

        return objects;
    }
    void track(const std::vector<ClusterResult>& cr, const TimePoint& t) {
        for (auto it = car_pool_.car_pool.begin(); it != car_pool_.car_pool.end();) {
            auto& car = it->second;

            car.predict_ekf(t);

            auto dt = std::chrono::duration<double>(t - car.last_update).count();

            if (dt > params_.lost_dt) {
                it = car_pool_.car_pool.erase(it);
            } else {
                ++it;
            }
        }
        auto matches = match(cr);
        for (const auto& [pool_id, input_id]: matches.pool_id_input_id) {
            auto& car = car_pool_.car_pool[pool_id];
            car.update(cr[input_id], t);
        }
        for (const auto& no_match_input_id: matches.no_match_input_id) {
            car_pool_.car_pool[current_id_] =
                CarTarget(car_target_config_, cr[no_match_input_id], t);
            AWAKENING_INFO("create a new car id: {}", current_id_);
            current_id_++;
        }
        car_pool_.time = t;
        return;
    }
    struct MatchResult {
        std::vector<std::pair<int, int>> pool_id_input_id;
        std::vector<int> no_match_input_id;
    };
    MatchResult match(const std::vector<ClusterResult>& cr) {
        MatchResult matches;
        if (cr.empty())
            return matches;
        if (car_pool_.car_pool.empty()) {
            matches.no_match_input_id.resize(cr.size());
            std::iota(matches.no_match_input_id.begin(), matches.no_match_input_id.end(), 0);
            return matches;
        }
        std::vector<std::pair<int, int>> pool_id_input_id;
        std::vector<int> no_match_input_id;
        struct Candidate {
            int track_id;
            int det_id;
            double score;
        };

        std::vector<Candidate> candidates;

        for (const auto& [id, target]: car_pool_.car_pool) {
            for (size_t j = 0; j < cr.size(); ++j) {
                double _match_score = target.get_match_score(cr[j]);
                if (_match_score > 0) {
                    candidates.push_back({ id, (int)j, _match_score });
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.score > b.score;
        });

        std::unordered_set<int> used_tracks;
        std::unordered_set<int> used_dets;

        for (const auto& c: candidates) {
            if (used_tracks.count(c.track_id) == 0 && used_dets.count(c.det_id) == 0) {
                pool_id_input_id.emplace_back(c.track_id, c.det_id);
                used_tracks.insert(c.track_id);
                used_dets.insert(c.det_id);
            }
        }
        for (size_t j = 0; j < cr.size(); ++j) {
            if (used_dets.count(j) == 0)
                no_match_input_id.push_back((int)j);
        }

        matches.pool_id_input_id = pool_id_input_id;
        matches.no_match_input_id = no_match_input_id;
        return matches;
    }
    std::pair<Eigen::Vector3f, Eigen::Vector3f> get_target_map_bbox() const {
        return std::make_pair(
            target_map_->key3DToWorld(target_map_->min_key),
            target_map_->key3DToWorld(target_map_->max_key)
        );
    }
    std::vector<Eigen::Vector3f>& get_target_map_pts() {
        return target_map_pts_;
    }
    struct Ceil {
        bool active = false;
    };
    struct Frame {
        std::vector<Eigen::Vector3f> pts;
        TimePoint t;
    };
    std::vector<Eigen::Vector3f> target_map_pts_;
    std::deque<Frame> frame_queue_;
    CarPool car_pool_;
    VoxelMap<Ceil>::Ptr target_map_;
    CarTarget::CarTargetConfig car_target_config_;
    int current_id_ = 0;
};
LidarLocation::LidarLocation(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}
LidarLocation::~LidarLocation() noexcept {}
// CarPool LidarLocation::detect(const std::vector<Eigen::Vector3f>& pts) {
//     return _impl->detect(pts);
// }
std::pair<Eigen::Vector3f, Eigen::Vector3f> LidarLocation::get_target_map_bbox() {
    return _impl->get_target_map_bbox();
}
std::vector<Eigen::Vector3f>& LidarLocation::get_target_map_pts() {
    return _impl->get_target_map_pts();
}
} // namespace awakening::radar_detect