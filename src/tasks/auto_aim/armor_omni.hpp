#pragma once
#include "tasks/auto_aim/armor_detect/armor_detector.hpp"
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/auto_aim/armor_tracker/armor_tracker.hpp"
#include "tasks/base/common.hpp"
#include "type.hpp"
#include "utils/drivers/uvc_camera.hpp"
#include <memory>
#include <opencv2/core/types.hpp>
#include <unordered_map>
#include <utility>
namespace awakening::auto_aim {
class ArmorOmni {
public:
    struct One {
        std::unique_ptr<UVCCamera> camera;
        std::unique_ptr<ArmorTracker> tracker;
        ArmorTarget target;
        CameraInfo camera_info;
        int frame_id;
        int cv_frame_id;
        double total_score;
        One(const YAML::Node& config,
            int frame_id,
            int cv_frame_id,
            const YAML::Node& tracker_config) {
            camera = std::make_unique<UVCCamera>(config["uvc_camera"]);
            camera->start();
            this->frame_id = frame_id;
            this->cv_frame_id = cv_frame_id;
            this->total_score = 0.0;
            camera_info.load(config["camera_info"]);
            tracker = std::make_unique<ArmorTracker>(tracker_config);
        }
    };
    ArmorOmni(const YAML::Node& config) {
        config_ = config;
        fps_ = config["fps"].as<double>();
        detector_ = std::make_unique<ArmorDetector>(config["armor_detector"]);
    }
    void emplace_one(const YAML::Node& config, int frame_id, int cv_frame_id) {
        ones_.emplace(frame_id, One(config, frame_id, cv_frame_id, config_["armor_tracker"]));
        ones_keys_.push_back(frame_id);
    }
    One& get_next() {
        if (ones_keys_.empty()) {
            throw std::runtime_error("empty");
        }

        current_ones_idx_ %= ones_keys_.size();
        int key = ones_keys_[current_ones_idx_++];

        return ones_.at(key);
    }
    ArmorTarget update() {
        double max_score = std::numeric_limits<double>::lowest();
        for (auto& [_, one]: ones_) {
            if (one.total_score > max_score) {
                max_score = one.total_score;
                best_target_ = one.target.fast_copy_without_ekf();
            }
        }
        return best_target_;
    }
    std::unique_ptr<ArmorDetector> detector_;
    double fps_ = 10;

private:
    YAML::Node config_;
    std::unordered_map<int, One> ones_;
    std::vector<int> ones_keys_;
    size_t current_ones_idx_ = 0;
    ArmorTarget best_target_;
};
} // namespace awakening::auto_aim