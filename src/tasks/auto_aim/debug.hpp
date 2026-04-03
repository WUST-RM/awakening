#pragma once
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "utils/buffer.hpp"
#include <opencv2/core/types.hpp>
namespace awakening::auto_aim {
struct AutoAimDebugCtx {
    utils::SWMR<Armors> armors_buffer;
    utils::SWMR<ArmorTarget> armor_target_buffer;
    utils::SWMR<ImageFrame> img_frame_buffer;
    utils::SWMR<CameraInfo> camera_info_buffer;
    utils::SWMR<cv::Rect> expanded_buffer;
    utils::SWMR<ISO3> camera_cv_in_odom_buffer;
    utils::SWMR<double> avg_latency_ms_buffer;

    Armors armors() {
        return armors_buffer.read();
    }
    ArmorTarget armor_target() {
        return armor_target_buffer.read();
    }
    ImageFrame img_frame() {
        return img_frame_buffer.read();
    }
    CameraInfo camera_info() {
        return camera_info_buffer.read();
    }
    cv::Rect expanded() {
        return expanded_buffer.read();
    }
    double avg_latency_ms() {
        return avg_latency_ms_buffer.read();
    }
    ISO3 camera_cv_in_odom() {
        return camera_cv_in_odom_buffer.read();
    }
};
void draw_auto_aim(cv::Mat& img, AutoAimDebugCtx& ctx);

} // namespace awakening::auto_aim