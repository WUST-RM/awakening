#pragma once
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "utils/buffer.hpp"
#include <mutex>
#include <opencv2/core/types.hpp>
#include <utility>
namespace awakening::auto_aim {
struct AutoAimDebugCtx {
    CameraInfo camera_info_;
    utils::SWMR<Armors> armors_buffer;
    utils::SWMR<ArmorTarget> armor_target_buffer;
    mutable std::mutex img_frame_mutex;
    ImageFrame img_frame_buffer;
    utils::SWMR<cv::Rect> expanded_buffer;
    utils::SWMR<double> avg_latency_ms_buffer;
    utils::SWMR<GimbalCmd> gimbal_cmd_buffer;
    utils::SWMR<AutoAimFsm> fsm_state_buffer;
    utils::SWMR<std::pair<double, double>> gimbal_yaw_pitch_buffer;
    Armors armors() const noexcept {
        return armors_buffer.read();
    }
    ArmorTarget armor_target() const noexcept {
        return armor_target_buffer.read();
    }
    void set_img_frame(const ImageFrame& img_frame) {
        std::lock_guard<std::mutex> lock(img_frame_mutex);
        img_frame_buffer = img_frame;
    }
    ImageFrame img_frame() const noexcept {
        std::lock_guard<std::mutex> lock(img_frame_mutex);
        return img_frame_buffer;
    }
    CameraInfo camera_info() const noexcept {
        return camera_info_;
    }
    cv::Rect expanded() const noexcept {
        return expanded_buffer.read();
    }
    double avg_latency_ms() const noexcept {
        return avg_latency_ms_buffer.read();
    }
    GimbalCmd gimbal_cmd() const noexcept {
        return gimbal_cmd_buffer.read();
    }
    AutoAimFsm fsm_state() const noexcept {
        return fsm_state_buffer.read();
    }
    std::pair<double, double> gimbal_yaw_pitch() const noexcept {
        return gimbal_yaw_pitch_buffer.read();
    }
};
void draw_auto_aim(cv::Mat& img, const AutoAimDebugCtx& ctx);
void write_debug_data(const AutoAimDebugCtx& ctx);

} // namespace awakening::auto_aim