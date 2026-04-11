#pragma once
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "utils/buffer.hpp"
#include "utils/common/type_common.hpp"
#include <mutex>
#include <opencv2/core/types.hpp>
#include <utility>
#include <vector>
namespace awakening::auto_aim {
struct AutoAimDebugCtx {
    CameraInfo camera_info_; // 基本不变数据，无锁

    utils::Locked<Armors> armors;
    utils::Locked<ArmorTarget> armor_target;
    utils::Locked<ImageFrame> img_frame;
    utils::Locked<cv::Rect> expanded;
    utils::Locked<double> avg_latency_ms;
    utils::Locked<GimbalCmd> gimbal_cmd;
    utils::Locked<AutoAimFsm> fsm_state;
    utils::Locked<std::pair<double, double>> gimbal_yaw_pitch;
    utils::Locked<std::vector<Vec3>> bullet_positions;

    CameraInfo camera_info() const noexcept {
        return camera_info_;
    }
};
void draw_auto_aim(cv::Mat& img, const AutoAimDebugCtx& ctx);
void write_debug_data(const AutoAimDebugCtx& ctx);

} // namespace awakening::auto_aim