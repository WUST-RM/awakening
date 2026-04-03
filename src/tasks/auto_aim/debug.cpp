#include "debug.hpp"
#include "utils/utils.hpp"
#include <fmt/format.h>
#include <utility>

namespace awakening::auto_aim {
void draw_auto_aim(cv::Mat& img, AutoAimDebugCtx& ctx) {
    if (img.empty()) {
        return;
    }
    auto armors = ctx.armors();
    auto armor_target = ctx.armor_target();
    auto camera_info = ctx.camera_info();
    auto camera_cv_in_odom = ctx.camera_cv_in_odom();
    const cv::Rect img_rect(0, 0, img.cols, img.rows);
    const cv::Rect roi = ctx.expanded() & img_rect;
    cv::rectangle(img, roi, cv::Scalar(255, 255, 255), 2);
    armors.draw(img);
    if (armor_target.check()) {
        auto target_state = armor_target.get_target_state();
        target_state.predict(Clock::now());
        auto armors_pose_in_odom = target_state.get_armors_pose(armor_target.target_number);
        for (auto& armor_pose_in_odom: armors_pose_in_odom) {
            auto armor_pose_in_camera_cv = camera_cv_in_odom.inverse() * armor_pose_in_odom;
            auto image_points = utils::reprojection(
                camera_info.camera_matrix,
                camera_info.distortion_coefficients,
                getArmorKeyPoints3D<cv::Point3f>(armor_target.target_number),
                armor_pose_in_camera_cv
            );
            using I = ArmorKeyPointsIndex;
            auto draw_line = [&](auto i, auto j) {
                cv::line(
                    img,
                    image_points[std::to_underlying(i)],
                    image_points[std::to_underlying(j)],
                    cv::Scalar(200, 255, 200),
                    1
                );
            };
            draw_line(I::LEFT_TOP, I::LEFT_BOTTOM);
            draw_line(I::LEFT_BOTTOM, I::RIGHT_BOTTOM);
            draw_line(I::RIGHT_BOTTOM, I::RIGHT_TOP);
            draw_line(I::RIGHT_TOP, I::LEFT_TOP);
        }
    }

    const std::string latency_str = fmt::format("Avg Latency: {:.2f}ms", ctx.avg_latency_ms());
    cv::putText(
        img,
        latency_str,
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2
    );
}
} // namespace awakening::auto_aim