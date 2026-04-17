#include "debug.hpp"
#include "angles.h"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "tasks/base/web.hpp"
#include "utils/common/type_common.hpp"
#include "utils/utils.hpp"
#include <fmt/format.h>
#include <opencv2/core/types.hpp>
#include <utility>

namespace awakening::auto_aim {
void draw_auto_aim(cv::Mat& img, const AutoAimDebugCtx& ctx) {
    if (img.empty()) {
        return;
    }
    auto armors = ctx.armors.get();
    auto armor_target = ctx.armor_target.get();
    auto camera_info = ctx.camera_info();
    auto cmd = ctx.gimbal_cmd.get();
    auto fsm = ctx.fsm_state.get();
    auto bullet_poss = ctx.bullet_positions.get();
    const cv::Rect img_rect(0, 0, img.cols, img.rows);
    const cv::Rect roi = ctx.expanded.get() & img_rect;
    if (roi.width > 0 && roi.height > 0) {
        cv::rectangle(img, roi, cv::Scalar(255, 255, 255), 2);
    }
    armors.draw(img);
    if (armor_target.check()) {
        auto target_state = armor_target.get_target_state();
        target_state.predict(Clock::now(), armor_target.target_number);
        auto armors_pose_in_odom = target_state.get_armors_pose(armor_target.target_number);
        auto odom_in_camera_cv = ctx.odom_in_camera_cv.get();
        for (int i = 0; i < armors_pose_in_odom.size(); ++i) {
            auto& armor_pose_in_camera_cv = odom_in_camera_cv * armors_pose_in_odom[i];
            if (armor_pose_in_camera_cv.translation().z() > 0.1) {
                auto image_points = utils::reprojection(
                    camera_info.camera_matrix,
                    camera_info.distortion_coefficients,
                    getArmorKeyPoints3D<cv::Point3f>(armor_target.target_number),
                    armor_pose_in_camera_cv
                );
                using I = ArmorKeyPointsIndex;
                auto draw_line = [&](auto _i, auto _j) {
                    cv::line(
                        img,
                        image_points[std::to_underlying(_i)],
                        image_points[std::to_underlying(_j)],
                        (i == cmd.select_id) ? cv::Scalar(255, 0, 255) : cv::Scalar(200, 255, 200),
                        2
                    );
                };
                draw_line(I::LEFT_TOP, I::LEFT_BOTTOM);
                draw_line(I::LEFT_BOTTOM, I::RIGHT_BOTTOM);
                draw_line(I::RIGHT_BOTTOM, I::RIGHT_TOP);
                draw_line(I::RIGHT_TOP, I::LEFT_TOP);
            }
        }
        {
            auto center_pose = ISO3::Identity();
            center_pose.translation() = target_state.pos();
            auto vel_pose = center_pose;
            vel_pose.translation() += target_state.vel();
            center_pose = odom_in_camera_cv * center_pose;
            vel_pose = odom_in_camera_cv * vel_pose;
            auto center_image_points = utils::reprojection(
                camera_info.camera_matrix,
                camera_info.distortion_coefficients,
                { cv::Point3f(0, 0, 0) },
                center_pose
            );
            auto vel_image_points = utils::reprojection(
                camera_info.camera_matrix,
                camera_info.distortion_coefficients,
                { cv::Point3f(0, 0, 0) },
                vel_pose
            );
            cv::Point2f center = center_image_points[0];
            cv::Point2f vel = vel_image_points[0];
            const double scale = 50.0;
            const double dy = scale * target_state.vyaw();
            const cv::Point2f start_pt = center;
            cv::Point2f end_pt = start_pt + cv::Point2f(0, dy);
            cv::arrowedLine(img, start_pt, end_pt, cv::Scalar(50, 255, 50), 3, cv::LINE_AA, 0, 0.1);
            end_pt = vel;
            cv::arrowedLine(img, start_pt, end_pt, cv::Scalar(50, 255, 50), 3, cv::LINE_AA, 0, 0.1);
            cv::circle(img, center, 5, cv::Scalar(50, 255, 50), -1);
            cv::putText(
                img,
                fmt::format("V_yaw: {:.2f}", target_state.vyaw()),
                center + cv::Point2f(0, -20),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(50, 255, 50),
                2
            );
        }
        if (cmd.aim_point.pose.translation().z() > 0.1) {
            auto aim_point_img_points = utils::reprojection(
                camera_info.camera_matrix,
                camera_info.distortion_coefficients,
                { cv::Point3f(0, 0, 0) },
                cmd.aim_point.pose
            );
            constexpr double R = 0.02;
            cv::Point2f center = aim_point_img_points[0];
            double r = camera_info.camera_matrix.at<double>(0, 0) * R
                / cmd.aim_point.pose.translation().z();
            cv::circle(img, center, r, cv::Scalar(255, 255, 255), 2);
            if (cmd.fire_advice) {
                int size = 50;
                cv::line(
                    img,
                    center + cv::Point2f(-size, -size),
                    center + cv::Point2f(size, size),
                    cv::Scalar(0, 0, 255),
                    2
                );
                cv::line(
                    img,
                    center + cv::Point2f(-size, size),
                    center + cv::Point2f(size, -size),
                    cv::Scalar(0, 0, 255),
                    2
                );
            }
            const double scale = 10.0;

            const double v_yaw = cmd.v_yaw;
            const double v_pitch = cmd.v_pitch;

            const double dx = -scale * v_yaw;
            const double dy = scale * v_pitch;

            const cv::Point2f start_pt = center;
            const cv::Point2f end_pt = start_pt + cv::Point2f(dx, dy);

            const cv::Scalar color_x = cv::Scalar(0, 215, 255);

            cv::arrowedLine(img, start_pt, end_pt, color_x, 4, cv::LINE_AA, 0, 0.2);
        }
    }
    {
        for (auto& p: bullet_poss) {
            ISO3 pose = ISO3::Identity();
            pose.translation() = p;
            if (p.z() > 0.2) {
                auto bullet_img_points = utils::reprojection(
                    camera_info.camera_matrix,
                    camera_info.distortion_coefficients,
                    { cv::Point3f(0, 0, 0) },
                    pose
                );
                constexpr double R = 0.017 / 2.0;
                cv::Point2f center = bullet_img_points[0];
                double r =
                    camera_info.camera_matrix.at<double>(0, 0) * R / pose.translation().norm();
                cv::circle(img, center, r, cv::Scalar(100, 255, 100), 3);
            }
        }
    }
    if (cmd.fire_advice) {
        std::string fire_str = "Fire!";
        cv::putText(
            img,
            fire_str,
            { img.cols / 2 - 100, 200 },
            cv::FONT_HERSHEY_SIMPLEX,
            2.85,
            cv::Scalar(0, 0, 255),
            2
        );
    }
    {
        auto now = std::chrono::system_clock::now();

        std::time_t t = std::chrono::system_clock::to_time_t(now);

        std::tm tm {};
        localtime_r(&t, &tm);

        auto duration = now.time_since_epoch();
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(duration - seconds).count();

        char time_buf[64];
        std::snprintf(
            time_buf,
            sizeof(time_buf),
            "%04d-%02d-%02d %02d:%02d:%02d.%03ld",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min,
            tm.tm_sec,
            milliseconds
        );
        std::string time_str = time_buf;

        const std::string latency_str =
            fmt::format("Avg Latency: {:.2f}ms", ctx.avg_latency_ms.get());

        int font = cv::FONT_HERSHEY_SIMPLEX;
        double scale = 0.8;
        int thickness = 2;

        int baseline = 0;

        cv::Size time_size = cv::getTextSize(time_str, font, scale, thickness, &baseline);

        int x = 10;
        int y = time_size.height + 10;

        cv::putText(
            img,
            time_str,
            cv::Point(x, y),
            font,
            scale,
            cv::Scalar(255, 255, 255),
            thickness
        );

        int line_gap = 5;
        int latency_y = y + time_size.height + line_gap;

        cv::putText(
            img,
            latency_str,
            cv::Point(x, latency_y),
            font,
            scale,
            cv::Scalar(255, 255, 255),
            thickness
        );
    }
    {
        constexpr int col_width = 8; // 每列占 8 个字符
        constexpr int precision = 2; // 小数位数

        auto format_col = [](double val) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%+*.*f", col_width - 1, precision, val);
            return std::string(buf);
        };
        const std::string yaw_cmd_str = "yaw:   p:" + format_col(cmd.yaw)
            + " v:" + format_col(cmd.v_yaw) + " a:" + format_col(cmd.a_yaw)
            + " enable:" + format_col(cmd.enable_yaw_diff);

        const std::string pitch_cmd_str = "pitch: p:" + format_col(cmd.pitch)
            + " v:" + format_col(cmd.v_pitch) + " a:" + format_col(cmd.a_pitch)
            + " enable:" + format_col(cmd.enable_pitch_diff);

        int font = cv::FONT_HERSHEY_SIMPLEX;
        double scale = 0.6;
        int thickness = 2;
        int baseline = 0;

        cv::Size yaw_size = cv::getTextSize(yaw_cmd_str, font, scale, thickness, &baseline);
        cv::Size pitch_size = cv::getTextSize(pitch_cmd_str, font, scale, thickness, &baseline);

        int max_width = std::max(yaw_size.width, pitch_size.width);
        int img_w = img.cols;
        int img_h = img.rows;

        int start_x = (img_w - max_width) / 2;
        int margin_bottom = 20;
        int line_gap = 10;

        int pitch_y = img_h - margin_bottom;
        int yaw_y = pitch_y - pitch_size.height - line_gap;

        cv::putText(
            img,
            yaw_cmd_str,
            cv::Point(start_x, yaw_y),
            font,
            scale,
            cv::Scalar(255, 255, 255),
            thickness
        );

        cv::putText(
            img,
            pitch_cmd_str,
            cv::Point(start_x, pitch_y),
            font,
            scale,
            cv::Scalar(255, 255, 255),
            thickness
        );
    }

    {
        std::string state_str = string_by_auto_aim_fsm(fsm);
        int baseline = 0;
        cv::Size text_size =
            cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 2.5, 2, &baseline);
        const int x = std::clamp(img.cols - text_size.width - 10, 0, img.cols - text_size.width);
        const int y = std::clamp(text_size.height + 10, text_size.height, img.rows - 1);

        cv::putText(
            img,
            state_str,
            { x, y },
            cv::FONT_HERSHEY_SIMPLEX,
            2.5,
            cv::Scalar(0, 0, 255),
            2
        );
        const std::string id_str =
            fmt::format("Attack: {}", string_by_armor_class(armor_target.target_number));
        const cv::Size id_size =
            cv::getTextSize(id_str, cv::FONT_HERSHEY_SIMPLEX, 1.6, 2, &baseline);

        // 保证在图像内
        const int id_x = std::clamp(img.cols - 300, 0, img.cols - id_size.width - 10);
        const int id_y = std::clamp(150, id_size.height, img.rows - 1);

        cv::putText(
            img,
            id_str,
            { id_x, id_y },
            cv::FONT_HERSHEY_SIMPLEX,
            1.6,
            cv::Scalar(255, 0, 255),
            2
        );
    }
    {
        cv::circle(img, cv::Point2i(img.cols / 2, img.rows / 2), 5, cv::Scalar(255, 255, 255), 2);
    }
}
void write_debug_data(const AutoAimDebugCtx& ctx) {
    static TimePoint start_time = Clock::now();
    static web::DebugDatas d;
    const auto now = Clock::now();
    const double t = std::chrono::duration<double>(now - start_time).count();
    static GimbalCmd last_cmd;
    static double last_yaw = 0.0;
    auto gimbal_yaw_pitch = ctx.gimbal_yaw_pitch.get();
    auto cmd = ctx.gimbal_cmd.get();
    cmd = cmd.appear ? cmd : last_cmd;
    last_cmd = cmd;
    auto armor_target = ctx.armor_target.get();
    auto target_state = armor_target.get_target_state();
    d.time_log.handle_once(t);
    auto un_warp = [&](double _yaw) {
        return last_yaw + angles::shortest_angular_distance_degrees(last_yaw, _yaw);
    };
    auto yaw = un_warp(cmd.yaw);
    d.yaw_log.handle_once(yaw);
    last_yaw = yaw;
    d.pitch_log.handle_once(cmd.pitch);
    d.target_yaw_log.handle_once(un_warp(cmd.target_yaw));
    d.target_pitch_log.handle_once(cmd.target_pitch);
    d.gimbal_yaw_log.handle_once(un_warp(gimbal_yaw_pitch.first));
    d.gimbal_pitch_log.handle_once(gimbal_yaw_pitch.second);
    d.control_v_yaw_log.handle_once(cmd.v_yaw);
    d.control_v_pitch_log.handle_once(cmd.v_pitch);
    d.control_a_yaw_log.handle_once(cmd.a_yaw);
    d.control_a_pitch_log.handle_once(cmd.a_pitch);
    d.fly_time_log.handle_once(cmd.fly_time);
    d.target_v_yaw_log.handle_once(target_state.vyaw());
    d.write();
}
} // namespace awakening::auto_aim