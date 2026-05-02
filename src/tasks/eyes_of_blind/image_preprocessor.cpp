#include "image_preprocessor.hpp"
#include "utils/logger.hpp"
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/node/node.h>

namespace awakening::eyes_of_blind {

struct ImagePreprocessor::Impl {
    // 参数结构
    struct Params {
        int crop_size = 800;
        int output_w = 400;
        int output_h = 400;
        int output_fps = 60; // 仅用于信息，预处理不依赖
        bool static_simplify = true;
        int motion_threshold = 14;
        int motion_erode_px = 1;
        int motion_dilate_px = 2;
        int motion_trail_frames = 3;
        double trail_disable_motion_ratio = 0.30;
        double bg_update_alpha = 0.01;
        double bg_blur_sigma = 1.2;
        int center_clear_size = 100;
        bool force_monochrome = false;

        void load(const YAML::Node& config) {
            crop_size = config["crop_size"].as<int>(crop_size);
            output_w = config["output_w"].as<int>(output_w);
            output_h = config["output_h"].as<int>(output_h);
            output_fps = config["fps"].as<int>(output_fps);
            static_simplify = config["static_simplify"].as<bool>(static_simplify);
            motion_threshold = config["motion_threshold"].as<int>(motion_threshold);
            motion_erode_px = config["motion_erode_px"].as<int>(motion_erode_px);
            motion_dilate_px = config["motion_dilate_px"].as<int>(motion_dilate_px);
            motion_trail_frames = config["motion_trail_frames"].as<int>(motion_trail_frames);
            trail_disable_motion_ratio =
                config["trail_disable_motion_ratio"].as<double>(trail_disable_motion_ratio);
            bg_update_alpha = config["bg_update_alpha"].as<double>(bg_update_alpha);
            bg_blur_sigma = config["bg_blur_sigma"].as<double>(bg_blur_sigma);
            center_clear_size = config["center_clear_size"].as<int>(center_clear_size);
            force_monochrome = config["force_monochrome"].as<bool>(force_monochrome);

            // 参数范围检查
            motion_trail_frames = std::clamp(motion_trail_frames, 0, 15);
            motion_erode_px = std::clamp(motion_erode_px, 0, 20);
            motion_dilate_px = std::clamp(motion_dilate_px, 0, 20);
            bg_update_alpha = std::clamp(bg_update_alpha, 0.001, 0.2);
            bg_blur_sigma = std::max(0.1, bg_blur_sigma);
        }
    } params_;

    // 状态变量
    cv::Mat background_gray_f32_;
    cv::Mat motion_erode_kernel_;
    cv::Mat motion_dilate_kernel_;
    std::deque<cv::Mat> motion_mask_history_;
    std::deque<cv::Mat> trail_frame_history_;

    explicit Impl(const YAML::Node& config) {
        params_.load(config);
    }

    cv::Mat process(const cv::Mat& input, cv::Mat* roi_out, cv::Mat* static_removed_out) {
        if (input.empty())
            return {};

        // 1. 中心裁剪
        int x = (input.cols - params_.crop_size) / 2;
        int y = (input.rows - params_.crop_size) / 2;
        x = std::max(0, x);
        y = std::max(0, y);
        int w = std::min(params_.crop_size, input.cols - x);
        int h = std::min(params_.crop_size, input.rows - y);
        cv::Mat cropped = input(cv::Rect(x, y, w, h));

        // 2. 缩放
        cv::Mat resized;
        cv::resize(
            cropped,
            resized,
            cv::Size(params_.output_w, params_.output_h),
            0,
            0,
            cv::INTER_LINEAR
        );
        if (roi_out) {
            resized.copyTo(*roi_out);
        }

        cv::Mat working = resized;
        if (!params_.static_simplify) {
            if (static_removed_out) {
                working.copyTo(*static_removed_out);
            }
            return working;
        }

        // 3. 动静分离 (static_simplify = true)
        cv::Mat gray;
        cv::cvtColor(working, gray, cv::COLOR_BGR2GRAY);

        if (background_gray_f32_.empty()) {
            gray.convertTo(background_gray_f32_, CV_32F);
            return working;
        }

        cv::Mat bg_u8;
        cv::convertScaleAbs(background_gray_f32_, bg_u8);

        cv::Mat diff;
        cv::absdiff(gray, bg_u8, diff);

        cv::Mat motion_mask;
        cv::threshold(diff, motion_mask, params_.motion_threshold, 255, cv::THRESH_BINARY);

        // 形态学操作
        if (params_.motion_erode_px > 0) {
            if (motion_erode_kernel_.empty()) {
                int k = 2 * params_.motion_erode_px + 1;
                motion_erode_kernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
            }
            cv::erode(motion_mask, motion_mask, motion_erode_kernel_, cv::Point(-1, -1), 1);
        }
        if (params_.motion_dilate_px > 0) {
            if (motion_dilate_kernel_.empty()) {
                int k = 2 * params_.motion_dilate_px + 1;
                motion_dilate_kernel_ =
                    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
            }
            cv::dilate(motion_mask, motion_mask, motion_dilate_kernel_, cv::Point(-1, -1), 1);
        }

        double motion_ratio_raw =
            static_cast<double>(cv::countNonZero(motion_mask)) / motion_mask.total();
        bool suppress_trail = (motion_ratio_raw >= params_.trail_disable_motion_ratio);

        // 中心保护区域
        if (params_.center_clear_size > 0) {
            int clear = std::min({ params_.center_clear_size, working.cols, working.rows });
            int x0 = std::max(0, working.cols / 2 - clear / 2);
            int y0 = std::max(0, working.rows / 2 - clear / 2);
            int cw = std::min(clear, working.cols - x0);
            int ch = std::min(clear, working.rows - y0);
            cv::rectangle(motion_mask, cv::Rect(x0, y0, cw, ch), cv::Scalar(255), cv::FILLED);
        }

        // 决定彩色掩码
        cv::Mat color_mask;
        if (params_.force_monochrome) {
            if (params_.center_clear_size > 0) {
                color_mask = cv::Mat::zeros(working.size(), CV_8UC1);
                int clear = std::min({ params_.center_clear_size, working.cols, working.rows });
                int x0 = std::max(0, working.cols / 2 - clear / 2);
                int y0 = std::max(0, working.rows / 2 - clear / 2);
                int cw = std::min(clear, working.cols - x0);
                int ch = std::min(clear, working.rows - y0);
                cv::rectangle(color_mask, cv::Rect(x0, y0, cw, ch), cv::Scalar(255), cv::FILLED);
            } else {
                color_mask = cv::Mat::zeros(working.size(), CV_8UC1);
            }
        } else {
            color_mask = motion_mask;
        }

        // 背景灰度化 + 模糊
        cv::Mat static_base = working.clone();
        if (params_.force_monochrome) {
            cv::Mat gray_bg;
            cv::cvtColor(static_base, gray_bg, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray_bg, static_base, cv::COLOR_GRAY2BGR);
        }

        cv::Mat blurred_static;
        double sigma = std::max(0.1, params_.bg_blur_sigma);
        cv::GaussianBlur(static_base, blurred_static, cv::Size(), sigma, sigma);

        cv::Mat focused = blurred_static.clone();
        working.copyTo(focused, color_mask);

        if (static_removed_out) {
            focused.copyTo(*static_removed_out);
        }

        // 运动拖影
        if (params_.motion_trail_frames > 0) {
            motion_mask_history_.push_back(motion_mask.clone());
            trail_frame_history_.push_back(working.clone());
            size_t max_hist = static_cast<size_t>(params_.motion_trail_frames + 1);
            while (motion_mask_history_.size() > max_hist)
                motion_mask_history_.pop_front();
            while (trail_frame_history_.size() > max_hist)
                trail_frame_history_.pop_front();

            if (!suppress_trail && motion_mask_history_.size() > 1
                && motion_mask_history_.size() == trail_frame_history_.size())
            {
                cv::Mat trail_mask = motion_mask.clone();
                cv::Mat trail_img = working.clone();
                for (size_t i = 0; i < motion_mask_history_.size() - 1; ++i) {
                    cv::bitwise_or(trail_mask, motion_mask_history_[i], trail_mask);
                    cv::max(trail_img, trail_frame_history_[i], trail_img);
                }
                trail_img.copyTo(focused, trail_mask);
            }
        } else {
            motion_mask_history_.clear();
            trail_frame_history_.clear();
        }

        // 更新背景模型
        cv::accumulateWeighted(gray, background_gray_f32_, params_.bg_update_alpha);

        return focused;
    }
};

ImagePreprocessor::ImagePreprocessor(const YAML::Node& config):
    _impl(std::make_unique<Impl>(config)) {}

ImagePreprocessor::~ImagePreprocessor() noexcept = default;

cv::Mat
ImagePreprocessor::process(const cv::Mat& input, cv::Mat* roi_out, cv::Mat* static_removed_out) {
    return _impl->process(input, roi_out, static_removed_out);
}

} // namespace awakening::eyes_of_blind