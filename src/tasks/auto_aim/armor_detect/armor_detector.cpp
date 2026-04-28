#include "armor_detector.hpp"
#include "armor_infer.hpp"
#include "tasks/auto_aim/type.hpp"
#include "utils/logger.hpp"
#include "utils/net_detector/net_detector_base.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#if USE_OPENVINO
    #include "utils/net_detector/openvino/net_detector_openvino.hpp"
#endif
#ifdef USE_TRT
    #include "utils/net_detector/tensorrt/net_detector_tensorrt.hpp"
#endif
#include <fstream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <utility>
#include <vector>
namespace awakening::auto_aim {
struct ArmorDetector::Impl {
    static constexpr const char* OPENVINO = "openvino";
    static constexpr const char* TENSORRT = "tensorrt";
    struct Params {
        struct NumberClassifierParams {
            std::string model_path;
            void load(const YAML::Node& config) {
                model_path = replace_root_dir(config["model_path"].as<std::string>());
            }
        };
        std::optional<NumberClassifierParams> number_classifier_params;
        struct ColorClassifierParams {
            double diff_threshold = 20.0;
            void load(const YAML::Node& config) {
                diff_threshold = config["diff_threshold"].as<double>();
            }
        };
        std::optional<ColorClassifierParams> color_classifier_params;
        void load(const YAML::Node& config) {
            if (config["number_classifier"]["enable"].as<bool>()) {
                number_classifier_params = NumberClassifierParams();
                number_classifier_params->load(config["number_classifier"]);
            }
            if (config["color_classifier"]["enable"].as<bool>()) {
                color_classifier_params = ColorClassifierParams();
                color_classifier_params->load(config["color_classifier"]);
            }
        }
    } params_;
    Impl(const YAML::Node& config) {
        params_.load(config);
        if (params_.number_classifier_params) {
            init_number_classifier();
        }
        armor_infer_ = ArmorInfer::create(config["armor_infer"]);
        auto backend = config["net_detector"]["backend"].as<std::string>();
        const double scale = armor_infer_->useNorm() ? 1.0 / 255.0f : 1.0f;
        auto format = armor_infer_->targetFormat();
        auto net_cfg = utils::NetDetectorBase::Config {
            .target_format = format,
            .preprocess_scale = scale,
            .target_w = armor_infer_->inputW(),
            .target_h = armor_infer_->inputH(),
        };
        bool backend_valid = false;
#ifdef USE_OPENVINO
        if (backend == OPENVINO) {
            backend_valid = true;
            net_detector_ = std::make_unique<utils::NetDetectorOpenVINO>(
                config["net_detector"][OPENVINO],
                net_cfg
            );
        }
#endif
#ifdef USE_TRT
        if (backend == TENSORRT) {
            backend_valid = true;
            net_detector_ = std::make_unique<utils::NetDetectorTensorrt>(
                config["net_detector"][TENSORRT],
                net_cfg
            );
        }
#endif
        if (!backend_valid) {
            throw std::runtime_error("Invalid backend");
        }
    }
    bool extract_number(const cv::Mat& src, Armor& armor) const noexcept {
        // Light length in image
        constexpr int light_length = 12;
        // Image size after warp
        constexpr int warp_height = 28;
        constexpr int small_armor_width = 32;
        constexpr int large_armor_width = 54;
        // Number ROI size
        const cv::Size roi_size(20, 28);
        constexpr float min_large_center_distance = 3.5f;

        if (src.empty() || src.cols < 10 || src.rows < 10) {
            AWAKENING_ERROR("[extractNumber] input src is empty or too small!");
            return false;
        }

        auto key_points = armor.net.key_points.points;

        const cv::Point2f& rb =
            key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_BOTTOM)].value();
        const cv::Point2f& rt =
            key_points[std::to_underlying(ArmorKeyPointsIndex::RIGHT_TOP)].value();
        const cv::Point2f& lt =
            key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_TOP)].value();
        const cv::Point2f& lb =
            key_points[std::to_underlying(ArmorKeyPointsIndex::LEFT_BOTTOM)].value();

        const float l1_len = cv::norm(rt - rb);
        const float l2_len = cv::norm(lt - lb);
        const cv::Point2f c1 = (rb + rt) * 0.5f;
        const cv::Point2f c2 = (lb + lt) * 0.5f;

        const float avg_light_len = 0.5f * (l1_len + l2_len);
        const float center_dist = avg_light_len > 1e-3f ? cv::norm(c1 - c2) / avg_light_len : 0.f;

        const bool is_large = center_dist > min_large_center_distance;

        cv::Point2f lights_vertices[4] = { lb, lt, rt, rb };

        const int top_light_y = (warp_height - light_length) / 2 - 1;
        const int bottom_light_y = top_light_y + light_length;
        const int warp_width = !is_large ? small_armor_width : large_armor_width;
        cv::Point2f target_vertices[4] = {
            cv::Point(0, bottom_light_y),
            cv::Point(0, top_light_y),
            cv::Point(warp_width - 1, top_light_y),
            cv::Point(warp_width - 1, bottom_light_y),
        };
        cv::Mat number_image;
        auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
        cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height));

        // Get ROI
        number_image =
            number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

        // Binarize
        cv::cvtColor(number_image, number_image, cv::COLOR_RGB2GRAY);
        cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        armor.number_classifier = Armor::NumberClassifierCtx();
        armor.number_classifier->number_img = std::move(number_image);
        return true;
    }
    void classify_color(const cv::Mat& src, Armor& armor, PixelFormat pixel_format) const noexcept {
        constexpr float light_width = 1.0f;
        constexpr float light_height = 5.0f;
        if (src.empty() || src.cols < 10 || src.rows < 10 || !params_.color_classifier_params) {
            AWAKENING_ERROR("[classifyColor] input src is empty or too small!");
            return;
        }
        auto& key_points = armor.net.key_points.points;
        auto getPt = [&](ArmorKeyPointsIndex idx) -> const cv::Point2f* {
            auto& opt = key_points[std::to_underlying(idx)];
            return opt ? &(*opt) : nullptr;
        };

        const cv::Point2f* rb = getPt(ArmorKeyPointsIndex::RIGHT_BOTTOM);
        const cv::Point2f* rt = getPt(ArmorKeyPointsIndex::RIGHT_TOP);
        const cv::Point2f* lt = getPt(ArmorKeyPointsIndex::LEFT_TOP);
        const cv::Point2f* lb = getPt(ArmorKeyPointsIndex::LEFT_BOTTOM);

        if (!rb || !rt || !lt || !lb)
            return;

        std::array<cv::RotatedRect, 2> lights_box;

        auto makeLight = [&](const cv::Point2f& top, const cv::Point2f& bottom) {
            cv::Point2f center = (top + bottom) * 0.5f;

            float height = cv::norm(top - bottom);
            float width = height / (light_height / light_width);

            float angle = std::atan2(top.y - bottom.y, top.x - bottom.x) * 180.0f / CV_PI;
            if (width > height) {
                std::swap(width, height);
                angle += 90.0f;
            }

            return cv::RotatedRect(center, cv::Size2f(width, height), angle);
        };

        lights_box[Armor::ColorClassifierCtx::RIGHT] = makeLight(*rt, *rb);
        lights_box[Armor::ColorClassifierCtx::LEFT] = makeLight(*lt, *lb);

        armor.color_classifier = Armor::ColorClassifierCtx();
        armor.color_classifier->lights_box = lights_box;

        armor.color_classifier = Armor::ColorClassifierCtx();
        armor.color_classifier->lights_box = lights_box;
        auto extractRotatedROI = [](const cv::Mat& src, const cv::RotatedRect& rect) {
            cv::Rect bbox = rect.boundingRect();
            bbox &= cv::Rect(0, 0, src.cols, src.rows);
            if (bbox.width <= 0 || bbox.height <= 0)
                return cv::Mat();
            return src(bbox);
        };

        auto judgeColor = [&](const cv::Mat& roi) {
            if (roi.empty() || roi.channels() < 3)
                return ArmorColor::NONE;
            cv::Scalar mean_val = cv::mean(roi);
            float R = 0.f, B = 0.f;
            switch (pixel_format) {
                case PixelFormat::BGR:
                    R = mean_val[2];
                    B = mean_val[0];
                    break;
                case PixelFormat::RGB:
                    B = mean_val[2];
                    R = mean_val[0];
                    break;
                case PixelFormat::GRAY:
                    return ArmorColor::NONE;
            }

            const float threshold = params_.color_classifier_params->diff_threshold;
            if (R - B > threshold)
                return ArmorColor::RED;
            if (B - R > threshold)
                return ArmorColor::BLUE;
            return ArmorColor::NONE;
        };
        for (int i = 0; i < 2; ++i) {
            cv::Mat roi = extractRotatedROI(src, lights_box[i]);
            armor.color_classifier->light_colors[i] = judgeColor(roi);
        }

        return;
    }

    void init_number_classifier() {
        if (!params_.number_classifier_params) {
            return;
        }
        const std::string model_path = params_.number_classifier_params->model_path;
        std::unique_ptr<cv::dnn::Net> number_net_ =
            std::make_unique<cv::dnn::Net>(cv::dnn::readNetFromONNX(model_path));

        if (number_net_->empty()) {
            throw std::runtime_error("Failed to load number classifier model" + model_path);
        } else {
            AWAKENING_DEBUG("Successfully loaded number classifier model from {}", model_path);
        }
        number_net_.reset();
    }
    bool classify_number_batch(std::vector<Armor*>& armors) const noexcept {
        static thread_local std::unique_ptr<cv::dnn::Net> thread_net;

        if (!thread_net) {
            thread_net = std::make_unique<cv::dnn::Net>(
                cv::dnn::readNetFromONNX(params_.number_classifier_params->model_path)
            );
            AWAKENING_DEBUG("Created thread-local number classifier model");
            if (thread_net->empty()) {
                AWAKENING_ERROR("Failed to load model");
                return false;
            }
        }

        std::vector<cv::Mat> images;
        std::vector<Armor*> valid_armors;

        for (auto* armor: armors) {
            if (armor && armor->number_classifier && !armor->number_classifier->number_img.empty())
            {
                images.emplace_back(armor->number_classifier->number_img);
                valid_armors.emplace_back(armor);
            }
        }

        if (images.empty())
            return false;

        cv::Mat blob;
        cv::dnn::blobFromImages(images, blob, 1.0 / 255.0);

        thread_net->setInput(blob);
        cv::Mat outputs = thread_net->forward();

        static const std::array<ArmorClass, 8> label_map = {
            ArmorClass::NO1, ArmorClass::NO2,     ArmorClass::NO3,    ArmorClass::NO4,
            ArmorClass::NO5, ArmorClass::OUTPOST, ArmorClass::SENTRY, ArmorClass::BASE
        };

        for (int i = 0; i < outputs.rows; ++i) {
            cv::Mat logits = outputs.row(i);

            double max_val;
            cv::minMaxLoc(logits, nullptr, &max_val);

            cv::Mat prob;
            cv::exp(logits - max_val, prob);
            prob /= cv::sum(prob)[0];

            double confidence;
            cv::Point class_id;
            cv::minMaxLoc(prob, nullptr, &confidence, nullptr, &class_id);

            int label = class_id.x;

            auto* armor = valid_armors[i];
            armor->number_classifier->confidence = confidence;

            if (label >= 0 && label < (int)label_map.size()) {
                armor->number_classifier->number = label_map[label];
            } else {
                armor->number_classifier->number = ArmorClass::UNKNOWN;
            }
        }

        return true;
    }

    std::vector<Armor> detect(const CommonFrame& frame) const {
        std::vector<Armor> result;
        const auto& src_img = frame.img_frame.src_img;
        const auto roi = src_img(frame.expanded);

        auto net_output = net_detector_->detect(roi, frame.img_frame.format);
        result = armor_infer_->process(net_output.output);
        if (net_output.resized_img.empty()) {
            return result;
        }
        if (params_.number_classifier_params) {
            std::vector<Armor*> batch_armors;
            for (auto& armor: result) {
                bool ok = extract_number(net_output.resized_img, armor);
                if (ok) {
                    batch_armors.push_back(&armor);
                }
            }
            classify_number_batch(batch_armors);
        }

        for (auto& armor: result) {
            if (params_.color_classifier_params) {
                classify_color(net_output.resized_img, armor, frame.img_frame.format);
            }
            armor.tidy();
            armor.transform(net_output.transform_matrix);
            armor.add_offset(frame.offset);
        }

        return result;
    }

    utils::NetDetectorBase::Ptr net_detector_;
    ArmorInfer::Ptr armor_infer_;
};
ArmorDetector::ArmorDetector(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}
ArmorDetector::~ArmorDetector() noexcept {
    _impl.reset();
}
std::vector<Armor> ArmorDetector::detect(const CommonFrame& frame) {
    return _impl->detect(frame);
}
} // namespace awakening::auto_aim