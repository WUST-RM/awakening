#include "armor_detector.hpp"
#include "armor_infer.hpp"
#include "tasks/auto_aim/type.hpp"
#include "utils/logger.hpp"
#include "utils/net_detector/net_detector_base.hpp"
#include "utils/net_detector/openvino/net_detector_openvino.hpp"
#include <fstream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <utility>
#include <wust_vl/video/icamera.hpp>
namespace awakening::auto_aim {
struct ArmorDetector::Impl {
    static constexpr const char* OPENVINO = "openvino";
    struct Params {
        struct NumberClassifierParams {
            std::string model_path;
            std::string label_path;
            void load(const YAML::Node& config) {
                model_path = config["model_path"].as<std::string>();
                label_path = config["label_path"].as<std::string>();
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
            initNumberClassifier();
        }
        armor_infer_ = ArmorInfer::create(config["armor_infer"]);
        auto backend = config["net_detector"]["backend"].as<std::string>();

        if (backend == OPENVINO) {
            const double scale = armor_infer_->useNorm() ? 255.0f : 1.0f;
            auto format = armor_infer_->targetFormat();
            net_detector_ = std::make_unique<utils::NetDetectorOpenVINO>(
                config["net_detector"][OPENVINO],
                format,
                scale
            );
        } else {
            throw std::runtime_error("Invalid backend");
        }
    }
    bool extractNumber(const cv::Mat& src, Armor& armor) const noexcept {
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
    void classifyColor(const cv::Mat& src, Armor& armor, PixelFormat pixel_format) const noexcept {
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
            // cv::Mat M, rotated, cropped;
            // M = cv::getRotationMatrix2D(rect.center, rect.angle, 1.0);
            // cv::warpAffine(src, rotated, M, src.size(), cv::INTER_LINEAR);
            // cv::getRectSubPix(rotated, rect.size, rect.center, cropped);
            //  return cropped;
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

    void initNumberClassifier() {
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

        const std::string label_path = params_.number_classifier_params->label_path;
        std::ifstream label_file(label_path);
        std::string line;

        class_names_.clear();

        while (std::getline(label_file, line)) {
            class_names_.push_back(line);
        }

        if (class_names_.empty()) {
            throw std::runtime_error("Failed to load labels from " + label_path);
        } else {
            AWAKENING_DEBUG(
                "Successfully loaded {} labels from {}",
                class_names_.size(),
                label_path
            );
        }
        number_net_.reset();
    }
    bool classifyNumber(Armor& armor) const noexcept {
        static thread_local std::unique_ptr<cv::dnn::Net> thread_net;
        if (!armor.number_classifier || armor.number_classifier->number_img.empty()) {
            return false;
        }

        if (!thread_net) {
            thread_net = std::make_unique<cv::dnn::Net>(
                cv::dnn::readNetFromONNX(params_.number_classifier_params->model_path)
            );
            AWAKENING_DEBUG("Loaded number classifier model for this thread");
            if (thread_net->empty()) {
                AWAKENING_ERROR("Failed to load thread-local number classifier model.");
                return false;
            }
        }

        const cv::Mat image = armor.number_classifier->number_img;
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob, 1.0 / 255.0);

        thread_net->setInput(blob);
        cv::Mat outputs = thread_net->forward();
        double max_val;
        cv::minMaxLoc(outputs, nullptr, &max_val);

        cv::Mat prob;
        cv::exp(outputs - max_val, prob);
        prob /= cv::sum(prob)[0];

        double confidence;
        cv::Point class_id;
        cv::minMaxLoc(prob, nullptr, &confidence, nullptr, &class_id);

        const int label_id = class_id.x;
        armor.number_classifier->confidence = confidence;

        static const std::map<int, ArmorClass> label_to_armor_number = {
            { 0, ArmorClass::NO1 },    { 1, ArmorClass::NO2 }, { 2, ArmorClass::NO3 },
            { 3, ArmorClass::NO4 },    { 4, ArmorClass::NO5 }, { 5, ArmorClass::OUTPOST },
            { 6, ArmorClass::SENTRY }, { 7, ArmorClass::BASE }
        };

        if (label_id < 8 && label_to_armor_number.find(label_id) != label_to_armor_number.end()) {
            armor.number = label_to_armor_number.at(label_id);

            return true;
        } else {
            armor.number = ArmorClass::UNKNOWN;
            armor.number_classifier->confidence = confidence;
            return false;
        }
    }

    std::vector<Armor> detect(const CommonFrame& frame) const {
        std::vector<Armor> result;
        Eigen::Matrix3f transform_matrix;
        const auto& src_img = frame.img_frame.src_img;
        const auto roi = src_img(frame.expanded);
        cv::Mat resized_img =
            utils::letterbox(roi, transform_matrix, armor_infer_->inputW(), armor_infer_->inputH());

        auto net_output = net_detector_->detect(resized_img, frame.img_frame.format);

        result = armor_infer_->process(net_output);
        for (auto& armor: result) {
            if (params_.number_classifier_params) {
                bool ok = extractNumber(resized_img, armor);
                if (ok) {
                    classifyNumber(armor);
                }
            }
            if (params_.color_classifier_params) {
                classifyColor(resized_img, armor, frame.img_frame.format);
            }

            armor.tidy();
            armor.transform(transform_matrix);
            armor.addOffset(frame.offset);
        }
        return result;
    }

    utils::NetDetectorBase::Ptr net_detector_;
    ArmorInfer::Ptr armor_infer_;
    std::vector<std::string> class_names_;
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