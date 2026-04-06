#include "net_detector_openvino.hpp"
#include "openvino/openvino.hpp"
#include "utils/logger.hpp"
#include "utils/net_detector/net_detector_base.hpp"
#include "utils/utils.hpp"
#include <atomic>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/properties.hpp>
#include <optional>
#include <string>
#include <utility>
namespace awakening::utils {
struct NetDetectorOpenVINO::Impl {
    struct Params {
        std::string model_path;
        std::string device_name;

        std::optional<ov::hint::PerformanceMode> perf_mode;
        // ov::hint::Priority priority = ov::hint::Priority::MEDIUM;
        std::optional<ov::hint::SchedulingCoreType> scheduling_core_type;
        std::optional<ov::hint::ExecutionMode> execution_mode;
        static ov::hint::PerformanceMode perfModeFromString(const std::string& s) {
            auto v = to_upper(s);
            if (v == "LATENCY")
                return ov::hint::PerformanceMode::LATENCY;
            if (v == "THROUGHPUT")
                return ov::hint::PerformanceMode::THROUGHPUT;
            if (v == "CUMULATIVE_THROUGHPUT")
                return ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
            throw std::runtime_error("Invalid perf_mode: " + s);
        }

        static ov::hint::Priority priorityFromString(const std::string& s) {
            auto v = to_upper(s);
            if (v == "LOW")
                return ov::hint::Priority::LOW;
            if (v == "MEDIUM")
                return ov::hint::Priority::MEDIUM;
            if (v == "HIGH")
                return ov::hint::Priority::HIGH;
            throw std::runtime_error("Invalid priority: " + s);
        }

        static ov::hint::SchedulingCoreType coreTypeFromString(const std::string& s) {
            auto v = to_upper(s);
            if (v == "ANY_CORE")
                return ov::hint::SchedulingCoreType::ANY_CORE;
            if (v == "PCORE_ONLY")
                return ov::hint::SchedulingCoreType::PCORE_ONLY;
            if (v == "ECORE_ONLY")
                return ov::hint::SchedulingCoreType::ECORE_ONLY;
            throw std::runtime_error("Invalid scheduling_core_type: " + s);
        }

        static ov::hint::ModelDistributionPolicy modelDistFromString(const std::string& s) {
            auto v = to_upper(s);
            if (v == "TENSOR_PARALLEL")
                return ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL;
            if (v == "PIPELINE_PARALLEL")
                return ov::hint::ModelDistributionPolicy::PIPELINE_PARALLEL;
            throw std::runtime_error("Invalid model_distribution_policy: " + s);
        }

        static ov::hint::ExecutionMode execModeFromString(const std::string& s) {
            auto v = to_upper(s);
            if (v == "PERFORMANCE")
                return ov::hint::ExecutionMode::PERFORMANCE;
            if (v == "ACCURACY")
                return ov::hint::ExecutionMode::ACCURACY;
            throw std::runtime_error("Invalid execution_mode: " + s);
        }

        void load(const YAML::Node& config) {
            if (config["model_path"]) {
                model_path = replace_root_dir(config["model_path"].as<std::string>());
            }

            if (config["device_name"]) {
                device_name = config["device_name"].as<std::string>();
            }

            if (config["perf_mode"]) {
                perf_mode = perfModeFromString(config["perf_mode"].as<std::string>());
            }

            // if (config["priority"]) {
            //     priority = priorityFromString(config["priority"].as<std::string>());
            // }

            if (config["scheduling_core_type"]) {
                scheduling_core_type =
                    coreTypeFromString(config["scheduling_core_type"].as<std::string>());
            }

            if (config["execution_mode"]) {
                execution_mode = execModeFromString(config["execution_mode"].as<std::string>());
            }
        }

        [[nodiscard]] ov::AnyMap anyMap() const {
            return ov::AnyMap {
                ov::hint::performance_mode(perf_mode.value()),
                // ov::hint::model_priority(priority),
                ov::hint::scheduling_core_type(scheduling_core_type.value()),
                ov::hint::execution_mode(execution_mode.value()),
            };
        }
    } params_;

    Impl(const YAML::Node& config, Config c) {
        params_.load(config);
        config_ = c;
        init();
    }
    void init() {
        resetting_ = true;

        auto any_map = params_.anyMap();

        if (!ov_core_) {
            ov_core_ = std::make_unique<ov::Core>();
            ov_core_->set_property(params_.device_name, any_map);
        }

        model_ = ov_core_->read_model(params_.model_path);

        ov::preprocess::PrePostProcessor ppp(model_);

        auto toColor = [](PixelFormat fmt) {
            switch (fmt) {
                case PixelFormat::RGB:
                    return ov::preprocess::ColorFormat::RGB;
                case PixelFormat::BGR:
                    return ov::preprocess::ColorFormat::BGR;
                case PixelFormat::GRAY:
                    return ov::preprocess::ColorFormat::GRAY;
            }
            throw std::runtime_error("Unsupported pixel format");
        };
        ppp.input()
            .tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(toColor(input_format_));

        ppp.input()
            .preprocess()
            .convert_element_type(ov::element::f32)
            .scale((1.0 / config_.preprocess_scale))
            .convert_color(toColor(config_.target_format));
        ppp.input().model().set_layout("NCHW");
        ppp.output().tensor().set_element_type(ov::element::f32);

        model_ = ppp.build();

        compiled_model_ = std::make_unique<ov::CompiledModel>(
            ov_core_->compile_model(model_, params_.device_name, any_map)
        );
        AWAKENING_INFO("OpenVINO model : {} compiled successfully!", params_.model_path);
        resetting_ = false;
    }
    ov::InferRequest createInferRequest() noexcept {
        return compiled_model_->create_infer_request();
    }
    ov::Tensor infer(const ov::Tensor& input_tensor, ov::InferRequest& infer_request) noexcept {
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        return infer_request.get_output_tensor();
    }
    ov::Tensor infer(const ov::Tensor& input_tensor) noexcept {
        auto infer_request = compiled_model_->create_infer_request();

        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        return infer_request.get_output_tensor();
    }
    ov::Tensor infer_thread_local(const ov::Tensor& input_tensor) noexcept {
        static thread_local std::unique_ptr<ov::InferRequest> infer_request;
        if (!infer_request) {
            infer_request = std::make_unique<ov::InferRequest>(createInferRequest());
        }

        infer_request->set_input_tensor(input_tensor);
        infer_request->infer();
        return infer_request->get_output_tensor();
    }

    OutPut detect(const cv::Mat& img, PixelFormat format) noexcept {
        if (resetting_ || img.empty()) {
            return {};
        }

        if (format != input_format_) {
            input_format_ = format;
            init();
        }
        OutPut output;
        output.resized_img =
            utils::letterbox(img, output.transform_matrix, config_.target_w, config_.target_h);
        const auto input = compiled_model_->input();
        ov::Tensor input_tensor(
            input.get_element_type(),
            input.get_shape(),
            output.resized_img.data
        );

        const auto output_tensor = infer_thread_local(input_tensor);
        const auto& shape = output_tensor.get_shape();

        auto ptr = output_tensor.data<float>();

        if (shape.size() == 3) {
            output.output = cv::Mat(shape[1], shape[2], CV_32F, ptr).clone();
        } else if (shape.size() == 4) {
            output.output = cv::Mat(shape[2], shape[3], CV_32F, ptr).clone();
        }

        return output;
    }
    std::atomic<bool> resetting_;
    std::unique_ptr<ov::Core> ov_core_;
    std::unique_ptr<ov::CompiledModel> compiled_model_;
    std::shared_ptr<ov::Model> model_;
    PixelFormat input_format_ = PixelFormat::BGR;
    Config config_;
};
NetDetectorOpenVINO::NetDetectorOpenVINO(const YAML::Node& config, Config c) {
    _impl = std::make_unique<NetDetectorOpenVINO::Impl>(config, c);
}
NetDetectorOpenVINO::~NetDetectorOpenVINO() noexcept {
    _impl.reset();
}
NetDetectorOpenVINO::OutPut
NetDetectorOpenVINO::detect(const cv::Mat& img, PixelFormat format) noexcept {
    return _impl->detect(img, format);
}
} // namespace awakening::utils