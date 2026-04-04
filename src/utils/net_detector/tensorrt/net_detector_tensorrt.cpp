#include "net_detector_tensorrt.hpp"
#include "utils/buffer.hpp"
#include "utils/common/image.hpp"
#include "utils/cuda/letter_box.hpp"
#include "utils/logger.hpp"
#include <NvOnnxParser.h>
#include <array>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <string>
namespace awakening::utils {
#define TRT_CHECK(expr) \
    do { \
        const auto _ret = (expr); \
        if (_ret != cudaSuccess) { \
            std::cerr << "\033[31mCUDA error: " << cudaGetErrorString(_ret) << " (" << #expr \
                      << ")\033[0m\n"; \
            std::abort(); \
        } \
    } while (0)
struct NetDetectorTensorrt::Impl {
    struct Params {
        std::string model_path;
        int copy_context_num = 1;
        double min_free_mem_ratio = 0.1;
        bool use_cuda_preproces = true;
        void load(const YAML::Node& config) {
            model_path = config["model_path"].as<std::string>();
            copy_context_num = config["copy_context_num"].as<int>();
            min_free_mem_ratio = config["min_free_mem_ratio"].as<double>();
            use_cuda_preproces = config["use_cuda_preproces"].as<bool>();
        }
    } params_;
    class TRTLogger: public nvinfer1::ILogger {
    public:
        explicit TRTLogger(
            nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING
        ):
            severity_(severity) {}
        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
            if (severity <= severity_) {
                std::cerr << msg << std::endl;
            }
        }
        nvinfer1::ILogger::Severity severity_;
    };
    static int64_t volume(const nvinfer1::Dims& dims) {
        int64_t v = 1;
        for (int i = 0; i < dims.nbDims; ++i)
            v *= dims.d[i];
        return v;
    }
    Impl(const YAML::Node& config, Config c) {
        config_ = c;

        params_.load(config);
        buildEngine(params_.model_path);
        auto tmp_ctx =
            std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());

        if (!tmp_ctx)
            throw std::runtime_error("Failed to create execution context");
        input_name_ = engine_->getIOTensorName(0);
        output_name_ = engine_->getIOTensorName(1);
        input_dims_ = nvinfer1::Dims4 { 1,
                                        config_.target_format == PixelFormat::GRAY ? 1 : 3,
                                        config_.target_h,
                                        config_.target_w };
        tmp_ctx->setInputShape(input_name_, input_dims_);
        auto dims = tmp_ctx->getTensorShape(input_name_);
        if (dims.nbDims == -1) {
            throw std::runtime_error("Input shape not specified");
        }

        input_dims_ = dims;
        output_dims_ = tmp_ctx->getTensorShape(output_name_);
        tmp_ctx.reset();
        input_sz_ = volume(input_dims_);
        output_sz_ = volume(output_dims_);
        if (params_.copy_context_num < 1) {
            params_.copy_context_num = 1;
        }
        for (int i = 0; i < params_.copy_context_num; ++i) {
            if (i > 0) {
                size_t free_mem, total_mem;
                cudaMemGetInfo(&free_mem, &total_mem);

                AWAKENING_DEBUG("Free GPU memory: {} MB", free_mem / 1024.0 / 1024.0);
                AWAKENING_DEBUG("Total GPU memory: {} MB", total_mem / 1024.0 / 1024.0);
                double free_mem_ratio =
                    static_cast<double>(free_mem) / static_cast<double>(total_mem);
                if (free_mem_ratio < params_.min_free_mem_ratio && i > 0) {
                    AWAKENING_WARN(
                        "GPU memory is not enough! Free GPU memory: {:.2f}%",
                        free_mem_ratio * 100
                    );
                    break;
                }
            }

            Ctx ctx;
            ctx.context.reset(engine_->createExecutionContext());
            ctx.letter_box = std::make_shared<__cuda::LetterBox>(config_);
            TRT_CHECK(cudaMalloc(&ctx.device_buffers[input_idx_], input_sz_ * sizeof(float)));
            TRT_CHECK(cudaMalloc(&ctx.device_buffers[output_idx_], output_sz_ * sizeof(float)));
            ctx.output_buffer.resize(output_sz_);
            TRT_CHECK(cudaStreamCreate(&ctx.stream));
            ctx_buffers_.addResource(std::move(ctx));
        }
    }
    void buildEngine(const std::string& onnx_path) {
        const std::string engine_path =
            onnx_path.substr(0, onnx_path.find_last_of('.')) + ".engine";

        runtime_.reset(nvinfer1::createInferRuntime(g_logger_));

        {
            std::ifstream fin(engine_path, std::ios::binary);
            if (fin.good()) {
                fin.seekg(0, std::ios::end);
                const size_t size = fin.tellg();
                fin.seekg(0, std::ios::beg);

                std::vector<char> data(size);
                fin.read(data.data(), size);

                engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
                if (engine_) {
                    AWAKENING_INFO("Loaded TensorRT engine: {}", engine_path);
                    return;
                }
                AWAKENING_ERROR("Failed to load TensorRT engine: {}", engine_path);
            }
        }

        AWAKENING_INFO("Building TensorRT engine from ONNX...");
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger_));

        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));

        auto parser =
            std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, g_logger_));

        if (!parser->parseFromFile(
                onnx_path.c_str(),
                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)
            ))
        {
            AWAKENING_ERROR("Failed to parse ONNX: {}", onnx_path);
            throw std::runtime_error("Failed to parse ONNX: " + onnx_path);
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
        auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
            builder->buildSerializedNetwork(*network, *config)
        );
        auto profile = builder->createOptimizationProfile();

        int c = config_.target_format == PixelFormat::GRAY ? 1 : 3;
        nvinfer1::Dims4 dims { 1, c, config_.target_h, config_.target_w };

        const char* input_name = network->getInput(0)->getName();

        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, dims);
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, dims);
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, dims);

        config->addOptimizationProfile(profile);

        engine_.reset(runtime_->deserializeCudaEngine(serialized->data(), serialized->size()));

        if (!engine_)
            throw std::runtime_error("Engine build failed");

        std::ofstream fout(engine_path, std::ios::binary);
        fout.write(static_cast<const char*>(serialized->data()), serialized->size());

        AWAKENING_INFO("Engine built & saved: {}", engine_path);
    }

    OutPut detect(const cv::Mat& img, PixelFormat format) noexcept {
        if (img.empty()) {
            return {};
        }
        OutPut output;
        const float* cpu_blob_ptr;
        if (!params_.use_cuda_preproces) { // 最大化ctx利用率,该部分不需要ctx则暂时不请求c
            output.resized_img =
                utils::letterbox(img, output.transform_matrix, config_.target_w, config_.target_h);
            auto swap_rb = format != config_.target_format;
            const cv::Mat blob = cv::dnn::blobFromImage(
                output.resized_img,
                config_.preprocess_scale,
                cv::Size(config_.target_w, config_.target_h),
                cv::Scalar(0, 0, 0),
                swap_rb
            );
            cpu_blob_ptr = blob.ptr<float>();
        }
        {
            auto r = ctx_buffers_.acquire();
            if (!r) {
                return output;
            }
            auto& ctx = *r;
            if (params_.use_cuda_preproces) {
                auto tensor = ctx.letter_box->letterbox_pitched(
                    img.data,
                    format,
                    img.cols,
                    img.rows,
                    img.step,
                    output.transform_matrix,
                    ctx.stream
                );
                if (!tensor) {
                    return output;
                }
                ctx.device_buffers[input_idx_] = tensor;
                output.resized_img = ctx.letter_box->tensorToMat(
                    static_cast<float*>(ctx.device_buffers[input_idx_]),
                    ctx.stream,
                    format != config_.target_format
                );
            } else {
                TRT_CHECK(cudaMemcpyAsync(
                    ctx.device_buffers[input_idx_],
                    cpu_blob_ptr,
                    input_sz_ * sizeof(float),
                    cudaMemcpyHostToDevice,
                    ctx.stream
                ));
            }

            ctx.context->setTensorAddress(input_name_, ctx.device_buffers[input_idx_]);
            ctx.context->setTensorAddress(output_name_, ctx.device_buffers[output_idx_]);

            if (!ctx.context->enqueueV3(ctx.stream)) {
                AWAKENING_ERROR("enqueueV3 failed");
                return output;
            }

            TRT_CHECK(cudaMemcpyAsync(
                ctx.output_buffer.data(),
                ctx.device_buffers[output_idx_],
                output_sz_ * sizeof(float),
                cudaMemcpyDeviceToHost,
                ctx.stream
            ));

            cudaStreamSynchronize(ctx.stream);

            output.output =
                cv::Mat(output_dims_.d[1], output_dims_.d[2], CV_32F, ctx.output_buffer.data())
                    .clone();
        }

        return output;
    }
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    TRTLogger g_logger_;
    struct Ctx {
        std::shared_ptr<nvinfer1::IExecutionContext> context;
        std::array<void*, 2> device_buffers;
        std::vector<float> output_buffer;
        cudaStream_t stream { nullptr };
        __cuda::LetterBox::Ptr letter_box;
    };
    ResourcePool<Ctx> ctx_buffers_;

    int input_idx_ { 0 }, output_idx_ { 1 };
    size_t input_sz_ { 0 }, output_sz_ { 0 };

    nvinfer1::Dims input_dims_ {};
    nvinfer1::Dims output_dims_ {};
    const char* input_name_ { nullptr };
    const char* output_name_ { nullptr };
    Config config_;
};
NetDetectorTensorrt::NetDetectorTensorrt(const YAML::Node& config, Config c) {
    _impl = std::make_unique<NetDetectorTensorrt::Impl>(config, c);
}
NetDetectorTensorrt::~NetDetectorTensorrt() noexcept {
    _impl.reset();
}
NetDetectorTensorrt::OutPut
NetDetectorTensorrt::detect(const cv::Mat& img, PixelFormat format) noexcept {
    return _impl->detect(img, format);
}
} // namespace awakening::utils