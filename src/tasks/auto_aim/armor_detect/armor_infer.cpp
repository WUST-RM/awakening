#include "armor_infer.hpp"
#include "tasks/auto_aim/type.hpp"
#include "utils/common/image.hpp"
#include <cstddef>
#include <memory>

namespace awakening::auto_aim {
static constexpr float MERGE_CONF_ERROR = 0.95f;
static constexpr float MERGE_MIN_IOU = 0.9f;
static constexpr float NMS_THRESHOLD = 0.35;
static constexpr int TOP_K = 128;
enum class Mode : int { TUP, RP, AT };
inline Mode modeFromString(const std::string& s) noexcept {
    std::string str = utils::to_upper(s);
    if (str == "TUP")
        return Mode::TUP;
    if (str == "RP")
        return Mode::RP;
    if (str == "AT")
        return Mode::AT;
    return Mode::TUP;
}
template<Mode M>
struct ModelTraits; // declare
// TUP
template<>
struct ModelTraits<Mode::TUP> {
    static constexpr int INPUT_W = 416;
    static constexpr int INPUT_H = 416;
    static constexpr int NUM_CLASSES = 8;
    static constexpr int NUM_COLORS = 4;
    static constexpr bool USE_NORM = false;
    static constexpr PixelFormat TARGET_FORMAT = PixelFormat::BGR;
    static constexpr std::array CLASSES = { ArmorClass::SENTRY,  ArmorClass::NO1,
                                            ArmorClass::NO2,     ArmorClass::NO3,
                                            ArmorClass::NO4,     ArmorClass::NO5,
                                            ArmorClass::OUTPOST, ArmorClass::BASE,
                                            ArmorClass::UNKNOWN };
    static constexpr std::array COLORS = { ArmorColor::BLUE,
                                           ArmorColor::RED,
                                           ArmorColor::PURPLE,
                                           ArmorColor::NONE };
};

// RP
template<>
struct ModelTraits<Mode::RP> {
    static constexpr int INPUT_W = 640;
    static constexpr int INPUT_H = 640;
    static constexpr int NUM_CLASSES = 9;
    static constexpr int NUM_COLORS = 4;
    static constexpr bool USE_NORM = true;
    static constexpr PixelFormat TARGET_FORMAT = PixelFormat::RGB;
    static constexpr std::array CLASSES = { ArmorClass::SENTRY,  ArmorClass::NO1,
                                            ArmorClass::NO2,     ArmorClass::NO3,
                                            ArmorClass::NO4,     ArmorClass::NO5,
                                            ArmorClass::OUTPOST, ArmorClass::BASE,
                                            ArmorClass::UNKNOWN };
    static constexpr std::array COLORS = { ArmorColor::BLUE,
                                           ArmorColor::RED,
                                           ArmorColor::PURPLE,
                                           ArmorColor::NONE };
};
template<>
struct ModelTraits<Mode::AT> {
    static constexpr int INPUT_W = 640;
    static constexpr int INPUT_H = 640;
    static constexpr int NUM_KPTS = 4;
    static constexpr bool USE_NORM = true;
    static constexpr PixelFormat TARGET_FORMAT = PixelFormat::RGB;
    static constexpr std::array<std::pair<ArmorColor, ArmorClass>, 64> CLASSES = { {
        { ArmorColor::BLUE, ArmorClass::SENTRY },    { ArmorColor::BLUE, ArmorClass::NO1 },
        { ArmorColor::BLUE, ArmorClass::NO2 },       { ArmorColor::BLUE, ArmorClass::NO3 },
        { ArmorColor::BLUE, ArmorClass::NO4 },       { ArmorColor::BLUE, ArmorClass::NO5 },
        { ArmorColor::BLUE, ArmorClass::OUTPOST },   { ArmorColor::BLUE, ArmorClass::BASE },
        { ArmorColor::BLUE, ArmorClass::SENTRY },    { ArmorColor::BLUE, ArmorClass::NO1 },
        { ArmorColor::BLUE, ArmorClass::NO2 },       { ArmorColor::BLUE, ArmorClass::NO3 },
        { ArmorColor::BLUE, ArmorClass::NO4 },       { ArmorColor::BLUE, ArmorClass::NO5 },
        { ArmorColor::BLUE, ArmorClass::OUTPOST },   { ArmorColor::BLUE, ArmorClass::BASE },
        { ArmorColor::RED, ArmorClass::SENTRY },     { ArmorColor::RED, ArmorClass::NO1 },
        { ArmorColor::RED, ArmorClass::NO2 },        { ArmorColor::RED, ArmorClass::NO3 },
        { ArmorColor::RED, ArmorClass::NO4 },        { ArmorColor::RED, ArmorClass::NO5 },
        { ArmorColor::RED, ArmorClass::OUTPOST },    { ArmorColor::RED, ArmorClass::BASE },
        { ArmorColor::RED, ArmorClass::SENTRY },     { ArmorColor::RED, ArmorClass::NO1 },
        { ArmorColor::RED, ArmorClass::NO2 },        { ArmorColor::RED, ArmorClass::NO3 },
        { ArmorColor::RED, ArmorClass::NO4 },        { ArmorColor::RED, ArmorClass::NO5 },
        { ArmorColor::RED, ArmorClass::OUTPOST },    { ArmorColor::RED, ArmorClass::BASE },
        { ArmorColor::NONE, ArmorClass::SENTRY },    { ArmorColor::NONE, ArmorClass::NO1 },
        { ArmorColor::NONE, ArmorClass::NO2 },       { ArmorColor::NONE, ArmorClass::NO3 },
        { ArmorColor::NONE, ArmorClass::NO4 },       { ArmorColor::NONE, ArmorClass::NO5 },
        { ArmorColor::NONE, ArmorClass::OUTPOST },   { ArmorColor::NONE, ArmorClass::BASE },
        { ArmorColor::NONE, ArmorClass::SENTRY },    { ArmorColor::NONE, ArmorClass::NO1 },
        { ArmorColor::NONE, ArmorClass::NO2 },       { ArmorColor::NONE, ArmorClass::NO3 },
        { ArmorColor::NONE, ArmorClass::NO4 },       { ArmorColor::NONE, ArmorClass::NO5 },
        { ArmorColor::NONE, ArmorClass::OUTPOST },   { ArmorColor::NONE, ArmorClass::BASE },
        { ArmorColor::PURPLE, ArmorClass::SENTRY },  { ArmorColor::PURPLE, ArmorClass::NO1 },
        { ArmorColor::PURPLE, ArmorClass::NO2 },     { ArmorColor::PURPLE, ArmorClass::NO3 },
        { ArmorColor::PURPLE, ArmorClass::NO4 },     { ArmorColor::PURPLE, ArmorClass::NO5 },
        { ArmorColor::PURPLE, ArmorClass::OUTPOST }, { ArmorColor::PURPLE, ArmorClass::BASE },
        { ArmorColor::PURPLE, ArmorClass::SENTRY },  { ArmorColor::PURPLE, ArmorClass::NO1 },
        { ArmorColor::PURPLE, ArmorClass::NO2 },     { ArmorColor::PURPLE, ArmorClass::NO3 },
        { ArmorColor::PURPLE, ArmorClass::NO4 },     { ArmorColor::PURPLE, ArmorClass::NO5 },
        { ArmorColor::PURPLE, ArmorClass::OUTPOST }, { ArmorColor::PURPLE, ArmorClass::BASE },
    } };
};
[[nodiscard]] inline double sigmoid(double x) noexcept {
    return x >= 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

[[nodiscard]] inline float rect_ioU(const cv::Rect2f& a, const cv::Rect2f& b) noexcept {
    const cv::Rect2f inter = a & b;
    const float inter_area = inter.area();
    const float union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0.f || std::isnan(union_area))
        return 0.f;
    return inter_area / union_area;
}
inline void nms_merge_sorted_bboxes(
    std::vector<Armor>& objs,
    std::vector<int>& out_indices,
    float nms_threshold
) {
    out_indices.clear();
    const size_t n = objs.size();

    for (size_t i = 0; i < n; ++i) {
        Armor& a = objs[i];
        bool keep = true;
        for (int idx: out_indices) {
            Armor& b = objs[idx];
            const float iou =
                rect_ioU(a.net.key_points.bounding_box(), b.net.key_points.bounding_box());
            if (std::isnan(iou) || iou > nms_threshold) {
                keep = false;
                if (a.number == b.number && a.color == b.color && iou > MERGE_MIN_IOU
                    && std::abs(a.net.confidence - b.net.confidence) < MERGE_CONF_ERROR)
                {
                    // accumulate points for later averaging
                    b.net.tmp_points.push_back(a.net.key_points.points);
                }
                break;
            }
        }
        if (keep)
            out_indices.push_back(static_cast<int>(i));
    }
}

inline std::vector<Armor> topk_and_nms(std::vector<Armor>& objs) {
    std::sort(objs.begin(), objs.end(), [](const Armor& a, const Armor& b) {
        return a.net.confidence > b.net.confidence;
    });

    if (static_cast<int>(objs.size()) > TOP_K)
        objs.resize(static_cast<size_t>(TOP_K));

    std::vector<int> indices;
    nms_merge_sorted_bboxes(objs, indices, NMS_THRESHOLD);

    std::vector<Armor> result;
    result.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); ++i) {
        result.push_back(std::move(objs[indices[i]]));
        auto& ro = result.back();
        if (ro.net.tmp_points.size() >= 1) {
            constexpr size_t N = std::to_underlying(ArmorKeyPointsIndex::N);
            std::array<cv::Point2f, N> accum {};
            std::array<int, N> count {};
            const auto& base_pts_opt = ro.net.key_points.points;
            for (size_t k = 0; k < N; ++k) {
                if (base_pts_opt[k]) {
                    accum[k] += *base_pts_opt[k];
                    count[k]++;
                }
            }
            for (const auto& pts_opt: ro.net.tmp_points) {
                for (size_t k = 0; k < N; ++k) {
                    if (pts_opt[k]) {
                        accum[k] += *pts_opt[k];
                        count[k]++;
                    }
                }
            }
            std::array<std::optional<cv::Point2f>, std::to_underlying(ArmorKeyPointsIndex::N)>
                final_pts {};
            for (size_t k = 0; k < std::to_underlying(ArmorKeyPointsIndex::N); ++k) {
                if (count[k] > 0) {
                    if (base_pts_opt[k]) {
                        final_pts[k] = accum[k] / static_cast<float>(count[k]);
                    }
                }
            }
            ro.net.key_points.points = final_pts;
            ro.net.tmp_points.clear();
        }
    }

    return result;
}
struct ArmorInfer::Impl {
    struct Params {
        double conf_threshold = 0.1;
        Mode mode;
        int input_w;
        int input_h;
        bool use_norm;
        PixelFormat target_format;
        template<typename M>
        void setMode() {
            input_w = M::INPUT_W;
            input_h = M::INPUT_H;
            use_norm = M::USE_NORM;
            target_format = M::TARGET_FORMAT;
        }
        void load(const YAML::Node& config) {
            auto mode_str = config["model_type"].as<std::string>();
            mode = modeFromString(mode_str);
            switch (mode) {
                case Mode::TUP: {
                    setMode<ModelTraits<Mode::TUP>>();
                    break;
                }
                case Mode::RP: {
                    setMode<ModelTraits<Mode::RP>>();
                    break;
                }
                case Mode::AT: {
                    setMode<ModelTraits<Mode::AT>>();
                    break;
                }
            }
            conf_threshold = config["conf_threshold"].as<double>();
        }
    } params_;
    Impl(const YAML::Node& config) {
        params_.load(config);
    }

    [[nodiscard]] std::vector<Armor> process(const cv::Mat& output_buffer) const {
        std::vector<Armor> results;
        results = post_process(output_buffer);
        return results;
    }
    [[nodiscard]] std::vector<Armor> post_process(const cv::Mat& output_buffer) const {
        if (output_buffer.empty())
            return {};
        switch (params_.mode) {
            case Mode::TUP:
                return post_processTUP_impl(output_buffer);
            case Mode::RP:
                return {};
            case Mode::AT:
                return post_processAT_impl(output_buffer);
        }
        return {};
    }
    std::vector<Armor> post_processTUP_impl(const cv::Mat& out) const {
        struct GridAndStride {
            int grid0;
            int grid1;
            int stride;
        };
        static std::optional<std::vector<GridAndStride>> _grid_strides;
        if (!_grid_strides) {
            auto generate_grids_and_stride = [&]() {
                std::vector<GridAndStride> grid_strides;
                for (int stride: { 8, 16, 32 }) {
                    const int num_w = inputW() / stride;
                    const int num_h = inputH() / stride;
                    grid_strides.reserve(grid_strides.size() + num_w * num_h);
                    for (int gy = 0; gy < num_h; ++gy) {
                        for (int gx = 0; gx < num_w; ++gx) {
                            grid_strides.push_back(GridAndStride { gx, gy, stride });
                        }
                    }
                }
                return grid_strides;
            };
            _grid_strides = generate_grids_and_stride();
        }
        const auto& grid_strides = _grid_strides.value();
        std::vector<Armor> out_objs;
        const int num_anchors =
            static_cast<int>(std::min<size_t>(grid_strides.size(), static_cast<size_t>(out.rows)));
        using I = ArmorKeyPointsIndex;
        for (int a = 0; a < num_anchors; ++a) {
            const float confidence = out.at<float>(a, 8);
            if (confidence < params_.conf_threshold) {
                continue;
            }

            const auto& gs = grid_strides[a];
            const int gx = gs.grid0, gy = gs.grid1, stride = gs.stride;

            // color & class
            const int color_offset = 9;
            const int num_colors = ModelTraits<Mode::TUP>::NUM_COLORS;
            const int num_classes = ModelTraits<Mode::TUP>::NUM_CLASSES;

            cv::Mat color_scores = out.row(a).colRange(color_offset, color_offset + num_colors);
            cv::Mat class_scores = out.row(a).colRange(
                color_offset + num_colors,
                color_offset + num_colors + num_classes
            );

            double max_color, max_class;
            cv::Point color_id, class_id;
            cv::minMaxLoc(color_scores, nullptr, &max_color, nullptr, &color_id);
            cv::minMaxLoc(class_scores, nullptr, &max_class, nullptr, &class_id);

            const float x1 = (out.at<float>(a, 0) + gx) * stride;
            const float y1 = (out.at<float>(a, 1) + gy) * stride;
            const float x2 = (out.at<float>(a, 2) + gx) * stride;
            const float y2 = (out.at<float>(a, 3) + gy) * stride;
            const float x3 = (out.at<float>(a, 4) + gx) * stride;
            const float y3 = (out.at<float>(a, 5) + gy) * stride;
            const float x4 = (out.at<float>(a, 6) + gx) * stride;
            const float y4 = (out.at<float>(a, 7) + gy) * stride;

            Armor obj;
            auto& net = obj.net;
            net = Armor::NetCtx();
            net.color = ModelTraits<Mode::TUP>::COLORS[color_id.x];
            net.number = ModelTraits<Mode::TUP>::CLASSES[class_id.x];
            auto& key_points = net.key_points;
            key_points.points[std::to_underlying(I::LEFT_TOP)] = cv::Point2f(x1, y1);
            key_points.points[std::to_underlying(I::LEFT_BOTTOM)] = cv::Point2f(x2, y2);
            key_points.points[std::to_underlying(I::RIGHT_BOTTOM)] = cv::Point2f(x3, y3);
            key_points.points[std::to_underlying(I::RIGHT_TOP)] = cv::Point2f(x4, y4);
            // net.tmp_points.push_back(key_points.points);
            net.confidence = confidence;
            out_objs.push_back(std::move(obj));
        }
        return topk_and_nms(out_objs);
    }
    std::vector<Armor> post_processAT_impl(const cv::Mat& out) const {
        std::vector<Armor> out_objs;

        constexpr int nkpt = ModelTraits<Mode::AT>::NUM_KPTS;
        constexpr int nk = nkpt * 2; // keypoints flattened
        auto max_det = out.rows;
        auto det_dim = out.cols;
        auto output_ptr = out.ptr<float>();
        using I = ArmorKeyPointsIndex;
        for (int i = 0; i < max_det; ++i) {
            const float* row = output_ptr + i * det_dim;
            float conf = row[4];
            if (!std::isfinite(conf) || conf < params_.conf_threshold)
                continue;

            float x1 = row[0];
            float y1 = row[1];
            float x2 = row[2];
            float y2 = row[3];
            int cls = static_cast<int>(row[5]);
            Armor obj;
            auto& net = obj.net;
            net = Armor::NetCtx();

            net.confidence = conf;
            auto color_num = ModelTraits<Mode::AT>::CLASSES[cls];
            net.color = color_num.first;
            net.number = color_num.second;
            auto getKeyPoints = [&](int k) {
                float kx = row[6 + 2 * k];
                float ky = row[6 + 2 * k + 1];
                return cv::Point2f(kx, ky);
            };
            auto& key_points = net.key_points;
            key_points.points[std::to_underlying(I::LEFT_TOP)] = getKeyPoints(0);
            key_points.points[std::to_underlying(I::LEFT_BOTTOM)] = getKeyPoints(1);
            key_points.points[std::to_underlying(I::RIGHT_BOTTOM)] = getKeyPoints(2);
            key_points.points[std::to_underlying(I::RIGHT_TOP)] = getKeyPoints(3);

            out_objs.emplace_back(std::move(obj));
        }

        return out_objs;
    }

    int inputW() const noexcept {
        return params_.input_w;
    }
    int inputH() const noexcept {
        return params_.input_h;
    }
    bool useNorm() const noexcept {
        return params_.use_norm;
    }
    PixelFormat targetFormat() const noexcept {
        return params_.target_format;
    }
};

ArmorInfer::ArmorInfer(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}
ArmorInfer::~ArmorInfer() noexcept {
    _impl.reset();
}
[[nodiscard]] std::vector<Armor> ArmorInfer::process(const cv::Mat& output_buffer) const {
    return _impl->process(output_buffer);
}

int ArmorInfer::inputW() const noexcept {
    return _impl->inputW();
}
int ArmorInfer::inputH() const noexcept {
    return _impl->inputH();
}
bool ArmorInfer::useNorm() const noexcept {
    return _impl->useNorm();
}
PixelFormat ArmorInfer::targetFormat() const noexcept {
    return _impl->targetFormat();
}
} // namespace awakening::auto_aim