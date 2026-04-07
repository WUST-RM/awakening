#pragma once
#include "type_common.hpp"
namespace awakening {
enum class PixelFormat : int { BGR = 0, GRAY, RGB };
inline PixelFormat string2PixelFormat(const std::string& str) {
    if (str == "BGR" || str == "bgr")
        return PixelFormat::BGR;
    if (str == "GRAY" || str == "gray")
        return PixelFormat::GRAY;
    if (str == "RGB" || str == "rgb")
        return PixelFormat::RGB;

    return PixelFormat::BGR; // Default
}
struct ImageFrame {
    cv::Mat src_img;
    PixelFormat format;
    TimePoint timestamp;
    ImageFrame clone() const {
        return ImageFrame { src_img.clone(), format, timestamp };
    }
    std::vector<uint8_t> serialize(int quality = 55) const {
        std::vector<uint8_t> buffer;

        // 压缩图像为 JPEG
        std::vector<uint8_t> img_data;
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, quality };
        cv::imencode(".jpg", src_img, img_data, params);

        // 保存压缩图像大小
        uint32_t data_size = static_cast<uint32_t>(img_data.size());
        buffer.insert(
            buffer.end(),
            reinterpret_cast<uint8_t*>(&data_size),
            reinterpret_cast<uint8_t*>(&data_size) + sizeof(data_size)
        );

        // 保存压缩图像数据
        buffer.insert(buffer.end(), img_data.begin(), img_data.end());

        // 保存 PixelFormat
        int fmt = static_cast<int>(format);
        buffer.insert(
            buffer.end(),
            reinterpret_cast<uint8_t*>(&fmt),
            reinterpret_cast<uint8_t*>(&fmt) + sizeof(fmt)
        );

        // 保存 timestamp
        int64_t ts = timestamp.time_since_epoch().count();
        buffer.insert(
            buffer.end(),
            reinterpret_cast<uint8_t*>(&ts),
            reinterpret_cast<uint8_t*>(&ts) + sizeof(ts)
        );

        return buffer;
    }

    // ---------------- 反序列化 ----------------
    static ImageFrame deserialize(const std::vector<uint8_t>& buffer) {
        ImageFrame frame;
        size_t offset = 0;

        // 压缩图像大小
        uint32_t data_size;
        std::memcpy(&data_size, buffer.data() + offset, sizeof(data_size));
        offset += sizeof(data_size);

        // 压缩图像数据
        std::vector<uint8_t> img_data(buffer.begin() + offset, buffer.begin() + offset + data_size);
        offset += data_size;

        // 解码 JPEG
        frame.src_img = cv::imdecode(img_data, cv::IMREAD_COLOR);

        // PixelFormat
        int fmt;
        std::memcpy(&fmt, buffer.data() + offset, sizeof(fmt));
        frame.format = static_cast<PixelFormat>(fmt);
        offset += sizeof(fmt);

        // timestamp
        int64_t ts;
        std::memcpy(&ts, buffer.data() + offset, sizeof(ts));
        frame.timestamp = Clock::now();
        offset += sizeof(ts);

        return frame;
    }
};
} // namespace awakening