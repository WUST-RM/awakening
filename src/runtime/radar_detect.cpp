#include "ascii_banner.hpp"
#include "tasks/base/common.hpp"
#include "tasks/radar_detect/detector.hpp"
#include "utils/common/image.hpp"
#include "utils/common/type_common.hpp"
#include "utils/logger.hpp"
#include "utils/signal_guard.hpp"
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <optional>
#include <yaml-cpp/node/parse.h>
using namespace awakening;

int main(int argc, char** argv) {
    print_banner();
    auto& signal = utils::SignalGuard::instance();
    logger::init(spdlog::level::trace);
    auto get_arg = [&](int i) -> std::optional<std::string> {
        if (i < argc) {
            AWAKENING_INFO("get args {} ", std::string(argv[i]));
            return std::make_optional(std::string(argv[i]));
        }
        return std::nullopt;
    };
    bool debug = false;
    std::string config_path;
    auto first_arg = get_arg(1);
    if (first_arg) {
        config_path = first_arg.value();
    } else {
        return 1;
    }
    auto config = YAML::LoadFile(config_path);
    radar_detect::Detector detector(config["detector"]);
    cv::VideoCapture cap("/home/hy/data/video_save/合工业1.avi");
    cap.set(cv::CAP_PROP_FPS, 60);

    if (!cap.isOpened()) {
        AWAKENING_ERROR("Failed to open video file.");
        return 1; // Exit if video cannot be opened
    }

    // Get the total number of frames in the video
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // Create a window for displaying video frames
    cv::namedWindow("Video Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Frame", 800, 600);

    // Create a trackbar (slider) to control video progress
    int slider_position = 0; // 当前的视频帧位置

    // 回调函数，当 Trackbar 发生变化时被调用
    auto trackbar_callback = [](int pos, void* userdata) {
        cv::VideoCapture* cap = (cv::VideoCapture*)userdata;
        cap->set(cv::CAP_PROP_POS_FRAMES, pos); // 设置视频播放到指定帧
    };
    if (total_frames > 0) {
        cv::createTrackbar(
            "Progress", // Trackbar 名称
            "Video Frame", // 显示窗口名称
            nullptr, // 不再使用 'value' 指针
            total_frames, // 最大值（视频的总帧数）
            trackbar_callback, // 回调函数
            &cap // 传递给回调函数的参数（视频文件句柄）
        );
    }

    // Loop to process frames
    while (true) {
        cv::Mat img;

        // Capture frame from video
        if (!cap.read(img)) {
            AWAKENING_INFO("End of video reached or failed to read frame.");
            break; // Break the loop if no more frames or error occurred
        }

        int current_frame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));

        // Draw the progress bar on the image (optional)
        cv::rectangle(
            img,
            cv::Point(0, img.rows - 20),
            cv::Point(img.cols, img.rows),
            cv::Scalar(0, 0, 0),
            -1
        ); // Background of progress bar
        float progress = static_cast<float>(current_frame) / total_frames;
        int progress_width = static_cast<int>(img.cols * progress);
        cv::rectangle(
            img,
            cv::Point(0, img.rows - 20),
            cv::Point(progress_width, img.rows),
            cv::Scalar(0, 255, 0),
            -1
        ); // Progress bar

        // Detect objects in the current frame
        CommonFrame frame;
        frame.img_frame.src_img = img;
        frame.img_frame.format = PixelFormat::BGR;
        frame.img_frame.timestamp = Clock::now();
        frame.expanded = cv::Rect(0, 0, img.cols, img.rows);
        frame.offset = cv::Point2f(0, 0);
        auto start = Clock::now();
        auto cars = detector.detect(frame);

        std::cout << cars.size() << std::endl;
        auto end = Clock::now();
        std::cout << "cost : "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms" << std::endl;
        for (const auto& car: cars) {
            car.draw(img);
        }
        // Show the image with the progress bar
        cv::imshow("Video Frame", img);

        // Update the slider position
        slider_position = current_frame;

        // Exit if user presses 'q'
        if (cv::waitKey(1) == 'q') {
            break;
        } else if (cv::waitKey(1) == 's') {
            cv::imwrite("out.jpg", img);
        }
    }

    // Release video capture and close windows
    cap.release();
    cv::destroyAllWindows();
}