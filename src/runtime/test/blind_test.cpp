#include "ascii_banner.hpp"
#include "tasks/eyes_of_blind/decoder.hpp"
#include "tasks/eyes_of_blind/encoder.hpp"
#include "utils/drivers/hik_camera.hpp"
#include "utils/drivers/serial_driver.hpp"
#include "utils/logger.hpp"
#include "utils/semaphore_guard.hpp"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/node/parse.h>
using namespace awakening;
struct CameraTag {};

using CameraIO = IOPair<CameraTag, ImageFrame>;
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
    Scheduler s;
    auto config = YAML::LoadFile(config_path);
    std::unique_ptr<SerialDriver> serial;

    if (config["serial"]["enable"].as<bool>()) {
        serial = std::make_unique<SerialDriver>(config["serial"], s);
    }

    auto camera_config = config["camera"];
    std::unique_ptr<HikCamera> camera;
    utils::SignalGuard::add_callback([&]() {
        if (camera) {
            camera->stop();
        }
    });

    camera = std::make_unique<HikCamera>(camera_config["hik_camera"], s);
    camera->init();
    if (!camera->running_) {
        return 0;
    }
    eyes_of_blind::Encoder encoder(config["encoder"]);
    eyes_of_blind::Decoder decoder;
    s.register_task<CameraIO>("blind", [&](CameraIO::second_type&& f) {
        if (f.src_img.empty()) {
            return;
        }
        static std::unique_ptr<std::counting_semaphore<>> detector_sem;
        if (!detector_sem) {
            detector_sem = std::make_unique<std::counting_semaphore<>>(1);
        }

        {
            bool got = detector_sem->try_acquire();
            utils::SemaphoreGuard guard(*detector_sem, got);
            if (got) {
                encoder.push_frame(f.src_img);
                eyes_of_blind::BlindSend pkg;
                while (true) {
                    if (encoder.try_pop_packet(pkg)) {
                        cv::Mat out;
                        if (serial) {
                            serial->write(utils::to_vector(pkg));
                        }
                        decoder.push_packet(pkg);
                        while (true) {
                            if (decoder.try_pop_frame(out)) {
                                cv::namedWindow("Decoded Frame", cv::WINDOW_NORMAL);
                                cv::imshow("Decoded Frame", out);
                                cv::waitKey(1);
                            } else {
                                break;
                            }
                        }

                    } else {
                        break;
                    }
                }
            }
        }
    });
    if (camera) {
        camera->start<CameraTag>("hik");
    }

    s.build();
    s.run();
    // encoder.start();
    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();

    for (int i = 0; i < 10; ++i) {
        AWAKENING_CRITICAL("改了东西记得同步其他有关的exe的src");
    }
    return 0;
}