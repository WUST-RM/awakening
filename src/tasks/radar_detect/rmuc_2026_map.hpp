#pragma once
#include "tasks/radar_detect/tracker.hpp"
#include "tasks/radar_detect/type.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <fstream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
namespace awakening::radar_detect {

class RMUC2026Map {
public:
    RMUC2026Map(const YAML::Node& config, SelfColor self_color) {
        auto map_path = config["map_path"].as<std::string>();
        image.image = cv::imread(map_path);
        image.self_color = self_color;
        d_factor_ = config["d_factor"].as<double>();
        cos_factor_ = config["cos_factor"].as<double>();
        auto guess_path = config["guess_path"].as<std::string>();
        auto guess_config = YAML::LoadFile(guess_path);
        SingleRobotGuess self_1(image, guess_config["self_1"]);
        SingleRobotGuess self_2(image, guess_config["self_2"]);
        SingleRobotGuess self_3(image, guess_config["self_3"]);
        SingleRobotGuess self_4(image, guess_config["self_4"]);
        SingleRobotGuess self_7(image, guess_config["self_7"]);
        SingleRobotGuess enemy_1(image, guess_config["enemy_1"]);
        SingleRobotGuess enemy_2(image, guess_config["enemy_2"]);
        SingleRobotGuess enemy_3(image, guess_config["enemy_3"]);
        SingleRobotGuess enemy_4(image, guess_config["enemy_4"]);
        SingleRobotGuess enemy_7(image, guess_config["enemy_7"]);
        if (self_color == SelfColor::RED) {
            self_1.set_car_class(CarClass::R1);
            robot_guesses[std::to_underlying(CarClass::R1)] = self_1;
            self_2.set_car_class(CarClass::R2);
            robot_guesses[std::to_underlying(CarClass::R2)] = self_2;
            self_3.set_car_class(CarClass::R3);
            robot_guesses[std::to_underlying(CarClass::R3)] = self_3;
            self_4.set_car_class(CarClass::R4);
            robot_guesses[std::to_underlying(CarClass::R4)] = self_4;
            self_7.set_car_class(CarClass::R7);
            robot_guesses[std::to_underlying(CarClass::R7)] = self_7;
            enemy_1.set_car_class(CarClass::B1);
            robot_guesses[std::to_underlying(CarClass::B1)] = enemy_1;
            enemy_2.set_car_class(CarClass::B2);
            robot_guesses[std::to_underlying(CarClass::B2)] = enemy_2;
            enemy_3.set_car_class(CarClass::B3);
            robot_guesses[std::to_underlying(CarClass::B3)] = enemy_3;
            enemy_4.set_car_class(CarClass::B4);
            robot_guesses[std::to_underlying(CarClass::B4)] = enemy_4;
            enemy_7.set_car_class(CarClass::B7);
            robot_guesses[std::to_underlying(CarClass::B7)] = enemy_7;
        } else {
            enemy_1.set_car_class(CarClass::R1);
            robot_guesses[std::to_underlying(CarClass::R1)] = enemy_1;
            enemy_2.set_car_class(CarClass::R2);
            robot_guesses[std::to_underlying(CarClass::R2)] = enemy_2;
            enemy_3.set_car_class(CarClass::R3);
            robot_guesses[std::to_underlying(CarClass::R3)] = enemy_3;
            enemy_4.set_car_class(CarClass::R4);
            robot_guesses[std::to_underlying(CarClass::R4)] = enemy_4;
            enemy_7.set_car_class(CarClass::R7);
            robot_guesses[std::to_underlying(CarClass::R7)] = enemy_7;
            self_1.set_car_class(CarClass::B1);
            robot_guesses[std::to_underlying(CarClass::B1)] = self_1;
            self_2.set_car_class(CarClass::B2);
            robot_guesses[std::to_underlying(CarClass::B2)] = self_2;
            self_3.set_car_class(CarClass::B3);
            robot_guesses[std::to_underlying(CarClass::B3)] = self_3;
            self_4.set_car_class(CarClass::B4);
            robot_guesses[std::to_underlying(CarClass::B4)] = self_4;
            self_7.set_car_class(CarClass::B7);
            robot_guesses[std::to_underlying(CarClass::B7)] = self_7;
        }
    }
    void edit() {
        for (auto& [key, guess]: robot_guesses) {
            guess.edit();
        }
    }
    Eigen::Vector3d predict_guess(
        const CarClass& cc,
        const Eigen::Vector3d& last_pos,
        const Eigen::Vector3d& v_vec
    ) noexcept {
        int key = std::to_underlying(cc);
        auto guess = robot_guesses[key];
        auto guesses = guess.get_guesses();
        Eigen::Vector3d predict = guesses.empty() ? Eigen::Vector3d::Zero() : guesses.front();
        double best_score = -1e9;

        for (const auto& point: guesses) {
            Eigen::Vector3d d_vector = point - last_pos;
            double dot_product = v_vec.dot(d_vector);
            double v_norm = v_vec.norm();
            double d_norm = d_vector.norm();
            double cos_sim = dot_product / (v_norm * d_norm + 1e-8);
            double d_score = std::exp(-d_norm * d_factor_);
            double score = cos_factor_ * cos_sim + (1 - cos_factor_) * d_score;
            if (score > best_score) {
                best_score = score;
                predict = point;
            }
        }
        return predict;
    }
    void dump_yaml(const std::string& path) const {
        SingleRobotGuess self_1;
        SingleRobotGuess self_2;
        SingleRobotGuess self_3;
        SingleRobotGuess self_4;
        SingleRobotGuess self_7;
        SingleRobotGuess enemy_1;
        SingleRobotGuess enemy_2;
        SingleRobotGuess enemy_3;
        SingleRobotGuess enemy_4;
        SingleRobotGuess enemy_7;
        if (image.self_color == SelfColor::RED) {
            self_1 = robot_guesses.at(std::to_underlying(CarClass::R1));
            self_2 = robot_guesses.at(std::to_underlying(CarClass::R2));
            self_3 = robot_guesses.at(std::to_underlying(CarClass::R3));
            self_4 = robot_guesses.at(std::to_underlying(CarClass::R4));
            self_7 = robot_guesses.at(std::to_underlying(CarClass::R7));

            enemy_1 = robot_guesses.at(std::to_underlying(CarClass::B1));
            enemy_2 = robot_guesses.at(std::to_underlying(CarClass::B2));
            enemy_3 = robot_guesses.at(std::to_underlying(CarClass::B3));
            enemy_4 = robot_guesses.at(std::to_underlying(CarClass::B4));
            enemy_7 = robot_guesses.at(std::to_underlying(CarClass::B7));
        } else {
            enemy_1 = robot_guesses.at(std::to_underlying(CarClass::R1));
            enemy_2 = robot_guesses.at(std::to_underlying(CarClass::R2));
            enemy_3 = robot_guesses.at(std::to_underlying(CarClass::R3));
            enemy_4 = robot_guesses.at(std::to_underlying(CarClass::R4));
            enemy_7 = robot_guesses.at(std::to_underlying(CarClass::R7));

            self_1 = robot_guesses.at(std::to_underlying(CarClass::B1));
            self_2 = robot_guesses.at(std::to_underlying(CarClass::B2));
            self_3 = robot_guesses.at(std::to_underlying(CarClass::B3));
            self_4 = robot_guesses.at(std::to_underlying(CarClass::B4));
            self_7 = robot_guesses.at(std::to_underlying(CarClass::B7));
        }

        std::ofstream out(path);
        auto insert = [&](const SingleRobotGuess& guess) {
            for (auto& [k, v]: guess.guess_positions) {
                out << "  " << k << ": [" << v.x() << ", " << v.y() << "]\n";
            }
        };
        out << "self_1:\n";
        insert(self_1);
        out << "self_2:\n";
        insert(self_2);
        out << "self_3:\n";
        insert(self_3);
        out << "self_4:\n";
        insert(self_4);
        out << "self_7:\n";
        insert(self_7);
        out << "enemy_1:\n";
        insert(enemy_1);
        out << "enemy_2:\n";
        insert(enemy_2);
        out << "enemy_3:\n";
        insert(enemy_3);
        out << "enemy_4:\n";
        insert(enemy_4);
        out << "enemy_7:\n";
        insert(enemy_7);
    }
    struct SingleRobotGuess {
        SingleRobotGuess() {}
        SingleRobotGuess(const Image& image, const YAML::Node& config) {
            image_ = image.clone();
            for (auto it: config) {
                std::string key = it.first.as<std::string>();
                auto arr = it.second;
                if (!arr.IsSequence() || arr.size() != 2)
                    throw std::runtime_error("Point YAML must be [x,y]: " + key);
                guess_positions[key] = Eigen::Vector2d(arr[0].as<double>(), arr[1].as<double>());
            }
        }
        void set_car_class(CarClass car_class) {
            car_class_ = car_class;
        }

        cv::Mat get_drawed_guesses(const Image& image) {
            cv::Mat img = image.image.clone();
            cv::putText(
                img,
                CarClass_to_str(car_class_),
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                cv::Scalar(0, 255, 0),
                2
            );
            for (const auto& pair: guess_positions) {
                const auto& point = pair.second;
                cv::Point2f img_point = uwb_to_image(image, point);
                cv::putText(
                    img,
                    pair.first,
                    img_point,
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 0),
                    2
                );
                cv::circle(img, img_point, 5, cv::Scalar(0, 255, 0), -1);
            }
            return img;
        }
        void edit() {
            cv::namedWindow("guess_map", cv::WINDOW_NORMAL);
            cv::resizeWindow("guess_map", 1000, 600);
            cv::setMouseCallback("guess_map", mouseCallback, this);
            std::vector<std::string> keys;
            for (auto& [k, _]: guess_positions)
                keys.push_back(k);
            const int padding = 20;
            const int max_panel_width_limit = 400;
            const int min_display_width = 600;
            int min = std::min(image_.image.cols, image_.image.rows);
            double display_scale = min_display_width / (double)min;
            display_ = { .image = image_.image.clone(), .self_color = image_.self_color };
            if (display_scale > 1.0) {
                cv::resize(
                    display_.image,
                    display_.image,
                    cv::Size(
                        display_.image.cols * display_scale,
                        display_.image.rows * display_scale
                    )
                );
            }
            while (true) {
                auto drawed = get_drawed_guesses(display_);
                int max_width = 0;
                int baseline = 0;

                for (size_t i = 0; i < keys.size(); i++) {
                    std::string text = (i == selected_index_ ? "> " : "  ") + keys[i];

                    auto size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

                    max_width = std::max(max_width, size.width);
                }

                int panel_width = max_width + padding * 2;
                panel_width = std::min(panel_width, max_panel_width_limit);

                cv::Mat
                    vis(drawed.rows, drawed.cols + panel_width, CV_8UC3, cv::Scalar(40, 40, 40));

                drawed.copyTo(vis(cv::Rect(0, 0, drawed.cols, drawed.rows)));
                int start_x = drawed.cols + padding;
                int start_y = 30;
                int line_h = 20;

                for (size_t i = 0; i < keys.size(); i++) {
                    std::string text = (i == selected_index_ ? "> " : "  ") + keys[i];

                    int baseline = 0;
                    auto size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

                    int x = start_x;
                    int y = start_y + i * line_h;

                    cv::rectangle(
                        vis,
                        cv::Point(x - 5, y - size.height - 2),
                        cv::Point(x + size.width + 5, y + 4),
                        cv::Scalar(20, 20, 20),
                        -1
                    );

                    cv::Scalar color = (i == selected_index_) ? cv::Scalar(0, 255, 255)
                                                              : cv::Scalar(220, 220, 220);

                    cv::putText(
                        vis,
                        text,
                        cv::Point(x, y),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    );
                }
                cv::imshow("guess_map", vis);

                int key = cv::waitKey(30);

                if (key == 'q')
                    break;

                if (key == 81 || key == 2424832) {
                    selected_index_--;
                    if (selected_index_ < 0)
                        selected_index_ = keys.size() - 1;
                }

                if (key == 83 || key == 2555904) {
                    selected_index_++;
                    if (selected_index_ >= (int)keys.size())
                        selected_index_ = 0;
                }
            }
        }
        static void mouseCallback(int event, int x, int y, int, void* userdata) {
            if (event != cv::EVENT_LBUTTONDOWN)
                return;

            auto* map = static_cast<SingleRobotGuess*>(userdata);
            auto p = image_to_uwb(map->image_, cv::Point2f(x, y));

            int idx = 0;
            for (auto& [k, v]: map->guess_positions) {
                if (idx == map->selected_index_) {
                    v = p;
                    std::cout << "Set " << k << " -> " << p.x() << ", " << p.y() << std::endl;
                    break;
                }
                idx++;
            }
        }
        void declare(const std::string& key, const Eigen::Vector2d& p) {
            guess_positions[key] = p;
        }
        std::vector<Eigen::Vector3d> get_guesses() const noexcept {
            std::vector<Eigen::Vector3d> guesses;
            for (const auto& [k, v]: guess_positions) {
                guesses.push_back(Eigen::Vector3d(v.x(), v.y(), 0));
            }
            return guesses;
        }
        std::unordered_map<std::string, Eigen::Vector2d> guess_positions;
        CarClass car_class_;
        Image image_;
        Image display_;
        int selected_index_ = 0;
    };
    std::unordered_map<int, SingleRobotGuess> robot_guesses;
    Image image;
    double d_factor_;
    double cos_factor_;
};
} // namespace awakening::radar_detect