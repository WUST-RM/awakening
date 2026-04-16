#pragma once
#include <array>
#include <cmath>
#include <iostream>
namespace awakening {
constexpr std::array<const char*, 5> ascii_banner = {
    "    ___ _       _____    __ __ _______   _______   ________",
    "   /   | |     / /   |  / //_// ____/ | / /  _/ | / / ____/",
    "  / /| | | /| / / /| | / ,<  / __/ /  |/ // //  |/ / / __  ",
    " / ___ | |/ |/ / ___ |/ /| |/ /___/ /|  // // /|  / /_/ /  ",
    "/_/  |_|__/|__/_/  |_/_/ |_/_____/_/ |_/___/_/ |_|\\____/   "
};
namespace {
    struct RGB {
        int r, g, b;
    };

    inline RGB hsv2rgb(float h, float s, float v) {
        float c = v * s;
        float x = c * (1 - std::fabs(std::fmod(h / 60.0f, 2) - 1));
        float m = v - c;

        float r = 0, g = 0, b = 0;
        if (h < 60) {
            r = c;
            g = x;
        } else if (h < 120) {
            r = x;
            g = c;
        } else if (h < 180) {
            g = c;
            b = x;
        } else if (h < 240) {
            g = x;
            b = c;
        } else if (h < 300) {
            r = x;
            b = c;
        } else {
            r = c;
            b = x;
        }

        return { int((r + m) * 255), int((g + m) * 255), int((b + m) * 255) };
    }
} // namespace
inline void print_banner() {
    constexpr const char* reset = "\033[0m";

    for (const auto& line: ascii_banner) {
        const int n = static_cast<int>(std::string_view(line).size());

        for (int i = 0; i < n; ++i) {
            // hue 从 0° → 360°
            float hue = 360.0f * i / std::max(1, n - 1);
            auto rgb = hsv2rgb(hue, 1.0f, 1.0f);

            std::cout << "\033[38;2;" << rgb.r << ";" << rgb.g << ";" << rgb.b << "m" << line[i];
        }
        std::cout << reset << '\n';
    }
}

} // namespace awakening