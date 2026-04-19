#pragma once

#include "param_deliver.h"
#include "utils/utils.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace awakening::logger {

static constexpr auto LOG_FOLDER_PATH_ARR = utils::concat(ROOT_DIR, "/log");
static constexpr std::string_view LOG_FOLDER_PATH(LOG_FOLDER_PATH_ARR.data());
static constexpr auto LOG_NAME = "awakening";
static constexpr size_t MAX_LOG_FILE_SIZE = 10 * 1024 * 1024; // 10MB
static constexpr size_t FOLDER_WARN_SIZE = 50 * 1024 * 1024; // 50MB

#define AWAKENING_TRACE(...) ::awakening::logger::get_logger()->trace(__VA_ARGS__)
#define AWAKENING_DEBUG(...) ::awakening::logger::get_logger()->debug(__VA_ARGS__)
#define AWAKENING_INFO(...) ::awakening::logger::get_logger()->info(__VA_ARGS__)
#define AWAKENING_WARN(...) ::awakening::logger::get_logger()->warn(__VA_ARGS__)
#define AWAKENING_ERROR(...) ::awakening::logger::get_logger()->error(__VA_ARGS__)
#define AWAKENING_CRITICAL(...) ::awakening::logger::get_logger()->critical(__VA_ARGS__)
inline std::shared_ptr<spdlog::logger>& get_logger() {
    static std::shared_ptr<spdlog::logger> logger = nullptr;
    return logger;
}

inline std::string get_current_log_file_path() {
    auto& logger = get_logger();
    if (!logger) {
        return {};
    }

    for (const auto& sink: logger->sinks()) {
        if (auto file_sink = std::dynamic_pointer_cast<spdlog::sinks::rotating_file_sink_mt>(sink))
        {
            return file_sink->filename();
        }
    }
    return {};
}

inline void
check_folder_size(const std::string& folder_path, std::size_t warn_size = FOLDER_WARN_SIZE) {
    if (!std::filesystem::exists(folder_path)) {
        return;
    }

    std::uintmax_t total_size = 0;
    for (const auto& entry: std::filesystem::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            total_size += entry.file_size();
        }
    }

    if (total_size >= warn_size) {
        AWAKENING_WARN(
            "Log folder {} size {} bytes exceeds warning threshold {} bytes",
            folder_path,
            total_size,
            warn_size
        );
    }
}

inline std::string generate_log_filename(const std::string& folder_path) {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm {};

#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    std::ostringstream oss;
    oss << folder_path << "/" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << ".log";
    return oss.str();
}

inline void init(spdlog::level::level_enum level = spdlog::level::info) {
    if (get_logger()) {
        return;
    }

    try {
        if (!std::filesystem::exists(LOG_FOLDER_PATH)) {
            std::filesystem::create_directories(LOG_FOLDER_PATH);
        }

        std::string file_path = generate_log_filename(std::string(LOG_FOLDER_PATH));

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            file_path,
            MAX_LOG_FILE_SIZE,
            3
        ); // max_files = 3

        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");

        std::vector<spdlog::sink_ptr> sinks { console_sink, file_sink };

        auto logger =
            std::make_shared<spdlog::logger>(std::string(LOG_NAME), sinks.begin(), sinks.end());

        logger->set_level(level);
        logger->flush_on(level);

        check_folder_size(std::string(LOG_FOLDER_PATH));

        spdlog::register_logger(logger);
        get_logger() = logger;

        logger->info("Logger initialized successfully. Current log file: {}", file_path);
        logger->info("Log folder: {}", LOG_FOLDER_PATH);

    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
    }
}

} // namespace awakening::logger