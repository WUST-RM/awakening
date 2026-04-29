#include "utils/utils.hpp"
#include <optional>
#include <string>
#define DEFINE_CONFIG_PATH(NAME, FILE) \
    static constexpr auto NAME##_ARR = utils::concat(ROOT_DIR, FILE); \
    static constexpr std::string_view NAME(NAME##_ARR.data());
namespace awakening {
DEFINE_CONFIG_PATH(OMNI_CONFIG_PATH, "/config/omni.yaml")
DEFINE_CONFIG_PATH(SENTRY_CONFIG_PATH, "/config/sentry.yaml")
DEFINE_CONFIG_PATH(LEG_CONFIG_PATH, "/config/leg.yaml")
inline std::optional<std::string> get_robot_config_path(std::string name)
{
    auto key = utils::to_upper(name);
    if (key == "OMNI") {
        return std::string(OMNI_CONFIG_PATH);
    } else if (key == "SENTRY") {
        return std::string(SENTRY_CONFIG_PATH);
    } else if (key == "LEG") {
        return std::string(LEG_CONFIG_PATH);
    }
    return std::nullopt;
}
} // namespace awakening
