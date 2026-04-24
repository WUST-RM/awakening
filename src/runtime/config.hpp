#include "utils/utils.hpp"
#define DEFINE_CONFIG_PATH(NAME, FILE) \
    static constexpr auto NAME##_ARR = utils::concat(ROOT_DIR, FILE); \
    static constexpr std::string_view NAME(NAME##_ARR.data());
namespace awakening {
DEFINE_CONFIG_PATH(OMNI_CONFIG_PATH, "/config/omni.yaml")
DEFINE_CONFIG_PATH(SENTRY_CONFIG_PATH, "/config/sentry.yaml")
} // namespace awakening
