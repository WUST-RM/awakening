# --------------------------------------------------------------------------------------------
# FindMvSDK.cmake (simplified fallback version)
# --------------------------------------------------------------------------------------------

# =========================
# 1. 默认 SDK 路径（可选环境变量）
# =========================
set(MvSDK_Path "$ENV{MVSDK_PATH}")

# 如果环境变量不存在，就不依赖它
if(NOT MvSDK_Path OR MvSDK_Path STREQUAL "")
  unset(MvSDK_Path)
endif()

# =========================
# 2. 查找头文件（优先 /usr/include）
# =========================
find_path(
  MvSDK_INCLUDE_DIR
  NAMES
    CameraApi.h
    CameraDefine.h
    CameraStatus.h
  PATHS
    /usr/include
    /usr/local/include
    ${MvSDK_Path}/include
    ${MvSDK_Path}/Includes
    ${MvSDK_Path}/Demo/VC++/Include
  NO_DEFAULT_PATH
)

# fallback：允许系统默认路径
if(NOT MvSDK_INCLUDE_DIR)
  find_path(
    MvSDK_INCLUDE_DIR
    NAMES CameraApi.h CameraDefine.h CameraStatus.h
  )
endif()

# =========================
# 3. 查找库文件（优先 /lib）
# =========================
find_library(
  MvSDK_LIB
  NAMES
    MVSDK
    libMVSDK.so
  PATHS
    /lib
    /usr/lib
    /usr/local/lib
    /lib/x86_64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    ${MvSDK_Path}/lib
    ${MvSDK_Path}/lib64
    ${MvSDK_Path}/lib/x64
  NO_DEFAULT_PATH
)

# fallback：允许系统默认路径
if(NOT MvSDK_LIB)
  find_library(
    MvSDK_LIB
    NAMES MVSDK libMVSDK.so
  )
endif()

# =========================
# 4. imported target
# =========================
if(MvSDK_LIB AND MvSDK_INCLUDE_DIR)

  if(NOT TARGET MvSDK::MvSDK)
    add_library(MvSDK::MvSDK SHARED IMPORTED GLOBAL)

    set_target_properties(MvSDK::MvSDK PROPERTIES
      IMPORTED_LOCATION "${MvSDK_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${MvSDK_INCLUDE_DIR}"
    )
  endif()

endif()

# =========================
# 5. variables
# =========================
set(MvSDK_LIBS MvSDK::MvSDK)
set(MvSDK_INCLUDE_DIRS ${MvSDK_INCLUDE_DIR})

# =========================
# 6. standard handle
# =========================
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  MvSDK
  REQUIRED_VARS MvSDK_LIB MvSDK_INCLUDE_DIR
)

# =========================
# 7. debug
# =========================
if(MvSDK_FOUND)
  message(STATUS "MvSDK found:")
  message(STATUS "  include: ${MvSDK_INCLUDE_DIR}")
  message(STATUS "  lib    : ${MvSDK_LIB}")
else()
  message(STATUS "MvSDK NOT found (checked /lib and /usr/include)")
endif()

mark_as_advanced(MvSDK_LIB MvSDK_INCLUDE_DIR)