cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0072 NEW)
project(awakening LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_BUILD_TYPE "Release")
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")


option(BUILD_WITH_TRT      "Enable TensorRT backend"   ON)
option(BUILD_WITH_OPENVINO "Enable OpenVINO backend"   ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)




find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(HikSDK REQUIRED)
find_package(TBB REQUIRED)
find_package(spdlog REQUIRED)

set(BUILD_DEFINITIONS "")
set(BUILD_LIBS "")


if(BUILD_WITH_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
    list(APPEND BUILD_DEFINITIONS USE_OPENVINO)
    list(APPEND BUILD_LIBS openvino::runtime openvino::frontend::onnx)
endif()

if(BUILD_WITH_TRT)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/hy/TensorRT-10.6.0.26")
    find_package(TensorRT REQUIRED)
    include_directories(${TensorRT_INCLUDE_DIR})
    include_directories(/usr/local/cuda/include)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)

    list(APPEND BUILD_DEFINITIONS USE_TRT)
    list(APPEND BUILD_LIBS TensorRT::TensorRT TensorRT::nvonnxparser CUDA::cudart)
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/param_deliver.h.in  
    ${CMAKE_BINARY_DIR}/src/param_deliver.h            
    @ONLY
)
include_directories(${PROJECT_NAME}_utils PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/3rdparty
    ${CMAKE_BINARY_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
file(GLOB_RECURSE UTILS_SOURCES CONFIGURE_DEPENDS src/utils/*.cpp)
if(BUILD_WITH_TRT)
    file(GLOB_RECURSE UTILS_CUDA_SOURCES CONFIGURE_DEPENDS src/utils/*.cu)
    list(APPEND UTILS_SOURCES ${UTILS_CUDA_SOURCES})
endif()
add_library(${PROJECT_NAME}_utils SHARED ${UTILS_SOURCES})
target_compile_definitions(${PROJECT_NAME}_utils PUBLIC ${BUILD_DEFINITIONS})
target_link_libraries(${PROJECT_NAME}_utils
    HikSDK::HikSDK
    yaml-cpp
    fmt::fmt
    Eigen3::Eigen
    Ceres::ceres
    TBB::tbb
    spdlog::spdlog
    ${OpenCV_LIBS}
    ${BUILD_LIBS}
)
file(GLOB_RECURSE AUTO_AIM_SOURCES CONFIGURE_DEPENDS src/tasks/auto_aim/*.cpp)
add_library(${PROJECT_NAME}_auto_aim SHARED ${AUTO_AIM_SOURCES})
target_link_libraries(${PROJECT_NAME}_auto_aim
     ${PROJECT_NAME}_utils
)
function(add_common_executable exe_name src_file)
    add_executable(${exe_name} ${src_file})
    target_link_libraries(${exe_name}  ${PROJECT_NAME}_utils)
endfunction()
add_common_executable(auto_aim src/runtime/auto_aim.cpp)
target_link_libraries(auto_aim
    ${PROJECT_NAME}_auto_aim
)