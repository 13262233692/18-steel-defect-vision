cmake_minimum_required(VERSION 3.18)
project(steel_defect_vision LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/wd4819")
endif()

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT TENSORRT_DIR)
    set(TENSORRT_DIR "$ENV{TENSORRT_DIR}" CACHE PATH "TensorRT root directory")
    if(NOT TENSORRT_DIR)
        message(FATAL_ERROR "TENSORRT_DIR not set. Please set it via -DTENSORRT_DIR=... or env var TENSORRT_DIR")
    endif()
endif()

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TENSORRT_DIR}/include
    NO_DEFAULT_PATH
)
find_library(TENSORRT_LIB
    NAMES nvinfer
    PATHS ${TENSORRT_DIR}/lib
    NO_DEFAULT_PATH
)
find_library(TENSORRT_ONNX_LIB
    NAMES nvinfer_plugin nvonnxparser
    PATHS ${TENSORRT_DIR}/lib
    NO_DEFAULT_PATH
)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
    message(FATAL_ERROR "TensorRT not found in ${TENSORRT_DIR}")
endif()
message(STATUS "TensorRT found: ${TENSORRT_DIR}")

if(NOT BASLER_DIR)
    set(BASLER_DIR "$ENV{BASLER_DIR}" CACHE PATH "Basler pylon SDK root directory")
endif()
if(BASLER_DIR)
    find_path(BASLER_INCLUDE_DIR
        NAMES pylc/GenApi/GenApi.h
        PATHS ${BASLER_DIR}/include
        NO_DEFAULT_PATH
    )
    find_library(BASLER_LIB
        NAMES PylonBase pylon
        PATHS ${BASLER_DIR}/lib ${BASLER_DIR}/lib64
        NO_DEFAULT_PATH
    )
    if(BASLER_INCLUDE_DIR AND BASLER_LIB)
        message(STATUS "Basler pylon SDK found: ${BASLER_DIR}")
        add_definitions(-DUSE_BASLER)
    else()
        message(WARNING "Basler SDK not fully found, falling back to simulator")
    endif()
endif()

if(NOT HIKVISION_DIR)
    set(HIKVISION_DIR "$ENV{HIKVISION_DIR}" CACHE PATH "Hikvision MVS SDK root directory")
endif()
if(HIKVISION_DIR)
    find_path(HIKVISION_INCLUDE_DIR
        NAMES MvCameraControl.h
        PATHS ${HIKVISION_DIR}/include
        NO_DEFAULT_PATH
    )
    find_library(HIKVISION_LIB
        NAMES MvCameraControl
        PATHS ${HIKVISION_DIR}/lib ${HIKVISION_DIR}/lib64
        NO_DEFAULT_PATH
    )
    if(HIKVISION_INCLUDE_DIR AND HIKVISION_LIB)
        message(STATUS "Hikvision MVS SDK found: ${HIKVISION_DIR}")
        add_definitions(-DUSE_HIKVISION)
    else()
        message(WARNING "Hikvision SDK not fully found")
    endif()
endif()

find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, will use embedded single-header")
    set(JSON_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party")
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

if(BASLER_INCLUDE_DIR)
    include_directories(${BASLER_INCLUDE_DIR})
endif()
if(HIKVISION_INCLUDE_DIR)
    include_directories(${HIKVISION_INCLUDE_DIR})
endif()
if(NOT nlohmann_json_FOUND)
    include_directories(${JSON_INCLUDE_DIR})
endif()

file(GLOB SOURCES_CPP "src/*.cpp")
file(GLOB SOURCES_CU  "src/*.cu")

add_executable(${PROJECT_NAME} ${SOURCES_CPP} ${SOURCES_CU})

target_link_libraries(${PROJECT_NAME}
    ${TENSORRT_LIB}
    ${OpenCV_LIBS}
    cudart
)

if(BASLER_LIB)
    target_link_libraries(${PROJECT_NAME} ${BASLER_LIB})
endif()
if(HIKVISION_LIB)
    target_link_libraries(${PROJECT_NAME} ${HIKVISION_LIB})
endif()
if(nlohmann_json_FOUND)
    target_link_libraries(${PROJECT_NAME} nlohmann_json::nlohmann_json)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
    )
endif()
