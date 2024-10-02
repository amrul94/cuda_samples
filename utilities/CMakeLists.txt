cmake_minimum_required(VERSION 3.28)
project(cuda_samples CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(SOURCE_FILES
    src/error_handling.cu
)

add_library(cuda_samples_utilities SHARED ${SOURCE_FILES})
add_library(cuda_samples::utilities ALIAS cuda_samples_utilities)

target_include_directories(cuda_samples_utilities
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
)