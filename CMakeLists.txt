cmake_minimum_required(VERSION 3.28)
project(cuda_samples CUDA)

set(CMAKE_CUDA_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(samples/cuda_by_example)