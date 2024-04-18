cmake_minimum_required(VERSION 3.28)
project(cuda_samples CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda_samples main.cu)

set_target_properties(cuda_samples PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
