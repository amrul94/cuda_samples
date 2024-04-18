add_executable(3_device_info 3_device_info.cu)

set_target_properties(3_device_info PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
