add_executable(${SAMPLES_SOURCE}_03_device_info main.cu)
set_target_properties(${SAMPLES_SOURCE}_03_device_info PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${SAMPLES_SOURCE}_03_device_info cuda_samples::utilities)
