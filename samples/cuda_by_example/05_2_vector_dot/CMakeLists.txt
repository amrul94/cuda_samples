add_executable(${SAMPLES_SOURCE}_05_2_vector_dot main.cu)
set_target_properties(${SAMPLES_SOURCE}_05_2_vector_dot PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${SAMPLES_SOURCE}_05_2_vector_dot cuda_samples::utilities)