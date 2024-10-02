add_executable(${SAMPLES_SOURCE}_04_1_vector_add main.cu)
set_target_properties(${SAMPLES_SOURCE}_04_1_vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
