add_executable(${SAMPLES_SOURCE}_05_1_wave_effect main.cu)
set_target_properties(${SAMPLES_SOURCE}_05_1_wave_effect PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${SAMPLES_SOURCE}_05_1_wave_effect
    cuda_samples::utilities
    OpenGL::OpenGL
    GLUT::GLUT
)
