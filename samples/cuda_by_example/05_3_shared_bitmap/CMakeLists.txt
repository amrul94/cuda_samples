add_executable(${SAMPLES_SOURCE}_05_3_shared_bitmap main.cu)
set_target_properties(${SAMPLES_SOURCE}_05_3_shared_bitmap PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${SAMPLES_SOURCE}_05_3_shared_bitmap
    cuda_samples::utilities
    OpenGL::OpenGL
    GLUT::GLUT
)