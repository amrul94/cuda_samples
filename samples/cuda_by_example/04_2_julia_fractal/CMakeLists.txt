add_executable(${SAMPLES_SOURCE}_04_2_julia_fractal main.cu)
set_target_properties(${SAMPLES_SOURCE}_04_2_julia_fractal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${SAMPLES_SOURCE}_04_2_julia_fractal OpenGL::OpenGL GLUT::GLUT Thrust)
