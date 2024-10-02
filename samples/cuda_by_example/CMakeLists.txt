set(SAMPLES_SOURCE "cuda_by_example")

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

add_subdirectory(03_device_info)
add_subdirectory(04_1_vector_add)
add_subdirectory(04_2_julia_fractal)
add_subdirectory(05_1_wave_effect)
add_subdirectory(05_2_vector_dot)
