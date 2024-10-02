set(SAMPLES_SOURCE "cuda_by_example")

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

message("GLUT found: ${GLUT_FOUND}")
message("OpenGL found: ${OpenGL_FOUND}")

add_subdirectory(03_device_info)
add_subdirectory(04_1_vector_add)
add_subdirectory(04_2_julia_fractal)
