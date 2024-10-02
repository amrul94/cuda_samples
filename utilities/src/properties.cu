#include "utilities/properties.cuh"

#include <algorithm>
#include <bit>
#include <iostream>
#include <vector>

#include "utilities/error_handling.cuh"

namespace utils {

size_t getMaxThreadsPerBlock() {
  int num_devices;
  HANDLE_ERROR(cudaGetDeviceCount(&num_devices));

  if (num_devices == 0) {
    std::cerr << "Cuda device not found!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  cudaDeviceProp prop{};
  std::vector<size_t> max_threads;
  max_threads.reserve(num_devices);

  for (int i = 0; i < num_devices; i++) {
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
    max_threads.push_back(prop.maxThreadsPerBlock);
  }

  return *std::ranges::min_element(max_threads);
}

dim3 getOptimalBlockSize(int num_dimensions) {
  const size_t max_threads_per_block = getMaxThreadsPerBlock();
  switch (num_dimensions) {
  case 1:
    return std::bit_floor(max_threads_per_block);
  case 2: {
    auto v = static_cast<unsigned int>(std::sqrt(max_threads_per_block));
    v = std::bit_floor(v);
    return {v, v};
  }
  case 3: {
    auto v = static_cast<unsigned int>(std::cbrt(max_threads_per_block));
    v = std::bit_floor(v);
    return {v, v, v};
  }
  default:
    std::cerr << "Cuda support only 1, 2 or 3 dimensional blocks!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

std::pair<dim3, dim3> getOptimalGridAndBlockSize(int num_dimensions,
                                                 unsigned int max_dim_size) {
  switch (num_dimensions) {
  case 3:
    return getOptimalGridAndBlockSize(
        num_dimensions, {max_dim_size, max_dim_size, max_dim_size});
  case 2:
    return getOptimalGridAndBlockSize(num_dimensions,
                                      {max_dim_size, max_dim_size});
  case 1:
    return getOptimalGridAndBlockSize(num_dimensions, dim3{max_dim_size});
  default:
    std::cerr << "Cuda support only 1, 2 or 3 dimensional blocks and blocks!"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

std::pair<dim3, dim3> getOptimalGridAndBlockSize(int num_dimensions,
                                                 dim3 max_dim_size) {
  const dim3 block_size = getOptimalBlockSize(num_dimensions);
  dim3 grid_size{};

  switch (num_dimensions) {
  case 3:
    grid_size.z = std::ceil(static_cast<float>(max_dim_size.z) /
                            static_cast<float>(block_size.z));
    [[fallthrough]];
  case 2:
    grid_size.y = std::ceil(static_cast<float>(max_dim_size.y) /
                            static_cast<float>(block_size.y));
    [[fallthrough]];
  case 1:
    grid_size.x = std::ceil(static_cast<float>(max_dim_size.x) /
                            static_cast<float>(block_size.x));
    return {grid_size, block_size};
  default:
    std::cerr << "Cuda support only 1, 2 or 3 dimensional blocks and blocks!"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace utils