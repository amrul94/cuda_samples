#include "utilities/properties.cuh"

#include <algorithm>
#include <bit>
#include <iostream>
#include <type_traits>
#include <vector>

#include "utilities/error_handling.cuh"

namespace utils {

namespace {

template <typename T>
concept Ariphmetic = std::is_arithmetic_v<T>;

inline unsigned int bitFloor(Ariphmetic auto num) {
  return std::bit_floor(static_cast<unsigned int>(num));
}

} // namespace

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

dim3 getBlockDim(size_t num_dimensions) {
  const size_t max_threads_per_block = getMaxThreadsPerBlock();
  switch (num_dimensions) {
  case 1:
    return std::bit_floor(max_threads_per_block);
  case 2: {
    unsigned int v = bitFloor(std::sqrt(max_threads_per_block));
    return {v, v};
  }
  case 3: {
    unsigned int v = bitFloor(std::cbrt(max_threads_per_block));
    return {v, v, v};
  }
  default:
    std::cerr << "Cuda support only 1, 2 or 3 dimensional blocks!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

unsigned int getGridDim(unsigned int data_dim, unsigned int block_dim) {
  return std::ceil(static_cast<float>(data_dim) /
                   static_cast<float>(block_dim));
}

dim3 getGridDim(dim3 data_dim, dim3 block_dim) {
  const unsigned int vx = getGridDim(data_dim.x, block_dim.x);
  const unsigned int vy = getGridDim(data_dim.y, block_dim.y);
  const unsigned int vz = getGridDim(data_dim.z, block_dim.z);
  return {vx, vy, vz};
}

std::pair<dim3, dim3> getGridAndBlockDims(unsigned int data_dim,
                                          size_t num_dimensions) {
  switch (num_dimensions) {
  case 3:
    return getGridAndBlockDims({data_dim, data_dim, data_dim}, num_dimensions);
  case 2:
    return getGridAndBlockDims({data_dim, data_dim}, num_dimensions);
  case 1:
    return getGridAndBlockDims(dim3{data_dim}, num_dimensions);
  default:
    std::cerr << "Cuda support only 1, 2 or 3 dimensional blocks and blocks!"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

std::pair<dim3, dim3> getGridAndBlockDims(dim3 data_dim,
                                          size_t num_dimensions) {
  const dim3 block_size = getBlockDim(num_dimensions);
  dim3 grid_size{};

  switch (num_dimensions) {
  case 3:
    grid_size.z = getGridDim(data_dim.z, block_size.z);
    [[fallthrough]];
  case 2:
    grid_size.y = getGridDim(data_dim.y, block_size.y);
    [[fallthrough]];
  case 1:
    grid_size.x = getGridDim(data_dim.x, block_size.x);
    return {grid_size, block_size};
  default:
    std::cerr << "Cuda support only 1, 2 or 3 dimensional blocks and blocks!"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace utils