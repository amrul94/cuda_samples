#include "utilities/properties.cuh"

#include <iostream>
#include <vector>
#include <algorithm>

#include "utilities/error_handling.cuh"

namespace utils {

__host__ size_t getMaxThreadsPerBlock() {
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

}