#ifndef CUDA_SAMPLES_COMMON_PROPERTIES_CUH
#define CUDA_SAMPLES_COMMON_PROPERTIES_CUH

#include <utility>

namespace utils {

size_t getMaxThreadsPerBlock();

dim3 getOptimalBlockSize(int num_dimensions);

std::pair<dim3, dim3> getOptimalGridAndBlockSize(int num_dimensions,
                                                 unsigned int max_dim_size);

std::pair<dim3, dim3> getOptimalGridAndBlockSize(int num_dimensions,
                                                 dim3 max_dim_size);

} // namespace utils

#endif // CUDA_SAMPLES_COMMON_PROPERTIES_CUH
