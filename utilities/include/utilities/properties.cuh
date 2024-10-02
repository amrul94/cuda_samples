#ifndef CUDA_SAMPLES_COMMON_PROPERTIES_CUH
#define CUDA_SAMPLES_COMMON_PROPERTIES_CUH

#include <cstddef>

namespace utils {

size_t getMaxThreadsPerBlock();

dim3 getBlockDim(size_t num_dimensions);

unsigned int getGridDim(unsigned int data_dim, unsigned int block_dim);

dim3 getGridDim(dim3 data_dim, dim3 block_dim);

std::pair<dim3, dim3> getGridAndBlockDims(unsigned int data_dim,
                                          size_t num_dimensions);

std::pair<dim3, dim3> getGridAndBlockDims(dim3 data_dim, size_t num_dimensions);

} // namespace utils

#endif // CUDA_SAMPLES_COMMON_PROPERTIES_CUH
