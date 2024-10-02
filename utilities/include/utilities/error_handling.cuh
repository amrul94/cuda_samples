#ifndef CUDA_SAMPLES_COMMON_ERROR_HANDLING_CUH
#define CUDA_SAMPLES_COMMON_ERROR_HANDLING_CUH

#include <cstdio>

namespace utils {

void handleError(cudaError_t err, const char *file, int line);

void handleError(void *pointer, const char *file, int line);

} // namespace utils

#define HANDLE_ERROR(ret) utils::handleError(ret, __FILE__, __LINE__)

#endif // CUDA_SAMPLES_COMMON_ERROR_HANDLING_CUH
