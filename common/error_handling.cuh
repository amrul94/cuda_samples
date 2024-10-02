//
// Created by amrulla on 18.04.2024.
//

#ifndef CUDA_SAMPLES_COMMON_ERROR_HANDLING_CUH
#define CUDA_SAMPLES_COMMON_ERROR_HANDLING_CUH

#include <cstdio>

static void handleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    std::exit(EXIT_FAILURE);
  }
}

static void handleError(void *pointer, const char *file, int line) {
  if (pointer == nullptr) {
    std::printf("Host memory failed in %s at line %d\n", file, line);
    std::exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(ret) handleError(ret, __FILE__, __LINE__)

#endif // CUDA_SAMPLES_COMMON_ERROR_HANDLING_CUH
