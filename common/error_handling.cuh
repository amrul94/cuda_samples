//
// Created by amrulla on 18.04.2024.
//

#ifndef CUDA_SAMPLES_COMMON_ERROR_HANDLING_HPP
#define CUDA_SAMPLES_COMMON_ERROR_HANDLING_HPP

#include <cstdio>

static void handleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

static void handleError(void *pointer, const char *file, int line) {
  if (pointer == nullptr) {
    printf("Host memory failed  in %s at line %d\n", file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (handleError(err, __FILE__, __LINE__))

#endif // CUDA_SAMPLES_COMMON_ERROR_HANDLING_HPP
