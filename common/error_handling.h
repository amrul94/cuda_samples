//
// Created by amrulla on 18.04.2024.
//

#ifndef CUDA_SAMPLES_COMMON_ERROR_HANDLING_HPP
#define CUDA_SAMPLES_COMMON_ERROR_HANDLING_HPP

#include <cstdio>

static void HandleError(cudaError_t err,
                        const char *file,
                        int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))


#define HANDLE_NULL(a)                                      \
    {                                                       \
        if (a == NULL) {                                    \
            printf("Host memory failed in %s at line %d\n", \
                   __FILE__, __LINE__);                     \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    }


#endif//CUDA_SAMPLES_COMMON_ERROR_HANDLING_HPP
