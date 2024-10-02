/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include <format>
#include <iostream>

#include "utilities/error_handling.cuh"
#include "utilities/properties.cuh"

__host__ size_t getVectorSize(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Incorrect number of arguments: " << argc - 1
              << " but expect 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  char *end = nullptr;
  size_t ret = strtoul(argv[1], &end, 10);

  if (errno == ERANGE) {
    std::perror("!! Problem is -> ");
    std::exit(EXIT_FAILURE);
  } else if (ret) {
    return ret;
  } else {
    std::cerr << "No number found input is -> " << argv[1] << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

__global__ void vectorAddKernel(const int *a, const int *b, int *c, size_t n) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < n) {
    c[tid] = a[tid] + b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

__host__ void vectorAdd(const int *h_a, const int *h_b, int *h_c, size_t n) {
  int *d_a, *d_b, *d_c;
  const size_t size = n * sizeof(int);

  HANDLE_ERROR(cudaMalloc((void **)&d_a, size));
  HANDLE_ERROR(cudaMalloc((void **)&d_b, size));
  HANDLE_ERROR(cudaMalloc((void **)&d_c, size));

  HANDLE_ERROR(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

  const auto [grid_dim, block_dim] = utils::getGridAndBlockDims(n, 1);
  vectorAddKernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);

  HANDLE_ERROR(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

int main(int argc, char *argv[]) {
  size_t vec_size = getVectorSize(argc, argv);
  std::cout << "Number of elements: " << vec_size << std::endl;

  auto *a = new int[vec_size];
  auto *b = new int[vec_size];
  auto *c = new int[vec_size];

  for (int i = 0; i < vec_size; ++i) {
    a[i] = i;
    b[i] = 2 * i;
  }

  vectorAdd(a, b, c, vec_size);

  bool success = true;
  for (int i = 0; i < vec_size; i++) {
    if ((a[i] + b[i]) != c[i]) {
      printf("Error:  %d + %d != %d\n", a[i], b[i], c[i]);
      success = false;
    }
  }
  if (success) {
    printf("We did it!\n");
  }

  delete[] a;
  delete[] b;
  delete[] c;
}
