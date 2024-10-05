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

#include <cstdio>

#include "utilities/error_handling.cuh"
#include "utilities/properties.cuh"

// TODO(amrulla): сделать параметром настраиваемым через командную строку.
constexpr unsigned int N = 33 * 1024;

constexpr float sumSquares(float x) { return (x * (x + 1) * (2 * x + 1) / 6); }

__global__ void dotKernel(const float *a, const float *b, float *c) {
  extern __shared__ float cache[];

  unsigned int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int cache_idx = threadIdx.x;

  float temp = 0;
  while (thread_idx < N) {
    temp += a[thread_idx] * b[thread_idx];
    thread_idx += blockDim.x * gridDim.x;
  }

  cache[cache_idx] = temp;
  __syncthreads();

  for (unsigned int i = blockDim.x / 2; i != 0; i /= 2) {
    if (cache_idx < i) {
      cache[cache_idx] += cache[cache_idx + i];
    }
    __syncthreads();
  }

  if (cache_idx == 0) {
    c[blockIdx.x] = cache[0];
  }
}

__host__ float dot(const float *a, const float *b, float *partial_c,
                   unsigned int block_dim, unsigned int grid_dim) {
  float *dev_a, *dev_b, *dev_partial_c;

  const size_t size = N * sizeof(float);
  const size_t partial_size = grid_dim * sizeof(float);
  const size_t cache_size = block_dim * sizeof(float);

  HANDLE_ERROR(cudaMalloc((void **)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void **)&dev_partial_c, partial_size));

  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  dotKernel<<<grid_dim, block_dim, cache_size>>>(dev_a, dev_b, dev_partial_c);

  HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, partial_size,
                          cudaMemcpyDeviceToHost));

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_partial_c);

  float c = 0;
  for (int i = 0; i < grid_dim; ++i) {
    c += partial_c[i];
  }

  return c;
}

int main() {
  const unsigned int block_dim = utils::getBlockDim(1).x;
  const unsigned int grid_dim = std::min(32u, utils::getGridDim(N, block_dim));

  auto *a = new float[N];
  auto *b = new float[N];
  auto *partial_c = new float[grid_dim];

  for (int i = 0; i < N; ++i) {
    const auto fi = static_cast<float>(i);
    a[i] = fi;
    b[i] = fi * 2;
  }

  const float c = dot(a, b, partial_c, block_dim, grid_dim);

  std::printf("Does GPU value %.6g = %.6g?\n", c,
              2 * sumSquares((float)(N - 1)));

  delete[] a;
  delete[] b;
  delete[] partial_c;
}