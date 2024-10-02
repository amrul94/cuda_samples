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

constexpr unsigned int N = 33 * 1024;
constexpr unsigned int threads_per_block = 256;
constexpr unsigned int blocks_per_grid =
    std::min(32u, (N + threads_per_block - 1) / threads_per_block);

constexpr float sumSquares(float x) { return (x * (x + 1) * (2 * x + 1) / 6); }

__global__ void dotKernel(const float *a, const float *b, float *c) {
  __shared__ float cache[threads_per_block];
  unsigned int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int cache_idx = threadIdx.x;

  float temp = 0;
  while (thread_idx < N) {
    temp += a[thread_idx] * b[thread_idx];
    thread_idx += blockDim.x * gridDim.x;
  }

  cache[cache_idx] = temp;
  __syncthreads();

  unsigned int i = blockDim.x / 2;
  while (i != 0) {
    if (cache_idx < i) {
      cache[cache_idx] += cache[cache_idx + i];
    }
    __syncthreads();
    i /= 2;
  }

  if (cache_idx == 0) {
    c[blockIdx.x] = cache[0];
  }
}

float dot(const float *a, const float *b, float *partial_c) {
  float *dev_a, *dev_b, *dev_partial_c;
  const size_t size = N * sizeof(float);
  const size_t partial_size = blocks_per_grid * sizeof(float);

  HANDLE_ERROR(cudaMalloc((void **)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void **)&dev_partial_c, partial_size));

  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  dotKernel<<<blocks_per_grid, threads_per_block>>>(dev_a, dev_b,
                                                    dev_partial_c);

  HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c,
                          blocks_per_grid * sizeof(float),
                          cudaMemcpyDeviceToHost));

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_partial_c);

  float c = 0;
  for (int i = 0; i < blocks_per_grid; ++i) {
    c += partial_c[i];
  }

  return c;
}

int main() {
  auto *a = new float[N];
  auto *b = new float[N];
  auto *partial_c = new float[blocks_per_grid];

  for (int i = 0; i < N; ++i) {
    const auto fi = static_cast<float>(i);
    a[i] = fi;
    b[i] = fi * 2;
  }

  const float c = dot(a, b, partial_c);

  std::printf("Does GPU value %.6g = %.6g?\n", c,
              2 * sumSquares((float)(N - 1)));

  delete[] a;
  delete[] b;
  delete[] partial_c;
}
