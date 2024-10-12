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

#include <numbers>

#include "samples/cuda_by_example/common/cpu_bitmap.h"
#include "utilities/error_handling.cuh"

// TODO(amrulla): сделать параметром настраиваемым через командную строку.
constexpr unsigned int DIM = 1024;

// TODO(amrulla): сделать параметром настраиваемым через командную строку
//  или получаемым автоматически.
constexpr unsigned int NUM_THREADS = 16;

__device__ float partialCalc(unsigned int v, float period) {
  const auto fv = static_cast<float>(v);
  return sinf(fv * 2.0f * std::numbers::pi_v<float> / period) + 1.0f;
}

__global__ void kernel(unsigned char *ptr) {
  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned int offset = x + y * blockDim.x * gridDim.x;

  __shared__ float shared[NUM_THREADS][NUM_THREADS];

  const float period = 128.0f;

  const float vx = partialCalc(x, period);
  const float vy = partialCalc(y, period);
  shared[threadIdx.x][threadIdx.y] = 255 * vx * vy / 4.0f;

  __syncthreads();

  constexpr size_t max_idx = NUM_THREADS - 1;
  const float color = shared[max_idx - threadIdx.x][max_idx - threadIdx.y];

  ptr[offset * 4 + 0] = 0;
  ptr[offset * 4 + 1] = static_cast<unsigned char>(color);
  ptr[offset * 4 + 2] = 0;
  ptr[offset * 4 + 3] = 255;
}

int main() {
  CPUBitmap bitmap{DIM, DIM};
  unsigned char *dev_bitmap;

  HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));

  constexpr dim3 grids{DIM / NUM_THREADS, DIM / NUM_THREADS};
  constexpr dim3 threads{NUM_THREADS, NUM_THREADS};

  kernel<<<grids, threads>>>(dev_bitmap);

  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(),
                          cudaMemcpyDeviceToHost));

  bitmap.display_and_exit();

  HANDLE_ERROR(cudaFree(dev_bitmap));
}
