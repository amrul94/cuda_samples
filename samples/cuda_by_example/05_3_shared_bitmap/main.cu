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

#include "samples/cuda_by_example/common/cpu_bitmap.h"
#include "utilities/error_handling.cuh"

// TODO(amrulla): сделать параметром настраиваемым через командную строку.
constexpr unsigned int DIM = 1024;
constexpr float PI = 3.1415926535897932f;

__global__ void kernel(unsigned char *ptr) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  __shared__ float shared[16][16];

  const float period = 128.0f;

  shared[threadIdx.x][threadIdx.y] =
      255 * (sinf(x * 2.0f * PI / period) + 1.0f) *
            (sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;

  __syncthreads();

  ptr[offset * 4 + 0] = 0;
  ptr[offset * 4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
  ptr[offset * 4 + 2] = 0;
  ptr[offset * 4 + 3] = 255;
}

int main() {
  CPUBitmap bitmap{DIM, DIM};
  unsigned char *dev_bitmap;

  HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));

  dim3 grids{DIM / 16, DIM / 16};
  dim3 threads{16, 16};

  kernel<<<grids, threads>>>(dev_bitmap);

  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(),
                          cudaMemcpyDeviceToHost));

  bitmap.display_and_exit();

  HANDLE_ERROR(cudaFree(dev_bitmap));
}
