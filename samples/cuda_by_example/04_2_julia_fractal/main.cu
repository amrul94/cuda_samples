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
#include <thrust/complex.h>

#include "common/error_handling.cuh"
#include "samples/cuda_by_example/common/cpu_bitmap.h"

constexpr int DIM = 1000;

__device__ int julia(int x, int y) {
  constexpr float half_dim = static_cast<float>(DIM) / 2;
  const auto fx = static_cast<float>(x);
  const auto fy = static_cast<float>(y);

  const float scale = 1.5;
  float jx = scale * (half_dim - fx) / half_dim;
  float jy = scale * (half_dim - fy) / half_dim;

  thrust::complex<float> c{-0.8, 0.156};
  thrust::complex<float> a{jx, jy};

  int i = 0;
  for (i = 0; i < 200; ++i) {
    a = a * a + c;
    if (thrust::norm(a) > 1000) {
      return 0;
    }
  }
  return 1;
}

__global__ void kernel(unsigned char *ptr) {
  int x = static_cast<int>(blockIdx.x);
  int y = static_cast<int>(blockIdx.y);
  size_t offset = x + y * gridDim.x;

  int julia_value = julia(x, y);
  ptr[offset * 4 + 0] = 255 * julia_value;
  ptr[offset * 4 + 1] = 0;
  ptr[offset * 4 + 2] = 0;
  ptr[offset * 4 + 3] = 255;
}

int main() {
  CPUBitmap bitmap{DIM, DIM};
  unsigned char *dev_bitmap;

  HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));

  dim3 grid(DIM, DIM);

  kernel<<<grid, 1>>>(dev_bitmap);

  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(),
                          cudaMemcpyDeviceToHost));

  bitmap.display_and_exit();
}
