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

#include "samples/cuda_by_example/common/cpu_anim.h"
#include "utilities/error_handling.cuh"
#include "utilities/properties.cuh"

constexpr int DIM = 1024;

struct DataBlock {
  unsigned char *dev_bitmap;
  CPUAnimBitmap *bitmap;
};

__global__ void kernel(unsigned char *ptr, int ticks) {
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int offset = x + y * blockDim.x * gridDim.x;

  constexpr auto half_dim = static_cast<float>(DIM) / 2;
  const float fx = static_cast<float>(x) - half_dim;
  const float fy = static_cast<float>(y) - half_dim;
  const float d = std::hypot(fx, fy);

  const float mul = std::cos(d / 10.0f - static_cast<float>(ticks) / 7.0f);
  const float div = d / 10.f + 1.0f;
  const auto grey = static_cast<unsigned char>(128.0f + 127.0f * mul / div);

  ptr[offset * 4 + 0] = 0;
  ptr[offset * 4 + 1] = grey;
  ptr[offset * 4 + 2] = grey;
  ptr[offset * 4 + 3] = 255;
}

__host__ void generateFrame(DataBlock *d, int ticks) {
  const auto [blocks, threads] = utils::getOptimalGridAndBlockSize(2, DIM);
  kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);
  HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap,
                          d->bitmap->image_size(), cudaMemcpyDeviceToHost));
}

__host__ void cleanup(DataBlock *d) { cudaFree(d->dev_bitmap); }

int main() {
  DataBlock data{};
  CPUAnimBitmap bitmap{DIM, DIM, &data};
  data.bitmap = &bitmap;

  HANDLE_ERROR(cudaMalloc((void **)&data.dev_bitmap, bitmap.image_size()));

  bitmap.anim_and_exit((void (*)(void *, int))generateFrame,
                       (void (*)(void *))cleanup);
}
