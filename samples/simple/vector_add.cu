//
// Created by amrulla on 19.04.2024.
//
#include "common/error_handling.cuh"

constexpr size_t N = 10;

__global__ void add(const int *a, const int *b, int *c) {
  unsigned int tid = blockIdx.x;
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;

  const size_t size = N * sizeof(int);
  HANDLE_ERROR(cudaMalloc((void **)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c, size));

  for (int i = 0; i < N; ++i) {
    a[i] = -i;
    b[i] = i * i;
  }

  HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice));

  add<<<N, 1>>>(dev_a, dev_b, dev_c);

  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}
