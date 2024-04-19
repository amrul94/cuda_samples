//
// Created by amrulla on 19.04.2024.
//

#include <format>
#include <iostream>

#include "common/error_handling.cuh"

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
    std::cout << "The number is " << ret << std::endl;
    return ret;
  } else {
    std::cerr << "No number found input is -> " << argv[1] << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

__global__ void vectorAddKernel(const float *a, const float *b, float *c,
                                size_t n) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}

__host__ void vectorAdd(const float *h_a, const float *h_b, float *h_c,
                        size_t n) {
  float *d_a, *d_b, *d_c;
  const size_t size = n * sizeof(float);

  HANDLE_ERROR(cudaMalloc((void **)&d_a, size));
  HANDLE_ERROR(cudaMalloc((void **)&d_b, size));
  HANDLE_ERROR(cudaMalloc((void **)&d_c, size));

  HANDLE_ERROR(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

  size_t grid_dim = std::ceil(static_cast<float>(n) / 256.0);
  vectorAddKernel<<<grid_dim, 256>>>(d_a, d_b, d_c, n);

  HANDLE_ERROR(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

int main(int argc, char *argv[]) {

  size_t n = getVectorSize(argc, argv);

  auto *a = new float[n];
  auto *b = new float[n];
  auto *c = new float[n];

  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(-i);
    b[i] = static_cast<float>(i * i);
  }

  vectorAdd(a, b, c, n);

  for (int i = 0; i < n; ++i) {
    std::printf("%f + %f = %f\n", a[i], b[i], c[i]);
  }

  delete[] a;
  delete[] b;
  delete[] c;
}
