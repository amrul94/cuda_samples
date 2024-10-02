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

#include "common/error_handling.cuh"

constexpr const char *isSupported(int flag) {
  return flag ? "supported" : "unsupported";
}

int main() {
  cudaDeviceProp prop{};
  int count;
  HANDLE_ERROR(cudaGetDeviceCount(&count));

  for (int i = 0; i < count; i++) {
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
    printf(" --- General information about the device %d ---\n", i);
    printf("Name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Number of asynchronous engines: %d\n", prop.asyncEngineCount);
    printf("Device is %s\n", prop.integrated ? "integrated" : "discrete");
    printf("Device can map host memory into CUDA address space: %s\n",
           prop.canMapHostMemory ? "yes" : "no");
    printf("Device %s possibly execute multiple kernels concurrently\n",
           prop.concurrentKernels ? "can" : "can't");
    printf("Device has ECC support: %s\n",
           prop.ECCEnabled ? "enabled" : "disabled");
    printf("PCI bus ID of the device: %d\n", prop.pciBusID);
    printf("PCI device ID of the device: %d\n", prop.pciDeviceID);
    printf("PCI domain ID of the device: %d\n", prop.pciDomainID);
    printf("Device supports stream priorities: %s\n",
           isSupported(prop.streamPrioritiesSupported));
    printf("Device is on a multi-GPU board: %s\n",
           prop.isMultiGpuBoard ? "yes" : "no");
    // TODO: последнее, что записал
    printf("Link between the device and the host supports native atomic "
           "operations: %s\n",
           isSupported(prop.hostNativeAtomicSupported));
    printf("\n");

    printf(" --- Device multiprocessor information %d ---\n", i);
    printf("Number of multiprocessors on device: %d\n",
           prop.multiProcessorCount);
    printf("Shared memory available per block in bytes: %ld\n",
           prop.sharedMemPerBlock);
    printf("Shared memory available per multiprocessor in bytes: %ld\n",
           prop.sharedMemPerMultiprocessor);
    printf("32-bit registers available per block: %d\n", prop.regsPerBlock);
    printf("32-bit registers available per multiprocessor: %d\n",
           prop.regsPerMultiprocessor);
    printf("Warp size in threads: %d\n", prop.warpSize);
    printf("Maximum number of threads per block: %d\n",
           prop.maxThreadsPerBlock);
    printf("Maximum size of each dimension of a block: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Maximum size of each dimension of a grid: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum resident threads per multiprocessor: %d\n\n",
           prop.maxThreadsPerMultiProcessor);

    printf(" --- Device memory information %d ---\n", i);
    printf("Global memory available on device in bytes: %ld\n",
           prop.totalGlobalMem);
    printf("Constant memory available on device in bytes: %ld\n",
           prop.totalConstMem);
    printf("Maximum pitch in bytes allowed by memory copies: %ld\n",
           prop.memPitch);
    printf("Device %s a unified address space with the host\n",
           prop.unifiedAddressing ? "shares" : "does not share");
    printf("Global memory bus width in bits: %d\n", prop.memoryBusWidth);
    printf("Size of L2 cache in bytes: %d\n", prop.l2CacheSize);
    printf(
        "Device's maximum l2 persisting lines capacity setting in bytes: %d\n",
        prop.persistingL2CacheMaxSize);
    printf("Device supports caching globals in L1: %s\n",
           isSupported(prop.globalL1CacheSupported));
    printf("Device supports caching locals in L1: %s\n",
           isSupported(prop.localL1CacheSupported));
    printf("Device supports allocating managed memory on this system: %s\n",
           isSupported(prop.managedMemory));
    printf("\n");

    printf(" --- Device texture memory information %d ---\n", i);
    printf("Alignment requirement for textures: %ld\n", prop.textureAlignment);
    printf("Pitch alignment requirement for texture references bound to "
           "pitched memory: %ld\n",
           prop.texturePitchAlignment);
    printf("Maximum 1D texture size: %d\n", prop.maxTexture1D);
    printf("Maximum 1D mipmapped texture size: %d\n", prop.maxTexture1DMipmap);
    printf("Maximum 2D texture dimensions: (%d, %d)\n", prop.maxTexture2D[0],
           prop.maxTexture2D[1]);
    printf("Maximum 2D mipmapped texture dimensions: (%d, %d)\n",
           prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]);
    printf("Maximum dimensions (width, height, pitch) for 2D textures bound to "
           "pitched memory: (%d, %d, %d)\n",
           prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1],
           prop.maxTexture2DLinear[2]);
    printf("Maximum 2D texture dimensions if texture gather operations have "
           "to be performed: (%d, %d)\n",
           prop.maxTexture2DGather[0], prop.maxTexture2DGather[1]);
    printf("Maximum 3D texture dimensions: (%d, %d, %d)\n",
           prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
    printf("Maximum alternate 3D texture dimensions: (%d, %d, %d)\n",
           prop.maxTexture3DAlt[0], prop.maxTexture3DAlt[1],
           prop.maxTexture3DAlt[2]);
    printf("Maximum 1D layered texture dimensions: (%d, %d)\n",
           prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);
    printf("Maximum 2D layered texture dimensions: (%d, %d, %d)\n",
           prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1],
           prop.maxTexture2DLayered[2]);
    printf("Maximum Cubemap texture dimensions: %d\n", prop.maxTextureCubemap);
    printf("Maximum Cubemap layered texture dimensions: (%d, %d)\n\n",
           prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]);

    printf(" --- Device surface memory information %d ---\n", i);
    printf("Alignment requirement for surfaces: %ld\n", prop.surfaceAlignment);
    printf("Maximum 1D surface size: %d\n", prop.maxSurface1D);
    printf("Maximum 2D surface dimensions: (%d, %d)\n", prop.maxSurface2D[0],
           prop.maxSurface2D[1]);
    printf("Maximum 3D surface dimensions: (%d, %d, %d)\n",
           prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]);
    printf("Maximum 1D layered surface dimensions: (%d, %d)\n",
           prop.maxSurface1DLayered[0], prop.maxSurface1DLayered[1]);
    printf("Maximum 2D layered surface dimensions: (%d, %d, %d)\n",
           prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1],
           prop.maxSurface2DLayered[2]);
    printf("Maximum Cubemap surface dimensions: %d\n", prop.maxSurfaceCubemap);
    printf("Maximum Cubemap layered surface dimensions: (%d, %d)\n\n",
           prop.maxSurfaceCubemapLayered[0], prop.maxSurfaceCubemapLayered[1]);
  }
}