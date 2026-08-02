#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "matmul_cpu.h"  // brings in N and Matrix so both sides agree on shape

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(err));                   \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                      \
    } while (0)

__global__ void matmul_gpu_kernel(const float* a, const float* b, Matrix c);

// Same outer product as matmul_cpu, but on the GPU. Takes care of the
// device malloc/copy/launch/free dance so callers just pass host pointers.
void matmul_gpu(const float h_a[N], const float h_b[N], Matrix h_c);
