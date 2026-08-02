#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "matmul_cpu.h"  // brings in N so both sides agree on shape

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(err));                   \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                      \
    } while (0)

__global__ void matmul_gpu_kernel(const float* a, const float* b, float* c);

// Same outer product as matmul_cpu, but on the GPU. Takes care of the
// device malloc/copy/launch/free dance so callers just pass host pointers.
// h_c is a host float** (array of row pointers); the device itself works
// on a flat buffer since it can't dereference host row pointers.
void matmul_gpu(const float* h_a, const float* h_b, float** h_c);
