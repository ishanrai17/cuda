// Build: nvcc -O2 test_matmul.cu matmul_gpu.cu -o test_matmul
// Run:   ./test_matmul
//
// Shape is overridable at build time, e.g. to get the old memory-bound
// outer-product case back:
//   nvcc -O2 -DMATMUL_M=8192 -DMATMUL_K=1 -DMATMUL_N=8192 \
//        test_matmul.cu matmul_gpu.cu -o test_matmul

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

#include "matmul_cpu.h"
#include "matmul_gpu.cuh"

float** alloc_matrix(int rows, int cols) {
    float** m = new float*[rows];
    for (int i = 0; i < rows; ++i) {
        m[i] = new float[cols];
    }
    return m;
}

void free_matrix(float** m, int rows) {
    for (int i = 0; i < rows; ++i) {
        delete[] m[i];
    }
    delete[] m;
}

int main() {
    // These are large enough that stack arrays would overflow, so heap-allocate.
    // a and b are flat row-major; only the outputs use row pointers.
    float* a = new float[static_cast<size_t>(M) * K];
    float* b = new float[static_cast<size_t>(K) * N];
    float** c_cpu = alloc_matrix(M, N);
    float** c_gpu = alloc_matrix(M, N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i) {
        a[i] = dist(rng);
    }
    for (size_t i = 0; i < static_cast<size_t>(K) * N; ++i) {
        b[i] = dist(rng);
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(a, b, c_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // CUDA events give more accurate GPU timing than wall-clock chrono would.
    // This also covers the host<->device transfer time, not just the kernel.
    cudaEvent_t gpu_start, gpu_stop;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_stop));

    CUDA_CHECK(cudaEventRecord(gpu_start));
    matmul_gpu(a, b, c_gpu);
    CUDA_CHECK(cudaEventRecord(gpu_stop));
    CUDA_CHECK(cudaEventSynchronize(gpu_stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop));
    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_stop));

    // CPU and GPU should agree everywhere; count anything that doesn't.
    // CPU and GPU sum the K products in the same order but the GPU may contract
    // multiply-add pairs into FMAs, so tolerance has to grow with K rather than
    // stay at the single-multiply epsilon the outer-product version used.
    int mismatches = 0;
    const float epsilon = 1e-4f * K;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (std::fabs(c_cpu[i][j] - c_gpu[i][j]) > epsilon) {
                ++mismatches;
            }
        }
    }

    // 2 FLOP per multiply-add, M*N*K of them.
    double gflop = 2.0 * M * N * K / 1e9;

    std::printf("Matmul: (%d, %d) @ (%d, %d) -> (%d, %d)\n", M, K, K, N, M, N);
    std::printf("%.2f GFLOP of work\n\n", gflop);
    std::printf("CPU: %.4f ms  (%.1f GFLOP/s)\n", cpu_ms, gflop / (cpu_ms / 1e3));
    std::printf("GPU: %.4f ms  (%.1f GFLOP/s)\n", static_cast<double>(gpu_ms),
                gflop / (gpu_ms / 1e3));
    std::printf("\nMismatches: %d / %d\n", mismatches, M * N);
    std::printf("Speedup (CPU / GPU): %.2fx\n", cpu_ms / gpu_ms);

    delete[] a;
    delete[] b;
    free_matrix(c_cpu, M);
    free_matrix(c_gpu, M);

    return mismatches == 0 ? 0 : 1;
}
