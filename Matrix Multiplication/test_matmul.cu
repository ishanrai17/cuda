// Build: nvcc -O2 test_matmul.cu matmul_gpu.cu -o test_matmul
// Run:   ./test_matmul

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

#include "matmul_cpu.h"
#include "matmul_gpu.cuh"

float** alloc_matrix() {
    float** m = new float*[N];
    for (int i = 0; i < N; ++i) {
        m[i] = new float[N];
    }
    return m;
}

void free_matrix(float** m) {
    for (int i = 0; i < N; ++i) {
        delete[] m[i];
    }
    delete[] m;
}

int main() {
    float a[N], b[N];
    float** c_cpu = alloc_matrix();
    float** c_gpu = alloc_matrix();

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        a[i] = dist(rng);
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
    int mismatches = 0;
    const float epsilon = 1e-4f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (std::fabs(c_cpu[i][j] - c_gpu[i][j]) > epsilon) {
                ++mismatches;
            }
        }
    }

    std::printf("Matmul: (%d, 1) @ (1, %d) -> (%d, %d)\n\n", N, N, N, N);
    std::printf("CPU: %.4f ms\n", cpu_ms);
    std::printf("GPU: %.4f ms\n", static_cast<double>(gpu_ms));
    std::printf("\nMismatches: %d / %d\n", mismatches, N * N);
    std::printf("Speedup (CPU / GPU): %.2fx\n", cpu_ms / gpu_ms);

    free_matrix(c_cpu);
    free_matrix(c_gpu);

    return mismatches == 0 ? 0 : 1;
}
