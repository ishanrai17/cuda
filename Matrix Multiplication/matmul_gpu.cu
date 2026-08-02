#include <cstring>

#include "matmul_gpu.cuh"

__global__ void matmul_gpu_kernel(const float* a, const float* b, float* c) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        c[i * N + j] = a[i] * b[j];
    }
}

void matmul_gpu(const float* h_a, const float* h_b, float** h_c) {
    float *d_a, *d_b, *d_c;
    size_t vec_bytes = static_cast<size_t>(N) * sizeof(float);
    size_t mat_bytes = static_cast<size_t>(N) * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_a, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, mat_bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, vec_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matmul_gpu_kernel<<<grid, block>>>(d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());

    // One big device->host copy into a flat scratch buffer, then fan it out into
    // the caller's (non-contiguous) row pointers with cheap host-side memcpys.
    // N separate cudaMemcpy calls instead of this is what made the GPU path
    // slower than the CPU one -- each cudaMemcpy has real fixed overhead.
    float* flat_c = new float[static_cast<size_t>(N) * N];
    CUDA_CHECK(cudaMemcpy(flat_c, d_c, mat_bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        std::memcpy(h_c[i], flat_c + i * N, vec_bytes);
    }
    delete[] flat_c;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}
