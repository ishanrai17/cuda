#include <cstring>

#include "matmul_gpu.cuh"

__global__ void matmul_gpu_kernel(const float* a, const float* b, float* c) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        // Each thread reduces over the full K dimension for its output element.
        // Consecutive threads in a warp have consecutive j, so the b[k * N + j]
        // read is coalesced; a[i * K + k] is broadcast across the row.
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += a[i * K + k] * b[k * N + j];
        }
        c[i * N + j] = acc;
    }
}

void matmul_gpu(const float* h_a, const float* h_b, float** h_c) {
    float *d_a, *d_b, *d_c;
    size_t a_bytes = static_cast<size_t>(M) * K * sizeof(float);
    size_t b_bytes = static_cast<size_t>(K) * N * sizeof(float);
    size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);
    size_t row_bytes = static_cast<size_t>(N) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_gpu_kernel<<<grid, block>>>(d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());

    // One big device->host copy into a flat scratch buffer, then fan it out into
    // the caller's (non-contiguous) row pointers with cheap host-side memcpys.
    // N separate cudaMemcpy calls instead of this is what made the GPU path
    // slower than the CPU one -- each cudaMemcpy has real fixed overhead.
    float* flat_c = new float[static_cast<size_t>(M) * N];
    CUDA_CHECK(cudaMemcpy(flat_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < M; ++i) {
        std::memcpy(h_c[i], flat_c + static_cast<size_t>(i) * N, row_bytes);
    }
    delete[] flat_c;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}
