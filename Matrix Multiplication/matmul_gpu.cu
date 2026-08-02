#include "matmul_gpu.cuh"

__global__ void matmul_gpu_kernel(const float* a, const float* b, Matrix c) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        c[i][j] = a[i] * b[j];
    }
}

void matmul_gpu(const float h_a[N], const float h_b[N], Matrix h_c) {
    float *d_a, *d_b;
    Matrix* d_c;
    size_t vec_bytes = static_cast<size_t>(N) * sizeof(float);
    size_t mat_bytes = sizeof(Matrix);

    CUDA_CHECK(cudaMalloc(&d_a, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, mat_bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, vec_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matmul_gpu_kernel<<<grid, block>>>(d_a, d_b, *d_c);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, mat_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}
