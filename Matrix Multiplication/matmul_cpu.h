#pragma once

// c = a @ b, where a is (M x K), b is (K x N), c is (M x N).
//
// K is the reduction depth, and it's what decides whether this benchmark is
// measuring arithmetic or just memory bandwidth. At K=1 the whole thing
// degenerates to an outer product: one multiply per output element, so ~0.25
// FLOP per byte written, and the GPU spends all its time on the PCIe copy-back
// instead of on math. A few hundred and up gives the GPU real work to amortize
// the transfer against.
//
// Overridable at build time, e.g. -DMATMUL_K=1 to get the old memory-bound
// outer-product case back.
#ifndef MATMUL_M
#define MATMUL_M 2048
#endif
#ifndef MATMUL_K
#define MATMUL_K 512
#endif
#ifndef MATMUL_N
#define MATMUL_N 2048
#endif

constexpr int M = MATMUL_M;
constexpr int K = MATMUL_K;
constexpr int N = MATMUL_N;

// a and b are flat row-major buffers (M*K and K*N floats); c is an array of M
// row pointers, each N floats wide.
inline void matmul_cpu(const float* a, const float* b, float** c) {
    // i-k-j ordering: walks b and c along rows so the inner loop stays
    // sequential in memory. An i-j-k loop would stride down b by N floats on
    // every step and thrash the cache, which would make the CPU baseline
    // artificially bad rather than the fair comparison we want.
    for (int i = 0; i < M; ++i) {
        float* c_row = c[i];
        for (int j = 0; j < N; ++j) {
            c_row[j] = 0.0f;
        }
        for (int k = 0; k < K; ++k) {
            const float a_ik = a[i * K + k];
            const float* b_row = b + k * N;
            for (int j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}
