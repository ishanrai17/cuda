#pragma once

constexpr int N = 8192;

// c = a @ b, outer product of a column vector and a row vector.
inline void matmul_cpu(const float* a, const float* b, float** c) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            c[i][j] = a[i] * b[j];
        }
    }
}
