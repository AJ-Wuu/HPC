#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row = idx / n, col = idx % n;
    for (int k = 0; k < n; k++) {
        C[idx] += A[row * n + k] * B[k * n + col];
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    unsigned int nBlocks = (n * n + threads_per_block - 1) / threads_per_block;
    matmul_kernel<<<nBlocks, threads_per_block>>>(A, B, C, n);
    cudaDeviceSynchronize();
}
