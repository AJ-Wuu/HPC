#include "mmul.h"
#include <cublas_v2.h>

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n) {
    /*
    cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta,
                           float *C, int ldc)
    */
    int lda = n, ldb = n, ldc = n;
    const float alpha = 1.0, beta = 1.0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
    cudaDeviceSynchronize();
}