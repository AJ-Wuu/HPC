#include "mmul.h"
#include "time.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    long n = atol(argv[1]);
    long n_tests = atol(argv[2]);
    long totalSize = n * n * sizeof(float);

    float *A, *B, *C; // stored in managed memory
    cudaMallocManaged(&A, totalSize);
    cudaMallocManaged(&B, totalSize);
    cudaMallocManaged(&C, totalSize);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            int index = i * n + j; // column-major order 
            A[index] = ((float)rand() / (RAND_MAX)) * 2 - 1;  // [-1.0, 1.0]
            B[index] = ((float)rand() / (RAND_MAX)) * 2 - 1;
            C[index] = 0.0;
        }
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_total = 0.0, ms;
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (int i = 0; i < n_tests; i++) {
        cudaEventRecord(start, 0);
        mmul(handle, A, B, C, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        ms_total += ms;

        /*
        for (int j = 0; j < n * n; j++) { // re-initialize C -- takes too long to finish
            C[j] = 0.0;
        }
        */
    }

    printf("%f\n", ms_total / n_tests);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cublasDestroy(handle);
}