#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include "time.h"
#include "matmul.cuh"

using namespace std;

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    long n = atol(argv[1]);
    long threads_per_block = atol(argv[2]);
    long totalSize = n * n * sizeof(float);

    float *dA, *dB, *dC;
    // cudaMallocManaged() simplifies memory access by eliminating the need for explicit memory allocations on host and device
    cudaMallocManaged((void **)&dA, totalSize);
    cudaMallocManaged((void **)&dB, totalSize);
    cudaMallocManaged((void **)&dC, totalSize);
    for (int i = 0; i < n * n; i++) {
        dA[i] = ((float)rand() / (RAND_MAX)) * 2 - 1;  // [-1.0, 1.0]
        dB[i] = ((float)rand() / (RAND_MAX)) * 2 - 1;
        dC[i] = 0.0;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul(dA, dB, dC, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("%f\n", dC[n * n - 1]);
    printf("%f\n", ms);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
