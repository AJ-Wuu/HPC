#include "scan.cuh"
#include "time.h"
#include <cuda.h>
#include <cstdio>
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    long n = atol(argv[1]);
    long threads_per_block = atol(argv[2]);

    float *input, *output;
    cudaMallocManaged(&input, n * sizeof(float));
    cudaMallocManaged(&output, n * sizeof(float));
    for (int i = 0; i < n; i++) {
        input[i] = ((float)rand() / (RAND_MAX)) * 2 - 1;  // [-1.0, 1.0]
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    scan(input, output, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("%f\n", output[n - 1]);
    printf("%f\n", ms);

    cudaFree(input);
    cudaFree(output);
}