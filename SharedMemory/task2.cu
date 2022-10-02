#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include "time.h"
#include "stencil.cuh"

using namespace std;

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    long n = atol(argv[1]);
    long R = atol(argv[2]);
    long threads_per_block = atol(argv[3]);

    float *image, *mask, *output;
    cudaMallocManaged((void **)&image, n * sizeof(float));
    cudaMallocManaged((void **)&mask, (2 * R + 1) * sizeof(float));
    cudaMallocManaged((void **)&output, n * sizeof(float));
    for (int i = 0; i < n; i++) {
        image[i] = ((float)rand() / (RAND_MAX)) * 2 - 1;  // [-1.0, 1.0]
    }
    for (int i = 0; i < 2 * R + 1; i++) {
        mask[i] = ((float)rand() / (RAND_MAX)) * 2 - 1;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    stencil(image, mask, output, n, R, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("%f\n", output[n - 1]);
    printf("%f\n", ms);

    cudaFree(image);
    cudaFree(mask);
    cudaFree(output);
}