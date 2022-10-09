#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include "time.h"
#include "reduce.cuh"

using namespace std;

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    long N = atol(argv[1]);
    long threads_per_block = atol(argv[2]);
    int temp = (N % 2 == 0) ? N/2 : N/2 + 1;
    int nBlocks = (temp + threads_per_block - 1) / threads_per_block;

    float *array = new float[N];
    for (int i=0; i<N; i++) {
        array[i] = ((float)rand() / (RAND_MAX)) * 2 - 1;  // [-1.0, 1.0]
    }

    float *input, *output;
    cudaMallocManaged((void **)&input, N * sizeof(float));
    cudaMallocManaged((void **)&output, nBlocks * sizeof(float));
    cudaMemcpy(input, array, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce(&input, &output, N, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("%f\n", output[0]);
    printf("%f\n", ms);

    cudaFree(input);
    cudaFree(output);
}
