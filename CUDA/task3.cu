#include <cuda.h>

#include <cstdio>
#include <cstdlib>

#include "time.h"
#include "vscale.cuh"

using namespace std;

const int nThreadsPerBlock = 512;

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    srand((unsigned int)time(NULL));
    int nBlocks = (n - 1) / nThreadsPerBlock + 1;
    int totalSize = n * sizeof(float);

    float *ha = (float *)malloc(totalSize);
    float *hb = (float *)malloc(totalSize);
    for (int i = 0; i < n; i++) {
        ha[i] = ((float)rand() / (RAND_MAX)) * 20 - 10;  // [-10.0, 10.0]
        hb[i] = ((float)rand() / (RAND_MAX));            // [0.0, 1.0]
    }

    float *da, *db;
    cudaMalloc((void **)&da, totalSize);
    cudaMalloc((void **)&db, totalSize);
    cudaMemcpy(da, ha, totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, totalSize, cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vscale<<<nBlocks, nThreadsPerBlock>>>(da, db, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;  // get the elapsed time in milliseconds
    cudaEventElapsedTime(&ms, start, stop);

    std::printf("%f\n", ms);
    std::printf("%f\n", hb[0]);
    std::printf("%f\n", hb[n - 1]);

    free(ha);
    free(hb);
    cudaFree(da);
    cudaFree(db);
}
