#include <cuda.h>

#include <cstdio>
#include <cstdlib>

#include "time.h"

using namespace std;

const int ARRAY_SIZE = 16;
const int nBlocks = 2;
const int nThreads = 8;

__global__ void task2(int *dA, int a) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int x = threadIdx.x;
    int y = blockIdx.x;
    dA[index] = a * x + y;
}

int main() {
    srand((unsigned int)time(NULL));
    int a = (int)rand();
    int totalSize = ARRAY_SIZE * sizeof(int);

    int *dA;
    cudaMalloc((void **)&dA, totalSize);

    task2<<<nBlocks, nThreads>>>(dA, a);
    cudaDeviceSynchronize();

    int *hA;
    hA = (int *)malloc(totalSize);
    cudaMemcpy(hA, dA, totalSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (i == ARRAY_SIZE - 1) {
            std::printf("%d", hA[i]);
        } else {
            std::printf("%d ", hA[i]);
        }
    }

    cudaFree(dA);
    free(hA);
}
