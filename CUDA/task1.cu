#include <cuda.h>

#include <cstdio>

using namespace std;

const unsigned int nThreads = 8;

__global__ void task1() {
    int count = 1;
    if (threadIdx.x < nThreads) {
        for (int i = 2; i <= threadIdx.x + 1; i++) {
            count *= i;
        }
        std::printf("%u!=%d\n", threadIdx.x + 1, count);
    }
}

int main() {
    task1<<<1, nThreads>>>();
    cudaDeviceSynchronize();
}
