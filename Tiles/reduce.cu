#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sharedArray[];
    int index = threadIdx.x + blockIdx.x * blockDim.x * 2; // two entries together
    // bring in value
    if (index + blockDim.x < n) {
        sharedArray[threadIdx.x] = g_idata[index] + g_idata[index + blockDim.x];
    }
    else if (index < n) {
        sharedArray[threadIdx.x] = g_idata[index];
    }
    else {
        sharedArray[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int temp = blockDim.x / 2; temp > 0; temp >>= 1) {
        if (threadIdx.x < temp) {
            sharedArray[threadIdx.x] += sharedArray[threadIdx.x + temp];
        }
        __syncthreads();
    }

    g_odata[blockIdx.x] = sharedArray[0];
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    int temp, nBlocks = 1;
    while (nBlocks > 0) {
        temp = (N % 2 == 0) ? N/2 : N/2 + 1;
        nBlocks = (temp + threads_per_block - 1) / threads_per_block;
        reduce_kernel<<<nBlocks, threads_per_block, sizeof(float)*threads_per_block>>>(*input, *output, N);
        cudaDeviceSynchronize();
        *input = *output;

        if (nBlocks == 1) {
            break;
        }
        N = nBlocks;
    }
}