#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    extern __shared__ float shared_array[]; // declare dynamic shared memory

    /*
        store in the order of image (block_size + R * 2), mask (R * 2 + 1), output (block_size)
        reference: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/

         shared_image   _mask  _output
        |_____________|_______|_______|
        |-R-|     |-R-|
            ↑
            i = threadIdx.x + R
     */
    float* shared_image = shared_array; // same as &shared_array[0]
    float* shared_mask = &shared_array[blockDim.x + R * 2];
    float* shared_output = &shared_array[blockDim.x + R * 4 + 1];
    int index_global = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.x + R;

    // assign values from global memory to shared memory
    // image
    shared_image[i] = (index_global < n) ? image[index_global] : 1.0;
    if (threadIdx.x < R) { // i < 0
        shared_image[i - R] = (index_global >= R) ? image[index_global - R] : 1.0;
    }
    if (threadIdx.x >= blockDim.x - R) { // i > n - 1
        shared_image[i + R] = (index_global < n - R) ? image[index_global + R] : 1.0;
    }
    // mask
    if (threadIdx.x < 2 * R + 1) {
        shared_mask[threadIdx.x] = mask[threadIdx.x];
    }
    // output
    shared_output[threadIdx.x] = 0.0;
    __syncthreads();

    // calculate convolution
    int j = (int)R * -1; // change R from unsigned to signed
    for (; j <= (int)R; j++) {
        shared_output[threadIdx.x] += shared_image[i + j] * shared_mask[j + (int)R];
    }

    // load result from shared memory to global memory
    if (index_global < n) { // valid index_global
        output[index_global] = shared_output[threadIdx.x];
    }
}

__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block) {
    unsigned int nBlocks = (n + threads_per_block - 1) / threads_per_block;

    // image_size  = threads_per_block + R * 2
    // mask_size   =                     R * 2 + 1
    // output_size = threads_per_block
    // shared_size = image + mask + output -> all in float
    unsigned int shared_size = (threads_per_block * 2 + R * 4 + 1) * sizeof(float);

    stencil_kernel<<<nBlocks, threads_per_block, shared_size>>>(image, mask, output, n, R);
    cudaDeviceSynchronize();
}
