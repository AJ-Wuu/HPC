#define CUB_STDERR // print CUDA runtime errors to console

#include "time.h"
#include "cub/util_debug.cuh"
#include <cuda.h>
#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // caching allocator for device memory

int main(int argc, char* argv[]) {
    srand((unsigned int)time(NULL));
    long n = atol(argv[1]);

    // fill in the host array
    float *h_in = new float[n];
    for (long i = 0; i < n; i++) {
        h_in[i] = ((float)rand() / (RAND_MAX)) * 2 - 1;  // [-1.0, 1.0]
    }

    // set up device arrays
    float* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) * n));
    // initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));
    // setup device output array
    float* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1));
    // request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float gpu_sum;
    CubDebugExit(cudaMemcpy(&gpu_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost));
    std::printf("%f\n", gpu_sum);
    std::printf("%f\n", ms);

    // cleanup
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}
