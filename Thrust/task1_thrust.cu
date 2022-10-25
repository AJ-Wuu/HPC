#include "time.h"
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;

// reference: https://stackoverflow.com/questions/12614164/generating-random-numbers-with-uniform-distribution-using-thrust
__host__ static __inline__ float rand_custom() {
    srand((unsigned int)time(NULL));
    return ((float)rand() / RAND_MAX) * 2 - 1.f;
}

int main(int argc, char* argv[]) {
    long n = atol(argv[1]);

    // generate random numbers on the host
    thrust::host_vector<float> h_vec(n);
    thrust::generate(h_vec.begin(), h_vec.end(), rand_custom);

    // transfer data to the device
    thrust::device_vector<float> d_vec = h_vec;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float result = thrust::reduce(d_vec.begin(), d_vec.end());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("%f\n", result);
    printf("%f\n", ms);
}
