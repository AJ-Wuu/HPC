#include "count.cuh"
#include "device_launch_parameters.h"
#include "time.h"
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <random>

using namespace std;

__host__ static __inline__ int rand_custom() {
    srand((unsigned int)time(NULL));
    int RANGE = 501; // integers in [0, 500]
    return (int)rand() % RANGE;
}

int main(int argc, char* argv[]) {
    long n = atol(argv[1]);

    thrust::host_vector<int> h_in(n);
    thrust::generate(h_in.begin(), h_in.end(), rand_custom);
    thrust::device_vector<int> d_in = h_in;
    thrust::device_vector<int> values;
    thrust::device_vector<int> counts;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    count(d_in, values, counts);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    int size = values.size();
    std::cout << values[size - 1] << std::endl;
    std::cout << counts[size - 1] << std::endl;
    std::printf("%f\n", ms);
}
