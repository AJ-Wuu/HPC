#include "count.cuh"
#include <cuda.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void count(const thrust::device_vector<int>& d_in, thrust::device_vector<int>& values, thrust::device_vector<int>& counts) {
    thrust::device_vector<int> d_in_asc = d_in;
    thrust::sort(d_in_asc.begin(), d_in_asc.end());

    // notice that the length of values and counts may not be equal to the length of d_in
    // use thrust::inner product to find the number of “jumps” (when a[i-1] != a[i]) as stepping through the sorted array
    size_t unique = thrust::inner_product(d_in_asc.begin(), d_in_asc.end() - 1, d_in_asc.begin() + 1, 0, thrust::plus<int>(), thrust::not_equal_to<int>()) + 1;
    values.resize(unique); // with the unique integers that appear in d_in in ascending order
    counts.resize(unique); // with the corresponding occurrences of these integers
    
    thrust::device_vector<int> temp(d_in.size(), 1);
    thrust::reduce_by_key(d_in_asc.begin(), d_in_asc.end(), temp.begin(), values.begin(), counts.begin());
}
