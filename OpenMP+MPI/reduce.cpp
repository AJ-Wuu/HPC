#include "reduce.h"

float reduce(const float* arr, const size_t l, const size_t r) {
    float result = 0;
#pragma omp parallel for simd reduction(+:result)
    for (size_t i = l; i < r; i++) {
        result += arr[i];
    }
    return result;
}