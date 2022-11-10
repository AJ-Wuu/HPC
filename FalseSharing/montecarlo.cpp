#include "montecarlo.h"

int montecarlo(const size_t n, const float *x, const float *y, const float radius) {
    int incircle = 0;
#pragma omp parallel
    {
#pragma omp for simd reduction(+ : incircle)
// #pragma omp for reduction(+ : incircle)
        for (size_t i = 0; i < n; i++) {
            // increment 1 if x^2 + y^2 < r^2
            // else, stay the same
            // note that "<" or "<=" doesn't really matter -- it's almost impossible to get "="
            incircle += (x[i] * x[i] + y[i] * y[i] < radius * radius);
        }
    }
    return incircle;
}