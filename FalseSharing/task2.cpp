#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <ratio>

#include "montecarlo.h"

using namespace std;
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char **argv) {
    long n = atol(argv[1]);
    long t = atol(argv[2]);
    srand((unsigned int)time(NULL));

    float *x = (float *)malloc(n * sizeof(float));
    float *y = (float *)malloc(n * sizeof(float));
    float r = 1;
    for (int i = 0; i < n; i++) {
        x[i] = ((float)rand() / (RAND_MAX)) * 2 * r - r;  // [-r, r]
        y[i] = ((float)rand() / (RAND_MAX)) * 2 * r - r;
    }

    high_resolution_clock::time_point start, end;
    double ms_total = 0;
    duration<double, std::milli> ms;
    int incircle;
    for (int i = 0; i < 10; i++) {
        start = high_resolution_clock::now();
        omp_set_num_threads(t);
        incircle = montecarlo(n, x, y, r);
        end = high_resolution_clock::now();
        ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

        ms_total += ms.count();
    }

    float pi = (float)(4 * incircle) / n;
    cout << pi << endl;
    cout << (ms_total / 10) << endl;

    free(x);
    free(y);
}
