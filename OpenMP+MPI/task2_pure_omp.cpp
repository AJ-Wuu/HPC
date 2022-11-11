#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <ratio>

#include "reduce.h"

using namespace std;
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char **argv) {
    long n = atol(argv[1]);
    long t = atol(argv[2]);
    srand((unsigned int)time(NULL));

    float *arr = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        arr[i] = ((float)rand() / (RAND_MAX)) * 2 - 1;  // [-1, 1]
    }
    float res = 0;

    high_resolution_clock::time_point start, end;
    duration<double, std::milli> ms;
    start = high_resolution_clock::now();
    omp_set_num_threads(t);
    res = reduce(arr, 0, n);
    end = high_resolution_clock::now();
    ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << res << endl;
    cout << ms.count() << endl;

    free(arr);
}
