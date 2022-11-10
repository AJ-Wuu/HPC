#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <ratio>

#include "cluster.h"

using namespace std;
using std::cout;
using std::sort;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char **argv) {
    long n = atol(argv[1]);
    long t = atol(argv[2]);
    srand((unsigned int)time(NULL));

    float *arr = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        arr[i] = ((float)rand() / (RAND_MAX)) * n;  // [0, n]
    }

    sort(arr, arr + n);

    float *centers = (float *)malloc(t * sizeof(float));
    for (int i = 0; i < t; i++) {
        centers[i] = ((i + 1) * 2 - 1) * n / (2 * t);
    }

    float *dists = (float *)malloc(t * sizeof(float));

    high_resolution_clock::time_point start, end;
    double ms_total = 0;
    duration<double, std::milli> ms;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < t; j++) {
            dists[j] = 0;
        }

        start = high_resolution_clock::now();
        cluster(n, t, arr, centers, dists);
        end = high_resolution_clock::now();
        ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

        ms_total += ms.count();
    }

    int max = -1, partition = 0;
    for (int i = 0; i < t; i++) {
        if (dists[i] > max) {
            max = dists[i];
            partition = i;
        }
    }

    cout << max << endl;
    cout << partition << endl;
    cout << (ms_total / 10) << endl;

    free(arr);
    free(centers);
    free(dists);
}