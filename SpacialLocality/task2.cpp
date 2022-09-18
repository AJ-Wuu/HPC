#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <ratio>

#include "convolution.h"

using namespace std;
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    srand((unsigned int)time(NULL));

    float *image = (float *)malloc(n * n * sizeof(float));
    float *mask = (float *)malloc(m * m * sizeof(float));
    float *output = (float *)malloc(n * n * sizeof(float));

    for (int i = 0; i < n * n; i++) {
        image[i] = ((float)rand() / (RAND_MAX)) * 20 - 10;  // [-10.0, 10.0]
    }
    for (int i = 0; i < m * m; i++) {
        mask[i] = ((float)rand() / (RAND_MAX)) * 2 - 1;  // [-1.0, 1.0]
    }

    high_resolution_clock::time_point start = high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double, std::milli> duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);  // convert the calculated duration to a double

    std::cout << duration_sec.count() << endl;
    std::cout << output[0] << endl;
    std::cout << output[n * n - 1] << endl;

    delete image;
    delete mask;
    delete output;
}