#include <cstdlib>
#include <iostream>

#include "convolution.h"

using namespace std;

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    int n = atoi(argv[1]);
    int m = 3;
    int t = atoi(argv[2]);

    float *image = (float *)malloc(n * n * sizeof(float));
    float mask[] = {0, 0, 1, 0, 1, 0, 1, 0, 0};
    float *output = (float *)malloc(n * n * sizeof(float));
    for (int i = 0; i < n * n; i++) {
        image[i] = ((float)rand() / (RAND_MAX)) * 20 - 10;  // [-10.0, 10.0]
        output[i] = 0;
    }

    omp_set_num_threads(t);
    double startTime = omp_get_wtime();
    convolve(image, output, n, mask, m);
    double endTime = omp_get_wtime();
    double ms = (endTime - startTime) * 1000;

    std::cout << output[0] << endl;
    std::cout << output[n * n - 1] << endl;
    std::cout << ms << endl;

    free(image);
    free(output);
}
