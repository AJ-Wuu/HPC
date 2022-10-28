#include <cstdlib>
#include <iostream>

#include "matmul.h"

using namespace std;

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    int n = atoi(argv[1]);
    int t = atoi(argv[2]);

    float *A = (float *)malloc(n * n * sizeof(float));
    float *B = (float *)malloc(n * n * sizeof(float));
    float *C = (float *)malloc(n * n * sizeof(float));
    for (int i = 0; i < n * n; i++) {
        A[i] = ((float)rand() / (RAND_MAX));
        B[i] = ((float)rand() / (RAND_MAX));
        C[i] = 0;
    }

    omp_set_num_threads(t);
    double startTime = omp_get_wtime();  // time unit = second
    mmul(A, B, C, n);
    double endTime = omp_get_wtime();
    double ms = (endTime - startTime) * 1000;

    std::cout << C[0] << endl;
    std::cout << C[n * n - 1] << endl;
    std::cout << ms << endl;

    free(A);
    free(B);
    free(C);
}
