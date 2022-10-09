#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include "time.h"
#include "matmul.cuh"

using namespace std;

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    long n = atol(argv[1]);
    long block_dim = atol(argv[2]);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    int *intA, *intB, *intC;
    cudaMallocManaged((void **)&intA, n * n * sizeof(int));
    cudaMallocManaged((void **)&intB, n * n * sizeof(int));
    cudaMallocManaged((void **)&intC, n * n * sizeof(int));
    for (int i = 0; i < n * n; i++) {
        intA[i] = i + 1; //(int)rand();
        intB[i] = i + 1; //(int)rand();
        intC[i] = 0;
    }
    cudaEventRecord(start);
    matmul_1(intA, intB, intC, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%d\n", intC[0]);
    printf("%d\n", intC[n * n - 1]);
    printf("%f\n", ms);
    cudaFree(intA);
    cudaFree(intB);
    cudaFree(intC);

    float *floatA, *floatB, *floatC;
    cudaMallocManaged((void **)&floatA, n * n * sizeof(float));
    cudaMallocManaged((void **)&floatB, n * n * sizeof(float));
    cudaMallocManaged((void **)&floatC, n * n * sizeof(float));
    for (int i = 0; i < n * n; i++) {
        floatA[i] = i + 1; //(float)rand();
        floatB[i] = i + 1; //(float)rand();
        floatC[i] = 0.0;
    }
    cudaEventRecord(start);
    matmul_2(floatA, floatB, floatC, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%f\n", floatC[0]);
    printf("%f\n", floatC[n * n - 1]);
    printf("%f\n", ms);
    cudaFree(floatA);
    cudaFree(floatB);
    cudaFree(floatC);

    double *doubleA, *doubleB, *doubleC;
    cudaMallocManaged((void **)&doubleA, n * n * sizeof(double));
    cudaMallocManaged((void **)&doubleB, n * n * sizeof(double));
    cudaMallocManaged((void **)&doubleC, n * n * sizeof(double));
    for (int i = 0; i < n * n; i++) {
        doubleA[i] = i + 1; //(double)rand();
        doubleB[i] = i + 1; //(double)rand();
        doubleC[i] = 0.0;
    }
    cudaEventRecord(start);
    matmul_3(doubleA, doubleB, doubleC, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%lf\n", doubleC[0]);
    printf("%lf\n", doubleC[n * n - 1]);
    printf("%f\n", ms);
    cudaFree(doubleA);
    cudaFree(doubleB);
    cudaFree(doubleC);
}
