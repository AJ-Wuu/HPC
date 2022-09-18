#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <ratio>

#include "matmul.h"

using namespace std;
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main() {
    int n = 1024;
    srand((unsigned int)time(NULL));

    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(n * n * sizeof(double));
    std::vector<double> A_vector;
    std::vector<double> B_vector;
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> duration_sec;

    for (int i = 0; i < n * n; i++) {
        double temp = ((double)rand() / (RAND_MAX));
        A[i] = temp;
        A_vector.push_back(temp);

        temp = ((double)rand() / (RAND_MAX));
        B[i] = temp;
        B_vector.push_back(temp);
    }
    std::cout << n << endl;

    start = high_resolution_clock::now();
    mmul1(A, B, C, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << duration_sec.count() << endl;
    std::cout << C[n * n - 1] << endl;

    start = high_resolution_clock::now();
    mmul2(A, B, C, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << duration_sec.count() << endl;
    std::cout << C[n * n - 1] << endl;

    start = high_resolution_clock::now();
    mmul3(A, B, C, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << duration_sec.count() << endl;
    std::cout << C[n * n - 1] << endl;

    start = high_resolution_clock::now();
    mmul4(A_vector, B_vector, C, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << duration_sec.count() << endl;
    std::cout << C[n * n - 1] << endl;

    delete A;
    delete B;
    delete C;
    delete &A_vector;
    delete &B_vector;
}