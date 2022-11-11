#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <ratio>

#include "mpi.h"
#include "reduce.h"

using namespace std;
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char **argv) {
    long n = atol(argv[1]);
    long t = atol(argv[2]);
    srand((unsigned int)time(NULL));

    float *arr = (float *)malloc(2 * n * sizeof(float));
    for (int i = 0; i < 2 * n; i++) {
        arr[i] = ((float)rand() / (RAND_MAX)) * 2 - 1;  // [-1, 1]
    }
    float res = 0, global_res = 0;

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    high_resolution_clock::time_point start, end;
    duration<double, std::milli> ms;
    MPI_Barrier(MPI_COMM_WORLD);
    start = high_resolution_clock::now();
    if (rank == 0) {
        omp_set_num_threads(t);
        res = reduce(arr, 0, n);
    } else {
        omp_set_num_threads(t);
        res = reduce(arr, n, 2 * n);
    }

    MPI_Reduce(&res, &global_res, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        end = high_resolution_clock::now();
        ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        cout << global_res << endl;
        cout << ms.count() << endl;
    }

    MPI_Finalize();
    free(arr);
}
