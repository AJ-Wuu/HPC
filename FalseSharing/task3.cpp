#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <ratio>

using namespace std;
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char **argv) {
    long n = atol(argv[1]);

    float *buffer_1 = (float *)malloc(n * sizeof(float));
    float *buffer_2 = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        buffer_1[i] = i;
        buffer_2[i] = i;
    }

    int rank;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    high_resolution_clock::time_point start, end;
    double t0, t1;
    if (rank == 0) {
        start = high_resolution_clock::now();
        MPI_Send(buffer_1, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(buffer_2, n, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &status);
        end = high_resolution_clock::now();
        t0 = std::chrono::duration_cast<duration<double, std::milli>>(end - start).count();
        MPI_Send(&t0, 1, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
    } else if (rank == 1) {
        start = high_resolution_clock::now();
        MPI_Recv(buffer_2, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Send(buffer_1, n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        end = high_resolution_clock::now();
        t1 = std::chrono::duration_cast<duration<double, std::milli>>(end - start).count();
        MPI_Recv(&t1, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);

        cout << (t0 + t1) << endl;  // inside the condition to ensure only output once
    }

    MPI_Finalize();

    free(buffer_1);
    free(buffer_2);
}
