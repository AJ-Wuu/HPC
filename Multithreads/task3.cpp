#include <cstdlib>
#include <iostream>
#include <random>

#include "msort.h"

using namespace std;

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int t = atoi(argv[2]);
    int ts = atoi(argv[3]);

    std::random_device rd;                               // obtain a random number from hardware
    std::mt19937 gen(rd());                              // seed the generator
    std::uniform_int_distribution<> distr(-1000, 1000);  // define the range

    int *arr = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        arr[i] = distr(gen);
    }

    omp_set_num_threads(t);
    double startTime = omp_get_wtime();
    msort(arr, n, ts);
    double endTime = omp_get_wtime();
    double ms = (endTime - startTime) * 1000;

    std::cout << arr[0] << endl;
    std::cout << arr[n - 1] << endl;
    std::cout << ms << endl;

    free(arr);
}
