#include <omp.h>

#include <iostream>

using namespace std;

int factorial(int num) {
    int result = 1;
    for (int i = 2; i <= num; i++) {
        result *= i;
    }
    return result;
}

int main() {
    int nThreads = 4;
    printf("Number of threads: %d\n", nThreads);

#pragma omp parallel num_threads(nThreads)
    {
        int myId = omp_get_thread_num();
        printf("I'm thread No. %d\n", myId);
    }
#pragma omp parallel for
    for (int i = 1; i <= 8; i++) {
        printf("%d!=%d\n", i, factorial(i));
    }
}