#include "msort.h"

#include <omp.h>

#include <algorithm>

void merge_sort(int* arr, const std::size_t n) {
    for (size_t i = 1; i < n; i++) {
        int j = i - 1;
        int temp = arr[i];
        while (j >= 0 && arr[j] > temp) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = temp;
    }
    return;
}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    if (n < threshold) {  // base case
        merge_sort(arr, n);
        return;
    }

#pragma omp parallel sections
    {
#pragma omp section
        msort(arr, n / 2, threshold);

#pragma omp section
        msort(arr + n / 2, (n + 1) / 2, threshold);
    }

    std::inplace_merge(arr, arr + n / 2, arr + n);
}