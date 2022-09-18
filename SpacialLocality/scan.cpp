#include "scan.h"

using namespace std;

void scan(const float *arr, float *output, std::size_t n) {
    output[0] = arr[0];  // store the first element

    for (size_t i = 1; i < n; i++) {
        output[i] = output[i - 1] + arr[i];
    }
}