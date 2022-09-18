#include <stdio.h>
#include <stdlib.h>

// High-Resolution Timers in C++
#include <chrono>    // the std::chrono namespace provides timer functions in C++
#include <iostream>  // iostream is not needed for timers, but we need it for cout
#include <ratio>     // std::ratio provides easy conversions between metric units

#include "scan.h"  // custom scan() function

using namespace std;
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char **argv) {
    int n = atoi(argv[1]);            // user input number
    srand((unsigned int)time(NULL));  // update the seed of the random generator

    float *input = (float *)malloc(n * sizeof(float));  // allocation
    float *output = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {                         // initialization
        input[i] = ((float)rand() / (RAND_MAX)) * 2 - 1;  // [-1.0, 1.0]
    }

    high_resolution_clock::time_point start = high_resolution_clock::now();
    scan(input, output, n);
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double, std::milli> duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);  // convert the calculated duration to a double

    // print out task results
    std::cout << duration_sec.count() << endl;  // durations are converted to milliseconds already thanks to std::chrono::duration_cast
    std::cout << output[0] << endl;
    std::cout << output[n - 1] << endl;

    // deallocation
    delete input;
    delete output;
}