#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <ratio>

#include "optimize.h"

using namespace std;
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char **argv) {
    srand((unsigned int)time(NULL));
    long n = atol(argv[1]);
    vec v = vec(n);
    data_t *data = new data_t[n];
    data_t dest1, dest2, dest3, dest4, dest5;
    for (int i = 0; i < n; i++) {
        data[i] = (int)rand() % 10;  // [0, 9]
    }
    v.data = data;

    high_resolution_clock::time_point start, end;
    duration<double, std::milli> ms;
    double ms_total = 0;
    for (int i = 0; i < 10; i++) {
        start = high_resolution_clock::now();
        optimize1(&v, &dest1);
        end = high_resolution_clock::now();
        ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        ms_total += ms.count();
    }
    cout << dest1 << endl;
    cout << (ms_total / 10) << endl;

    ms_total = 0;
    for (int i = 0; i < 10; i++) {
        start = high_resolution_clock::now();
        optimize2(&v, &dest2);
        end = high_resolution_clock::now();
        ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        ms_total += ms.count();
    }
    cout << dest2 << endl;
    cout << (ms_total / 10) << endl;

    ms_total = 0;
    for (int i = 0; i < 10; i++) {
        start = high_resolution_clock::now();
        optimize3(&v, &dest3);
        end = high_resolution_clock::now();
        ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        ms_total += ms.count();
    }
    cout << dest3 << endl;
    cout << (ms_total / 10) << endl;

    ms_total = 0;
    for (int i = 0; i < 10; i++) {
        start = high_resolution_clock::now();
        optimize4(&v, &dest4);
        end = high_resolution_clock::now();
        ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        ms_total += ms.count();
    }
    cout << dest4 << endl;
    cout << (ms_total / 10) << endl;

    ms_total = 0;
    for (int i = 0; i < 10; i++) {
        start = high_resolution_clock::now();
        optimize5(&v, &dest5);
        end = high_resolution_clock::now();
        ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        ms_total += ms.count();
    }
    cout << dest5 << endl;
    cout << (ms_total / 10) << endl;
}
