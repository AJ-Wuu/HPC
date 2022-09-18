#include "convolution.h"

using namespace std;

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    std::size_t temp = (m - 1) / 2;

    for (std::size_t x = 0; x < n; x++) {
        for (std::size_t y = 0; y < n; y++) {
            float curr = 0;

            for (std::size_t i = 0; i < m; i++) {
                for (std::size_t j = 0; j < m; j++) {
                    std::size_t index_a = x + i - temp;
                    std::size_t index_b = y + j - temp;
                    float imageVal = 0;
                    if ((index_a >= 0 && index_a < n) && (index_b >= 0 && index_b < n)) {
                        imageVal = image[index_a * n + index_b];
                    } else if ((index_a < 0 || index_a >= n) && (index_b < 0 || index_b >= n)) {
                        imageVal = 0;
                    } else {
                        imageVal = 1;
                    }

                    curr += mask[i * m + j] * imageVal;
                }
            }

            output[x * n + y] = curr;
        }
    }
}