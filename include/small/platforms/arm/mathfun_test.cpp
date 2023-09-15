#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "arm_mathfun.h" // Include the ARM CMSIS-DSP library

// Function to compute exponentiation using std::exp
void computeExpNaive(const float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = std::exp(input[i]);
    }
}

// Function to compute exponentiation using ARM NEON SIMD from arm_math
void computeExpNEON(const float* input, float* output, int size) {
    int numIterations = size / 4;
    for (int i = 0; i < numIterations; ++i) {
        float32x4_t in = vld1q_f32(&input[i * 4]);
        float32x4_t result = exp_ps(in); // ARM CMSIS-DSP exp_ps intrinsic
        vst1q_f32(&output[i * 4], result);
    }

    // Compute any remaining elements (not multiple of 4) using the naive method
    for (int i = numIterations * 4; i < size; ++i) {
        output[i] = std::exp(input[i]);
    }
}

int main() {
    const int size = 16; // Change this to your desired array size

    // Initialize input array with random values
    std::vector<float> input(size);
    for (int i = 0; i < size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Initialize output arrays
    std::vector<float> outputNaive(size);
    std::vector<float> outputNEON(size);

    // Compute exponentiation using both methods
    computeExpNaive(input.data(), outputNaive.data(), size);
    computeExpNEON(input.data(), outputNEON.data(), size);

    // Compare the results
    bool isEqual = true;
    for (int i = 0; i < size; ++i) {
        if (std::abs(outputNaive[i] - outputNEON[i]) > 1e-5) {
            isEqual = false;
            break;
        }
    }

    if (isEqual) {
        std::cout << "Results are equal!" << std::endl;
    } else {
        std::cout << "Results differ!" << std::endl;
    }

    return 0;
}
