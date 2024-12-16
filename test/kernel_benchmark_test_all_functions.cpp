//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2024 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126
//****************************************************************************


#include <math.h>
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <arm_fp16.h>
#include <small/op_type.hpp>
#include <small/utils/Timer.hpp>
#include <params.h>
#include <Buffer.hpp>
#include <intrinsics.h>
#include <arm_neon.h>
#include <random>

// Kernel parameters
#ifndef KERNEL_C_ob
#define KERNEL_C_ob FLOAT_C_ob
#endif

#ifndef KERNEL_W_ob
#define KERNEL_W_ob FLOAT_W_ob
#endif

// Benchmark configuration
#define FREQ 1.5
#define TRIALS 100
#define RUNS 1000
#define NUM_IMPLEMENTATIONS 1
#define NUM_SIZES 14
#define float_OP_TYPE small::OP_CONV
#define float_G_b 1
#define float_UNROLL 1
#define float_OP_CLASS 2
#define STRIDE 1

// Global performance tracking
double min_layer_timers[NUM_IMPLEMENTATIONS][NUM_SIZES];
double avg_layer_timers[NUM_IMPLEMENTATIONS][NUM_SIZES];
double total_layer_timers[NUM_IMPLEMENTATIONS][NUM_SIZES] = {0};
double layer_flops[NUM_IMPLEMENTATIONS][NUM_SIZES];

// Pre-defined layer configurations
struct LayerConfig {
    int H;
    int W;
    int C;
    int K;
    int R;
    int S;
    int stride;
    int padding;
};

const LayerConfig layer_configs[NUM_SIZES] = {
    {112, 112, 32, 32, 3, 3, STRIDE, 1},  // Layer 1
    {56, 56, 64, 64, 3, 3, STRIDE, 1},    // Layer 2
    {28, 28, 128, 128, 3, 3, STRIDE, 1},  // Layer 3
    {14, 14, 256, 256, 3, 3, STRIDE, 1},  // Layer 4
    {7, 7, 512, 512, 3, 3, STRIDE, 1},    // Layer 5
    {56, 56, 32, 32, 3, 3, STRIDE, 1},    // Layer 6
    {28, 28, 64, 64, 3, 3, STRIDE, 1},    // Layer 7
    {14, 14, 128, 128, 3, 3, STRIDE, 1},  // Layer 8
    {7, 7, 256, 256, 3, 3, STRIDE, 1},    // Layer 9
    {28, 28, 32, 32, 3, 3, STRIDE, 1},    // Layer 10
    {14, 14, 64, 64, 3, 3, STRIDE, 1},    // Layer 11
    {7, 7, 128, 128, 3, 3, STRIDE, 1},    // Layer 12
    {14, 14, 32, 32, 3, 3, STRIDE, 1},    // Layer 13
    {7, 7, 64, 64, 3, 3, STRIDE, 1}       // Layer 14
};

// Reference implementations class
class ReferenceOps {
public:
    static void conv_reference(const float16_t* a, const float16_t* b,
                             float16_t* c, int W_ob, int C_ob, int step) {
        for (int kk = 0; kk < W_ob; kk++) {
            for (int jj = 0; jj < C_ob; jj++) {
                c[kk * C_ob + jj] += a[kk * step] * b[jj];
            }
        }
    }

    static void dw_conv_reference(const float16_t* a, const float16_t* b,
                                float16_t* c, int W_ob, int C_ob, int step) {
        for (int kk = 0; kk < W_ob; kk++) {
            for (int jj = 0; jj < C_ob; jj++) {
                c[kk * C_ob + jj] += a[kk * step + jj] * b[jj];
            }
        }
    }

    static void max_reference(const float16_t* a, float16_t* c,
                            int W_ob, int C_ob, int step) {
        for (int kk = 0; kk < W_ob; kk++) {
            for (int jj = 0; jj < C_ob; jj++) {
                c[kk * C_ob + jj] = std::max(c[kk * C_ob + jj], 
                                           a[kk * step + jj]);
            }
        }
    }

    static void fused_relu_reference(const float16_t* a, float16_t* c,
                                   int W_ob, int C_ob) {
        for (int kk = 0; kk < W_ob; kk++) {
            for (int jj = 0; jj < C_ob; jj++) {
                c[kk * C_ob + jj] = std::max(float16_t(0), 
                                           a[kk * C_ob + jj]);
            }
        }
    }

    static void exp_reference(const float16_t* a, float16_t* c,
                            int W_ob, int C_ob) {
        for (int kk = 0; kk < W_ob; kk++) {
            for (int jj = 0; jj < C_ob; jj++) {
                c[kk * C_ob + jj] = expf16(a[kk * C_ob + jj]);
            }
        }
    }

    static void reduce_channel_reference(float16_t* c, int W_ob, int C_ob) {
        for (int kk = 0; kk < W_ob; kk++) {
            float16_t sum = 0;
            for (int jj = 0; jj < C_ob; jj++) {
                sum += c[kk * C_ob + jj];
            }
            c[kk * C_ob] = sum;
            for (int jj = 1; jj < C_ob; jj++) {
                c[kk * C_ob + jj] = 0;
            }
        }
    }
};

// Main benchmark class
class MacroBenchmark {
private:
    struct TestConfig {
        int W_ob;        // Width of output block
        int C_ob;        // Channels per output block
        int step;        // Step size for strided operations
        int W_last;      // Size for end operations
        int _UNROLL;     // Unroll factor for conv operations
        int trials;      // Number of benchmark trials
        int warmup;      // Number of warmup iterations
        int H;           // Height for layer configurations
        int W;           // Width for layer configurations
        int C;           // Input channels
        int K;           // Output channels
        int R;           // Filter height
        int S;           // Filter width
        int stride;      // Stride value
        int padding;     // Padding value
    };

    small::Timer timer;
    std::ofstream log_file;
    const char* log_filename = "simd_benchmark_results.txt";

// Helper Methods
    bool verify_results(const float16_t* simd_result, 
                       const float16_t* ref_result,
                       int size, 
                       float tolerance = 1e-3f) {
        int mismatches = 0;
        float max_diff = 0.0f;
        float total_diff = 0.0f;
        
        for (int i = 0; i < size; i++) {
            float diff = std::abs(static_cast<float>(simd_result[i] - ref_result[i]));
            max_diff = std::max(max_diff, diff);
            total_diff += diff;
            
            if (diff > tolerance) {
                mismatches++;
                if (mismatches <= 10) {
                    log_file << "Mismatch at " << i << ": "
                            << "SIMD=" << simd_result[i] 
                            << " Reference=" << ref_result[i]
                            << " Diff=" << diff << std::endl;
                }
            }
        }
        
        log_file << "Max difference: " << max_diff << std::endl;
        log_file << "Average difference: " << total_diff/size << std::endl;
        log_file << "Total mismatches: " << mismatches << std::endl;
        
        return mismatches == 0;
    }

    template<typename T>
    T* allocate_aligned(size_t size) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 32, size * sizeof(T)) != 0) {
            throw std::runtime_error("Failed to allocate aligned memory");
        }
        memset(ptr, 0, size * sizeof(T));  // Initialize to zero
        return reinterpret_cast<T*>(ptr);
    }

    void fill_random(float16_t* data, size_t size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<float16_t>(dis(gen));
        }
    }

    void record_timing(int impl_idx, int size_idx, double time) {
        min_layer_timers[impl_idx][size_idx] = 
            std::min(min_layer_timers[impl_idx][size_idx], time);
        total_layer_timers[impl_idx][size_idx] += time;
        avg_layer_timers[impl_idx][size_idx] = 
            total_layer_timers[impl_idx][size_idx] / TRIALS;
    }

    double measure_performance(const std::function<void()>& test_fn, 
                             int warmup_runs = 10) {
        // Warmup runs
        for (int i = 0; i < warmup_runs; i++) {
            test_fn();
        }

        // Actual measurement
        timer.start();
        for (int i = 0; i < RUNS; i++) {
            test_fn();
        }
        return timer.stop();
    }

public:
    MacroBenchmark() : log_file(log_filename) {
        if (!log_file.is_open()) {
            throw std::runtime_error("Failed to open log file");
        }
        log_file << std::fixed << std::setprecision(6);
        
        // Initialize timing arrays
        for (int i = 0; i < NUM_IMPLEMENTATIONS; i++) {
            for (int j = 0; j < NUM_SIZES; j++) {
                min_layer_timers[i][j] = DBL_MAX;
                avg_layer_timers[i][j] = 0.0;
                layer_flops[i][j] = 0.0;
            }
        }
    }

    ~MacroBenchmark() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }

    // Test Methods
    void test_data_movement(const TestConfig& config) {
        log_file << "\n=== Testing Data Movement Operations ===\n";
        
        // Test zero operations
        test_zero_operations(config);
        
        // Test load/store operations
        test_load_store_operations(config);
        
        // Test strided operations
        test_strided_operations(config);
        
        // Test upsampling operations
        test_upsampling_operations(config);
    }

    void test_zero_operations(const TestConfig& config) {
        log_file << "\nTesting Zero Operations" << std::endl;

        // Test: FLOAT_ZERO_TILE_C_FP16
        {
            log_file << "\nTesting FLOAT_ZERO_TILE_C_FP16" << std::endl;
            
            float16_t* output = allocate_aligned<float16_t>(config.W_ob * config.C_ob);
            float16x8_t c_tile_v[config.W_ob * (config.C_ob / FLOAT_SIMD_FP16)];
            
            auto zero_test = [&]() {
                FLOAT_ZERO_TILE_C_FP16(config.W_ob, config.C_ob);
                FLOAT_STORE_TILE_C_FP16(output, config.W_ob, config.C_ob);
            };

            double time = measure_performance(zero_test);
            
            // Verify all zeros
            bool passed = true;
            for (int i = 0; i < config.W_ob * config.C_ob; i++) {
                if (output[i] != 0.0f) {
                    passed = false;
                    log_file << "Non-zero value at " << i << ": " << output[i] << std::endl;
                    break;
                }
            }
            
            log_file << "Zero test: " << (passed ? "PASSED" : "FAILED") << "\n";
            log_file << "Average time: " << time/RUNS << " seconds\n";
            
            free(output);
        }

        // Test: FLOAT_ZERO_END_C_FP16
        {
            log_file << "\nTesting FLOAT_ZERO_END_C_FP16" << std::endl;
            
            float16_t* output = allocate_aligned<float16_t>(config.W_last * config.C_ob);
            float16x8_t* c_tile = reinterpret_cast<float16x8_t*>(
                allocate_aligned<float16_t>(config.W_last * config.C_ob));
            
            auto zero_end_test = [&]() {
                FLOAT_ZERO_END_C_FP16(config.W_last, config.C_ob);
                FLOAT_STORE_END_C_FP16(output, config.W_last, config.C_ob);
            };

            double time = measure_performance(zero_end_test);
            
            // Verify all zeros
            bool passed = true;
            for (int i = 0; i < config.W_last * config.C_ob; i++) {
                if (output[i] != 0.0f) {
                    passed = false;
                    log_file << "Non-zero value at " << i << ": " << output[i] << std::endl;
                    break;
                }
            }
            
            log_file << "Zero end test: " << (passed ? "PASSED" : "FAILED") << "\n";
            log_file << "Average time: " << time/RUNS << " seconds\n";
            
            free(output);
            free(c_tile);
        }
    }

void test_load_store_operations(const TestConfig& config) {
        log_file << "\nTesting Load/Store Operations" << std::endl;

        // Test: FLOAT_LOAD_TILE_C_FP16 and FLOAT_STORE_TILE_C_FP16
        {
            log_file << "\nTesting FLOAT_LOAD_TILE_C_FP16 and FLOAT_STORE_TILE_C_FP16" << std::endl;
            
            float16_t* input = allocate_aligned<float16_t>(config.W_ob * config.C_ob);
            float16_t* output = allocate_aligned<float16_t>(config.W_ob * config.C_ob);
            float16x8_t c_tile_v[config.W_ob * (config.C_ob / FLOAT_SIMD_FP16)];
            
            // Initialize input with pattern
            for (int i = 0; i < config.W_ob * config.C_ob; i++) {
                input[i] = static_cast<float16_t>(i);
            }

            auto load_store_test = [&]() {
                FLOAT_LOAD_TILE_C_FP16(input, config.W_ob, config.C_ob);
                FLOAT_STORE_TILE_C_FP16(output, config.W_ob, config.C_ob);
            };

            // Performance measurement
            double time = measure_performance(load_store_test);
            
            // Verify correctness
            bool passed = verify_results(input, output, config.W_ob * config.C_ob);
            
            // Calculate bandwidth
            double bytes_transferred = 2.0 * config.W_ob * config.C_ob * sizeof(float16_t) * RUNS;
            double bandwidth = bytes_transferred / (time * 1e9);  // GB/s
            
            log_file << "Load/Store test: " << (passed ? "PASSED" : "FAILED") << "\n"
                    << "Average time: " << time/RUNS << " seconds\n"
                    << "Memory bandwidth: " << bandwidth << " GB/s\n";

            free(input);
            free(output);
        }

        // Test: FLOAT_LOAD_END_C_FP16 and FLOAT_STORE_END_C_FP16
        {
            log_file << "\nTesting FLOAT_LOAD_END_C_FP16 and FLOAT_STORE_END_C_FP16" << std::endl;
            
            float16_t* input = allocate_aligned<float16_t>(config.W_last * config.C_ob);
            float16_t* output = allocate_aligned<float16_t>(config.W_last * config.C_ob);
            float16x8_t* c_tile = reinterpret_cast<float16x8_t*>(
                allocate_aligned<float16_t>(config.W_last * config.C_ob));
            
            // Initialize input
            for (int i = 0; i < config.W_last * config.C_ob; i++) {
                input[i] = static_cast<float16_t>(i);
            }

            auto load_store_end_test = [&]() {
                FLOAT_LOAD_END_C_FP16(input, config.W_last, config.C_ob);
                FLOAT_STORE_END_C_FP16(output, config.W_last, config.C_ob);
            };

            double time = measure_performance(load_store_end_test);
            bool passed = verify_results(input, output, config.W_last * config.C_ob);
            
            double bytes_transferred = 2.0 * config.W_last * config.C_ob * sizeof(float16_t) * RUNS;
            double bandwidth = bytes_transferred / (time * 1e9);
            
            log_file << "Load/Store end test: " << (passed ? "PASSED" : "FAILED") << "\n"
                    << "Average time: " << time/RUNS << " seconds\n"
                    << "Memory bandwidth: " << bandwidth << " GB/s\n";

            free(input);
            free(output);
            free(c_tile);
        }
    }

    void test_strided_operations(const TestConfig& config) {
        log_file << "\nTesting Strided Operations" << std::endl;

        // Test: FLOAT_LOAD_TILE_C_strided_FP16
        {
            log_file << "\nTesting FLOAT_LOAD_TILE_C_strided_FP16" << std::endl;
            
            const int stride = 2;
            float16_t* input = allocate_aligned<float16_t>(config.W_ob * stride * config.C_ob);
            float16_t* output = allocate_aligned<float16_t>(config.W_ob * config.C_ob);
            float16_t* ref = allocate_aligned<float16_t>(config.W_ob * config.C_ob);
            float16x8_t c_tile_v[config.W_ob * (config.C_ob / FLOAT_SIMD_FP16)];
            
            // Initialize input with pattern
            for (int i = 0; i < config.W_ob * stride * config.C_ob; i++) {
                input[i] = static_cast<float16_t>(i);
            }

            // Prepare reference data
            for (int kk = 0; kk < config.W_ob; kk++) {
                for (int jj = 0; jj < config.C_ob; jj++) {
                    ref[kk * config.C_ob + jj] = input[kk * stride * config.C_ob + jj];
                }
            }

            auto strided_test = [&]() {
                FLOAT_LOAD_TILE_C_strided_FP16(input, stride * config.C_ob, 
                                              config.W_ob, config.C_ob);
                FLOAT_STORE_TILE_C_FP16(output, config.W_ob, config.C_ob);
            };

            double time = measure_performance(strided_test);
            bool passed = verify_results(ref, output, config.W_ob * config.C_ob);
            
            log_file << "Strided load test: " << (passed ? "PASSED" : "FAILED") << "\n"
                    << "Average time: " << time/RUNS << " seconds\n";

            free(input);
            free(output);
            free(ref);
        }

        // Test: FLOAT_LOAD_END_C_strided_FP16
        {
            log_file << "\nTesting FLOAT_LOAD_END_C_strided_FP16" << std::endl;
            
            const int stride = 2;
            float16_t* input = allocate_aligned<float16_t>(config.W_last * stride * config.C_ob);
            float16_t* output = allocate_aligned<float16_t>(config.W_last * config.C_ob);
            float16_t* ref = allocate_aligned<float16_t>(config.W_last * config.C_ob);
            float16x8_t* c_tile = reinterpret_cast<float16x8_t*>(
                allocate_aligned<float16_t>(config.W_last * config.C_ob));
            
            // Initialize input
            for (int i = 0; i < config.W_last * stride * config.C_ob; i++) {
                input[i] = static_cast<float16_t>(i);
            }

            // Prepare reference
            for (int kk = 0; kk < config.W_last; kk++) {
                for (int jj = 0; jj < config.C_ob; jj++) {
                    ref[kk * config.C_ob + jj] = input[kk * stride * config.C_ob + jj];
                }
            }

            auto strided_end_test = [&]() {
                FLOAT_LOAD_END_C_strided_FP16(input, stride * config.C_ob, 
                                            config.W_last, config.C_ob);
                FLOAT_STORE_END_C_FP16(output, config.W_last, config.C_ob);
            };

            double time = measure_performance(strided_end_test);
            bool passed = verify_results(ref, output, config.W_last * config.C_ob);
            
            log_file << "Strided load end test: " << (passed ? "PASSED" : "FAILED") << "\n"
                    << "Average time: " << time/RUNS << " seconds\n";

            free(input);
            free(output);
            free(ref);
            free(c_tile);
        }
    }

void test_upsampling_operations(const TestConfig& config) {
        log_file << "\nTesting Upsampling Operations" << std::endl;

        // Test: FLOAT_LOAD_TILE_C_upsample_FP16
        {
            log_file << "\nTesting FLOAT_LOAD_TILE_C_upsample_FP16" << std::endl;
            
            const int stride = 2;
            const int _C_ib = config.C_ob;
            float16_t* input = allocate_aligned<float16_t>((config.W_ob/stride) * config.C_ob);
            float16_t* output = allocate_aligned<float16_t>(config.W_ob * config.C_ob);
            
            // Initialize input
            fill_random(input, (config.W_ob/stride) * config.C_ob);

            float16x8_t c_0_0, c_0_1, c_1_0, c_1_1;
            float16x8_t c_2_0, c_2_1, c_3_0, c_3_1;
            
            auto upsample_test = [&]() {
                FLOAT_LOAD_TILE_C_upsample_FP16(input, stride, _C_ib, config.W_ob, config.C_ob);
            };

            double time = measure_performance(upsample_test);
            
            log_file << "Upsampling load time: " << time/RUNS << " seconds\n";

            free(input);
            free(output);
        }

        // Test: FLOAT_LOAD_END_C_upsample_FP16
        {
            log_file << "\nTesting FLOAT_LOAD_END_C_upsample_FP16" << std::endl;
            
            const int stride = 2;
            const int _C_ib = config.C_ob;
            float16_t* input = allocate_aligned<float16_t>((config.W_last/stride) * config.C_ob);
            float16_t* output = allocate_aligned<float16_t>(config.W_last * config.C_ob);
            float16x8_t* c_tile = reinterpret_cast<float16x8_t*>(
                allocate_aligned<float16_t>(config.W_last * config.C_ob));
            
            fill_random(input, (config.W_last/stride) * config.C_ob);

            auto upsample_end_test = [&]() {
                FLOAT_LOAD_END_C_upsample_FP16(input, stride, _C_ib, config.W_last, config.C_ob);
            };

            double time = measure_performance(upsample_end_test);
            
            log_file << "Upsampling load end time: " << time/RUNS << " seconds\n";

            free(input);
            free(output);
            free(c_tile);
        }
    }

    void test_computation(const TestConfig& config) {
        log_file << "\n=== Testing Computation Operations ===\n";
        
        test_convolution_operations(config);
        test_max_operations(config);
        test_activation_operations(config);
        test_reduction_operations(config);
    }

    void test_convolution_operations(const TestConfig& config) {
        log_file << "\nTesting Convolution Operations" << std::endl;

        // Test: FLOAT_CONV_TILE_C_FP16
        {
            log_file << "\nTesting FLOAT_CONV_TILE_C_FP16" << std::endl;
            
            float16_t* a = allocate_aligned<float16_t>(config.W_ob * config.step);
            float16_t* b = allocate_aligned<float16_t>(config.C_ob);
            float16_t* c_simd = allocate_aligned<float16_t>(config.W_ob * config.C_ob);
            float16_t* c_ref = allocate_aligned<float16_t>(config.W_ob * config.C_ob);

            // Initialize data
            fill_random(a, config.W_ob * config.step);
            fill_random(b, config.C_ob);

            float16x8_t c_tile_v[config.W_ob * (config.C_ob / FLOAT_SIMD_FP16)];

            auto conv_test = [&]() {
                FLOAT_ZERO_TILE_C_FP16(config.W_ob, config.C_ob);
                FLOAT_CONV_TILE_C_FP16(config.step, a, b, config.W_ob, config.C_ob);
                FLOAT_STORE_TILE_C_FP16(c_simd, config.W_ob, config.C_ob);
            };

            // Reference implementation
            ReferenceOps::conv_reference(a, b, c_ref, config.W_ob, config.C_ob, config.step);

            double time = measure_performance(conv_test);
            bool passed = verify_results(c_simd, c_ref, config.W_ob * config.C_ob);
            
            // Calculate FLOPS
            double total_ops = 2.0 * config.W_ob * config.C_ob * RUNS;  // multiply-add counts as 2
            double gflops = (total_ops / time) / 1e9;
            
            log_file << "Convolution test: " << (passed ? "PASSED" : "FAILED") << "\n"
                    << "Average time: " << time/RUNS << " seconds\n"
                    << "GFLOPS: " << gflops << "\n";

            free(a);
            free(b);
            free(c_simd);
            free(c_ref);
        }

        // Test: FLOAT_DW_TILE_C_FP16
        {
            log_file << "\nTesting FLOAT_DW_TILE_C_FP16" << std::endl;
            
            float16_t* a = allocate_aligned<float16_t>(config.W_ob * config.step);
            float16_t* b = allocate_aligned<float16_t>(config.C_ob);
            float16_t* c_simd = allocate_aligned<float16_t>(config.W_ob * config.C_ob);
            float16_t* c_ref = allocate_aligned<float16_t>(config.W_ob * config.C_ob);

            fill_random(a, config.W_ob * config.step);
            fill_random(b, config.C_ob);

            float16x8_t c_tile_v[config.W_ob * (config.C_ob / FLOAT_SIMD_FP16)];

            auto dw_conv_test = [&]() {
                FLOAT_ZERO_TILE_C_FP16(config.W_ob, config.C_ob);
                FLOAT_DW_TILE_C_FP16(config.step, a, b, config.W_ob, config.C_ob);
                FLOAT_STORE_TILE_C_FP16(c_simd, config.W_ob, config.C_ob);
            };

            ReferenceOps::dw_conv_reference(a, b, c_ref, config.W_ob, config.C_ob, config.step);

            double time = measure_performance(dw_conv_test);
            bool passed = verify_results(c_simd, c_ref, config.W_ob * config.C_ob);
            
            double total_ops = 2.0 * config.W_ob * config.C_ob * RUNS;
            double gflops = (total_ops / time) / 1e9;
            
            log_file << "Depthwise Convolution test: " << (passed ? "PASSED" : "FAILED") << "\n"
                    << "Average time: " << time/RUNS << " seconds\n"
                    << "GFLOPS: " << gflops << "\n";

            free(a);
            free(b);
            free(c_simd);
            free(c_ref);
        }

void test_max_operations(const TestConfig& config) {
        log_file << "\nTesting Max Operations" << std::endl;

        // Test: FLOAT_MAX_TILE_C_FP16
        {
            log_file << "\nTesting FLOAT_MAX_TILE_C_FP16" << std::endl;
            
            float16_t* a = allocate_aligned<float16_t>(config.W_ob * config.step);
            float16_t* c_simd = allocate_aligned<float16_t>(config.W_ob * config.C_ob);
            float16_t* c_ref = allocate_aligned<float16_t>(config.W_ob * config.C_ob);

            // Initialize data
            fill_random(a, config.W_ob * config.step);
            fill_random(c_simd, config.W_ob * config.C_ob);
            memcpy(c_ref, c_simd, config.W_ob * config.C_ob * sizeof(float16_t));

            float16x8_t c_tile_v[config.W_ob * (config.C_ob / FLOAT_SIMD_FP16)];

            auto max_test = [&]() {
                FLOAT_LOAD_TILE_C_FP16(c_simd, config.W_ob, config.C_ob);
                FLOAT_MAX_TILE_C_FP16(config.step, a, config.W_ob, config.C_ob);
                FLOAT_STORE_TILE_C_FP16(c_simd, config.W_ob, config.C_ob);
            };

            // Reference implementation
            ReferenceOps::max_reference(a, c_ref, config.W_ob, config.C_ob, config.step);

            double time = measure_performance(max_test);
            bool passed = verify_results(c_simd, c_ref, config.W_ob * config.C_ob);
            
            log_file << "Max operation test: " << (passed ? "PASSED" : "FAILED") << "\n"
                    << "Average time: " << time/RUNS << " seconds\n";

            free(a);
            free(c_simd);
            free(c_ref);
        }

        // Test: FLOAT_MAX_END_C_FP16
        {
            log_file << "\nTesting FLOAT_MAX_END_C_FP16" << std::endl;
            
            float16_t* a = allocate_aligned<float16_t>(config.W_last * config.step);
            float16_t* c_simd = allocate_aligned<float16_t>(config.W_last * config.C_ob);
            float16_t* c_ref = allocate_aligned<float16_t>(config.W_last * config.C_ob);
            float16x8_t* c_cur = reinterpret_cast<float16x8_t*>(
                allocate_aligned<float16_t>(config.W_last * config.C_ob));

            fill_random(a, config.W_last * config.step);
            fill_random(c_simd, config.W_last * config.C_ob);
            memcpy(c_ref, c_simd, config.W_last * config.C_ob * sizeof(float16_t));

            auto max_end_test = [&]() {
                FLOAT_MAX_END_C_FP16(config.step, a, c_cur, config.W_last, config.C_ob);
            };

            ReferenceOps::max_reference(a, c_ref, config.W_last, config.C_ob, config.step);

            double time = measure_performance(max_end_test);
            bool passed = verify_results(c_simd, c_ref, config.W_last * config.C_ob);
            
            log_file << "Max end operation test: " << (passed ? "PASSED" : "FAILED") << "\n"
                    << "Average time: " << time/RUNS << " seconds\n";

            free(a);
            free(c_simd);
            free(c_ref);
            free(c_cur);
        }
    }

    void test_activation_operations(const TestConfig& config) {
        log_file << "\n=== Testing Activation Operations ===\n";

        // Test: FLOAT_FUSED_RELU_TILE_C_FP16
        {
            log_file << "\nTesting FLOAT_FUSED_RELU_TILE_C_FP16" << std::endl;
            
            float16_t* a = allocate_aligned<float16_t>(config.W_ob * config.step);
            float16_t* c_simd = allocate_aligned<float16_t>(config.W_ob * config.C_ob);
            float16_t* c_ref = allocate_aligned<float16_t>(config.W_ob * config.C_ob);

            // Include negative values for ReLU testing
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
            for (int i = 0; i < config.W_ob * config.step; i++) {
                a[i] = static_cast<float16_t>(dis(gen));
            }

            float16x8_t c_tile_v[config.W_ob * (config.C_ob / FLOAT_SIMD_FP16)];

            auto relu_test = [&]() {
                FLOAT_FUSED_RELU_TILE_C_FP16(config.step, a, config.W_ob, config.C_ob);
                FLOAT_STORE_TILE_C_FP16(c_simd, config.W_ob, config.C_ob);
            };

            // Reference implementation
            ReferenceOps::fused_relu_reference(a, c_ref, config.W_ob, config.C_ob);

            double time = measure_performance(relu_test);
            bool passed = verify_results(c_simd, c_ref, config.W_ob * config.C_ob);
            
            log_file << "ReLU operation test: " << (passed ? "PASSED" : "FAILED") << "\n"
                    << "Average time: " << time/RUNS << " seconds\n";

            free(a);
            free(c_simd);
            free(c_ref);
        }

        // Test: FLOAT_FUSED_RELU_END_C_FP16
        {
            log_file << "\nTesting FLOAT_FUSED_RELU_END_C_FP16" << std::endl;
            
            float16_t* a = allocate_aligned<float16_t>(config.W_last * config.step);
            float16_t* c_simd = allocate_aligned<float16_t>(config.W_last * config.C_ob);
            float16_t* c_ref = allocate_aligned<float16_t>(config.W_last * config.C_ob);
            float16x8_t* c_cur = reinterpret_cast<float16x8_t*>(
                allocate_aligned<float16_t>(config.W_last * config.C_ob));

            // Initialize with positive and negative values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
            for (int i = 0; i < config.W_last * config.step; i++) {
                a[i] = static_cast<float16_t>(dis(gen));
            }

            auto relu_end_test = [&]() {
                FLOAT_FUSED_RELU_END_C_FP16(config.step, a, c_cur, config.W_last, config.C_ob);
            };

            ReferenceOps::fused_relu_reference(a, c_ref, config.W_last, config.C_ob);

            double time = measure_performance(relu_end_test);
            bool passed = verify_results(c_simd, c_ref, config.W_last * config.C_ob);
            
            log_file << "ReLU end operation test: " << (passed ? "PASSED" : "FAILED") << "\n"
                    << "Average time: " << time/RUNS << " seconds\n";

            free(a);
            free(c_simd);
            free(c_ref);
            free(c_cur);
        }

void test_exp_operations(const TestConfig& config) {
        log_file << "\n=== Testing Exponential Operations ===\n";

        // Test: FLOAT_EXP_TILE_C_FP16
        {
            log_file << "\nTesting FLOAT_EXP_TILE_C_FP16" << std::endl;
            
            float16_t* a = allocate_aligned<float16_t>(config.W_ob * config.step);
            float16_t* c_simd = allocate_aligned<float16_t>(config.W_ob * config.C_ob);
            float16_t* c_ref = allocate_aligned<float16_t>(config.W_ob * config.C_ob);

            // Initialize with values suitable for exp
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-5.0f, 5.0f);  // Reasonable range for exp
            for (int i = 0; i < config.W_ob * config.step; i++) {
                a[i] = static_cast<float16_t>(dis(gen));
            }

            float16x8_t c_0_0, c_0_1, c_1_0, c_1_1;
            float16x8_t c_2_0, c_2_1, c_3_0, c_3_1;

            auto exp_test = [&]() {
                FLOAT_EXP_TILE_C_FP16(config.step, a, config.W_ob, config.C_ob);
            };

            // Reference implementation
            ReferenceOps::exp_reference(a, c_ref, config.W_ob, config.C_ob);

            double time = measure_performance(exp_test);
            bool passed = verify_results(c_simd, c_ref, config.W_ob * config.C_ob);
            
            log_file << "Exp operation test: " << (passed ? "PASSED" : "FAILED") << "\n"
                    << "Average time: " << time/RUNS << " seconds\n";

            free(a);
            free(c_simd);
            free(c_ref);
        }

        // Test: FLOAT_FUSED_EXP_END_C_FP16
        {
            log_file << "\nTesting FLOAT_FUSED_EXP_END_C_FP16" << std::endl;
            
            float16_t* a = allocate_aligned<float16_t>(config.W_last * config.step);
            float16_t* c_simd = allocate_aligned<float16_t>(config.W_last * config.C_ob);
            float16_t* c_ref = allocate_aligned<float16_t>(config.W_last * config.C_ob);
            float16x8_t* c_cur = reinterpret_cast<float16x8_t*>(
                allocate_aligned<float16_t>(config.W_last * config.C_ob));

            // Initialize with suitable values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
            for (int i = 0; i < config.W_last * config.step; i++) {
                a[i] = static_cast<float16_t>(dis(gen));
            }

            auto exp_end_test = [&]() {
                FLOAT_FUSED_EXP_END_C_FP16(config.step, a, c_cur, config.W_last, config.C_ob);
            };

            ReferenceOps::exp_reference(a, c_ref, config.W_last, config.C_ob);

            double time = measure_performance(exp_end_test);
            bool passed = verify_results(c_simd, c_ref, config.W_last * config.C_ob);
            
            log_file << "Exp end operation test: " << (passed ? "PASSED" : "FAILED") << "\n"
                    << "Average time: " << time/RUNS << " seconds\n";

            free(a);
            free(c_simd);
            free(c_ref);
            free(c_cur);
        }
    }

    void test_reduction_operations(const TestConfig& config) {
        log_file << "\n=== Testing Reduction Operations ===\n";

        // Test: FLOAT_REDUCE_CHANNEL_END_C_FP16
        {
            log_file << "\nTesting FLOAT_REDUCE_CHANNEL_END_C_FP16" << std::endl;
            
            float16_t* c_simd = allocate_aligned<float16_t>(config.W_ob * config.C_ob);
            float16_t* c_ref = allocate_aligned<float16_t>(config.W_ob * config.C_ob);

            // Initialize with random values
            fill_random(c_simd, config.W_ob * config.C_ob);
            memcpy(c_ref, c_simd, config.W_ob * config.C_ob * sizeof(float16_t));

            auto reduce_test = [&]() {
                FLOAT_REDUCE_CHANNEL_END_C_FP16(config.W_ob, config.C_ob);
            };

            // Reference implementation
            ReferenceOps::reduce_channel_reference(c_ref, config.W_ob, config.C_ob);

            double time = measure_performance(reduce_test);
            bool passed = verify_results(c_simd, c_ref, config.W_ob * config.C_ob);
            
            log_file << "Channel reduction test: " << (passed ? "PASSED" : "FAILED") << "\n"
                    << "Average time: " << time/RUNS << " seconds\n";

            free(c_simd);
            free(c_ref);
        }

        // Test: FLOAT_REDUCE_div_C_FP16
        {
            log_file << "\nTesting FLOAT_REDUCE_div_C_FP16" << std::endl;
            
            float16_t* c_simd = allocate_aligned<float16_t>(config.W_ob * config.C_ob);
            float16_t* O = allocate_aligned<float16_t>(config.C_ob);
            float16_t d = static_cast<float16_t>(1.0f / config.W_ob);  // Normalization factor

            // Initialize data
            fill_random(c_simd, config.W_ob * config.C_ob);
            memset(O, 0, config.C_ob * sizeof(float16_t));

            auto reduce_div_test = [&]() {
                FLOAT_REDUCE_div_C_FP16(O, d, config.W_ob, config.C_ob);
            };

            double time = measure_performance(reduce_div_test);
            
            log_file << "Reduce with division test completed\n"
                    << "Average time: " << time/RUNS << " seconds\n";

            free(c_simd);
            free(O);
        }

        // more reduction tests if needed can be added herr
    }

// Main benchmark runner method
    void run_all_benchmarks() {
        log_file << "\n====== Starting SIMD FP16 Macro Benchmarks ======\n";
        log_file << "Configuration: TRIALS=" << TRIALS << ", RUNS=" << RUNS << "\n\n";

        // Layer configurations matching the global layer_configs array
        const TestConfig test_configs[] = {
            // Small configuration
            {
                .W_ob = 4,
                .C_ob = 32,
                .step = 32,
                .W_last = 2,
                ._UNROLL = 4,
                .trials = TRIALS,
                .warmup = 10,
                .H = 112,
                .W = 112,
                .C = 32,
                .K = 32,
                .R = 3,
                .S = 3,
                .stride = 1,
                .padding = 1
            },
            // Medium configuration
            {
                .W_ob = 4,
                .C_ob = 64,
                .step = 64,
                .W_last = 3,
                ._UNROLL = 4,
                .trials = TRIALS,
                .warmup = 10,
                .H = 56,
                .W = 56,
                .C = 64,
                .K = 64,
                .R = 3,
                .S = 3,
                .stride = 1,
                .padding = 1
            },
            // Large configuration
            {
                .W_ob = 4,
                .C_ob = 128,
                .step = 128,
                .W_last = 4,
                ._UNROLL = 4,
                .trials = TRIALS,
                .warmup = 10,
                .H = 28,
                .W = 28,
                .C = 128,
                .K = 128,
                .R = 3,
                .S = 3,
                .stride = 1,
                .padding = 1
            }
            // Add more configurations as needed...
        };

        for (int config_idx = 0; config_idx < sizeof(test_configs)/sizeof(TestConfig); config_idx++) {
            const auto& config = test_configs[config_idx];
            
            log_file << "\n\n====== Testing Configuration " << config_idx + 1 << " ======\n";
            log_file << "W_ob: " << config.W_ob << ", C_ob: " << config.C_ob 
                    << ", H: " << config.H << ", W: " << config.W << "\n";

            try {
                // Data movement tests
                test_data_movement(config);
                
                // Computation tests
                test_computation(config);
                
                // Max operation tests
                test_max_operations(config);
                
                // Activation function tests
                test_activation_operations(config);
                
                // Exponential function tests
                test_exp_operations(config);
                
                // Reduction operation tests
                test_reduction_operations(config);
                
                // Record timing statistics
                record_timing_stats(config_idx);
                
            } catch (const std::exception& e) {
                log_file << "Error during configuration " << config_idx + 1 
                        << ": " << e.what() << std::endl;
            }
        }

        // Print summary statistics
        print_summary_statistics();
    }

private:
    void record_timing_stats(int config_idx) {
        // Calculate aggregated statistics for the current configuration
        double total_time = 0.0;
        double min_time = DBL_MAX;
        double max_time = 0.0;
        
        for (int impl = 0; impl < NUM_IMPLEMENTATIONS; impl++) {
            min_time = std::min(min_time, min_layer_timers[impl][config_idx]);
            max_time = std::max(max_time, avg_layer_timers[impl][config_idx]);
            total_time += total_layer_timers[impl][config_idx];
        }
        
        log_file << "\nTiming Statistics for Configuration " << config_idx + 1 << ":\n"
                << "Minimum time: " << min_time << " seconds\n"
                << "Maximum time: " << max_time << " seconds\n"
                << "Average time: " << total_time / (NUM_IMPLEMENTATIONS * TRIALS) 
                << " seconds\n";
    }

    void print_summary_statistics() {
        log_file << "\n====== Benchmark Summary ======\n";
        
        // Print performance statistics for each implementation and configuration
        for (int impl = 0; impl < NUM_IMPLEMENTATIONS; impl++) {
            log_file << "\nImplementation " << impl + 1 << " Statistics:\n";
            
            for (int size = 0; size < NUM_SIZES; size++) {
                log_file << "Configuration " << size + 1 << ":\n"
                        << "  Min time: " << min_layer_timers[impl][size] << " seconds\n"
                        << "  Avg time: " << avg_layer_timers[impl][size] << " seconds\n"
                        << "  GFLOPS: " << layer_flops[impl][size] / 1e9 << "\n";
            }
        }
        
        log_file << "\nBenchmark completed successfully.\n";
    }
};

// Main function
int main(int argc, char* argv[]) {
    try {
        MacroBenchmark benchmark;
        benchmark.run_all_benchmarks();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}