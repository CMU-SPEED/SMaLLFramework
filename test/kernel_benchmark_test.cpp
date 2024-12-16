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
// #include <small.h>
#include <small/utils/Timer.hpp>
#include <params.h>
#include <Buffer.hpp>
#include <intrinsics.h>  // Including FP16 intrinsics header

#ifndef KERNEL_C_ob
#define KERNEL_C_ob FLOAT_C_ob  
#endif

#ifndef KERNEL_W_ob
#define KERNEL_W_ob FLOAT_W_ob  
#endif

#define FREQ 1.5

#define TRIALS 100 
#define RUNS 1000
#define NUM_IMPLEMENTATIONS 1
#define NUM_SIZES 14


#define float_OP_TYPE small::OP_CONV
#define float_G_b 1
#define float_UNROLL 1
#define float_OP_CLASS 2

//define stride
#define STRIDE 1

double min_layer_timers[NUM_IMPLEMENTATIONS][NUM_SIZES];
double avg_layer_timers[NUM_IMPLEMENTATIONS][NUM_SIZES];
double total_layer_timers[NUM_IMPLEMENTATIONS][NUM_SIZES] = {0};
double layer_flops[NUM_IMPLEMENTATIONS][NUM_SIZES];

// macros for the FP16 convolution operation
#define FP16_ABSTRACT_OP(op_type, op_class, step, a_cur, b_cur, _O_wb, _C_ob) \
    if constexpr (op_type == small::OP_CONV)                                  \
    {                                                                         \
        if constexpr (op_class == 1)                                          \
        {                                                                     \
           /*FLOAT_DW_TILE_C_FP16(step, a_cur, b_cur, _O_wb, _C_ob);  not implemented yet*/            \
        }                                                                     \
        else if constexpr (op_class == 2)                                     \
        {                                                                     \
            FLOAT_CONV_TILE_C_FP16(step, a_cur, b_cur, _O_wb, _C_ob);         \
        }                                                                     \
    }

//kernel benchmark: FP16 intrinsics
template <int W_ob, int C_ob, int G_b, int _UNROLL, int stride, small::OpType OP_TYPE, int8_t OP_CLASS>
void kernel_benchmark_fp16(
    const int64_t m, const int64_t n, const int64_t k,
    const __fp16 *I, const __fp16 *W, __fp16 *O)
{
    int32_t constexpr step = C_ob * stride;
    FLOAT_DEF_TILE_C_FP16(W_ob, C_ob);
    FLOAT_LOAD_TILE_C_FP16(O, W_ob, C_ob);
    const __fp16 *a_cur = I;
    const __fp16 *b_cur = W;

#pragma GCC unroll 16
    for (int p = 0; p < k; p += G_b * _UNROLL)
    {
        FP16_ABSTRACT_OP(OP_TYPE, OP_CLASS, step, a_cur, b_cur, W_ob, C_ob);
        b_cur += n * G_b * _UNROLL;
        a_cur += G_b * _UNROLL;
    }
    FLOAT_STORE_TILE_C_FP16(O, W_ob, C_ob);
}

// Main function to run FP16 convolution tests
int main()
{
    const int n = KERNEL_C_ob;
    const int m = KERNEL_W_ob;


    printf("m: %d, n: %d\n Minimum timing over %d trials, each trial averages over %d runs\n", m, n, TRIALS, RUNS);
    printf("k, ops, time (ns), ops/cyc, total time (ms)\n");

    int k_sizes[NUM_SIZES] = {16, 32, 64, 96, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 32768};

    for (int size = 0; size < NUM_SIZES; size++)
    {
        int k = k_sizes[size];
        printf("m = %d, n = %d, k = %d\n", m, n, k);
        int64_t ops = m * n * k * ((float_OP_TYPE == small::OP_CONV) ? 2 : 1);
        printf("%d, %ld, ", k, ops);

        __fp16 *I, *W, *O;
        __fp16 *shared_buffer;
        posix_memalign((void **)&shared_buffer, 64, (m * k + k * n + m * n) * sizeof(__fp16));

        I = shared_buffer;
        W = I + m * k;
        O = W + k * n;

        // Initialize input, weights, and output
        __fp16 *cur_ptr = I;
        for (int i = 0; i < m * k; i++)
        {
            *(cur_ptr++) = (__fp16)(2.0 * ((float)rand() / RAND_MAX) - 1);
        }

        cur_ptr = W;
        for (int i = 0; i < k * n; i++)
        {
            *(cur_ptr++) = (__fp16)(2.0 * ((float)rand() / RAND_MAX) - 1);
        }

        cur_ptr = O;
        for (int i = 0; i < m * n; i++)
        {
            *(cur_ptr++) = 0.0f;
        }

        small::Timer timer;

        for (int trial = 0; trial < TRIALS; trial++)
        {
            cur_ptr = O;
            for (int i = 0; i < m * n; i++)
            {
                *(cur_ptr++) = 0.0f;
            }

            timer.start();
            for (int r = 0; r < RUNS; r++)
            {
                kernel_benchmark_fp16<KERNEL_W_ob, KERNEL_C_ob, float_G_b, float_UNROLL, STRIDE, float_OP_TYPE, float_OP_CLASS>(m, n, k, I, W, O);
            }
            timer.stop();

            total_layer_timers[0][size] = timer.elapsed();
            avg_layer_timers[0][size] = timer.elapsed() / RUNS;
            min_layer_timers[0][size] = (trial == 0) ? avg_layer_timers[0][size] : std::min(min_layer_timers[0][size], avg_layer_timers[0][size]);
        }

        printf("%f, %f, %f\n", min_layer_timers[0][size], (1.0 * ops) / (min_layer_timers[0][size] * FREQ), total_layer_timers[0][size] / 1e6);

        free(shared_buffer);
    }

    return 0;
}


/*
#define float_OP_CLASS 1

// Update FP16_ABSTRACT_OP macro to use the depthwise convolution
#define FP16_ABSTRACT_OP(op_type, op_class, step, a_cur, b_cur, _O_wb, _C_ob) \
    if constexpr (op_type == small::OP_CONV)                                  \
    {                                                                         \
        if constexpr (op_class == 1)                                          \
        {                                                                     \
            FLOAT_DW_TILE_C_FP16(step, a_cur, b_cur, _O_wb, _C_ob);          \
        }                                                                     \
        else if constexpr (op_class == 2)                                     \
        {                                                                     \
            FLOAT_CONV_TILE_C_FP16(step, a_cur, b_cur, _O_wb, _C_ob);        \
        }                                                                     \
    }

// Update print statement for clarity
printf("Testing Depthwise Convolution FP16\n");
printf("m: %d, n: %d\n Minimum timing over %d trials, each trial averages over %d runs\n", m, n, TRIALS, RUNS);
*/