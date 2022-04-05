
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include<time.h>
// Pooling driver

#define GEMM 0
#define L 0

#define VERBOSE 0
#define FUSION 1
#define STRIDE 1

#define COMB 0
#ifndef BUFFER
#define BUFFER 0
#endif
#define PREFETCH 1

#define H_TILE 0
#define POOLING 1

#include "config.h"

#if uarch == ZEN2
#include "src/kernels/zen2/params.h"
#elif uarch == REF
#include "src/kernels/reference/params.h"
#elif uarch == ARM
#include "src/kernels/arm/params.h"
#endif

#include "src/utils.h"
#include "test/interface.h"

#include "src/interface.h"
// Problem size

// Timing Utils
// static __inline__ unsigned long long rdtsc(void)
// {
//     unsigned hi, lo;
//     __asm__ __volatile__("rdtsc"
//                          : "=a"(lo), "=d"(hi));
//     return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
// }

// #define print_flops(ops, time)                   \
//     {                                            \
//         printf("%.4lf\t", (ops) / (1.0 * time)); \
//     }
// #define print_cycles(time)                 \
//     {                                      \
//         printf("%.0lf, \t", 1.0 * (time)); \
//     }

// #define AVG(accum, trials, avg)       \
//     {                                 \
//         avg = (1.0 * accum) / trials; \
//     }

// #define MIN(a, b)            \
//     {                        \
//         a = (b < a) ? b : a; \
//     }
// #define ACCUM_time(a, b) \
//     {                    \
//         a += b;          \
//     }

#define LIMIT 1e-2

#define CONV 0
#define PARTIAL_CONV 1 // under development
#define DW_CONV 2      // under development
#define GROUP_CONV 3   // under development
#define POOL 4
#define RELU 5

// #ifndef LAYER
// #define LAYER
// #endif

int main(int argc, char **argv)
{
    // printf("%d \t %d\t ", BUFFER, PREFETCH);
    if (argc < 5)
    {
        printf("USAGE: torch_pool <Input Channels> <Input Height> <Input Width> <kernel heightand width> <stride> <padding 'v' or 'f'> <Output Channels> \n");
        return 0;
    }

    printf("layer %d \n", LAYER);
    int C_i = atoi(argv[1]);

    int N = atol(argv[2]);
    int M = atol(argv[3]);

#if LAYER != RELU
    int kernel_size = atol(argv[4]);
    int stride = atol(argv[5]);
    char padding = argv[6][0];
    // int N, M;

    int padding_elements;

    if (padding == 'v')
    {
        padding_elements = 0;
    }
    else if (padding == 'f')
    {
        padding_elements = (kernel_size) / (2);
    }
#else
    // int N = (output_rows);
    // int M = (output_cols);
#endif
#if LAYER == CONV
    uint32_t C_o = atol(argv[7]);
#else
    uint32_t C_o = C_i;
#endif

    print_build_info_check();

    // #endif
    //unsigned long long t0, t1;

    // Direct Convolution Setup
    //  Copy layer weights to temporaries
    // torch::Tensor weights = test_weights; // layer->weight;

    uint32_t in_dimensions = (C_i * N * M);
#if LAYER < RELU
    uint32_t output_rows = (2/stride)*(padding_elements) + (N - kernel_size)/stride + 1;
    uint32_t output_cols = (2 / stride) * (padding_elements) + (M - kernel_size) / stride + 1;
    uint32_t out_dimensions = (C_o * output_rows * output_cols);
#else
    uint32_t out_dimensions = in_dimensions;
#endif

    float *input_dc = alloc(in_dimensions);
    float *out_dc = alloc(out_dimensions);
    float *out_check_dc = alloc(out_dimensions);
    std::vector<uint32_t> intermediate_block_dimensions;

#if PARALLEL
    // uint32_t num_threads = 1;
    // if (char const *nt_str = std::getenv("OMP_NUM_THREADS"))
    // {
    //     int32_t nt(atoi(nt_str));
    //     if (nt > 0)
    //         num_threads = nt;
    // }
#endif

    unsigned long long sum = ULLONG_MAX; //, sum_pool = ULLONG_MAX;
    // volatile unsigned long long sum_fused = ULLONG_MAX;
    // volatile unsigned long long sum_conv = ULLONG_MAX;
    std::vector<uint64_t> unfused_timing;


        // Initialize Outputs to 0

        // 3x3 unfused
        // Copy Inputs to their flat buffers
        init(input_dc, in_dimensions);

#if LAYER == CONV
        uint32_t filter_dimensions = (C_i * C_o * kernel_size * kernel_size);
#endif
#if LAYER < POOL
        float *filter_dc = alloc(filter_dimensions);

        init(filter_dc, in_dimensions);

#endif
    {
        //bool check = 0;

        sum = ULLONG_MAX;
#if LAYER == RELU
        // direct_convolution_naive<W_ob, C_ob, C_o, 1 , 'v', 'a'>(1 , 1, 1, C_i, 1, C_i, N, M, input_dc, filter_dc, out_check_dc);
        check_ReLUActivation(0, C_i, N, M, input_dc, out_check_dc);
#elif LAYER == POOL
        Maxpool2D(0, kernel_size, stride, padding, C_i, N, M, input_dc, out_check_dc);
#elif LAYER == CONV
        check_Conv2D(0, kernel_size, stride, padding, C_o, C_i, N, M, input_dc, filter_dc, out_check_dc);
#endif

// direct_convolution_naive<W_ob, C_ob, C_o, 1 , 'v', 'a'>(1 , 1, 1, C_i, 1, C_i, N, M, input_dc, filter_dc, out_check_dc);
#if LAYER == RELU
        ReLUActivation(0, C_i, N, M, input_dc, out_dc);
#elif LAYER == POOL
        Maxpool2D(0, kernel_size, stride, padding, C_i, N, M, input_dc, out_dc);
#elif LAYER == CONV
        Conv2D(0, kernel_size, stride, padding, C_o, C_i, N, M, input_dc, filter_dc, out_dc);
#endif

        assert(equals(out_dimensions, out_check_dc, out_dc, 1e-4));
        diff = 0;
        for (int run = 0; run < RUNS; run++)
        {
            // t0 = rdtsc();
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

#if LAYER == RELU
            check_ReLUActivation(0, C_i, N, M, input_dc, out_check_dc);
#elif LAYER == POOL
            Maxpool2D(0, kernel_size, stride, padding, C_i, N, M, input_dc, out_check_dc);
#elif LAYER == CONV
            check_Conv2D(0, kernel_size, stride, padding, C_o, C_i, N, M, input_dc, filter_dc, out_dc);
#endif
            // t1 = rdtsc();
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);

            diff = time_difference(time1, time2);
            MIN(sum, diff);

        }

        print_cycles(sum);

        sum = ULLONG_MAX;
        diff = 0;
        for (int run = 0; run < RUNS; run++)
        {
            // t0 = rdtsc();
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
#if LAYER == RELU
            ReLUActivation(0, C_i, N, M, input_dc, out_dc);
#elif LAYER == POOL
            Maxpool2D(0, kernel_size, stride, padding, C_i, N, M, input_dc, out_dc);
#elif LAYER == CONV
            Conv2D(0, kernel_size, stride, padding, C_o, C_i, N, M, input_dc, filter_dc, out_dc); //     #endif
#endif
            // t1 = rdtsc();
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
            diff = time_difference(time1, time2);
            MIN(sum, diff);

        }

        print_cycles(sum);
        printf("\n");

        fflush(0);
    }
    free(input_dc);
#if LAYER == CONV
    free(filter_dc);
#endif
    free(out_check_dc);
    free(out_dc);
}
