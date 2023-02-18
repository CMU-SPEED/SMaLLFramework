
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
#include <time.h>

#include <small.h>
#include "utils.h"
#include "check_interface.h"

/// @todo Which of these defines are needed?
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

// from config.h, consider making command line options

#define RUNS 1000
#ifndef PARALLEL
#define PARALLEL 1
#endif

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

//****************************************************************************
//****************************************************************************
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

    int input_height = atol(argv[2]);
    int input_width = atol(argv[3]);

    int kernel_size = atol(argv[4]);
    int stride = atol(argv[5]);
    char padding = argv[6][0];

    uint8_t t_pad = 0, b_pad = 0;
    uint8_t l_pad = 0, r_pad = 0;

    //int padding_front = 0, padding_back = 0;

    if (padding == 'f')
    {
        small::calc_padding(input_height, kernel_size, stride, t_pad, b_pad);
        small::calc_padding(input_width, kernel_size, stride, l_pad, r_pad);

        // padding_front = r_pad;
        // padding_back = b_pad;
        // t_pad = 0;
        // b_pad = 0;
        // l_pad = 0;
    }

#if LAYER == CONV || LAYER == PARTIAL_CONV
    uint32_t C_o = atol(argv[7]);
#else
    //uint32_t C_o = C_i;
#endif

    print_build_info_check();

    //unsigned long long t0, t1;

    // Direct Convolution Setup
    //  Copy layer weights to temporaries
    // torch::Tensor weights = test_weights; // layer->weight;

    uint32_t in_dimensions = (C_i * input_height * input_width);
#if LAYER == CONV || LAYER == PARTIAL_CONV
    uint32_t output_rows =  ((input_height + t_pad + b_pad) - kernel_size)/stride + 1;
    uint32_t output_cols =  ((input_width + l_pad + r_pad) - kernel_size) / stride + 1;
    uint32_t out_dimensions = (C_o * output_rows * output_cols);
#else
    uint32_t out_dimensions = in_dimensions;
#endif

    float *input_dc = alloc(in_dimensions);
    // fprintf(stderr, "i %lu\n", input_dc);
    // fflush(stderr);

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
    // init_arange<C_ob>(input_dc, input_height, input_width, C_i);
    init(input_dc, in_dimensions);

#if LAYER == CONV || LAYER == PARTIAL_CONV
    uint32_t filter_dimensions = (C_i * C_o * kernel_size * kernel_size);
#elif LAYER == DW_CONV
    uint32_t filter_dimensions = (C_i * kernel_size * kernel_size);
#endif
#if LAYER < POOL
    float *filter_dc = alloc(filter_dimensions);
    // fprintf(stderr, "f %lu\n", filter_dc);
    // fflush(stderr);
    // printf("%d\n", in_dimensions);
    // init_ones(filter_dc, filter_dimensions);
    init(filter_dc, filter_dimensions);

#endif
    //bool check = 0;

    sum = ULLONG_MAX;

    printf("checking\n");

#if LAYER == RELU
    check_ReLUActivation(C_i,
                         input_height, input_width,
                         input_dc, out_check_dc);
#elif LAYER == POOL
    check_Maxpool2D(kernel_size, stride,
                    t_pad, b_pad, l_pad, r_pad,
                    C_i,
                    input_height, input_width,
                    input_dc, out_check_dc);
#elif LAYER == DW_CONV
    check_DepthwiseConv2D(kernel_size, stride,
                          t_pad, b_pad, l_pad, r_pad,
                          C_i,
                          input_height, input_width,
                          input_dc, filter_dc, out_check_dc);
#elif LAYER == CONV
    check_Conv2D(kernel_size, stride,
                 t_pad, b_pad, l_pad, r_pad,
                 C_o, C_i,
                 input_height, input_width,
                 input_dc, filter_dc, out_check_dc);
#elif LAYER == PARTIAL_CONV
    check_PartialConv2D(kernel_size, stride,
                        t_pad, b_pad, l_pad, r_pad,
                        C_o, C_i,
                        input_height, input_width,
                        input_dc, filter_dc, out_check_dc);

// #elif LAYER == FC
    // Dense(0, C_o, C_i, input_dc, filter_dc, out_check_dc);
#endif

    printf("computed with scalar kernels\n");

#if LAYER == RELU
    small::ReLUActivation(C_i,
                          input_height, input_width,
                          input_dc, out_dc);
#elif LAYER == POOL
    small::Maxpool2D(kernel_size, stride,
                     t_pad, b_pad, l_pad, r_pad,
                     C_i,
                     input_height, input_width,
                     input_dc, out_dc);
#elif LAYER == DW_CONV
    small::DepthwiseConv2D(kernel_size, stride,
                           t_pad, b_pad, l_pad, r_pad,
                           C_i,
                           input_height, input_width,
                           input_dc, filter_dc, out_dc);
#elif LAYER == CONV
    small::Conv2D(kernel_size, stride,
                  t_pad, b_pad, l_pad, r_pad,
                  C_o, C_i,
                  input_height, input_width,
                  input_dc, filter_dc, out_dc);
#elif LAYER == PARTIAL_CONV
    small::PartialConv2D(kernel_size, stride,
                         t_pad, b_pad, l_pad, r_pad,
                         C_o, C_i,
                         input_height, input_width,
                         input_dc, filter_dc, out_dc);
// #elif LAYER == FC
    // small::Dense(C_o, C_i, input_dc, filter_dc, out_dc);
#endif

    assert(equals(out_dimensions, out_check_dc, out_dc, 1e-4));
    diff = 0;
    for (int run = 0; run < RUNS; run++)
    {
        // t0 = rdtsc();
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

#if LAYER == RELU
        check_ReLUActivation(C_i,
                             input_height, input_width,
                             input_dc, out_check_dc);
#elif LAYER == POOL
        check_Maxpool2D(kernel_size, stride,
                        t_pad, b_pad, l_pad, r_pad,
                        C_i,
                        input_height, input_width,
                        input_dc, out_check_dc);
#elif LAYER == DW_CONV
        check_DepthwiseConv2D(kernel_size, stride,
                              t_pad, b_pad, l_pad, r_pad,
                              C_i,
                              input_height, input_width,
                              input_dc, filter_dc, out_check_dc);
#elif LAYER == CONV
        check_Conv2D(kernel_size, stride,
                     t_pad, b_pad, l_pad, r_pad,
                     C_o, C_i,
                     input_height, input_width,
                     input_dc, filter_dc, out_check_dc);
#elif LAYER == PARTIAL_CONV
        check_PartialConv2D(kernel_size, stride,
                            t_pad, b_pad, l_pad, r_pad,
                            C_o, C_i,
                            input_height, input_width,
                            input_dc, filter_dc, out_check_dc);
#endif

        // t1 = rdtsc();
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);

        diff = time_difference(time1, time2);
        sum = std::min<unsigned long long>(sum, diff);
        //MIN(sum, diff);

    }

    print_cycles(sum);
    auto sum_reference = sum;
    sum = ULLONG_MAX;
    diff = 0;
    for (int run = 0; run < RUNS; run++)
    {
        // t0 = rdtsc();
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
#if LAYER == RELU
        small::ReLUActivation(C_i,
                              input_height, input_width,
                              input_dc, out_dc);
#elif LAYER == POOL
        small::Maxpool2D(kernel_size, stride,
                         t_pad, b_pad, l_pad, r_pad,
                         C_i,
                         input_height, input_width,
                         input_dc, out_dc);
#elif LAYER == DW_CONV
        small::DepthwiseConv2D(kernel_size, stride,
                               t_pad, b_pad, l_pad, r_pad,
                               C_i,
                               input_height, input_width,
                               input_dc, filter_dc, out_dc);
#elif LAYER == CONV
        small::Conv2D(kernel_size, stride,
                      t_pad, b_pad, l_pad, r_pad,
                      C_o, C_i,
                      input_height, input_width,
                      input_dc, filter_dc, out_dc);
#elif LAYER == PARTIAL_CONV
        small::PartialConv2D(kernel_size, stride,
                             t_pad, b_pad, l_pad, r_pad,
                             C_o, C_i,
                             input_height, input_width,
                             input_dc, filter_dc, out_dc);
#endif
        // t1 = rdtsc();
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
        diff = time_difference(time1, time2);
        sum = std::min<unsigned long long>(sum, diff);
        //MIN(sum, diff);

    }

    print_cycles(sum);
    printf("\n");

    fflush(0);

    printf("%.4f ", (sum_reference*1.0)/(sum*1.0));

    auto output_els = out_dimensions;
    float compute_ops = 0.0;
    float throughput = 0.0;

#if LAYER == RELU
    compute_ops = output_els*(1.0);
    throughput = NUM_MAX * SIMD;
#elif LAYER == POOL
    compute_ops = output_els * (kernel_size*kernel_size);
    throughput = NUM_MAX * SIMD;
#elif LAYER == DW_CONV
    compute_ops = output_els * (kernel_size * kernel_size*2.0);
    throughput = (NUM_FMA * 2.0 * SIMD);
#elif LAYER == CONV
    compute_ops = output_els * (kernel_size * kernel_size * C_i * 2.0);
    throughput = (NUM_FMA * 2.0 * SIMD);
#elif LAYER == PARTIAL_CONV
    /// @todo CONFIRM THESE NUMBERS
    compute_ops = output_els * (kernel_size * kernel_size * C_i * 2.0);
    throughput = (NUM_FMA * 2.0 * SIMD);
#elif LAYER == FC
    compute_ops = output_els * (C_i * 2.0);
    throughput = (NUM_FMA * 2.0 * SIMD);
#endif

    //float peak_cycles = compute_ops/throughput;
    //float scaled_peak_cycles = peak_cycles;
    const int num_th = atoi(std::getenv("OMP_NUM_THREADS"));
    printf(" %.0f %.2f \n", throughput*num_th, (compute_ops) / (1.0 * sum));

    free(input_dc);
#if LAYER < POOL
    free(filter_dc);
#endif
    free(out_check_dc);
    free(out_dc);
}
