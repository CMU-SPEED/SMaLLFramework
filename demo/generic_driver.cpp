//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>

#include <small.h>
#include <small/utils/Timer.hpp>
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

#ifndef LAYER
#define LAYER CONV
#endif
//****************************************************************************
//****************************************************************************
int main(int argc, char **argv)
{
    if (argc < 5)
    {
        printf("USAGE: %s <Input Channels> <Input Height> <Input Width> <kernel heightand width> <stride> <padding 'v' or 'f'> <Output Channels> \n", argv[0]);
        return 0;
    }

#if defined(QUANTIZED)
    using Buffer = small::QUInt8Buffer;
#else
    using Buffer = small::FloatBuffer;
#endif

    printf("layer %d \n", LAYER);
    int C_i = atoi(argv[1]);

    int input_height = atol(argv[2]);
    int input_width = atol(argv[3]);

    int kernel_size = atol(argv[4]);
    int stride = atol(argv[5]);
    char padding = argv[6][0];

    uint8_t t_pad = 0, b_pad = 0;
    uint8_t l_pad = 0, r_pad = 0;

    if (padding == 'f')
    {
        small::calc_padding(input_height, kernel_size, stride, t_pad, b_pad);
        small::calc_padding(input_width, kernel_size, stride, l_pad, r_pad);
    }

#if LAYER == CONV || LAYER == PARTIAL_CONV
    uint32_t C_o = atol(argv[7]);
#else
    //uint32_t C_o = C_i;
#endif

    print_build_info_check<Buffer>();

    //unsigned long long t0, t1;

    // Direct Convolution Setup
    //  Copy layer weights to temporaries
    // torch::Tensor weights = test_weights; // layer->weight;

    /// @todo Consider changing dimensions to size_t
    uint32_t in_dimensions = (C_i * input_height * input_width);
#if LAYER == CONV || LAYER == PARTIAL_CONV
    uint32_t output_rows =  ((input_height + t_pad + b_pad) - kernel_size)/stride + 1;
    uint32_t output_cols =  ((input_width + l_pad + r_pad) - kernel_size) / stride + 1;
    uint32_t out_dimensions = (C_o * output_rows * output_cols);
#else
    uint32_t out_dimensions = in_dimensions;
#endif

    Buffer input_dc(in_dimensions);
    Buffer out_dc(out_dimensions);
    Buffer out_check_dc(out_dimensions);
    std::vector<uint32_t> intermediate_block_dimensions;

    std::vector<uint64_t> unfused_timing;

    // Initialize Outputs to 0

    // Copy Inputs to their flat buffers
    init(input_dc, in_dimensions);

#if LAYER == CONV || LAYER == PARTIAL_CONV
    uint32_t filter_dimensions = (C_i * C_o * kernel_size * kernel_size);
#elif LAYER == DW_CONV
    uint32_t filter_dimensions = (C_i * kernel_size * kernel_size);
#endif

#if LAYER < POOL
    Buffer filter_dc(filter_dimensions);
    init(filter_dc, filter_dimensions);
#endif

    printf("Computing with reference and platform kernels.\n");

#if LAYER == RELU
    check_ReLUActivation(C_i,
                         input_height, input_width,
                         input_dc, out_check_dc);
    small::ReLUActivation(C_i,
                          input_height, input_width,
                          input_dc, out_dc);
#elif LAYER == POOL
    check_MaxPool2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                    t_pad, b_pad, l_pad, r_pad,
                    C_i,
                    input_height, input_width,
                    input_dc, out_check_dc);
    small::MaxPool2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                     t_pad, b_pad, l_pad, r_pad,
                     C_i,
                     input_height, input_width,
                     input_dc, out_dc);
#elif LAYER == DW_CONV
    check_DepthwiseConv2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                          t_pad, b_pad, l_pad, r_pad,
                          C_i,
                          input_height, input_width,
                          input_dc, filter_dc, out_check_dc);
    small::DepthwiseConv2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                           t_pad, b_pad, l_pad, r_pad,
                           C_i,
                           input_height, input_width,
                           input_dc, filter_dc, out_dc);
#elif LAYER == CONV
    check_Conv2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                 t_pad, b_pad, l_pad, r_pad,
                 C_o, C_i,
                 input_height, input_width,
                 input_dc, filter_dc, out_check_dc);
    small::Conv2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                  t_pad, b_pad, l_pad, r_pad,
                  C_o, C_i,
                  input_height, input_width,
                  input_dc, filter_dc, out_dc);

#elif LAYER == PARTIAL_CONV
    check_PartialConv2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                        t_pad, b_pad, l_pad, r_pad,
                        C_o, C_i,
                        input_height, input_width,
                        input_dc, filter_dc, out_check_dc);
    small::PartialConv2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                         t_pad, b_pad, l_pad, r_pad,
                         C_o, C_i,
                         input_height, input_width,
                         input_dc, filter_dc, out_dc);

// #elif LAYER == FC
    // check_Dense(C_o, C_i, input_dc, filter_dc, out_check_dc);
    // small::Dense(C_o, C_i, input_dc, filter_dc, out_dc);
#endif

    printf("Checking correctness.\n");
    assert(equals(out_dimensions, out_check_dc, out_dc, 1e-4));
    printf("end of correctness check\n");

    printf("Computing with reference and platform kernels %d times.\n", RUNS);

    small::Timer t;
    double ref_time = std::numeric_limits<double>::max();

    for (int run = 0; run < RUNS; run++)
    {
        t.start();

#if LAYER == RELU
        check_ReLUActivation(C_i,
                             input_height, input_width,
                             input_dc, out_check_dc);
#elif LAYER == POOL
        check_MaxPool2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                        t_pad, b_pad, l_pad, r_pad,
                        C_i,
                        input_height, input_width,
                        input_dc, out_check_dc);
#elif LAYER == DW_CONV
        check_DepthwiseConv2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                              t_pad, b_pad, l_pad, r_pad,
                              C_i,
                              input_height, input_width,
                              input_dc, filter_dc, out_check_dc);
#elif LAYER == CONV
        check_Conv2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                     t_pad, b_pad, l_pad, r_pad,
                     C_o, C_i,
                     input_height, input_width,
                     input_dc, filter_dc, out_check_dc);
#elif LAYER == PARTIAL_CONV
        check_PartialConv2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                            t_pad, b_pad, l_pad, r_pad,
                            C_o, C_i,
                            input_height, input_width,
                            input_dc, filter_dc, out_check_dc);
#endif

        t.stop();
        ref_time = std::min<double>(ref_time, t.elapsed());
    }

    print_cycles(ref_time);


    double platform_time = std::numeric_limits<double>::max();
    for (int run = 0; run < RUNS; run++)
    {
        t.start();
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
#if LAYER == RELU
        small::ReLUActivation(C_i,
                              input_height, input_width,
                              input_dc, out_dc);
#elif LAYER == POOL
        small::MaxPool2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                         t_pad, b_pad, l_pad, r_pad,
                         C_i,
                         input_height, input_width,
                         input_dc, out_dc);
#elif LAYER == DW_CONV
        small::DepthwiseConv2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                               t_pad, b_pad, l_pad, r_pad,
                               C_i,
                               input_height, input_width,
                               input_dc, filter_dc, out_dc);
#elif LAYER == CONV
        small::Conv2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                      t_pad, b_pad, l_pad, r_pad,
                      C_o, C_i,
                      input_height, input_width,
                      input_dc, filter_dc, out_dc);
#elif LAYER == PARTIAL_CONV
        small::PartialConv2D<small::FloatBuffer>(kernel_size, kernel_size, stride,
                             t_pad, b_pad, l_pad, r_pad,
                             C_o, C_i,
                             input_height, input_width,
                             input_dc, filter_dc, out_dc);
#endif
        t.stop();
        platform_time = std::min<double>(platform_time, t.elapsed());
    }

    print_cycles(platform_time);
    printf("\n");


    printf("reference time / platform time = %.4f \n", (ref_time/platform_time));
    fflush(0);

    auto output_els = out_dimensions;
    double compute_ops = 0.0;
    double throughput = 0.0;

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
    throughput = (FLOAT_NUM_FMA * 2.0 * FLOAT_SIMD);
#elif LAYER == PARTIAL_CONV
    /// @todo CONFIRM THESE NUMBERS
    compute_ops = output_els * (kernel_size * kernel_size * C_i * 2.0);
    throughput = (NUM_FMA * 2.0 * SIMD);
#elif LAYER == FC
    compute_ops = output_els * (C_i * 2.0);
    throughput = (NUM_FMA * 2.0 * SIMD);
#endif

    //dtype peak_cycles = compute_ops/throughput;
    //dtype scaled_peak_cycles = peak_cycles;
    const int num_th = atoi(std::getenv("OMP_NUM_THREADS"));
    printf(" %.0f %.2f \n", throughput*num_th, (compute_ops / platform_time));
}
