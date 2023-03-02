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
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm> // std::min_element
#include <iterator>
#include <array>
#include <iostream>
// #include <functional>
#include <numeric>

// typedef float dtype;

#include <small.h>
#include "utils.h"

/// @todo Which of these defines are needed?
#ifndef RUNS
#define RUNS 1000
#endif
#ifndef PARALLEL
#define PARALLEL 0
#endif

#define PREFETCH 1

#define H_TILE 0
#define POOLING 1

#define LIMIT 1e-2

#define CONV 0
#define PARTIAL_CONV 1 // under development
#define DW_CONV 2      // under development
#define GROUP_CONV 3   // under development
#define POOL 4
#define RELU 5

#ifndef LAYER
#define LAYER DW_CONV
#endif

//****************************************************************************
// The output of the block is stored in I
// The weights must have been copied into F_1x1 and F_dw beforehand
inline void dscnn_block(
    std::array<uint32_t, 2> in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride,          // DWise Covolution parameters
    uint32_t output_channels, // 1x1 Convolution parameters
    uint8_t t_pad,
    uint8_t b_pad,
    uint8_t l_pad,
    uint8_t r_pad,
    dtype *I,
    dtype *F_dw,
    dtype *F_1x1,
    dtype *O_intermediate,
    dtype *O)
{

    DepthwiseConv2D(2, kernel_size, stride, t_pad, b_pad, l_pad, r_pad, input_channels, in_dims[0], in_dims[1], I, F_dw, O_intermediate);
    //uint32_t o_h, o_w;
    //op_dim(in_dims[0] + t_pad + b_pad, stride, kernel_size, o_h);
    //op_dim(in_dims[1] + l_pad + r_pad, stride, kernel_size, o_w);
    uint32_t o_h = output_dim(in_dims[0] + t_pad + b_pad, stride, kernel_size);
    uint32_t o_w = output_dim(in_dims[1] + l_pad + r_pad, stride, kernel_size);
    ReLUActivation(1, input_channels, o_h, o_w, O_intermediate, O_intermediate);
    Conv2D(0, 1, 1, 0, 0, 0, 0, output_channels, input_channels, o_h, o_w, O_intermediate, F_1x1, O);
    ReLUActivation(1, output_channels, o_h, o_w, O, O);
}

#define REDUCTION_C(layer_num) layer_params[layer_num][0]
#define GROUP_C(layer_num) layer_params[layer_num][1]
#define GROUPS(layer_num) layer_params[layer_num][2]
#define REDUCTION_HW(layer_num) layer_params[layer_num][3]
#define STRIDE(layer_num) layer_params[layer_num][4]

// In the case of non-square kernels
#define REDUCTION_H(layer_num) layer_params[layer_num][3]
#define REDUCTION_W(layer_num) layer_params[layer_num][9]

#define SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad) layer_params[layer_num][5] = t_pad, layer_params[layer_num][6] = b_pad, layer_params[layer_num][7] = l_pad, layer_params[layer_num][8] = r_pad;
#define PADDING(layer_num) layer_params[layer_num][5], layer_params[layer_num][6], layer_params[layer_num][7], layer_params[layer_num][8]

#define BOTTOM_PADDING(layer_num) layer_params[layer_num][6]

#define RIGHT_PADDING(layer_num) layer_params[layer_num][8]

#define PADDING_TORCH(layer_num) layer_params[layer_num][7], layer_params[layer_num][8], layer_params[layer_num][5], layer_params[layer_num][6]

#define I_WIDTH(layer_num) intermediate_dims[layer_num][0]
#define I_HEIGHT(layer_num) intermediate_dims[layer_num][1]

#define O_HEIGHT(layer_num) (((I_HEIGHT(layer_num - 1) + layer_params[layer_num - 1][5] + layer_params[layer_num - 1][6]) - REDUCTION_HW(layer_num - 1)) / STRIDE(layer_num - 1) + 1)
#define O_WIDTH(layer_num) (((I_WIDTH(layer_num - 1) + layer_params[layer_num - 1][7] + layer_params[layer_num - 1][8]) - REDUCTION_HW(layer_num - 1)) / STRIDE(layer_num - 1) + 1)

#define OUTPUT_DIMS(layer_num)                  \
    {                                           \
        O_HEIGHT(layer_num), O_WIDTH(layer_num) \
    }

#define INPUT_NUMEL(layer_num) \
    (O_HEIGHT(layer_num) * O_WIDTH(layer_num) * GROUP_C(layer_num - 1) * GROUPS(layer_num - 1))

int main(int argc, char **argv)
{

    int C_i = 3;
    uint32_t N = 49;
    uint32_t M = 10;
    int num_classes = 16;

    // uint32_t check_blocks = atol(argv[5]);
    if (num_classes % 16 != 0)
    {
        printf("Number of output classes must be a multiple of 16\n");
        exit(-1);
    }

    //Create input tensor
    uint32_t input_dimensions = C_i*N*M;
    dtype *input_dc = alloc<dtype>(input_dimensions);
    init(input_dc, input_dimensions);

    // std::vector<std::vector<uint64_t>> implementations;

    // calculate total number of weight elements
    //uint32_t total_num_weights = 0;

    uint16_t layer_params[30][10] = {1};

    std::vector<std::array<uint32_t, 2>> intermediate_dims;

    uint8_t t_pad, b_pad, r_pad, l_pad;


    // Set up model parameters
    int layer_num = 0;
    uint32_t max_numel_inter_0 = 0, max_numel_inter_1 = 0;

    intermediate_dims.push_back(std::array<uint, 2>({M, N}));
    // conv
    REDUCTION_C(layer_num) = C_i; // input channels
    GROUP_C(layer_num) = 64;      // output channels
    GROUPS(layer_num) = 1;
    REDUCTION_H(layer_num) = 3; // kernel size
    REDUCTION_W(layer_num) = 3; //
    STRIDE(layer_num) = 2;      // stride
    printf("%d %d %d %d\n", N, M, I_HEIGHT(layer_num), I_WIDTH(layer_num));
    CALC_PADDING(I_HEIGHT(layer_num), REDUCTION_H(layer_num), STRIDE(layer_num), t_pad, b_pad);
    CALC_PADDING(I_WIDTH(layer_num), REDUCTION_W(layer_num), STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad)
    printf("layer %d paddibg: %d %d %d %d\n", layer_num, PADDING(layer_num));
    layer_num++; // 1
    intermediate_dims.push_back(std::array<uint, 2>({5, 25}));
    std::cout << "conv " << layer_num << "  " << I_HEIGHT(layer_num) << " " << I_WIDTH(layer_num) << " " << GROUP_C(layer_num - 1) << std::endl;
    auto inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_0 = (inter_dim > max_numel_inter_0)? inter_dim : max_numel_inter_0;
    // common set up for model architecture
    auto ds_blocks = 4;
    int layer_strides[] = {1, 1, 1, 1};
    // dwise 1
    for (int ds_layer = 0; ds_layer < ds_blocks; ds_layer++)
    {
        auto channel_multiplier = layer_strides[ds_layer];

        REDUCTION_C(layer_num) = 1; // input channels
        GROUP_C(layer_num) = 1;
        GROUPS(layer_num) = GROUP_C(layer_num - 1);  // output channels
        REDUCTION_HW(layer_num) = 3;                 // kernel size
        STRIDE(layer_num) = layer_strides[ds_layer]; // stride
        CALC_PADDING(I_HEIGHT(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), t_pad, b_pad);
        CALC_PADDING(I_WIDTH(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad)
        layer_num++; // 2
        intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
        // std::cout << "dw " << layer_num << "  " << I_HEIGHT(layer_num) << " " << I_WIDTH(layer_num) << " " << GROUP_C(layer_num - 2) << std::endl;

        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_1 = (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;

        REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
        GROUP_C(layer_num) = (GROUPS(layer_num - 1)) * channel_multiplier;
        GROUPS(layer_num) = 1;
        REDUCTION_HW(layer_num) = 1;
        STRIDE(layer_num) = 1;
        SET_PADDING(layer_num, 0, 0, 0, 0)
        layer_num++; // 3
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_0 = (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;
        intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
        // std::cout << intermediate_dims[layer_num - 1][0] << " " << intermediate_dims[layer_num - 1][1] << std::endl;
        std::cout << "1x1 " << layer_num << "  " << I_HEIGHT(layer_num) << " " << I_WIDTH(layer_num) << " " << GROUP_C(layer_num - 1) << std::endl;
    }
    // pooling dims
    printf("%d pool layer num %d %d\n", layer_num, I_HEIGHT(layer_num), I_WIDTH(layer_num));
    REDUCTION_C(layer_num) = 1;
    GROUP_C(layer_num) = 1;
    GROUPS(layer_num) = GROUP_C(layer_num - 1);
    REDUCTION_H(layer_num) = I_HEIGHT(layer_num);
    REDUCTION_W(layer_num) = I_WIDTH(layer_num);
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0)
    layer_num++;
    // fc dims
    printf("size of intermediate buffers from configuration: %d %d\n", max_numel_inter_0, max_numel_inter_1);

    auto layer_num_total = layer_num - 1;
    printf("Layer num total: %d", layer_num_total);
    // Direct Convolution Setup

    //bool check = 1;
    // #if PARALLEL
    //     uint32_t num_threads = atoi(std::getenv("OMP_NUM_THREADS"));
    // #endif

    //  Copy layer weights to temporaries
    // std::vector<uint32_t> filter_dimensions;

    dtype *filter_fc_dc; //, *filter_conv_dc, *filter_1x1_1_dc, *filter_dw_1_dc;
    std::vector<dtype *> filter_ptrs;

    // torch::Tensor weights;
    for (int l = 0; l < layer_num_total; l++)
    {
        dtype *filter_ptr;
        // weights = layers[l]->weight; // conv_1x1->weight;
        uint32_t filter_dimensions = REDUCTION_H(l) * REDUCTION_W(l) * REDUCTION_C(l) * GROUP_C(l) * GROUPS(l);
        filter_ptr = alloc<dtype>(filter_dimensions);
        init(filter_ptr, filter_dimensions);
        filter_ptrs.push_back(filter_ptr);
    }

    uint32_t filter_dimensions = GROUP_C(layer_num_total) * num_classes;
    printf("Fc filter dims %d x %d\n", GROUP_C(layer_num_total-1) , num_classes);
    filter_fc_dc = alloc<dtype>(filter_dimensions);
    init(filter_fc_dc, filter_dimensions);
    filter_ptrs.push_back(filter_fc_dc);

    dtype *inter_0_dc = alloc<dtype>(max_numel_inter_0);
    dtype *inter_1_dc = alloc<dtype>(max_numel_inter_1);
    dtype *output_dc = alloc<dtype>(num_classes);

    //uint32_t inter_h, inter_w;

    // C_i = 3;
    // C_o = 32;
    // stride = 2;
    // kernel_size = 3;
    // char padding = 'f';

    layer_num = 0;

    Conv2D_rect(0, REDUCTION_H(layer_num), REDUCTION_W(layer_num), STRIDE(layer_num), PADDING(layer_num), GROUP_C(layer_num), REDUCTION_C(layer_num), I_HEIGHT(layer_num), I_WIDTH(layer_num), input_dc, filter_ptrs[layer_num], inter_0_dc);
    layer_num++;
    ReLUActivation(1, GROUP_C(0), I_HEIGHT(layer_num), I_WIDTH(layer_num), inter_0_dc, inter_0_dc);


    std::cout << "H: " << I_HEIGHT(layer_num) << " W: " << I_WIDTH(layer_num) << " C:" << GROUP_C(0) << std::endl;

    for (int ds_layer = 0; ds_layer < ds_blocks; ds_layer++)
    {
        dscnn_block(
            intermediate_dims[layer_num], GROUPS(layer_num), // Input dimensions
            REDUCTION_HW(layer_num),
            STRIDE(layer_num),      // DWise Covolution parameters
            GROUP_C(layer_num + 1), // 1x1 Convolution parameters
            PADDING(layer_num),
            inter_0_dc,
            filter_ptrs[layer_num],
            filter_ptrs[layer_num + 1],
            inter_1_dc,
            inter_0_dc);
        layer_num += 2;
    }
    // printf("calling pool %d %d \n", layer_num, layers.size());
    MaxPool2D_rect(0, REDUCTION_H(layer_num), REDUCTION_W(layer_num), STRIDE(layer_num), PADDING(layer_num), GROUPS(layer_num), I_HEIGHT(layer_num), I_WIDTH(layer_num), inter_0_dc, inter_1_dc);
    // Dense(1, num_classes, GROUP_C(layer_num - 1), inter_1_dc, filter_fc_dc, output_dc);
    Conv2D(0, 1, 1, 0, 0, 0, 0, num_classes, GROUP_C(layer_num_total - 1), 1, 1, inter_1_dc, filter_ptrs[layer_num - 1], output_dc);

    printf("\n");

    unsigned long long sum_small; //, t0, t1;
    sum_small = ULLONG_MAX;
    std::vector<unsigned long long> small_timing;
    for (int r = 0; r < RUNS; r++)
    {
      //t0 = rdtsc();
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

      layer_num = 0;

        Conv2D_rect(0, REDUCTION_H(layer_num), REDUCTION_W(layer_num), STRIDE(layer_num), PADDING(layer_num), GROUP_C(layer_num), REDUCTION_C(layer_num), I_HEIGHT(layer_num), I_WIDTH(layer_num), input_dc, filter_ptrs[layer_num], inter_0_dc);
        layer_num++;
        ReLUActivation(1, GROUP_C(0), I_HEIGHT(layer_num), I_WIDTH(layer_num), inter_0_dc, inter_0_dc);

        // std::cout << "H: " << I_HEIGHT(layer_num) << " W: " << I_WIDTH(layer_num) << " C:" << GROUP_C(0) << std::endl;

        for (int ds_layer = 0; ds_layer < ds_blocks; ds_layer++)
        {
            dscnn_block(
                intermediate_dims[layer_num], GROUPS(layer_num), // Input dimensions
                REDUCTION_HW(layer_num),
                STRIDE(layer_num),      // DWise Covolution parameters
                GROUP_C(layer_num + 1), // 1x1 Convolution parameters
                PADDING(layer_num),
                inter_0_dc,
                filter_ptrs[layer_num],
                filter_ptrs[layer_num + 1],
                inter_1_dc,
                inter_0_dc);
            layer_num += 2;
        }
        // printf("calling pool %d %d \n", layer_num, layers.size());
        MaxPool2D_rect(0, REDUCTION_H(layer_num), REDUCTION_W(layer_num), STRIDE(layer_num), PADDING(layer_num), GROUPS(layer_num), I_HEIGHT(layer_num), I_WIDTH(layer_num), inter_0_dc, inter_1_dc);
        // Dense(1, num_classes, GROUP_C(layer_num - 1), inter_1_dc, filter_fc_dc, output_dc);
        Conv2D(0, 1, 1, 0, 0, 0, 0, num_classes, GROUP_C(layer_num_total - 1), 1, 1, inter_1_dc, filter_ptrs[layer_num - 1], output_dc);
        //t1 = rdtsc();
        //MIN(sum_small, (t1 - t0));
        //small_timing.push_back((t1 - t0));
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
        auto diff = time_difference(time1, time2);
        sum_small = std::min<unsigned long long>(sum_small, diff);
        //MIN(sum_small, diff);
    }

    print_cycles(sum_small);
    print_stats(small_timing, "SMaLL");
    printf("%d\n", atoi(std::getenv("OMP_NUM_THREADS")));
    // std::cout<<small_timing;
    free(input_dc);
    for (size_t l = 0; l < filter_ptrs.size(); l++)
    {
        free(filter_ptrs[l]);
    }
    printf("deallocing %ld filters\n", filter_ptrs.size());

    //free(inter_1_dc);
    printf("deallocing intermediates\n");

    //free(inter_0_dc);

    printf("deallocing intermediates\n");
    free(output_dc);
}
