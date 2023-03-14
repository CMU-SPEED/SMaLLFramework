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
#include <vector>

#include <small.h>
#include "utils.h"

/// @todo Which of these defines are needed?
#ifndef RUNS
#define RUNS 1000
#endif
#ifndef PARALLEL
#define PARALLEL 0
#endif

//****************************************************************************

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

//****************************************************************************
// The output of the block is stored in O
//
inline void dscnn_block(
    std::array<uint32_t, 2> const &in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t output_channels,
    uint8_t t_pad,
    uint8_t b_pad,
    uint8_t l_pad,
    uint8_t r_pad,
    small::FloatBuffer const &I,
    small::FloatBuffer const &F_dw,
    small::FloatBuffer const &F_1x1,
    small::FloatBuffer       &O_intermediate,
    small::FloatBuffer       &O)
{
    small::DepthwiseConv2D(kernel_size, stride,
                           t_pad, b_pad, l_pad, r_pad,
                           input_channels,
                           in_dims[0], in_dims[1],
                           I, F_dw, O_intermediate);

    uint32_t o_h = small::output_dim(in_dims[0] + t_pad + b_pad,
                                     stride, kernel_size);
    uint32_t o_w = small::output_dim(in_dims[1] + l_pad + r_pad,
                                     stride, kernel_size);

    small::ReLUActivation(input_channels,
                          o_h, o_w,
                          O_intermediate, O_intermediate);
    small::Conv2D(1, 1,
                  0, 0, 0, 0,
                  output_channels, input_channels,
                  o_h, o_w,
                  O_intermediate, F_1x1, O);
    small::ReLUActivation(output_channels, o_h, o_w, O, O);
}

//****************************************************************************
small::FloatBuffer &
model_inference(uint32_t layer_num_total,
                uint16_t layer_params[30][10],
                std::vector<std::array<uint32_t, 2>> const &intermediate_dims,
                std::vector<small::FloatBuffer *> const &filter_buf_ptrs,
                small::FloatBuffer const &input_dc,
                small::FloatBuffer       &inter_0_dc,
                small::FloatBuffer       &inter_1_dc)
{
    auto layer_num = 0;
    small::Conv2D_rect(REDUCTION_H(layer_num), REDUCTION_W(layer_num),
                       STRIDE(layer_num), PADDING(layer_num),
                       GROUP_C(layer_num), REDUCTION_C(layer_num),
                       I_HEIGHT(layer_num), I_WIDTH(layer_num),
                       input_dc,
                       *filter_buf_ptrs[layer_num],
                       inter_0_dc);

    layer_num++;
    small::ReLUActivation(GROUP_C(0),
                          I_HEIGHT(layer_num), I_WIDTH(layer_num),
                          inter_0_dc,
                          inter_0_dc);

    //std::cout << "H: " << I_HEIGHT(layer_num)
    //          << " W: " << I_WIDTH(layer_num)
    //          << " C:" << GROUP_C(0) << std::endl;

    auto ds_blocks = 4;
    for (int ds_layer = 0; ds_layer < ds_blocks; ds_layer++)
    {
        dscnn_block(
            intermediate_dims[layer_num], GROUPS(layer_num), // Input dimensions
            REDUCTION_HW(layer_num),
            STRIDE(layer_num),
            GROUP_C(layer_num + 1),
            PADDING(layer_num),
            inter_0_dc,
            *filter_buf_ptrs[layer_num],
            *filter_buf_ptrs[layer_num + 1],
            inter_1_dc,
            inter_0_dc);
        layer_num += 2;
    }

    // printf("calling pool %d %d \n", layer_num, layers.size());
    small::MaxPool2D_rect(REDUCTION_H(layer_num), REDUCTION_W(layer_num),
                          STRIDE(layer_num), PADDING(layer_num),
                          GROUPS(layer_num),
                          I_HEIGHT(layer_num), I_WIDTH(layer_num),
                          inter_0_dc,
                          inter_1_dc);

    uint32_t num_classes = 16;  /// @todo get from layer params
    small::Conv2D(1, 1,
                  0, 0, 0, 0,
                  num_classes, GROUP_C(layer_num_total - 1),
                  1, 1,
                  inter_1_dc,
                  *filter_buf_ptrs[layer_num - 1], // layer_num??
                  inter_0_dc);
    return inter_0_dc;
}

//****************************************************************************
//****************************************************************************
void inference()
{
    uint32_t C_i = 3;
    uint32_t N = 49;
    uint32_t M = 10;
    uint32_t num_classes = 16;  // must be a multiple of 16

    //Create input tensor
    uint32_t input_dimensions = C_i * N * M;
    small::FloatBuffer input_dc(input_dimensions);
    init(input_dc, input_dimensions);

    // ================================================

    uint16_t layer_params[30][10] = {1};
    std::vector<std::array<uint32_t, 2>> intermediate_dims;

    uint8_t t_pad, b_pad, r_pad, l_pad;

    // Set up model parameters
    int layer_num = 0;
    uint32_t max_numel_inter_0 = 0, max_numel_inter_1 = 0;

    intermediate_dims.push_back(std::array<uint, 2>({M, N}));

    // conv
    REDUCTION_C(layer_num) = C_i;
    GROUP_C(layer_num) = 64;
    GROUPS(layer_num) = 1;
    REDUCTION_H(layer_num) = 3;
    REDUCTION_W(layer_num) = 3;
    STRIDE(layer_num) = 2;
    //printf("%d %d %d %d\n", N, M, I_HEIGHT(layer_num), I_WIDTH(layer_num));
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_H(layer_num),
                        STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num),  REDUCTION_W(layer_num),
                        STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
    //printf("layer %d paddibg: %d %d %d %d\n", layer_num, PADDING(layer_num));

    layer_num++; // 1
    intermediate_dims.push_back(std::array<uint, 2>({5, 25}));
    //std::cout << "conv " << layer_num << "  "
    //          << I_HEIGHT(layer_num) << " " << I_WIDTH(layer_num) << " "
    //          << GROUP_C(layer_num - 1) << std::endl;

    auto inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_0 =
        (inter_dim > max_numel_inter_0)? inter_dim : max_numel_inter_0;
    // common set up for model architecture
    auto ds_blocks = 4;

    const int layer_strides[] = {1, 1, 1, 1};
    // dwise 1
    for (int ds_layer = 0; ds_layer < ds_blocks; ds_layer++)
    {
        auto channel_multiplier = layer_strides[ds_layer];

        REDUCTION_C(layer_num) = 1; // input channels
        GROUP_C(layer_num) = 1;
        GROUPS(layer_num) = GROUP_C(layer_num - 1);  // output channels
        REDUCTION_HW(layer_num) = 3;                 // kernel size
        STRIDE(layer_num) = layer_strides[ds_layer]; // stride
        small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                            STRIDE(layer_num), t_pad, b_pad);
        small::calc_padding(I_WIDTH(layer_num), REDUCTION_HW(layer_num),
                            STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

        layer_num++; // 2
        intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
        // std::cout << "dw " << layer_num << "  "
        //           << I_HEIGHT(layer_num) << " " << I_WIDTH(layer_num) << " "
        //           << GROUP_C(layer_num - 2) << std::endl;

        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_1 =
            (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;
        REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
        GROUP_C(layer_num) = (GROUPS(layer_num - 1)) * channel_multiplier;
        GROUPS(layer_num) = 1;
        REDUCTION_HW(layer_num) = 1;
        STRIDE(layer_num) = 1;
        SET_PADDING(layer_num, 0, 0, 0, 0);

        layer_num++; // 3
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_0 =
            (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;
        intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
        // std::cout << intermediate_dims[layer_num - 1][0] << " "
        //           << intermediate_dims[layer_num - 1][1] << std::endl;
        std::cout << "1x1 " << layer_num << "  " << I_HEIGHT(layer_num) << " "
                  << I_WIDTH(layer_num) << " " << GROUP_C(layer_num - 1)
                  << std::endl;
    }
    // pooling dims
    //printf("%d pool layer num %d %d\n",
    //       layer_num, I_HEIGHT(layer_num), I_WIDTH(layer_num));
    REDUCTION_C(layer_num) = 1;
    GROUP_C(layer_num) = 1;
    GROUPS(layer_num) = GROUP_C(layer_num - 1);
    REDUCTION_H(layer_num) = I_HEIGHT(layer_num);
    REDUCTION_W(layer_num) = I_WIDTH(layer_num);
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0);

    layer_num++;

    /// @todo MISSING A LAYER (compare to dscnn.cpp)

    // fc dims
    printf("size of intermediate buffers from configuration: %d %d\n",
           max_numel_inter_0, max_numel_inter_1);

    auto layer_num_total = layer_num - 1;

    //printf("Layer num total: %d", layer_num_total);

    //  Copy layer weights to temporaries
    std::vector<small::FloatBuffer *> filter_buf_ptrs;

    for (int l = 0; l < layer_num_total; l++)
    {
        uint32_t filter_dimensions =
            REDUCTION_H(l) * REDUCTION_W(l) * REDUCTION_C(l) *
            GROUP_C(l) * GROUPS(l);

        small::FloatBuffer *filter_buf_ptr =
            new small::FloatBuffer(filter_dimensions);
        init(*filter_buf_ptr, filter_dimensions);
        filter_buf_ptrs.push_back(filter_buf_ptr);
    }

    uint32_t filter_dimensions = GROUP_C(layer_num_total) * num_classes;
    //printf("Fc filter dims %d x %d\n", GROUP_C(layer_num_total-1) , num_classes);
    small::FloatBuffer *filter_fc_dc_ptr =
        new small::FloatBuffer(filter_dimensions);
    init(*filter_fc_dc_ptr, filter_dimensions);
    filter_buf_ptrs.push_back(filter_fc_dc_ptr);

    // allocate space for intermediate outputs
    // (use the max sizes calculated previously)
    small::FloatBuffer inter_0_dc(max_numel_inter_0);
    small::FloatBuffer inter_1_dc(max_numel_inter_1);
    //small::FloatBuffer output_dc(num_classes);

    //auto &output_dc =
        model_inference(layer_num_total, layer_params, intermediate_dims,
                        filter_buf_ptrs,
                        input_dc,
                        inter_0_dc,
                        inter_1_dc);

    unsigned long long sum_small; //, t0, t1;
    sum_small = ULLONG_MAX;
    std::vector<unsigned long long> small_timing;
    for (int r = 0; r < RUNS; r++)
    {
        // t0 = rdtsc();
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

        //auto &output_dc =
            model_inference(layer_num_total, layer_params, intermediate_dims,
                            filter_buf_ptrs,
                            input_dc, inter_0_dc, inter_1_dc);

        // t1 = rdtsc();
        // MIN(sum_small, (t1 - t0));
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
        auto diff = time_difference(time1, time2);
        sum_small = std::min<unsigned long long>(sum_small, diff);
        small_timing.push_back(diff);
    }

    print_cycles(sum_small);
    print_stats(small_timing, "SMaLL");

    printf("deallocing %ld filters\n", filter_buf_ptrs.size());
    for (size_t l = 0; l < filter_buf_ptrs.size(); l++)
    {
        delete filter_buf_ptrs[l];
    }
}

//****************************************************************************
// For non-arduino platforms.  ... move to driver.cpp?
//****************************************************************************
#ifndef NANO33BLE
int main()
{
    inference();
    return 0;
}
#endif
