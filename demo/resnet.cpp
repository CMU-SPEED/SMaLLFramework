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

#define PARALLEL 1

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <limits>

#include <small.h>
#include <small/utils/Timer.hpp>
#include "utils.h"

/// @todo Which of these defines are needed?
#ifndef RUNS
#define RUNS 100
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
/* This is the runtime recording:

   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:16,ichans:3,img:32x32,I,F,O)
   ReLUActivation(chans:16,img:32x32,I,O)

   ** First Stack **
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:16,ichans:16,img:32x32,I,F,O)
   ReLUActivation(chans:16,img:32x32,I,O)
   PartialConv2D(k:3,s:1,pad:[1,1,1,1],ochans:16,ichans:16,img:32x32,I,F,O)
   ReLUActivation(chans:16,img:32x32,I,O)

   ** Second Stack **
   Conv2D(k:3,s:2,pad:[0,1,0,1],ochans:32,ichans:16,img:32x32,I,F,O)
   ReLUActivation(chans:32,img:16x16,I,O)

?  Conv2D(k:1,s:2,pad:[0,0,0,0],ochans:32,ichans:16,img:32x32,I,F,O)

   PartialConv2D(k:3,s:1,pad:[1,1,1,1],ochans:32,ichans:32,img:16x16,I,F,O)
   ReLUActivation(chans:32,img:16x16,I,O)

   ** Third Stack **
   Conv2D(k:3,s:2,pad:[0,1,0,1],ochans:64,ichans:32,img:16x16,I,F,O)
   ReLUActivation(chans:64,img:8x8,I,O)

?  Conv2D(k:1,s:2,pad:[0,0,0,0],ochans:64,ichans:32,img:16x16,I,F,O)

   PartialConv2D(k:3,s:1,pad:[1,1,1,1],ochans:64,ichans:64,img:8x8,I,F,O)
   ReLUActivation(chans:64,img:8x8,I,O)

   ** Final Classification Layer ** (Keras Model: AveragePooling2D + Dense)
   MaxPool2D(k:8,s:1,pad:[0,0,0,0],chans:64,img:8x8,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:16,ichans:64,img:1x1,I,F,O)

*/

//****************************************************************************
// The output of the block is stored in I
// The weights must have been copied into F_1x1 and F_dw beforehand
template <bool scale_channels>
inline void resnet_block(
    uint32_t in_dims[2], uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride,          // DWise Covolution parameters
    uint32_t output_channels, // 1x1 Convolution parameters
    uint8_t t_pad_0,
    uint8_t b_pad_0,
    uint8_t l_pad_0,
    uint8_t r_pad_0,
    uint8_t t_pad_1,
    uint8_t b_pad_1,
    uint8_t l_pad_1,
    uint8_t r_pad_1,
    dtype *I,
    dtype *F_conv0,
    dtype *F_conv1,
    dtype *F_conv_1x1,
    dtype *O_intermediate,
    dtype *O)
{
    Conv2D(2, kernel_size, stride, t_pad_0, b_pad_0, l_pad_0, r_pad_0, output_channels, input_channels, in_dims[0], in_dims[1], I, F_conv0, O_intermediate);
    uint32_t o_h = output_dim(in_dims[0] + t_pad_0 + b_pad_0, stride, kernel_size);
    uint32_t o_w = output_dim(in_dims[1] + l_pad_0 + r_pad_0, stride, kernel_size);
    ReLUActivation(1, output_channels, o_h, o_w, O_intermediate, O_intermediate);
    if (scale_channels)
    {
        Conv2D(2, 1, stride, 0, 0, 0, 0, output_channels, input_channels, in_dims[0], in_dims[1], I, F_conv_1x1, O);
    }
    PartialConv2D(0, kernel_size, 1, t_pad_1, b_pad_1, l_pad_1, r_pad_1, output_channels, output_channels, o_h, o_w, O_intermediate, F_conv1, O);
    ReLUActivation(1, output_channels, o_h, o_w, O, O);
}

#define REDUCTION_C(layer_num) layer_params[layer_num][0]
#define GROUP_C(layer_num) layer_params[layer_num][1]
#define GROUPS(layer_num) layer_params[layer_num][2]
#define REDUCTION_HW(layer_num) layer_params[layer_num][3]
#define STRIDE(layer_num) layer_params[layer_num][4]
#define TYPE(layer_num) layer_params[layer_num][9]

#define SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad) layer_params[layer_num][5] = t_pad, layer_params[layer_num][6] = b_pad, layer_params[layer_num][7] = l_pad, layer_params[layer_num][8] = r_pad;
#define PADDING(layer_num) layer_params[layer_num][5], layer_params[layer_num][6], layer_params[layer_num][7], layer_params[layer_num][8]

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
dtype *model_inference(uint32_t layer_num_total, uint16_t layer_params[30][10], uint32_t intermediate_dims[30][2], dtype *filter_ptrs[30], dtype *input_dc, dtype *inter_0_dc, dtype *inter_1_dc, dtype *inter_2_dc)
{
    auto layer_num = 0;
    Conv2D(0, REDUCTION_HW(layer_num), STRIDE(layer_num), PADDING(layer_num), GROUP_C(layer_num), REDUCTION_C(layer_num), I_HEIGHT(layer_num), I_WIDTH(layer_num), input_dc, filter_ptrs[layer_num], inter_0_dc);
    layer_num++;
    ReLUActivation(1, GROUP_C(0), I_HEIGHT(layer_num), I_WIDTH(layer_num), inter_0_dc, inter_0_dc);

    resnet_block<0>(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
                    REDUCTION_HW(layer_num),
                    STRIDE(layer_num), // Params for the first convolution
                    GROUP_C(layer_num),
                    PADDING(layer_num),
                    PADDING(layer_num + 1),
                    inter_0_dc,
                    filter_ptrs[layer_num],
                    filter_ptrs[layer_num + 1],
                    NULL,
                    inter_1_dc,
                    inter_0_dc);

    layer_num += 2;
    auto resnet_blocks = 3;
    auto num_filters = layer_num_total - 1;
    for (int ds_layer = 1; ds_layer < resnet_blocks; ds_layer++)
    {
        dtype *O_intermediate = inter_2_dc;
        resnet_block<1>(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
                        REDUCTION_HW(layer_num),
                        STRIDE(layer_num), // Params for the first convolution
                        GROUP_C(layer_num),
                        PADDING(layer_num), PADDING(layer_num + 1),
                        inter_0_dc,
                        filter_ptrs[layer_num],
                        filter_ptrs[layer_num + 1],
                        filter_ptrs[layer_num + 2],
                        inter_1_dc,
                        O_intermediate);

        layer_num += 3;
        // Since channels were scaled, switch the pointers between inter_2 and inter_0
        inter_2_dc = inter_0_dc;
        inter_0_dc = O_intermediate;
    }

    Maxpool2D(0, REDUCTION_HW(layer_num), STRIDE(layer_num), PADDING(layer_num), GROUPS(layer_num), I_HEIGHT(layer_num), I_WIDTH(layer_num), inter_0_dc, inter_1_dc);
    Conv2D(0, 1, 1, 0, 0, 0, 0, GROUP_C(layer_num_total - 1), REDUCTION_C(layer_num_total - 1), 1, 1, inter_1_dc, filter_ptrs[num_filters - 1], inter_0_dc);
    return inter_0_dc;
}

//****************************************************************************
int main()
{
    int C_i = 3;
    uint32_t N = 32;
    uint32_t M = 32;
    int num_classes = 16;

    // Create input tensor
    uint32_t input_dimensions = C_i * N * M;
    dtype *input_dc = alloc<dtype>(input_dimensions);
    init(input_dc, input_dimensions);

    // ================================================

    uint16_t layer_params[30][10] = {1};
    uint32_t intermediate_dims[30][2];

    uint8_t t_pad, b_pad, r_pad, l_pad;

    // Set up model parameters
    uint32_t max_numel_inter_0 = 0, max_numel_inter_1 = 0;

    intermediate_dims[0][0] = M;
    intermediate_dims[0][1] = N;

    // conv
    REDUCTION_C(0) = C_i; // input channels
    GROUP_C(0) = 16;      // output channels
    GROUPS(0) = 1;
    REDUCTION_HW(0) = 3; // kernel size
    STRIDE(0) = 1;       // stride
    CALC_PADDING(I_HEIGHT(0), REDUCTION_HW(0), STRIDE(0), t_pad, b_pad);
    CALC_PADDING(I_WIDTH(0), REDUCTION_HW(0), STRIDE(0), l_pad, r_pad);
    SET_PADDING(0, t_pad, b_pad, l_pad, r_pad);

    intermediate_dims[1][0] = O_WIDTH(1);
    intermediate_dims[1][1] = O_HEIGHT(1);
    auto inter_dim = INPUT_NUMEL(1);
    max_numel_inter_0 = (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

    int layer_num = 1;
    // common set up for model architecture
    auto resnet_blocks = 3;
    int layer_strides[] = {1, 2, 2};
    // dwise 1
    for (int ds_layer = 0; ds_layer < resnet_blocks; ds_layer++)
    {
        int channel_multiplier = (ds_layer > 0) ? 2 : 1;

        uint32_t in_channels = GROUP_C(layer_num - 1); // output channels from the previous block

        REDUCTION_C(layer_num) = in_channels; // input channels
        GROUP_C(layer_num) = in_channels * channel_multiplier;
        GROUPS(layer_num) = 1;                       // output channels
        REDUCTION_HW(layer_num) = 3;                 // kernel size
        STRIDE(layer_num) = layer_strides[ds_layer]; // stride
        CALC_PADDING(I_HEIGHT(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), t_pad, b_pad);
        CALC_PADDING(I_WIDTH(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

        layer_num++; // 2,4,7
        intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
        intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);

        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_1 = (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;

        REDUCTION_C(layer_num) = GROUP_C(layer_num - 1);
        GROUP_C(layer_num) = GROUP_C(layer_num - 1);
        GROUPS(layer_num) = 1;
        REDUCTION_HW(layer_num) = 3;
        STRIDE(layer_num) = 1;
        CALC_PADDING(I_HEIGHT(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), t_pad, b_pad);
        CALC_PADDING(I_WIDTH(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

        layer_num++; // 3,5,8
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_0 = (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

        if (channel_multiplier != 1)
        {
            intermediate_dims[layer_num][0] = O_WIDTH(layer_num - 2);
            intermediate_dims[layer_num][1] = O_HEIGHT(layer_num - 2);
            REDUCTION_C(layer_num) = in_channels; // input channels
            GROUP_C(layer_num) = in_channels * channel_multiplier;
            GROUPS(layer_num) = 1;       // output channels
            REDUCTION_HW(layer_num) = 1; // kernel size
            STRIDE(layer_num) = 2;       // stride
            SET_PADDING(layer_num, 0, 0, 0, 0);
            layer_num++; // 6,9
            inter_dim = INPUT_NUMEL(layer_num);
            max_numel_inter_0 = (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;
        }
        intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
        intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);
    }

    // pooling dims
    REDUCTION_C(layer_num) = 1;
    GROUP_C(layer_num) = 1;
    GROUPS(layer_num) = GROUP_C(layer_num - 1);
    REDUCTION_HW(layer_num) = I_HEIGHT(layer_num);
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0);

    layer_num++;
    intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
    intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);

    REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
    GROUP_C(layer_num) = num_classes;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 1;
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0);
    layer_num++;

    // fc dims
    uint32_t layer_num_total = layer_num;
    auto num_filters = layer_num_total - 1;

#if SUMMARY == 1
    for (uint32_t i = 0; i < layer_num_total; i++)
    {
        printf("layer %d: ", i);
	printf(" input_dims: %d %d ", I_HEIGHT(i), I_WIDTH(i));
        for (auto j = 0; j < 10; j++)
        {
            printf("%d, ", layer_params[i][j]);
        }
	printf("\b\b");
	//printf("input dims: %d %d ", I_HEIGHT(i+1), I_WIDTH(i+1));
        printf("\n");
    }
#endif

    // std::vector<uint32_t> filter_dimensions;

    dtype *filter_fc_dc;
    dtype *filter_ptrs[30];
    for (uint32_t l = 0; l < num_filters - 1; l++)
    {
        dtype *filter_ptr;
        uint32_t filter_dimensions = REDUCTION_HW(l) * REDUCTION_HW(l) * REDUCTION_C(l) * GROUP_C(l) * GROUPS(l);
        filter_ptr = alloc<dtype>(filter_dimensions);
        init(filter_ptr, filter_dimensions);
        filter_ptrs[l] = filter_ptr;
    }

    uint32_t filter_dimensions = GROUP_C(layer_num_total - 1) * REDUCTION_C(layer_num_total - 1);
    filter_fc_dc = alloc<dtype>(filter_dimensions);
    init(filter_fc_dc, filter_dimensions);
    filter_ptrs[num_filters - 1] = filter_fc_dc;

    // copy input
    // allocate space for intermediate outputs (use the max sizes calculated previously)
    dtype *inter_0_dc = alloc<dtype>(max_numel_inter_0);
    dtype *inter_1_dc = alloc<dtype>(max_numel_inter_1);
    dtype *inter_2_dc = alloc<dtype>(max_numel_inter_0);
    dtype *output_dc;

    //======================================================
    small::Timer my_timer;

    std::cerr << "Warm up run (ORIG)" << std::endl;
    my_timer.start();
    output_dc = model_inference(layer_num_total, layer_params, intermediate_dims, filter_ptrs, input_dc, inter_0_dc, inter_1_dc, inter_2_dc);
    my_timer.stop();
    printf("\nElapsed time: %lf ns.\n", my_timer.elapsed());

    // Compare the results
    std::cout << "\nCHECK RESULTS: Num output elements: " << num_classes
              << std::endl;
    for (auto ix = 0; ix < num_classes; ++ix)
    {
        std::cout << "Output " << ix << ": " << output_dc[ix] << std::endl;
    }

    //======================================================

    double min_small = std::numeric_limits<double>::max();
    std::vector<double> small_timing;

    for (int r = 0; r < RUNS; r++)
    {
        my_timer.start();

        output_dc = model_inference(layer_num_total, layer_params, intermediate_dims, filter_ptrs, input_dc, inter_0_dc, inter_1_dc, inter_2_dc);

        my_timer.stop();
        auto diff = my_timer.elapsed();
        min_small = std::min<double>(min_small, diff);
        small_timing.push_back(diff);
    }

    std::cout << "\nMinimum time: " << min_small << " ns.\n";
    const int num_th = atoi(std::getenv("OMP_NUM_THREADS"));
    std::cout << "Num Threads: " << num_th << std::endl;
    print_stats(small_timing, "SMaLL:resnet");

    free(input_dc);

    for (size_t l = 0; l < num_filters; l++)
    {
        free(filter_ptrs[l]);
    }

    free(inter_0_dc);
    free(inter_1_dc);
    free(inter_2_dc);
}
