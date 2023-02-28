#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>

typedef uint8_t dtype;

#include "include/small.h"
// #include "include/utils.h"

/// @todo Which of these defines are needed?
#ifndef RUNS
#define RUNS 1
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
    uint32_t in_dims[2], uint32_t input_channels, // Input dimensions
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

dtype *model_inference(uint32_t layer_num_total, uint16_t layer_params[30][10], uint32_t intermediate_dims[30][2], dtype *filter_ptrs[30], dtype *input_dc, dtype *inter_0_dc, dtype *inter_1_dc)
{
    int layer_num = 0;

    int num_filters = layer_num_total - 1;
    Conv2D_rect(0, REDUCTION_H(layer_num), REDUCTION_W(layer_num), STRIDE(layer_num), PADDING(layer_num), GROUP_C(layer_num), REDUCTION_C(layer_num), I_HEIGHT(layer_num), I_WIDTH(layer_num), input_dc, filter_ptrs[layer_num], inter_0_dc);
    layer_num++;
    ReLUActivation(1, GROUP_C(0), I_HEIGHT(layer_num), I_WIDTH(layer_num), inter_0_dc, inter_0_dc);
    auto ds_blocks = 4;
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
    MaxPool2D_rect(0, REDUCTION_H(layer_num), REDUCTION_W(layer_num), STRIDE(layer_num), PADDING(layer_num), GROUPS(layer_num), I_HEIGHT(layer_num), I_WIDTH(layer_num), inter_0_dc, inter_1_dc);
    layer_num++;
    Conv2D(0, 1, 1, 0, 0, 0, 0, GROUP_C(layer_num), REDUCTION_C(layer_num), 1, 1, inter_1_dc, filter_ptrs[num_filters - 1], inter_0_dc);

    return inter_0_dc;
}

//****************************************************************************
//****************************************************************************
void inference() {
    dtype buffer[100000];
    dtype *buffer_ptr;

    int C_i = 3;
    uint32_t N = 49;
    uint32_t M = 10;
    int num_classes = 16;

    uint32_t input_dimensions = C_i*N*M;
    dtype *input_dc = buffer_ptr; buffer_ptr += input_dimensions;
    init(input_dc, input_dimensions);

    uint16_t layer_params[30][10] = {1};
    uint32_t intermediate_dims[30][2];

    uint8_t t_pad, b_pad, r_pad, l_pad;

    // Set up model parameters
    uint32_t max_numel_inter_0 = 0, max_numel_inter_1 = 0;

    intermediate_dims[0][0] = M;
    intermediate_dims[0][1] = N;

    // conv
    REDUCTION_C(0) = C_i;
    GROUP_C(0) = 64;
    GROUPS(0) = 1;
    REDUCTION_H(0) = 3;
    REDUCTION_W(0) = 1;
    STRIDE(0) = 2;
    CALC_PADDING(I_HEIGHT(0), REDUCTION_H(0), STRIDE(0), t_pad, b_pad);
    CALC_PADDING(I_WIDTH(0), REDUCTION_W(0), STRIDE(0), l_pad, r_pad);
    SET_PADDING(0, t_pad, b_pad, l_pad, r_pad);
    SET_PADDING(0, 1, 1, 1, 1);

    intermediate_dims[1][0] = 5;
    intermediate_dims[1][1] = 25;

    auto inter_dim = INPUT_NUMEL(1);
    max_numel_inter_0 = (inter_dim > max_numel_inter_0)? inter_dim : max_numel_inter_0;
    auto ds_blocks = 4;

    int layer_num = 1;

    const int layer_strides[] = {1, 1, 1, 1};
    // dwise 1
    for (int ds_layer = 0; ds_layer < ds_blocks; ds_layer++) {
        auto channel_multiplier = layer_strides[ds_layer];

        REDUCTION_C(layer_num) = 1; // input channels
        GROUP_C(layer_num) = 1;
        GROUPS(layer_num) = GROUP_C(layer_num - 1);  // output channels
        REDUCTION_HW(layer_num) = 3;                 // kernel size
        REDUCTION_W(layer_num) = 3;
        STRIDE(layer_num) = layer_strides[ds_layer]; // stride
        CALC_PADDING(I_HEIGHT(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), t_pad, b_pad);
        CALC_PADDING(I_WIDTH(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad)
        layer_num++; // 2
        intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
        intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);

        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_1 = (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;
        REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
        GROUP_C(layer_num) = (GROUPS(layer_num - 1)) * channel_multiplier;
        GROUPS(layer_num) = 1;
        REDUCTION_HW(layer_num) = 1;
        REDUCTION_W(layer_num) = 1;
        STRIDE(layer_num) = 1;
        SET_PADDING(layer_num, 0, 0, 0, 0)
        layer_num++; // 3
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_0 = (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;
        intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
        intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);
    }
    // pooling dims
    REDUCTION_C(layer_num) = 1;
    GROUP_C(layer_num) = 1;
    GROUPS(layer_num) = GROUP_C(layer_num - 1);
    REDUCTION_H(layer_num) = I_HEIGHT(layer_num);
    REDUCTION_W(layer_num) = I_WIDTH(layer_num);
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0)
    layer_num++;

    intermediate_dims[layer_num][0] = 1;
    intermediate_dims[layer_num][1] = 1;
    REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
    GROUP_C(layer_num) = num_classes;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 1;
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0)
    layer_num++;

    auto layer_num_total = layer_num;
    auto num_filters = layer_num_total - 1;

    //  Copy layer weights to temporaries
    dtype *filter_fc_dc; 
    dtype *filter_ptrs[30];
    for (int l = 0; l < num_filters-1; l++) {
        dtype *filter_ptr;
        uint32_t filter_dimensions = REDUCTION_H(l) * REDUCTION_W(l) * REDUCTION_C(l) * GROUP_C(l) * GROUPS(l);
        filter_ptr = buffer_ptr; buffer_ptr += filter_dimensions;
        filter_ptrs[l] = filter_ptr;
    }

    uint32_t filter_dimensions = REDUCTION_C(layer_num_total-1) * GROUP_C(layer_num_total -1);
    filter_fc_dc = buffer_ptr; buffer_ptr += filter_dimensions;
    filter_ptrs[num_filters - 1] = filter_fc_dc;

    dtype *inter_0_dc = buffer_ptr; buffer_ptr += max_numel_inter_0;
    dtype *inter_1_dc = buffer_ptr; buffer_ptr += max_numel_inter_1;
    dtype *output_dc;


    mbed::Timer t;
    t.start();
    for (int r = 0; r < RUNS; r++) {
        output_dc = model_inference(layer_num_total, layer_params, intermediate_dims, filter_ptrs, input_dc, inter_0_dc, inter_1_dc);
    }
    t.stop();

    Serial.println(t.elapsed_time().count());
}
