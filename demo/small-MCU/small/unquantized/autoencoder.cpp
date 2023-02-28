#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>

// #include "mbed.h"

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



dtype * model_inference(uint32_t layer_num_total, uint16_t layer_params[30][10], dtype * filter_ptrs[30], dtype *input_dc, dtype *inter_0_dc, dtype *inter_1_dc)
{
    int layer_num = 0;
    Conv2D(0, 1, 1, 0, 0, 0, 0, GROUP_C(layer_num), REDUCTION_C(layer_num), 1, 1, input_dc, filter_ptrs[layer_num], inter_0_dc);
    ReLUActivation(1, GROUP_C(layer_num), 1, 1, inter_0_dc, inter_0_dc);

    dtype *out_inter_dc = inter_1_dc;
    for (int cur_layer = 1; cur_layer < layer_num_total; cur_layer++)
    {

        Conv2D(0, 1, 1, 0, 0, 0, 0, GROUP_C(layer_num), REDUCTION_C(layer_num), 1, 1, inter_0_dc, filter_ptrs[layer_num], out_inter_dc);
        ReLUActivation(1, GROUP_C(layer_num), 1, 1, out_inter_dc, inter_1_dc);
        layer_num++;
        inter_1_dc = inter_0_dc;
        inter_0_dc = out_inter_dc;
        out_inter_dc = inter_1_dc;
    }
    return inter_0_dc;
}

//****************************************************************************
//****************************************************************************
void inference() {
    dtype buffer[100000];
    dtype *buffer_ptr;
    int C_i = 128;
    uint32_t N = 1;
    uint32_t M = 1;
    int num_classes = 16;
    // // Create and Initialize small tensors
    // Create input tensor
    uint32_t input_dimensions = C_i * N * M;
    // dtype *input_dc = alloc<dtype>(input_dimensions);
    dtype *input_dc = buffer_ptr;
    buffer_ptr += input_dimensions;
    init(input_dc, input_dimensions);

    // calculate total number of weight elements

    uint16_t layer_params[30][10] = {1};

    uint32_t intermediate_dims[30][2];

    // Set up model parameters
    auto layer_num_total = 9;
    int layer_num = 0;
    uint32_t max_numel_inter_0 = 128, max_numel_inter_1 = 128;

    intermediate_dims[layer_num][0] = 1;
    intermediate_dims[layer_num][1] = 1;
    // conv
    REDUCTION_C(layer_num) = C_i; // input channels
    GROUP_C(layer_num) = 128;      // output channels
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 1; // kernel size
    STRIDE(layer_num) = 1;      // stride
    SET_PADDING(layer_num, 0, 0, 0, 0)
    layer_num++;
    intermediate_dims[layer_num][0] = 1;
    intermediate_dims[layer_num][1] = 1;

    // common set up for model architecture
    for (int cur_layer = 1; cur_layer < layer_num_total-1; cur_layer++)
    {

        REDUCTION_C(layer_num) = GROUP_C(layer_num - 1); // input channels
        GROUP_C(layer_num) = GROUP_C(layer_num - 1);
        GROUPS(layer_num) = 1;  // output channels
        REDUCTION_HW(layer_num) = 1;                 // kernel size
        STRIDE(layer_num) = 1; // stride
        SET_PADDING(layer_num, 0, 0, 0, 0)
        layer_num++; // 2

        intermediate_dims[layer_num][0] = 1;
        intermediate_dims[layer_num][1] = 1;
    }
    REDUCTION_C(layer_num) = GROUP_C(layer_num-1);
    GROUP_C(layer_num) = num_classes;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) =   1;
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0)
    layer_num++;
    intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
    intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);

    // Direct Convolution Setup

    dtype * filter_ptrs[30];
    for (int l = 0; l < layer_num_total; l++)
    {
        dtype *filter_ptr;
        uint32_t filter_dimensions = REDUCTION_HW(l) * REDUCTION_HW(l) * REDUCTION_C(l) * GROUP_C(l) * GROUPS(l);
        filter_ptr = buffer_ptr;
        buffer_ptr += filter_dimensions;
        init(filter_ptr, filter_dimensions);
        filter_ptrs[l] = filter_ptr;
    }

    dtype *inter_0_dc = buffer_ptr; buffer_ptr += max_numel_inter_0;
    dtype *inter_1_dc = buffer_ptr; buffer_ptr += max_numel_inter_1;
    dtype *output_dc;
    
    mbed::Timer t;
    t.start();
    for (int r = 0; r < RUNS; r++)
    {
        output_dc = model_inference(layer_num_total, layer_params, filter_ptrs, input_dc, inter_0_dc, inter_1_dc);
    }
    t.stop();
    
    Serial.println(t.elapsed_time().count());
}
