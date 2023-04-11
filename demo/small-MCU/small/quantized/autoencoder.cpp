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

#include <small.h>

/// @todo Which of these defines are needed?
#ifndef RUNS
#define RUNS 1
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
small::QUInt8Buffer &model_inference(
    uint32_t layer_num_total,
    uint16_t layer_params[30][10],
    small::QUInt8Buffer     **filter_buf_ptrs,
    small::QUInt8Buffer const &input_dc,
    small::QUInt8Buffer       &inter_0_dc,
    small::QUInt8Buffer       &inter_1_dc)
{
    auto layer_num = 0;
    small::Conv2D(1, 1,
                  0, 0, 0, 0,
                  GROUP_C(layer_num), REDUCTION_C(layer_num),
                  1, 1,
                  input_dc,
                  *filter_buf_ptrs[layer_num],
                  inter_0_dc);

    small::ReLUActivation(GROUP_C(layer_num),
                          1, 1,
                          inter_0_dc, inter_0_dc);

    for (uint32_t cur_layer = 1; cur_layer < layer_num_total; cur_layer++)
    {
        layer_num++;
        small::Conv2D(1, 1,
                      0, 0, 0, 0,
                      GROUP_C(layer_num), REDUCTION_C(layer_num),
                      1, 1,
                      inter_0_dc,
                      *filter_buf_ptrs[layer_num],
                      inter_1_dc);

        small::ReLUActivation(GROUP_C(layer_num),
                              1, 1,
                              inter_1_dc,
                              inter_1_dc);

        inter_0_dc.swap(inter_1_dc);
    }
    return inter_0_dc;
}

//****************************************************************************
//****************************************************************************
void inference()
{
    uint32_t C_i = 128;
    uint32_t N = 1;
    uint32_t M = 1;
    uint32_t num_classes = 16;

    // Create input tensor
    uint32_t input_dimensions = C_i * N * M;
    small::QUInt8Buffer input_dc(input_dimensions);
    small::init(input_dc, input_dimensions);

    // ================================================

    uint16_t layer_params[30][10] = {1};
    uint32_t intermediate_dims[30][2];

    // Set up model parameters
    auto layer_num_total = 9U;
    uint32_t max_numel_inter_0 = 128, max_numel_inter_1 = 128;

    uint32_t layer_num = 0;
    intermediate_dims[layer_num][0] = 1;
    intermediate_dims[layer_num][1] = 1;

    // conv
    REDUCTION_C(layer_num) = C_i; // input channels
    GROUP_C(layer_num) = 128;     // output channels
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 1;  // kernel size
    STRIDE(layer_num) = 1;        // stride
    SET_PADDING(layer_num, 0, 0, 0, 0);

    layer_num++;
    intermediate_dims[layer_num][0] = 1;
    intermediate_dims[layer_num][1] = 1;

    // common set up for model architecture
    for (uint32_t cur_layer = 1; cur_layer+1 < layer_num_total; cur_layer++)
    {
        REDUCTION_C(layer_num) = GROUP_C(layer_num - 1); // input channels
        GROUP_C(layer_num) = GROUP_C(layer_num - 1);     // output channels
        GROUPS(layer_num) = 1;
        REDUCTION_HW(layer_num) = 1;                 // kernel size
        STRIDE(layer_num) = 1; // stride
        SET_PADDING(layer_num, 0, 0, 0, 0);

        layer_num++; // 2
        intermediate_dims[layer_num][0] = 1;
        intermediate_dims[layer_num][1] = 1;
    }

    REDUCTION_C(layer_num) = GROUP_C(layer_num-1);
    GROUP_C(layer_num) = num_classes;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) =   1;
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0);

    layer_num++;
    intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
    intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);

    small::QUInt8Buffer *filter_buf_ptrs[30];

    // Direct Convolution Setup
    for (uint32_t l = 0; l < layer_num_total; l++)
    {
        uint32_t filter_dimensions =
            REDUCTION_HW(l) * REDUCTION_HW(l) * REDUCTION_C(l) *
            GROUP_C(l) * GROUPS(l);
        small::QUInt8Buffer *filter_buf_ptr =
            small::alloc_buffer(filter_dimensions);
        init(*filter_buf_ptr, filter_dimensions);
        filter_buf_ptrs[l] = filter_buf_ptr;
    }

    // allocating class on stack and data on 'heap'
    small::QUInt8Buffer inter_0_dc(max_numel_inter_0*4); /// @todo potential alignment issues
    small::QUInt8Buffer inter_1_dc(max_numel_inter_1*4);

    // always returns a reference to inter_0_dc
    auto &output_dc =
        model_inference(layer_num_total, layer_params,
                        filter_buf_ptrs,
                        input_dc,
                        inter_0_dc,
                        inter_1_dc);

#ifdef NANO33BLE
    mbed::Timer t;
    t.start();
    for (int r = 0; r < RUNS; r++)
    {
        //auto &output_dc =
            model_inference(layer_num_total, layer_params,
                            filter_buf_ptrs,
                            input_dc,
                            inter_0_dc,
                            inter_1_dc);
    }
    t.stop();
    Serial.println(t.elapsed_time().count());
#else
    for (size_t ix = 0; ix < num_classes; ix++)
    {
        printf("Output class %ld result: %d\n", ix, output_dc[ix]);
    }
#endif

    printf("deallocing %d filters\n", layer_num_total);

    for (size_t l = 0; l < layer_num_total; l++)
    {
        small::free_buffer(filter_buf_ptrs[l]);
    }

#if defined(NANO33BLE)
    small::detail::free_all();
#endif
}

//****************************************************************************
/// @todo For non-arduino platforms.  ... move to driver.cpp?
//****************************************************************************
// #ifndef NANO33BLE
// int main()
// {
//     inference();
//     return 0;
// }
// #endif
