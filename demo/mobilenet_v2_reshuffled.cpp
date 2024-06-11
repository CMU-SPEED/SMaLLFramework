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
#include <vector>

#include <small.h>
#include <small/utils/Timer.hpp>
#include "utils.h"

/// @todo Which of these defines are needed?
#ifndef RUNS
#define RUNS 10
#endif

#define SUMMARY 1
//****************************************************************************
/* This is the runtime recording:

   Conv2D(k:3,s:2,pad:[0,1,0,1],ochans:32,ichans:3,img:96x96,I,F,O)
   ReLUActivation(chans:32,img:48x48,I,O)
0
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:32,img:48x48,I,F,O)     s1
   ReLUActivation(chans:32,img:48x48,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:64,ichans:32,img:48x48,I,F,O)   2x
   ReLUActivation(chans:64,img:48x48,I,O)
1
   DepthwiseConv2D(k:3,s:2,pad:[0,1,0,1],chans:64,img:48x48,I,F,O)     s2
   ReLUActivation(chans:64,img:24x24,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:64,img:24x24,I,F,O)  2x
   ReLUActivation(chans:128,img:24x24,I,O)
2
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:128,img:24x24,I,F,O)    s1
   ReLUActivation(chans:128,img:24x24,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:24x24,I,F,O) 1x
   ReLUActivation(chans:128,img:24x24,I,O)
3
   DepthwiseConv2D(k:3,s:2,pad:[0,1,0,1],chans:128,img:24x24,I,F,O)    s2
   ReLUActivation(chans:128,img:12x12,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:256,ichans:128,img:12x12,I,F,O) 2x
   ReLUActivation(chans:256,img:12x12,I,O)
4
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:256,img:12x12,I,F,O)    s1
   ReLUActivation(chans:256,img:12x12,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:256,ichans:256,img:12x12,I,F,O) 1x
   ReLUActivation(chans:256,img:12x12,I,O)
5
   DepthwiseConv2D(k:3,s:2,pad:[0,1,0,1],chans:256,img:12x12,I,F,O)    s2
   ReLUActivation(chans:256,img:6x6,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:256,img:6x6,I,F,O)   2x
   ReLUActivation(chans:512,img:6x6,I,O)
6
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:512,img:6x6,I,F,O)      s1
   ReLUActivation(chans:512,img:6x6,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:512,img:6x6,I,F,O)   1x
   ReLUActivation(chans:512,img:6x6,I,O)
7
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:512,img:6x6,I,F,O)      s1
   ReLUActivation(chans:512,img:6x6,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:512,img:6x6,I,F,O)   1x
   ReLUActivation(chans:512,img:6x6,I,O)
8
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:512,img:6x6,I,F,O)      s1
   ReLUActivation(chans:512,img:6x6,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:512,img:6x6,I,F,O)   1x
   ReLUActivation(chans:512,img:6x6,I,O)
9
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:512,img:6x6,I,F,O)      s1
   ReLUActivation(chans:512,img:6x6,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:512,img:6x6,I,F,O)   1x
   ReLUActivation(chans:512,img:6x6,I,O)
10
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:512,img:6x6,I,F,O)      s1
   ReLUActivation(chans:512,img:6x6,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:512,img:6x6,I,F,O)   1x
   ReLUActivation(chans:512,img:6x6,I,O)
11
   DepthwiseConv2D(k:3,s:2,pad:[0,1,0,1],chans:512,img:6x6,I,F,O)      s2
   ReLUActivation(chans:512,img:3x3,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:1024,ichans:512,img:3x3,I,F,O)  2x
   ReLUActivation(chans:1024,img:3x3,I,O)
12
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:1024,img:3x3,I,F,O)     s1
   ReLUActivation(chans:1024,img:3x3,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:1024,ichans:1024,img:3x3,I,F,O) 1x
   ReLUActivation(chans:1024,img:3x3,I,O)
FINAL
   MaxPool2D(k:3,s:1,pad:[0,0,0,0],chans:1024,img:3x3,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:16,ichans:1024,img:1x1,I,F,O)
*/

#include<small/Layer.hpp>
#include<small/DepthwiseConv2DLayer.hpp>
#include<small/Conv2DLayer.hpp>
#include<small/MaxPool2DLayer.hpp>
#include<small/ReLULayer.hpp>


#define TIME_LAYER 1

template <class BufferT>
std::vector<small::Layer<BufferT>*> create_model(
    std::vector<BufferT*> const &filters)
{
    std::vector<small::Layer<BufferT>*> layers;

    // settings for first layers
    uint32_t kernel_size = 3;
    uint32_t stride = 2;
    uint32_t input_size = 96;
    uint32_t input_channels = 3, output_channels = 32;
    uint32_t num_classes = 16;
    size_t   filter_num = 0;

    small::shape_type input_shape(
        {1UL, input_channels, input_size, input_size});

    small::Layer<BufferT> *prev =
        new small::Conv2DLayer<BufferT>(input_shape,
                                        kernel_size, kernel_size,
                                        stride, small::PADDING_F,
                                        output_channels,
                                        *filters[filter_num], true);
    layers.push_back(prev);

    prev = new small::ReLULayer<BufferT>(prev->output_shape());
    layers.push_back(prev);

    size_t   const num_blocks{13};
    uint32_t const block_strides[]   = {1,2,1,2,1,2,1,1,1,1,1,2,1};
    uint32_t const channel_multiplier[] = {2,2,1,2,1,2,1,1,1,1,1,2,1};
    for (auto block_num = 0U; block_num < num_blocks; ++block_num)
    {
        ++filter_num;
        kernel_size = 3;

        prev = new small::DepthwiseConv2DLayer<BufferT>(
            prev->output_shape(),
            kernel_size, kernel_size, block_strides[block_num],
            small::PADDING_F,
            *filters[filter_num], true);
        layers.push_back(prev);

        prev = new small::ReLULayer<BufferT>(prev->output_shape());
        layers.push_back(prev);

        // =======================================================

        ++filter_num;
        kernel_size = 1;
        stride = 1;
        output_channels =
            prev->output_shape()[small::CHANNEL]*channel_multiplier[block_num];

        prev = new small::Conv2DLayer<BufferT>(prev->output_shape(),
                                               kernel_size, kernel_size,
                                               stride, small::PADDING_V,
                                               output_channels,
                                               *filters[filter_num], true);
        layers.push_back(prev);


        prev = new small::ReLULayer<BufferT>(prev->output_shape());
        layers.push_back(prev);
    }

    kernel_size = 3;
    stride = 1;

    prev = new small::MaxPool2DLayer<BufferT>(prev->output_shape(),
                                              kernel_size, kernel_size,
                                              stride, small::PADDING_V);
    layers.push_back(prev);

    ++filter_num;
    kernel_size = 1;
    prev = new small::Conv2DLayer<BufferT>(prev->output_shape(),
                                           kernel_size, kernel_size, stride,
                                           small::PADDING_V,
                                           num_classes,
                                           *filters[filter_num], true);
    layers.push_back(prev);

    return layers;
}

//****************************************************************************
template <class BufferT>
small::Tensor<BufferT> &model_inference(
    std::vector<small::Layer<BufferT>*> const &layers,
    small::Tensor<BufferT>              const &input_dc,
    small::Tensor<BufferT>                    &inter_0_dc,
    small::Tensor<BufferT>                    &inter_1_dc)
{
    size_t layer_num = 0;
    layers[layer_num++]->compute_output({&input_dc}, &inter_0_dc);   // Conv2D
    layers[layer_num++]->compute_output({&inter_0_dc}, &inter_0_dc); // ReLU

    auto ds_blocks = 13;
    for (auto ix = 0U; ix < ds_blocks; ++ix)
    {
        layers[layer_num++]->compute_output({&inter_0_dc}, &inter_1_dc); // DWConv
        layers[layer_num++]->compute_output({&inter_1_dc}, &inter_1_dc); // ReLU
        layers[layer_num++]->compute_output({&inter_1_dc}, &inter_0_dc); // Conv2D
        layers[layer_num++]->compute_output({&inter_0_dc}, &inter_0_dc); // ReLU
    }

    layers[layer_num++]->compute_output({&inter_0_dc}, &inter_1_dc); // MaxPool2D
    layers[layer_num++]->compute_output({&inter_1_dc}, &inter_0_dc); // Conv2D
    return inter_0_dc;
    // return inter_1_dc;
}

//****************************************************************************

#define REDUCTION_C(layer_num) layer_params[layer_num][0]
#define GROUP_C(layer_num) layer_params[layer_num][1]
#define GROUPS(layer_num) layer_params[layer_num][2]
#define REDUCTION_HW(layer_num) layer_params[layer_num][3]
#define STRIDE(layer_num) layer_params[layer_num][4]
#define TYPE(layer_num) layer_params[layer_num][9]

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

//timers
double layer_timers[3][15];
double min_layer_timers[3][15];
double avg_layer_timers[3][15];

template <class BufferT>
inline void dscnn_block(
    std::array<uint32_t, 2> const &in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t output_channels,
    uint8_t t_pad,
    uint8_t b_pad,
    uint8_t l_pad,
    uint8_t r_pad,
    BufferT const &I,
    BufferT const &F_1x1,
    BufferT const &F_dw,
    BufferT       &O_intermediate,
    BufferT       &O)
{
    /**/


    uint32_t i_w = in_dims[0];
    uint32_t i_h = in_dims[1];

    small::Conv2D(1, 1, 1,
                  0, 0, 0, 0,
                  output_channels, input_channels,
                  i_h, i_w,
                  I, F_1x1, O_intermediate);
    small::ReLUActivation(output_channels, i_h, i_w, O_intermediate, O_intermediate);

    /**/

    small::DepthwiseConv2D(kernel_size, kernel_size, stride,
                           t_pad, b_pad, l_pad, r_pad,
                           output_channels,
                           i_h, i_w,
                           O_intermediate, F_dw, O);

    uint32_t o_w_p2 = small::output_dim(i_w + l_pad + r_pad,
                                     stride, kernel_size);
    uint32_t o_h_p2 = small::output_dim(i_h + t_pad + b_pad,
                                     stride, kernel_size);

    small::ReLUActivation(output_channels,
                          o_h_p2, o_w_p2,
                          O, O);

}

template <class BufferT>
inline void fused_ewise_dscnn_block(
    std::array<uint32_t, 2> const &in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t output_channels,
    uint8_t t_pad,
    uint8_t b_pad,
    uint8_t l_pad,
    uint8_t r_pad,
    BufferT const &I,
    BufferT const &F_1x1,
    BufferT const &F_dw,
    BufferT &O_intermediate,
    BufferT &O)
{
    /**/

    uint32_t i_w = in_dims[0];
    uint32_t i_h = in_dims[1];

    small::Conv2D_ReLU(1, 1, 1,
                  0, 0, 0, 0,
                  output_channels, input_channels,
                  i_h, i_w,
                  I, F_1x1, O_intermediate);

    small::DepthwiseConv2D_ReLU(kernel_size, kernel_size, stride,
                           t_pad, b_pad, l_pad, r_pad,
                           output_channels,
                           i_h, i_w,
                           O_intermediate, F_dw, O);
}

template <class BufferT>
inline void fused_dscnn_block(
    std::array<uint32_t, 2> const &in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t output_channels,
    uint8_t t_pad,
    uint8_t b_pad,
    uint8_t l_pad,
    uint8_t r_pad,
    BufferT const &I,
    BufferT const &F_1x1,
    BufferT const &F_dw,
    BufferT &O_intermediate,
    BufferT &O)
{
    small::Conv2D_ReLU_DepthwiseConv2D_ReLU( 1, 1, 1, 
                                            0, 0, 0, 0, 
                                            kernel_size, kernel_size, stride,
                           t_pad, b_pad, l_pad, r_pad,
                           output_channels, input_channels,
                         in_dims[0], in_dims[1],
                         I,
                         F_1x1,
                         O_intermediate,
                         F_dw,
                         O);
}

//****************************************************************************
template <class BufferT>
BufferT &
model_inference(uint32_t layer_num_total,
                uint16_t layer_params[30][10],
                std::vector<std::array<uint32_t, 2>> const &intermediate_dims,
                std::vector<BufferT *> const &filter_buf_ptrs,
                BufferT const &input_dc,
                BufferT       &inter_0_dc,
                BufferT       &inter_1_dc)
{
    auto layer_num = 0;
    #if TIME_LAYER
    small::Timer my_timer;

    my_timer.start();
    #endif
    small::Conv2D(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
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
    
    /**/

    // {
    std::array<uint32_t, 2> const &in_dims = intermediate_dims[layer_num];
    uint32_t input_channels = GROUPS(layer_num);
    uint32_t kernel_size = REDUCTION_HW(layer_num);
    uint32_t stride = STRIDE(layer_num);
    uint32_t output_channels = GROUP_C(layer_num + 1);
    uint8_t t_pad = layer_params[layer_num][5];
    uint8_t b_pad = layer_params[layer_num][6];
    uint8_t l_pad = layer_params[layer_num][7];
    uint8_t r_pad = layer_params[layer_num][8];
    BufferT const &I = inter_0_dc;
    BufferT const &F_dw = *filter_buf_ptrs[layer_num];
    BufferT const &F_1x1 = *filter_buf_ptrs[layer_num + 1];
    BufferT       &O_intermediate = inter_1_dc;
    BufferT       &O = inter_0_dc;

    small::DepthwiseConv2D(kernel_size, kernel_size, stride,
                        t_pad, b_pad, l_pad, r_pad,
                        input_channels,
                        in_dims[1], in_dims[0],
                        inter_0_dc, F_dw, inter_1_dc);

    uint32_t o_w = small::output_dim(in_dims[0] + l_pad + r_pad,
                                    stride, kernel_size);
    uint32_t o_h = small::output_dim(in_dims[1] + t_pad + b_pad,
                                    stride, kernel_size);

    small::ReLUActivation(input_channels,
                        o_h, o_w,
                        inter_1_dc, inter_1_dc);

    layer_num++;
    #if TIME_LAYER
    my_timer.stop();
    layer_timers[0][0] = my_timer.elapsed();
    #endif
    auto ds_blocks = 12;
    for (int ds_layer = 0; ds_layer < ds_blocks; ds_layer++)
    {
       #if TIME_LAYER
       my_timer.start();
       #endif
        dscnn_block(
            intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
            REDUCTION_HW(layer_num+1),
            STRIDE(layer_num+1),
            GROUP_C(layer_num),
            PADDING(layer_num+1),
            inter_1_dc,
            *filter_buf_ptrs[layer_num],
            *filter_buf_ptrs[layer_num + 1],
            inter_0_dc,
            inter_1_dc);

        layer_num += 2;
        #if TIME_LAYER
        my_timer.stop();
        layer_timers[0][1+ds_layer] = my_timer.elapsed();
        #endif
    }


    #if TIME_LAYER
    my_timer.start();
    #endif
    small::Conv2D(1, 1, 1,
                    0, 0, 0, 0,
                    GROUP_C(layer_num), REDUCTION_C(layer_num),
                    I_HEIGHT(layer_num), I_WIDTH(layer_num), inter_1_dc, *filter_buf_ptrs[layer_num], inter_0_dc);

    small::ReLUActivation(GROUP_C(layer_num), I_HEIGHT(layer_num), I_WIDTH(layer_num), inter_0_dc, inter_0_dc);
    layer_num++;


    small::MaxPool2D(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                     STRIDE(layer_num), PADDING(layer_num),
                     GROUPS(layer_num),
                     I_HEIGHT(layer_num), I_WIDTH(layer_num),
                        inter_0_dc,
                        inter_1_dc);
    #if TIME_LAYER
    my_timer.stop();
    layer_timers[0][13] = my_timer.elapsed();
    #endif
    #if TIME_LAYER
    my_timer.start();
    #endif
    uint32_t num_classes = 16;  /// @todo get from layer params
    small::Conv2D(1, 1, 1,
                  0, 0, 0, 0,
                  num_classes, 1024,  /// @todo get from layer params
                  1, 1,
                  inter_1_dc,
                  *filter_buf_ptrs[filter_buf_ptrs.size() - 1], // layernum-1?
                  inter_0_dc);
    #if TIME_LAYER
    my_timer.stop();
    layer_timers[0][14] = my_timer.elapsed();
    #endif
    return inter_1_dc;
}

//****************************************************************************
template <class BufferT>
BufferT &
fused_ewise_model_inference(uint32_t layer_num_total,
                uint16_t layer_params[30][10],
                std::vector<std::array<uint32_t, 2>> const &intermediate_dims,
                std::vector<BufferT *> const &filter_buf_ptrs,
                BufferT const &input_dc,
                BufferT &inter_0_dc,
                BufferT &inter_1_dc)
{

    auto layer_num = 0;
    // small::Conv2D(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
    //               STRIDE(layer_num), PADDING(layer_num),
    //               GROUP_C(layer_num), REDUCTION_C(layer_num),
    //               I_HEIGHT(layer_num), I_WIDTH(layer_num),
    //               input_dc,
    //               *filter_buf_ptrs[layer_num],
    //               inter_0_dc);
    #if TIME_LAYER
    small::Timer my_timer;
    my_timer.start();
    #endif
    small::Conv2D_ReLU(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                       STRIDE(layer_num), PADDING(layer_num),
                       GROUP_C(layer_num), REDUCTION_C(layer_num),
                       I_HEIGHT(layer_num), I_WIDTH(layer_num),
                       input_dc,
                       *filter_buf_ptrs[layer_num],
                       inter_0_dc);
    layer_num++;
    // small::ReLUActivation(GROUP_C(0),
    //                       I_HEIGHT(layer_num), I_WIDTH(layer_num),
    //                       inter_0_dc,
    //                       inter_0_dc);
    std::array<uint32_t, 2> const &in_dims = intermediate_dims[layer_num];
    uint32_t input_channels = GROUPS(layer_num);
    uint32_t kernel_size = REDUCTION_HW(layer_num);
    uint32_t stride = STRIDE(layer_num);
    uint32_t output_channels = GROUP_C(layer_num + 1);
    uint8_t t_pad = layer_params[layer_num][5];
    uint8_t b_pad = layer_params[layer_num][6];
    uint8_t l_pad = layer_params[layer_num][7];
    uint8_t r_pad = layer_params[layer_num][8];
    BufferT const &I = inter_0_dc;
    BufferT const &F_dw = *filter_buf_ptrs[layer_num];
    BufferT const &F_1x1 = *filter_buf_ptrs[layer_num + 1];
    BufferT &O_intermediate = inter_1_dc;
    BufferT &O = inter_0_dc;

    small::DepthwiseConv2D_ReLU(kernel_size, kernel_size, stride,
                           t_pad, b_pad, l_pad, r_pad,
                           input_channels,
                           in_dims[1], in_dims[0],
                           inter_0_dc, F_dw, inter_1_dc);
    #if TIME_LAYER
    my_timer.stop();
    layer_timers[1][0] = my_timer.elapsed();
    #endif
    uint32_t o_w = small::output_dim(in_dims[0] + l_pad + r_pad,
                                     stride, kernel_size);
    uint32_t o_h = small::output_dim(in_dims[1] + t_pad + b_pad,
                                     stride, kernel_size);

    // small::ReLUActivation(input_channels,
    //                       o_h, o_w,
    //                       inter_1_dc, inter_1_dc);

    layer_num++;

    auto ds_blocks = 12;
    for (int ds_layer = 0; ds_layer < ds_blocks; ds_layer++)
    {
        #if TIME_LAYER
        my_timer.start();
        #endif
        fused_ewise_dscnn_block(
            intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
            REDUCTION_HW(layer_num + 1),
            STRIDE(layer_num + 1),
            GROUP_C(layer_num),
            PADDING(layer_num + 1),
            inter_1_dc,
            *filter_buf_ptrs[layer_num],
            *filter_buf_ptrs[layer_num + 1],
            inter_0_dc,
            inter_1_dc);
        #if TIME_LAYER
        my_timer.stop();
        layer_timers[1][1+ds_layer] = my_timer.elapsed();
            #endif
        layer_num += 2
        ;
    }

    #if TIME_LAYER
    my_timer.start();
    #endif
    small::Conv2D_ReLU(1, 1, 1,
                  0, 0, 0, 0,
                  GROUP_C(layer_num), REDUCTION_C(layer_num),
                  I_HEIGHT(layer_num), I_WIDTH(layer_num), inter_1_dc, *filter_buf_ptrs[layer_num], inter_0_dc);

    // small::ReLUActivation(GROUP_C(layer_num), I_HEIGHT(layer_num), I_WIDTH(layer_num), inter_0_dc, inter_0_dc);
    layer_num++;

    small::MaxPool2D(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                     STRIDE(layer_num), PADDING(layer_num),
                     GROUPS(layer_num),
                     I_HEIGHT(layer_num), I_WIDTH(layer_num),
                     inter_0_dc,
                     inter_1_dc);
    #if TIME_LAYER
    my_timer.stop();
    layer_timers[1][13] = my_timer.elapsed();
    #endif
    #if TIME_LAYER
    my_timer.start();
    #endif
    uint32_t num_classes = 16; /// @todo get from layer params
    small::Conv2D(1, 1, 1,
                  0, 0, 0, 0,
                  num_classes, 1024, /// @todo get from layer params
                  1, 1,
                  inter_1_dc,
                  *filter_buf_ptrs[filter_buf_ptrs.size() - 1], // layernum-1?
                  inter_0_dc);
    #if TIME_LAYER
    my_timer.stop();
    layer_timers[1][14] = my_timer.elapsed();
    #endif
    return inter_0_dc;
    // return inter_1_dc;
}

//****************************************************************************
template <class BufferT>
BufferT &
fused_model_inference(uint32_t layer_num_total,
                uint16_t layer_params[30][10],
                std::vector<std::array<uint32_t, 2>> const &intermediate_dims,
                std::vector<BufferT *> const &filter_buf_ptrs,
                BufferT const &input_dc,
                BufferT &inter_0_dc,
                BufferT &inter_1_dc,
                BufferT &inter_0_buffer_dc)
{
    

    auto layer_num = 0;
    #if TIME_LAYER
    small::Timer my_timer;
    my_timer.start();
    #endif
    // small::Conv2D_ReLU(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
    //                                         STRIDE(layer_num), PADDING(layer_num),
    //                                         GROUP_C(layer_num), REDUCTION_C(layer_num),
    //                                         I_HEIGHT(layer_num), I_WIDTH(layer_num),
    //                                         input_dc,
    //                                         *filter_buf_ptrs[layer_num],
    //                                         inter_0_dc);

    small::Conv2D_ReLU_DepthwiseConv2D_ReLU(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num), STRIDE(layer_num),
                                            PADDING(layer_num),
                                            REDUCTION_HW(layer_num + 1), REDUCTION_HW(layer_num + 1), STRIDE(layer_num + 1),
                                            PADDING(layer_num+1),
                                            GROUP_C(layer_num), REDUCTION_C(layer_num),
                                            I_HEIGHT(layer_num), I_WIDTH(layer_num),
                                            input_dc,
                                            *filter_buf_ptrs[layer_num],
                                            inter_0_buffer_dc,
                                            *filter_buf_ptrs[layer_num+1],
                                            inter_1_dc);

    layer_num++;

    /*

        std::array<uint32_t, 2> const &in_dims = intermediate_dims[layer_num];
        uint32_t input_channels = GROUPS(layer_num);
        uint32_t kernel_size = REDUCTION_HW(layer_num);
        uint32_t stride = STRIDE(layer_num);
        uint32_t output_channels = GROUP_C(layer_num + 1);
        uint8_t t_pad = layer_params[layer_num][5];
        uint8_t b_pad = layer_params[layer_num][6];
        uint8_t l_pad = layer_params[layer_num][7];
        uint8_t r_pad = layer_params[layer_num][8];
        BufferT const &I = inter_0_dc;
        BufferT const &F_dw = *filter_buf_ptrs[layer_num];
        // BufferT const &F_1x1 = *filter_buf_ptrs[layer_num + 1];
        BufferT &O_intermediate = inter_1_dc;
        BufferT &O = inter_0_dc;

        small::DepthwiseConv2D_ReLU(kernel_size, kernel_size, stride,
                               t_pad, b_pad, l_pad, r_pad,
                               input_channels,
                               in_dims[1], in_dims[0],
                               inter_0_dc, F_dw, inter_1_dc);
                               */

        layer_num++;
        #if TIME_LAYER
        my_timer.stop();
        layer_timers[2][0] = my_timer.elapsed();
        #endif
    /**/



    auto ds_blocks = 12;
    // printf("1:%d 0:%d\n", inter_1_dc.data(), inter_0_dc.data());

    for (int ds_layer = 0; ds_layer < ds_blocks; ds_layer++)
    {
        BufferT &I = inter_1_dc;
        BufferT &O_intermediate = inter_0_buffer_dc;
        BufferT &O = inter_0_dc;
        // printf("layer_num %d \n", layer_num);
        #if TIME_LAYER
        my_timer.start();
        #endif
        fused_dscnn_block(
            intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
            REDUCTION_HW(layer_num+1),
            STRIDE(layer_num+1),
            GROUP_C(layer_num),
            PADDING(layer_num+1),
            I,
            *filter_buf_ptrs[layer_num],
            *filter_buf_ptrs[layer_num + 1],
            O_intermediate,
            O);

        #if TIME_LAYER
        my_timer.stop();
        layer_timers[2][1+ds_layer] = my_timer.elapsed();
        #endif

        layer_num += 2;

        
        inter_1_dc = inter_0_dc;

        // printf("1:%d 0:%d\n", inter_1_dc.data(), inter_0_dc.data());
    }

    /**/

    // printf("calling conv 1x1 %d %d \n", layer_num, layer_num_total);

    // in_dims = intermediate_dims[layer_num];
    // input_channels = REDUCTION_C(layer_num);
    // output_channels = GROUP_C(layer_num);

    BufferT const &F_1x1 = *filter_buf_ptrs[layer_num];

#if TIME_LAYER
    my_timer.start();
#endif
    small::Conv2D_ReLU(1, 1, 1,
                       0, 0, 0, 0,
                       GROUP_C(layer_num), REDUCTION_C(layer_num),
                       intermediate_dims[layer_num][0], intermediate_dims[layer_num][1],
                       inter_1_dc, F_1x1, inter_0_dc);
    // small::ReLUActivation(output_channels, o_h, o_w, O, O);
    layer_num++;

    // /**/

    // printf("calling pool %d %d \n", layer_num, layer_num_total);
    small::MaxPool2D(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                     STRIDE(layer_num), PADDING(layer_num),
                     GROUPS(layer_num),
                     I_HEIGHT(layer_num), I_WIDTH(layer_num),
                     inter_0_dc,
                     inter_1_dc);

    // small::Conv2D_ReLU_Maxpool2D(1, 1, 1,
    //                    0, 0, 0, 0,
    //                    REDUCTION_HW(layer_num + 1), REDUCTION_HW(layer_num + 1), STRIDE(layer_num + 1),
    //                    PADDING(layer_num + 1),
    //                    output_channels, input_channels,
    //                    intermediate_dims[layer_num][0], intermediate_dims[layer_num][1],
    //                    inter_1_dc, F_1x1, inter_0_buffer_dc, inter_0_dc);
    // inter_1_dc = inter_0_dc;
    // // small::ReLUActivation(output_channels, o_h, o_w, O, O);

    // // printf("calling pool %d %d \n", layer_num, layer_num_total);
    // small::MaxPool2D(
    //                  GROUPS(layer_num),
    //                  I_HEIGHT(layer_num), I_WIDTH(layer_num),
    //                  inter_0_dc,
    //                  inter_1_dc);
    #if TIME_LAYER
    my_timer.stop();
    layer_timers[2][13] = my_timer.elapsed();
    #endif

    layer_num++;
    #if TIME_LAYER
    my_timer.start();
    #endif
    uint32_t num_classes = 16; /// @todo get from layer params
    small::Conv2D(1, 1, 1,
                  0, 0, 0, 0,
                  num_classes, 1024, /// @todo get from layer params
                  1, 1,
                  inter_1_dc,
                  *filter_buf_ptrs[filter_buf_ptrs.size() - 1], // layernum-1?
                  inter_0_dc);
    #if TIME_LAYER
    my_timer.stop();
    layer_timers[2][14] = my_timer.elapsed();
    #endif

    return inter_0_dc;
}

//****************************************************************************
//****************************************************************************
template <class BufferT>
void inference()
{
    uint32_t C_i = 3;
    uint32_t N = 96;
    uint32_t M = 96;
    uint32_t num_classes = 16;  // Must be a multiple of 16

    if (num_classes % 16 != 0)
    {
        throw std::invalid_argument(
            "Number of output classes must be a multiple of 16.");
    }

    //Create input tensor
    uint32_t input_dimensions = C_i * N * M;
    BufferT input_dc(input_dimensions);
    init(input_dc, input_dimensions);

    // ================================================

    uint16_t layer_params[30][10] = {1};
    std::vector<std::array<uint32_t, 2>> intermediate_dims;

    uint8_t t_pad, b_pad, r_pad, l_pad;

    // Set up model parameters
    int layer_num = 0;
    uint32_t max_numel_inter_0 = 0, max_numel_inter_1 = 0;

    intermediate_dims.push_back(std::array<uint, 2>({N, M}));

    // conv
    REDUCTION_C(layer_num) = C_i;
    GROUP_C(layer_num) = 32;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 3;
    STRIDE(layer_num) = 2;
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                        STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                        STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

    layer_num++; // 1
    intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));

    auto inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_0 =
        (inter_dim > max_numel_inter_0)? inter_dim : max_numel_inter_0;

    // common set up for model architecture
    auto ds_blocks = 13;

    const int layer_strides[] = {1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1};
    // dwise 1
    for (int ds_layer = 0; ds_layer < ds_blocks; ds_layer++)
    {
        int channel_multiplier = 2;
        if (ds_layer >= 2)
        {
            channel_multiplier = layer_strides[ds_layer];
        }

        REDUCTION_C(layer_num) = 1; // input channels
        GROUP_C(layer_num) = 1;
        GROUPS(layer_num) = GROUP_C(layer_num - 1);  // output channels
        REDUCTION_HW(layer_num) = 3;                 // kernel size
        STRIDE(layer_num) = layer_strides[ds_layer]; // stride
        small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                            STRIDE(layer_num), t_pad, b_pad);
        small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                            STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

        layer_num++; // 2
        intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));

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
    }
    // pooling dims
    REDUCTION_C(layer_num) = 1;
    GROUP_C(layer_num) = 1;
    GROUPS(layer_num) = GROUP_C(layer_num - 1);
    REDUCTION_HW(layer_num) = I_HEIGHT(layer_num);
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0);

    layer_num++;

    auto layer_num_total = layer_num - 1;  /// @todo is this (-1) right?

#if SUMMARY == 1
    printf("Layer num total: %d", layer_num_total);
    for (auto i = 0; i < layer_num_total; i++)
    {
        printf("layer %d: ", i);
        printf(" input_dims: %d %d ", I_HEIGHT(i), I_WIDTH(i));
        for (auto j = 0; j < 10; j++)
        {
            printf("%d, ", layer_params[i][j]);
        }
        printf("\b\b");
        // printf("input dims: %d %d ", I_HEIGHT(i+1), I_WIDTH(i+1));
        printf("\n");
    }
#endif

    //  Copy layer weights to temporaries
    std::vector<BufferT *> filter_buf_ptrs;

    for (auto l = 0; l < layer_num_total; l++)
    {
        uint32_t filter_dimensions =
            REDUCTION_HW(l) * REDUCTION_HW(l) * REDUCTION_C(l) *
            GROUP_C(l) * GROUPS(l);

        BufferT *filter_buf_ptr =
            new BufferT(filter_dimensions);
        init(*filter_buf_ptr, filter_dimensions);
        filter_buf_ptrs.push_back(filter_buf_ptr);
    }
    uint32_t filter_dimensions = GROUP_C(layer_num_total-1) * num_classes;
    BufferT *filter_fc_dc_ptr =
        new BufferT(filter_dimensions);
    init(*filter_fc_dc_ptr, filter_dimensions);
    filter_buf_ptrs.push_back(filter_fc_dc_ptr);

    BufferT inter_0_dc(max_numel_inter_1);
    BufferT inter_1_dc(max_numel_inter_1);
    BufferT inter_0_buffer_dc(147456*4);


    //======================================================
    small::Timer my_timer;

    std::cerr << "Warm up run (ORIG)" << std::endl;
    my_timer.start();
    auto &output_dc =
        fused_model_inference(layer_num_total, layer_params, intermediate_dims,
                        filter_buf_ptrs,
                        input_dc,
                        inter_0_dc,
                        inter_1_dc, inter_0_buffer_dc);
    my_timer.stop();
    printf("\nElapsed time: %lf ns.\n", my_timer.elapsed());

    //======================================================

    auto layers(create_model<BufferT>(filter_buf_ptrs));

    small::Tensor<BufferT> input_tensor({1, 3, 96, 96}, input_dc);
#if defined(QUANTIZED)
    small::Tensor<small::QUInt8Buffer> inter_0_tensor(max_numel_inter_0*2);
    small::Tensor<small::QUInt8Buffer> inter_1_tensor(max_numel_inter_1);
#else
    small::Tensor<small::FloatBuffer> inter_0_tensor(max_numel_inter_0);
    small::Tensor<small::FloatBuffer> inter_1_tensor(max_numel_inter_1);
#endif

    std::cerr << "Warm up run (LAYERS)" << std::endl;
    my_timer.start();
    auto &output_tensor =
        model_inference(layers, input_tensor, inter_0_tensor, inter_1_tensor);
    my_timer.stop();
    printf("\nElapsed time: %lf ns.\n", my_timer.elapsed());

    // Compare the results
    size_t num_outputs = layers.back()->output_size();
    std::cout << "\nCHECK RESULTS: Num output elements: " << num_outputs << std::endl;
    bool check =1;
    for (size_t ix = 0; ix < num_outputs; ++ix)
    {
        bool cur_check = output_dc[ix] == output_tensor.buffer()[ix];
        std::cout << "Current, new " << ix
                  << ": " << (float)output_dc[ix]
                  << ", " << (float)output_tensor.buffer()[ix]
                  << ((cur_check) ? " (pass)" : " (fail)")
                  << std::endl;
        check &= cur_check;
    }
    assert(check);

    // clean up model (move to model class destructor when built
    for (auto layer : layers) delete layer;
    //======================================================

    double min_small = std::numeric_limits<double>::max();
    std::vector<double> small_timing;
    std::cout << "Performing timing "<< RUNS <<" runs unfused ...\n";
    for (int r = 0; r < RUNS; r++)
    {
        my_timer.start();

        for (int i = 0; i < 100; i++)
        {
            // auto &output_dc =
            model_inference(layer_num_total, layer_params, intermediate_dims,
                            filter_buf_ptrs,
                            input_dc, inter_0_dc, inter_1_dc);
        }
        my_timer.stop();
        auto diff = my_timer.elapsed() / 100;
        min_small = std::min<double>(min_small, diff);
        small_timing.push_back(diff);


    #if TIME_LAYER
        int impl = 0;

        for (int i = 0; i < 100; i++)
        {
            model_inference(layer_num_total, layer_params, intermediate_dims,
                                        filter_buf_ptrs,
                                        input_dc, inter_0_dc, inter_1_dc);
            for (int timer = 0; timer < 15; timer++)
            {
                if (0 < i)
                {
                    avg_layer_timers[impl][timer] += layer_timers[impl][timer];
                }
                else
                {
                    avg_layer_timers[impl][timer] = layer_timers[impl][timer];
                }
            }
        }
        for (int timer = 0; timer < 15; timer++)
        {
            avg_layer_timers[impl][timer] /= 100;
            if (0 < r)
            {
                min_layer_timers[impl][timer] = std::min<double>(min_layer_timers[impl][timer], avg_layer_timers[impl][timer]);
            }
            else
            {
                min_layer_timers[impl][timer] = avg_layer_timers[impl][timer];
            }
        }
    #endif
    }

    std::cout << "Minimum time: " << min_small << " ns.\n";

    double min_small_fused_ewise = std::numeric_limits<double>::max();
    // std::vector<double> small_fused_ewise_timing;
    std::cout << "Performing timing runs ewise layers fused...\n";
    for (int r = 0; r < RUNS; r++)
    {
        my_timer.start();
        for(int i = 0; i < 100; i++)
        {
        fused_ewise_model_inference(layer_num_total, layer_params, intermediate_dims,
                              filter_buf_ptrs,
                              input_dc, inter_0_dc, inter_1_dc);
        }

        my_timer.stop();
        auto diff = (my_timer.elapsed())/100;
        min_small_fused_ewise = std::min<double>(min_small_fused_ewise, diff);

        #if TIME_LAYER
        int impl = 1;

        for (int i = 0; i < 100; i++)
        {
            fused_ewise_model_inference(layer_num_total, layer_params, intermediate_dims,
                                  filter_buf_ptrs,
                                  input_dc, inter_0_dc, inter_1_dc);
            for (int timer = 0; timer < 15; timer++)
            {
                if (0 < i)
                {
                    avg_layer_timers[impl][timer] += layer_timers[impl][timer];
                }
                else
                {
                    avg_layer_timers[impl][timer] = layer_timers[impl][timer];
                }
            }
        }
        for (int timer = 0; timer < 15; timer++)
        {
            avg_layer_timers[impl][timer] /= 100;
            if (0 < r)
            {
                min_layer_timers[impl][timer] = std::min<double>(min_layer_timers[impl][timer], avg_layer_timers[impl][timer]);
            }
            else
            {
                min_layer_timers[impl][timer] = avg_layer_timers[impl][timer];
            }
        }
        #endif
        // small_fused_ewise_timing.push_back(diff);
    }

    std::cout << "Minimum time: " << min_small_fused_ewise << " ns.\n";

    double min_small_fused = std::numeric_limits<double>::max();
    // std::vector<double> small_fused_timing;
    std::cout << "Performing timing runs fused block...\n";
    for (int r = 0; r < RUNS; r++)
    {
        my_timer.start();
        for(int i = 0; i < 100; i++)
        {
        fused_model_inference(layer_num_total, layer_params, intermediate_dims,
                        filter_buf_ptrs,
                        input_dc, inter_0_dc, inter_1_dc, inter_0_buffer_dc);

        }
        my_timer.stop();
        auto diff = my_timer.elapsed()/100;
        min_small_fused = std::min<double>(min_small_fused, diff);

        #if TIME_LAYER
        int impl = 2;
        for (int i = 0; i < 100; i++) {
            fused_model_inference(layer_num_total, layer_params, intermediate_dims,
                                  filter_buf_ptrs,
                                  input_dc, inter_0_dc, inter_1_dc, inter_0_buffer_dc);
            for (int timer = 0; timer < 15; timer++){
                if (0 < i) {
                    avg_layer_timers[impl][timer] += layer_timers[impl][timer];
                } else {
                    avg_layer_timers[impl][timer] = layer_timers[impl][timer];
                }
            }
        }
        for (int timer = 0; timer < 15; timer++)
        {
            avg_layer_timers[impl][timer] /= 100;
            if (0 < r) {
                min_layer_timers[impl][timer] = std::min<double>(min_layer_timers[impl][timer], avg_layer_timers[impl][timer]);
            } else {
                min_layer_timers[impl][timer] = avg_layer_timers[impl][timer];

            }
        }
        #endif
        // small_fused_timing.push_back(diff);
    }

    std::cout << "Minimum time: " << min_small_fused << " ns.\n";


    #if TIME_LAYER
    double sum_unfused = 0, sum_ewise = 0, sum_fused = 0;
    for(int layer = 0 ; layer < 15; layer++)
    {
        printf("%d , ",layer);
        for(int impl = 0; impl < 3; impl++)
        {
            printf("%f ,", min_layer_timers[impl][layer]);
        }
        printf("\n");
        sum_unfused += min_layer_timers[0][layer];
        sum_ewise   += min_layer_timers[1][layer];
        sum_fused   += min_layer_timers[2][layer];
    }

    printf("sum , %f, %f, %f \n", sum_unfused, sum_ewise, sum_fused);
    #endif

    printf("sum , %f, %f, %f \n", min_small, min_small_fused_ewise, min_small_fused);


    int num_th = 1;
#if PARALLEL == 1
    char const *env_nt(std::getenv("OMP_NUM_THREADS"));
    if (nullptr != env_nt)
    {
        num_th = atoi(std::getenv("OMP_NUM_THREADS"));
    }
#endif
    std::cout << "Num Threads: " << num_th << std::endl;
#if PARALLEL_DIST == ELEMENTAL
    printf("ELEMENTAL\n");
#else
    printf("BLOCK\n");
#endif
    // print_stats(small_fused_timing, "\nSMaLL:mobilenet");

    printf("deallocing %ld filters\n", filter_buf_ptrs.size());
    for (size_t l = 0; l < filter_buf_ptrs.size(); l++)
    {
        small::free_buffer(filter_buf_ptrs[l]);
    }

#if defined(NANO33BLE)
    small::detail::free_all();
#endif
}

//****************************************************************************
// For non-arduino platforms.  ... move to driver.cpp?
//****************************************************************************
#ifndef NANO33BLE
int main()
{
#if defined(QUANTIZED)
    inference<small::QUInt8Buffer>();
#else
    inference<small::FloatBuffer>();
#endif

    return 0;
}
#endif
