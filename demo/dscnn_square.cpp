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
#define RUNS 100
#endif

//****************************************************************************
/* This is the runtime recording:

   Conv2D(k:3,s:2,pad:[1,1,0,1],ochans:64,ichans:3,img:49x10,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)

   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:64,ichans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)

   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:64,ichans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)

   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:64,ichans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)

   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:64,ichans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)

   MaxPool2D_rect(k:25x5,s:1,pad:[0,0,0,0],chans:64,img:25x5,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:16,ichans:64,img:1x1,I,F,O)
 */

#include<small/Layer.hpp>
#include<small/DepthwiseConv2DLayer.hpp>
#include<small/Conv2DLayer.hpp>
#include<small/MaxPool2DLayer.hpp>
#include<small/ReLULayer.hpp>

template <class BufferT>
std::vector<small::Layer<BufferT>*> create_model(
    std::vector<BufferT*> const &filters)
{
    std::vector<small::Layer<BufferT>*> layers;

    // settings for first layer
    uint32_t kernel_size = 3;
    uint32_t stride = 2;
    uint32_t input_height = 49, input_width = 10;
    uint32_t input_channels = 3, output_channels = 64;
    size_t   filter_num = 0;

    small::shape_type input_shape(
        {1UL, input_channels, input_height, input_width});

    small::Layer<BufferT> *prev =
        new small::Conv2DLayer<BufferT>(input_shape,
                                        kernel_size, kernel_size,
                                        stride, small::PADDING_F,
                                        output_channels,
                                        *filters[filter_num], true);
    layers.push_back(prev);

    input_height = 25;
    input_width  = 5;

    prev = new small::ReLULayer<BufferT>(prev->output_shape());
    layers.push_back(prev);

    stride = 1;
    uint32_t num_channels = 64;

    // Four blocks of dw/relu/conv/relu
    for (auto ix = 0U; ix < 4; ++ix)
    {
        ++filter_num;
        kernel_size = 3;
        prev = new small::DepthwiseConv2DLayer<BufferT>(
            prev->output_shape(),
            kernel_size, kernel_size, stride, small::PADDING_F,
            *filters[filter_num], true);
        layers.push_back(prev);

        prev = new small::ReLULayer<BufferT>(prev->output_shape());
        layers.push_back(prev);

        ++filter_num;
        kernel_size = 1;
        prev = new small::Conv2DLayer<BufferT>(prev->output_shape(),
                                               kernel_size, kernel_size,
                                               stride, small::PADDING_V,
                                               num_channels,
                                               *filters[filter_num], true);
        layers.push_back(prev);

        prev = new small::ReLULayer<BufferT>(prev->output_shape());
        layers.push_back(prev);
    }

    prev = new small::MaxPool2DLayer<BufferT>(prev->output_shape(),
                                              input_height, input_width,
                                              stride, small::PADDING_V);
    layers.push_back(prev);

    ++filter_num;
    kernel_size = 1;
    input_channels = 64;
    output_channels = 16;

    input_height = 1;
    input_width  = 1;
    prev = new small::Conv2DLayer<BufferT>(prev->output_shape(),
                                           kernel_size, kernel_size,
                                           stride, small::PADDING_V,
                                           output_channels,
                                           *filters[filter_num], true);
    layers.push_back(prev);
    return layers;
}

//****************************************************************************
template <class BufferT>
small::Tensor<BufferT> &model_inference(
    std::vector<small::Layer<BufferT>*> const &layers,
    small::Tensor<BufferT>                             const &input_dc,
    small::Tensor<BufferT>                                   &inter_0_dc,
    small::Tensor<BufferT>                                   &inter_1_dc)
{
    size_t layer_num = 0;
    layers[layer_num++]->compute_output({&input_dc}, &inter_0_dc);   // Conv2D
    layers[layer_num++]->compute_output({&inter_0_dc}, &inter_0_dc); // ReLU

    for (auto ix = 0U; ix < 4; ++ix)
    {
        layers[layer_num++]->compute_output({&inter_0_dc}, &inter_1_dc); // DWConv
        layers[layer_num++]->compute_output({&inter_1_dc}, &inter_1_dc); // ReLU
        layers[layer_num++]->compute_output({&inter_1_dc}, &inter_0_dc); // Conv2D
        layers[layer_num++]->compute_output({&inter_0_dc}, &inter_0_dc); // ReLU
    }

    layers[layer_num++]->compute_output({&inter_0_dc}, &inter_1_dc); // MaxPool2D
    layers[layer_num++]->compute_output({&inter_1_dc}, &inter_0_dc); // Conv2D
    return inter_0_dc;
}

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
template <class BufferT>
inline void dscnn_block(
    uint32_t in_dims[2], uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t output_channels,
    uint8_t t_pad,
    uint8_t b_pad,
    uint8_t l_pad,
    uint8_t r_pad,
    BufferT const &I,
    BufferT const &F_dw,
    BufferT const &F_1x1,
    BufferT       &O_intermediate,
    BufferT       &O)
{
    small::DepthwiseConv2D(kernel_size, kernel_size, stride,
                           t_pad, b_pad, l_pad, r_pad,
                           input_channels,
                           in_dims[1], in_dims[0],
                           I, F_dw, O_intermediate);

    uint32_t o_w = small::output_dim(in_dims[0] + l_pad + r_pad,
                                     stride, kernel_size);
    uint32_t o_h = small::output_dim(in_dims[1] + t_pad + b_pad,
                                     stride, kernel_size);

    small::ReLUActivation(input_channels,
                          o_h, o_w,
                          O_intermediate, O_intermediate);
    small::Conv2D(1, 1, 1,
                  0, 0, 0, 0,
                  output_channels, input_channels,
                  o_h, o_w,
                  O_intermediate, F_1x1, O);
    small::ReLUActivation(output_channels, o_h, o_w, O, O);
}

//****************************************************************************
template <class BufferT>
BufferT &
model_inference(uint32_t layer_num_total,
                uint16_t layer_params[30][10],
                uint32_t intermediate_dims[30][2],
                std::vector<BufferT *> const &filter_buf_ptrs,
                BufferT const &input_dc,
                BufferT       &inter_0_dc,
                BufferT       &inter_1_dc)
{
    auto layer_num = 0;
    int num_filters = layer_num_total - 1;
    small::Conv2D(
        REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
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

    /// @todo WARNING quantized version has "layer_num = layer_num_total - 2;"
    small::MaxPool2D(REDUCTION_H(layer_num), REDUCTION_W(layer_num),
                     STRIDE(layer_num), PADDING(layer_num),
                     GROUPS(layer_num),
                     I_HEIGHT(layer_num), I_WIDTH(layer_num),
                     inter_0_dc,
                     inter_1_dc);

    layer_num++;
    small::Conv2D(1, 1, 1,
                  0, 0, 0, 0,
                  GROUP_C(layer_num), REDUCTION_C(layer_num),
                  1, 1,
                  inter_1_dc,
                  *filter_buf_ptrs[num_filters - 1],
                  inter_0_dc);

    return inter_0_dc;
}

//****************************************************************************
//****************************************************************************
template <class BufferT>
void inference()
{
    uint32_t C_i = 3;
    uint32_t N = 49;  // height
    uint32_t M = 10;  // width
    uint32_t num_classes = 16;  // must be a multiple of 16

    // Create input tensor
    uint32_t input_dimensions = C_i * N * M;
    BufferT input_dc(input_dimensions);
    init(input_dc, input_dimensions);

    // ================================================

    uint16_t layer_params[30][10] = {1};
    uint32_t intermediate_dims[30][2];

    uint8_t t_pad, b_pad, r_pad, l_pad;

    // Set up model parameters
    int layer_num = 0;
    uint32_t max_numel_inter_0 = 0, max_numel_inter_1 = 0;

    intermediate_dims[layer_num][0] = M;  // width
    intermediate_dims[layer_num][1] = N;  // height

    // conv
    REDUCTION_C(layer_num) = C_i;
    GROUP_C(layer_num) = 64;
    GROUPS(layer_num) = 1;
    REDUCTION_H(layer_num) = 3;
    REDUCTION_W(layer_num) = 3;
    STRIDE(layer_num) = 2;
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_H(layer_num),
                        STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num),  REDUCTION_W(layer_num),
                        STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

    layer_num++; // 1
    intermediate_dims[layer_num][0] = 5;   // width
    intermediate_dims[layer_num][1] = 25;  // height

    auto inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_0 =
        (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

    // common set up for model architecture
    auto ds_blocks = 4;
    const int layer_strides[] = {1, 1, 1, 1};
    // dwise 1
    for (int ds_layer = 0; ds_layer < ds_blocks; ds_layer++)
    {
        int channel_multiplier = layer_strides[ds_layer];

        REDUCTION_C(layer_num) = 1; // input channels
        GROUP_C(layer_num) = 1;
        GROUPS(layer_num) = GROUP_C(layer_num - 1);  // output channels
        REDUCTION_H(layer_num) = 3;                  // kernel size
        REDUCTION_W(layer_num) = 3;
        STRIDE(layer_num) = layer_strides[ds_layer]; // stride
        small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                            STRIDE(layer_num), t_pad, b_pad);
        small::calc_padding(I_WIDTH(layer_num), REDUCTION_HW(layer_num),
                            STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

        layer_num++; // 2
        intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
        intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);

        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_1 =
            (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;
        REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
        GROUP_C(layer_num) = (GROUPS(layer_num - 1)) * channel_multiplier;
        GROUPS(layer_num) = 1;
        REDUCTION_H(layer_num) = 1;
        REDUCTION_W(layer_num) = 1;
        STRIDE(layer_num) = 1;
        SET_PADDING(layer_num, 0, 0, 0, 0);

        layer_num++; // 3
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_0 =
            (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;
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
    SET_PADDING(layer_num, 0, 0, 0, 0);

    layer_num++;
    intermediate_dims[layer_num][0] = 1;
    intermediate_dims[layer_num][1] = 1;
    REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
    GROUP_C(layer_num) = num_classes;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 1;
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0);
    layer_num++;

    size_t layer_num_total = layer_num;
    size_t num_filters = layer_num_total - 1;

#if SUMMARY == 1
    printf("Layer num total: %d\n", layer_num_total);
    for (auto i = 0; i < layer_num_total; i++)
    {
        printf("layer %d: ", i);
        printf(" input_dims: %d %d ", I_HEIGHT(i), I_WIDTH(i));
        for (auto j = 0; j < 10; j++)
        {
            printf("%d, ", layer_params[i][j]);
        }
        printf(", intermediate dims: %d %d\n",
               intermediate_dims[i][0], intermediate_dims[i][1]);
    }
#endif

    //  Copy layer weights to temporaries
    std::vector<BufferT *> filter_buf_ptrs;

    //std::cout << "num filters = " << num_filters-1 << std::endl;
    for (size_t l = 0; l < num_filters - 1; l++)  // was layer_num_total
    {
        uint32_t filter_dimensions =
            REDUCTION_H(l) * REDUCTION_W(l) * REDUCTION_C(l) *
            GROUP_C(l) * GROUPS(l);

        BufferT *filter_buf_ptr =
            new BufferT(filter_dimensions);
        init(*filter_buf_ptr, filter_dimensions);
        filter_buf_ptrs.push_back(filter_buf_ptr);
    }

    //uint32_t filter_dimensions = GROUP_C(layer_num_total) * num_classes;
    uint32_t filter_dimensions =
        REDUCTION_C(layer_num_total - 1) * GROUP_C(layer_num_total - 1);
    BufferT *filter_fc_dc_ptr =
        new BufferT(filter_dimensions);
    init(*filter_fc_dc_ptr, filter_dimensions);
    filter_buf_ptrs.push_back(filter_fc_dc_ptr);

    // allocate space for intermediate outputs
    // (use the max sizes calculated previously)
#if defined(QUANTIZED)
    BufferT inter_0_dc(max_numel_inter_0*4);
    BufferT inter_1_dc(max_numel_inter_1*4);
#else
    BufferT inter_0_dc(max_numel_inter_0);
    BufferT inter_1_dc(max_numel_inter_1);
#endif

    //======================================================
    small::Timer my_timer;

    std::cerr << "Warm up run (ORIG)" << std::endl;
    my_timer.start();
    auto &output_dc =
        model_inference(layer_num_total, layer_params, intermediate_dims,
                        filter_buf_ptrs,
                        input_dc,
                        inter_0_dc,
                        inter_1_dc);
    my_timer.stop();
    printf("\nElapsed time: %lf ns.\n", my_timer.elapsed());

    //======================================================

    double min_small = std::numeric_limits<double>::max();
    std::vector<double> small_timing;

    std::cout << "Performing timing runs...\n";
    for (int r = 0; r < RUNS; r++)
    {
        my_timer.start();

        //auto &output_dc =
            model_inference(layer_num_total, layer_params, intermediate_dims,
                            filter_buf_ptrs,
                            input_dc,
                            inter_0_dc,
                            inter_1_dc);

        my_timer.stop();
        auto diff = my_timer.elapsed();
        min_small = std::min<double>(min_small, diff);
        small_timing.push_back(diff);
    }

    std::cout << "Minimum time: " << min_small << " ns.\n";

    int num_th = 1;
#if PARALLEL == 1
    char const *env_nt(std::getenv("OMP_NUM_THREADS"));
    if (nullptr != env_nt)
    {
        num_th = atoi(std::getenv("OMP_NUM_THREADS"));
    }
#endif
    std::cout << "Num Threads: " << num_th << std::endl;
    print_stats(small_timing, "SMaLL:dscnn_square");

    //======================================================
    auto layers(create_model<BufferT>(filter_buf_ptrs));

    small::Tensor<BufferT> input_tensor({1, 3, 49, 10}, input_dc); // B, C_i, N, M
#if defined(QUANTIZED)
    small::Tensor<BufferT> inter_0_tensor(max_numel_inter_0*4);
    small::Tensor<BufferT> inter_1_tensor(max_numel_inter_1*4);
#else
    small::Tensor<BufferT> inter_0_tensor(max_numel_inter_0);
    small::Tensor<BufferT> inter_1_tensor(max_numel_inter_1);
    std::cout << "buffer sizes: " << max_numel_inter_0 << ", "
              << max_numel_inter_1 << std::endl;
#endif

    std::cerr << "Warm up run (LAYERS)" << std::endl;
    my_timer.start();
    auto &output_tensor =
        model_inference(layers, input_tensor,
                        inter_0_tensor, inter_1_tensor);
    my_timer.stop();
    printf("\nElapsed time: %lf ns.\n", my_timer.elapsed());

    // Compare the results
    size_t num_outputs = layers.back()->output_size();
    std::cout << "\nCHECK RESULTS: Num output elements: " << num_outputs
              << std::endl;
    for (size_t ix = 0; ix < num_outputs; ++ix)
    {
        std::cout << "Current, new " << ix
                  << ": " << (float)output_dc[ix]
                  << ", " << (float)output_tensor.buffer()[ix]
                  << ((output_dc[ix] == output_tensor.buffer()[ix]) ? " (pass)" : " (fail)")
                  << std::endl;
    }

    // clean up model (move to model class destructor when built
    for (auto layer : layers) delete layer;
    //======================================================

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
