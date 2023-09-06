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

//#define DAG_DEBUG
//#define BUFFER_DEBUG
//#define DEBUG_LAYERS
//#define SUMMARY 1

#include <acutest.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>

#include <small.h>
#include <small/models/TinyYoloV2.hpp>
#include <small/models/TinyYoloV2DAG.hpp>
#include <small/utils/Timer.hpp>

#include "test_utils.hpp"

#define RUNS 5

//****************************************************************************
/* This is the runtime recording (for check_blocks==6):

   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:16,ichans:3,img:416x416,I,F,O)
   ReLUActivation(chans:16,img:416x416,I,O)
   MaxPool2D(k:2,s:2,pad:[0,0,0,0],chans:16,img:416x416,I,O)
   layer 2
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:32,ichans:16,img:208x208,I,F,O)
   ReLUActivation(chans:32,img:208x208,I,O)
   MaxPool2D(k:2,s:2,pad:[0,0,0,0],chans:32,img:208x208,I,O)
   layer 4
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:64,ichans:32,img:104x104,I,F,O)
   ReLUActivation(chans:64,img:104x104,I,O)
   MaxPool2D(k:2,s:2,pad:[0,0,0,0],chans:64,img:104x104,I,O)
   layer 6
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:128,ichans:64,img:52x52,I,F,O)
   ReLUActivation(chans:128,img:52x52,I,O)
   MaxPool2D(k:2,s:2,pad:[0,0,0,0],chans:128,img:52x52,I,O)
   layer 8
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:256,ichans:128,img:26x26,I,F,O)
   ReLUActivation(chans:256,img:26x26,I,O)
   MaxPool2D(k:2,s:2,pad:[0,0,0,0],chans:256,img:26x26,I,O)
   layer 10
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:512,ichans:256,img:13x13,I,F,O)
   ReLUActivation(chans:512,img:13x13,I,O)
   MaxPool2D(k:2,s:1,pad:[0,1,0,1],chans:512,img:13x13,I,O)

   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:512,ichans:256,img:13x13,I,F,O)
   ReLUActivation(chans:512,img:13x13,I,O)

   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:512,ichans:512,img:13x13,I,F,O)
   ReLUActivation(chans:512,img:13x13,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:16,ichans:512,img:13x13,I,F,O)
   ReLUActivation(chans:16,img:13x13,I,O)

 */

//****************************************************************************
// Function call implementation of tinyyolov2
//****************************************************************************

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
// The output of the block is stored in O
//
template <class BufferT>
inline void yolo_block(
    std::array<uint32_t, 2> in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride, // Covolution parameters
    uint32_t output_channels,
    uint8_t t_pad,
    uint8_t b_pad,
    uint8_t l_pad,
    uint8_t r_pad,
    uint32_t kernel_size_pool,
    uint32_t stride_pool, // MaxPooling parameters
    uint8_t t_pad_pool,
    uint8_t b_pad_pool,
    uint8_t l_pad_pool,
    uint8_t r_pad_pool,
    BufferT const &I,
    BufferT const &F_conv,
    BufferT       &O_intermediate,
    BufferT       &O)
{
    small::Conv2D(kernel_size, kernel_size, stride,
                  t_pad, b_pad, l_pad, r_pad,
                  output_channels, input_channels,
                  in_dims[0], in_dims[1],
                  I, F_conv, O_intermediate);

    uint32_t o_h = small::output_dim(in_dims[0] + t_pad + b_pad,
                                     stride, kernel_size);
    uint32_t o_w = small::output_dim(in_dims[1] + l_pad + r_pad,
                                     stride, kernel_size);

    small::ReLUActivation(output_channels,   /// @todo should this be input_channels?
                          o_h, o_w,
                          O_intermediate, O_intermediate);
    // std::cout << "ReLU(Y): " << O_intermediate[0]
    //           << "\t" << O_intermediate[1]
    //           << "\t" << O_intermediate[2]
    //           << "\t" << O_intermediate[3]
    //           << std::endl;

    small::MaxPool2D(kernel_size_pool, kernel_size_pool, stride_pool,
                     t_pad_pool, b_pad_pool, l_pad_pool, r_pad_pool,
                     output_channels,
                     o_h, o_w,
                     O_intermediate, O);
    // small::ReLUActivation(output_channels, o_h, o_w, O, O);
}

//****************************************************************************
template <class BufferT>
inline void conv_block(
    std::array<uint32_t, 2> in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride, // Convolution parameters
    uint32_t output_channels,
    uint8_t t_pad,
    uint8_t b_pad,
    uint8_t l_pad,
    uint8_t r_pad,
    BufferT const &I,
    BufferT const &F_conv,
    BufferT       &O_intermediate,
    BufferT       &O)  /// @todo not used
{
    small::Conv2D(kernel_size, kernel_size, stride,
                  t_pad, b_pad, l_pad, r_pad,
                  output_channels, input_channels,
                  in_dims[0], in_dims[1],
                  I, F_conv, O_intermediate);

    uint32_t o_h = small::output_dim(in_dims[0] + t_pad + b_pad,
                                     stride, kernel_size);
    uint32_t o_w = small::output_dim(in_dims[1] + l_pad + r_pad,
                                     stride, kernel_size);
    small::ReLUActivation(output_channels,
                          o_h, o_w,
                          O_intermediate, O_intermediate);
    // std::cout << "ReLU(C): " << O_intermediate[0]
    //           << "\t" << O_intermediate[1]
    //           << "\t" << O_intermediate[2]
    //           << "\t" << O_intermediate[3]
    //           << std::endl;
}

//****************************************************************************
template <class BufferT>
BufferT &
model_inference(uint32_t layer_num_total,
                uint16_t layer_params[30][10],
                std::vector<std::array<uint32_t, 2>> intermediate_dims,
                std::vector<BufferT *> const &filter_buf_ptrs,
                BufferT const &input_dc,
                BufferT       &inter_0_dc,
                BufferT       &inter_1_dc)
{
    uint32_t ds_blocks(6U);

    auto layer_num = 0;
    yolo_block(intermediate_dims[layer_num],
               REDUCTION_C(layer_num), // Input dimensions
               REDUCTION_HW(layer_num),
               STRIDE(layer_num), // Params for the  convolution
               GROUP_C(layer_num),
               PADDING(layer_num),
               REDUCTION_HW(layer_num + 1),
               STRIDE(layer_num + 1), // Params for the pooling
               PADDING(layer_num + 1),
               input_dc,
               *filter_buf_ptrs[layer_num],
               inter_1_dc,
               inter_0_dc);

    // printf("intermediate_output shape: ");print_shape(layer_num, inter_0);
    layer_num += 2;
    for (uint32_t ds_layer = 1U; ds_layer < ds_blocks; ds_layer++)
    {
        //printf("layer %d\n", layer_num);
        yolo_block(intermediate_dims[layer_num],
                   REDUCTION_C(layer_num), // Input dimensions
                   REDUCTION_HW(layer_num),
                   STRIDE(layer_num), // Params for the convolution
                   GROUP_C(layer_num),
                   PADDING(layer_num),
                   REDUCTION_HW(layer_num + 1),
                   STRIDE(layer_num + 1), // Params for the pooling
                   PADDING(layer_num + 1),
                   inter_0_dc,
                   *filter_buf_ptrs[ds_layer],
                   inter_1_dc,
                   inter_0_dc);

        layer_num += 2;
    }

    // //Epilogue layers with no pooling

    for (int epilogue_layer = 0; epilogue_layer < 3; epilogue_layer++)
    {
        conv_block(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
                   REDUCTION_HW(layer_num),
                   STRIDE(layer_num), // Params for the  convolution
                   GROUP_C(layer_num),
                   PADDING(layer_num),
                   inter_0_dc,
                   *filter_buf_ptrs[ds_blocks + epilogue_layer],
                   inter_1_dc,
                   inter_0_dc);
        layer_num++;

        inter_0_dc.swap(inter_1_dc);
        //float *tmp = inter_0_dc;
        //inter_0_dc = inter_1_dc;
        //inter_1_dc = tmp;
    }

    return inter_0_dc;
}

//****************************************************************************
//****************************************************************************
template <class BufferT>
void build_baseline_tinyyolov2(
    uint32_t C_i,         // 128, 1, 1, 16(8), ...
    uint32_t N, // I_h
    uint32_t M, // I_w
    uint32_t num_classes,
    uint32_t check_blocks,
    size_t&  layer_num_total,
    uint16_t layer_params[30][10],
    std::vector<std::array<uint,2>> &intermediate_dims,
    std::vector<BufferT *>& filter_buf_ptrs,
    size_t& max_numel_inter_0,
    size_t& max_numel_inter_1,
    size_t& num_outputs)
{
    uint8_t t_pad, b_pad, r_pad, l_pad;

    // Set up model parameters
    int layer_num = 0;
    int pool_layer_num = 0, conv_layer_num = 0;

    max_numel_inter_0 = 0;
    max_numel_inter_1 = 0;
    uint32_t inter_dim = 0;

    intermediate_dims.push_back(std::array<uint, 2>({N, M}));  //(height, width)

    // conv
    REDUCTION_C(layer_num) = C_i; // input channels
    GROUP_C(layer_num) = 16;      // output channels
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 3; // kernel size
    STRIDE(layer_num) = 1;       // stride
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                        STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                        STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
    TYPE(layer_num) = 1; // conv2d

    layer_num++; // 1
    conv_layer_num++;
    intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
    inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_1 =   // @todo Should this be inter_0?
        (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;
    TYPE(layer_num) = 0; // not conv2d

    // Pool
    REDUCTION_C(layer_num) = 1; // input channels
    GROUP_C(layer_num) = 1;
    GROUPS(layer_num) = GROUP_C(layer_num - 1); // output channels
    REDUCTION_HW(layer_num) = 2;                // kernel size
    STRIDE(layer_num) = 2;                      // stride
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                        STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                        STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

    layer_num++; // 2
    pool_layer_num++;
    intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
    inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_0 =
        (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

    // common set up for model architecture
    uint32_t ds_blocks = (check_blocks < 6) ? check_blocks : 6;
    int layer_strides[] = {2, 2, 2, 2, 2, 1};
    //int num_channels = 16;
    int channel_multiplier = 2;
    // dwise 1
    for (uint32_t ds_layer = 1U; ds_layer < ds_blocks; ds_layer++)  ///@todo verify start at 1
    {
        REDUCTION_C(layer_num) = GROUPS(layer_num - 1);                   // input channels
        GROUP_C(layer_num) = channel_multiplier * REDUCTION_C(layer_num); // output channels
        GROUPS(layer_num) = 1;
        REDUCTION_HW(layer_num) = 3;
        STRIDE(layer_num) = 1;
        small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                            STRIDE(layer_num), t_pad, b_pad);
        small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                            STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
        TYPE(layer_num) = 1; // conv2d

        layer_num++; // 3
        conv_layer_num++;
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_1 =
            (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;
        intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
        // std::cout << intermediate_dims[layer_num - 1][0] << " " << intermediate_dims[layer_num - 1][1] << std::endl;

        // Pool
        REDUCTION_C(layer_num) = 1; // input channels
        GROUP_C(layer_num) = 1;
        GROUPS(layer_num) = GROUP_C(layer_num - 1);  // output channels
        REDUCTION_HW(layer_num) = 2;                 // kernel size
        STRIDE(layer_num) = layer_strides[ds_layer]; // stride
        if (STRIDE(layer_num) == 1)
        {
            SET_PADDING(layer_num, 0, 1, 0, 1);
        }
        else
        {
            small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                                STRIDE(layer_num), t_pad, b_pad);
            small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                                STRIDE(layer_num), l_pad, r_pad);
            SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
        }
        TYPE(layer_num) = 0; // not conv2d

        layer_num++; // 2
        pool_layer_num++;
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_0 =
            (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;
        intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
    }

    REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
    GROUP_C(layer_num) = GROUPS(layer_num - 1) * 2;
    GROUPS(layer_num) = 1;       // GROUP_C(layer_num - 1);
    REDUCTION_HW(layer_num) = 3; // I_HEIGHT(layer_num);
    STRIDE(layer_num) = 1;
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                        STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                        STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
    TYPE(layer_num) = 1; // conv2d

    layer_num++;
    conv_layer_num++;
    inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_1 =
        (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;
    intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));

    REDUCTION_C(layer_num) = GROUP_C(layer_num - 1);
    GROUP_C(layer_num) = GROUP_C(layer_num - 1);
    GROUPS(layer_num) = 1;       // GROUP_C(layer_num - 1);
    REDUCTION_HW(layer_num) = 3; // I_HEIGHT(layer_num);
    STRIDE(layer_num) = 1;
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                        STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                        STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
    TYPE(layer_num) = 1; // conv2d

    layer_num++;
    conv_layer_num++;
    inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_0 =
        (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

    intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));

    REDUCTION_C(layer_num) = GROUP_C(layer_num - 1);
    GROUP_C(layer_num) = num_classes;
    GROUPS(layer_num) = 1;       // GROUP_C(layer_num - 1);
    REDUCTION_HW(layer_num) = 1; // I_HEIGHT(layer_num);
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0);
    TYPE(layer_num) = 1; // conv2d

    layer_num++;
    conv_layer_num++;
    inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_1 =
        (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;
    intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));

    layer_num_total = layer_num;
#if SUMMARY == 1
    //__________Summarize Network____________________________________
    printf("layer num, H_in, W_in, F_c, K, G, F_h, stride, t_pad, b_pad, l_pad, r_pad\n");
    for (auto i = 0UL; i < layer_num_total; i++)
    {
        printf("%ld, ", i);
        printf("HxW: %d, %d, layer_params: ", I_HEIGHT(i), I_WIDTH(i));
        for (auto j = 0; j < 10; j++)
        {
            printf("%d, ", layer_params[i][j]);
        }
        printf("intermediate dims: %d %d ",
               intermediate_dims[i][0], intermediate_dims[i][1]);
        printf("\n");
    }

    printf("set up %d pool layers, %d conv layers, total: %ld\n",
           pool_layer_num, conv_layer_num, layer_num_total);
#endif

    for (size_t l = 0; l < layer_num_total; ++l) //conv_layer_num; l++)
    {
        if (1 == TYPE(l))
        {
            //float *filter_ptr;
            // weights = layers[l]->weight; // conv_1x1->weight;
            uint32_t filter_dimensions =
                REDUCTION_HW(l) * REDUCTION_HW(l) * REDUCTION_C(l) *
                GROUP_C(l) * GROUPS(l);
            BufferT *filter_buf_ptr =
                new BufferT(filter_dimensions);
            //filter_ptr = alloc(filter_dimensions);
            init(*filter_buf_ptr, filter_dimensions);
            filter_buf_ptrs.push_back(filter_buf_ptr);
#if SUMMARY == 1
            std::cout << l << ": init filter buf, size = " << filter_dimensions
                      << "\t" << (*filter_buf_ptr)[0]
                      << "\t" << (*filter_buf_ptr)[1]
                      << "\t" << (*filter_buf_ptr)[2]
                      << "..." << std::endl;
#endif
        }
    }
    printf("\n");

    /// @todo assert(filter_buf_ptrs.size() == num_filters)
    num_outputs =
        num_classes*intermediate_dims.back()[0]*intermediate_dims.back()[1];
}

//****************************************************************************
//****************************************************************************
void test_tinyyolov2(void)
{
#if defined QUANTIZED
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // input parameters
    uint32_t C_i = 3;
    uint32_t N = 416; // 608
    uint32_t M = 416; // 608
    uint32_t num_classes = 16;
    uint32_t check_blocks = 6;

    // ================================================

    //************************************************************************
    // Baseline (function call) model
    //************************************************************************

    // "model" params
    size_t   layer_num_total = 0;
    uint16_t layer_params[30][10] = {1};
    std::vector<std::array<uint32_t, 2>> intermediate_dims;
    size_t   max_numel_inter_0 = 0;
    size_t   max_numel_inter_1 = 0;
    std::vector<BufferT*> filter_buf_ptrs;
    size_t   num_outputs = 0;

    build_baseline_tinyyolov2(C_i, N, M, num_classes, check_blocks,
                              layer_num_total,
                              layer_params,
                              intermediate_dims,
                              filter_buf_ptrs,
                              max_numel_inter_0,
                              max_numel_inter_1,
                              num_outputs);

    // ================================================

    std::cerr << "Intermediate buffer sizes: "
              << max_numel_inter_0 << ", " << max_numel_inter_1
              << std::endl;

#if defined QUANTIZED
    BufferT inter_0_dc(max_numel_inter_0*2);  /// @todo HACK need to determine correct size
    BufferT inter_1_dc(max_numel_inter_1*2);  /// @todo HACK need to determine correct size
#else
    BufferT inter_0_dc(max_numel_inter_0);
    BufferT inter_1_dc(max_numel_inter_1);
#endif

    // Create input tensor
    uint32_t input_dimensions = C_i * N * M;
    BufferT input_dc(input_dimensions);
    init(input_dc, input_dimensions);

#if VERBOSE == 1
    std::cout << "\nInput buffer:\n";
    for (size_t ix = 0; ix < N*M; ++ix)
    {
        std::cout << "(" << ix/M << "," << ix%M << ")";
        for (size_t iy = 0; iy < C_i; ++iy)
        {
            std::cout << "\t" << input_dc[iy*(N*M) + ix];
        }
        std::cout << std::endl;
    }
#endif

    //======================================================
    small::Timer my_timer;

    std::cerr << "\nWarm up run (ORIG)" << std::endl;

    // HACK: due to odd buffer swaps
    if (inter_0_dc.size() != max_numel_inter_0)
        inter_0_dc.swap(inter_1_dc);

    my_timer.start();
    auto &output_tmp =
        model_inference(layer_num_total, layer_params,
                        intermediate_dims,
                        filter_buf_ptrs,
                        input_dc,
                        inter_0_dc,
                        inter_1_dc);
    my_timer.stop();

    // copy the output for comparison in subsequent runs.
    BufferT output_answers(output_tmp);
    printf("\nElapsed time: %lf ns.\n", my_timer.elapsed());

    //======================================================
    // Timing runs
    //======================================================
    std::vector<double> small_timing;

    for (int r = 0; r < RUNS; r++)
    {
        std::cout << "Baseline run: " << r;
        // HACK: due to odd buffer swaps
        if (inter_0_dc.size() != max_numel_inter_0)
            inter_0_dc.swap(inter_1_dc);

        my_timer.start();

        auto &output_dc =
            model_inference(layer_num_total, layer_params,
                            intermediate_dims,
                            filter_buf_ptrs,
                            input_dc,
                            inter_0_dc,
                            inter_1_dc);

        my_timer.stop();
        auto diff = my_timer.elapsed();
        small_timing.push_back(diff);
        std::cout << ": " << diff << " ns.\n";

        // Test that the answer stays the same through multiple invocations
        bool passed = true;
        for (size_t ix = 0; ix < num_outputs; ++ix)
        {
            bool same_value = (output_answers[ix] == output_dc[ix]);

#if SUMMARY == 1
            std::cout << (same_value ? "pass: " : "FAIL: ")
                      << "baseline (first run), baseline output " << ix
                      << ": " << (float)output_answers[ix]
                      << " ?= " << (float)output_dc[ix]
                      << std::endl;
#endif
            passed &= same_value;
        }
        TEST_CHECK(passed);
    }

    //************************************************************************
    // Model class
    //************************************************************************
    std::vector<double> layer_timing;
    {
        small::shape_type input_shape({1UL, C_i, N, M});

        small::TinyYoloV2<BufferT> model(input_shape,
                                         filter_buf_ptrs, true);

        small::Tensor<BufferT> input_tensor(input_shape, input_dc);

        //***********
        std::cerr << "\nWarm up run (LAYERS)" << std::endl;
        auto output_tensors = model.inference(&input_tensor);
        BufferT output_buffer = output_tensors[0]->buffer();
        //***********

        TEST_CHECK(1 == output_tensors.size());
        std::cerr << "Output sizes: " << num_outputs << " ?= "
                  <<  output_tensors[0]->size() << std::endl;
        TEST_CHECK(num_outputs == output_tensors[0]->size());

        // Compare outputs
        bool passed = true;
        std::cout << "\nCHECK RESULTS: Num output elements: "
                  << num_outputs << std::endl;

        for (size_t ix = 0; ix < num_outputs; ++ix)
        {
            bool same_value = (output_answers[ix] == output_tensors[0]->buffer()[ix]);

#if SUMMARY == 1
            std::cout << (same_value ? "pass: " : "FAIL: ")
                      << "baseline, Model output " << ix
                      << ": " << (float)output_answers[ix]
                      << " ?= " << (float)output_tensors[0]->buffer()[ix]
                      << std::endl;
#endif
            passed &= same_value;
        }
        TEST_CHECK(passed);

        //======================================================
        // Timing runs
        //======================================================

        for (int r = 0; r < RUNS; r++)
        {
            std::cout << "Model run: " << r;
            my_timer.start();

            //***********
            model.inference(&input_tensor);
            //***********

            my_timer.stop();
            auto diff = my_timer.elapsed();
            layer_timing.push_back(diff);
            std::cout << ": " << diff << " ns.\n";

            // Test that the answer stays the same through multiple invocations
            bool passed = true;
            for (size_t ix = 0; ix < num_outputs; ++ix)
            {
                bool same_value = (output_answers[ix] == output_tensors[0]->buffer()[ix]);

#if SUMMARY == 1
                std::cout << (same_value ? "pass: " : "FAIL: ")
                          << "baseline (first run), Model output " << ix
                          << ": " << (float)output_answers[ix]
                          << " ?= " << (float)output_tensors[0]->buffer()[ix]
                          << std::endl;
#endif
                passed &= same_value;
            }
            TEST_CHECK(passed);
        }
    }

    //************************************************************************
    // DAGModel class
    //************************************************************************
    std::vector<double> dag_timing;
    {
        small::shape_type input_shape({1UL, C_i, N, M});

        small::TinyYoloV2DAG<BufferT> model(input_shape, filter_buf_ptrs, true);

        small::Tensor<BufferT> input_tensor(input_shape, input_dc);

        //***********
        std::cerr << "\nWarm up run (DAG)" << std::endl;
        auto output_tensors = model.inference(&input_tensor);
        BufferT output_buffer = output_tensors[0]->buffer();
        //***********

        TEST_CHECK(1 == output_tensors.size());
        std::cerr << "Output sizes: " << num_outputs << " ?= "
                  <<  output_tensors[0]->size() << std::endl;
        TEST_CHECK(num_outputs == output_tensors[0]->size());

        // Compare outputs
        bool passed = true;
        std::cout << "\nCHECK RESULTS: Num output elements: "
                  << num_outputs << std::endl;

        for (size_t ix = 0; ix < num_outputs; ++ix)
        {
            bool same_value = (output_answers[ix] == output_tensors[0]->buffer()[ix]);

#if SUMMARY == 1
            std::cout << (same_value ? "pass: " : "FAIL: ")
                      << "baseline, Model output " << ix
                      << ": " << (float)output_answers[ix]
                      << " ?= " << (float)output_tensors[0]->buffer()[ix]
                      << std::endl;
#endif
            passed &= same_value;
        }
        TEST_CHECK(passed);

        //======================================================
        // Timing runs
        //======================================================

        for (int r = 0; r < RUNS; r++)
        {
            std::cout << "DAG run: " << r;
            my_timer.start();

            //***********
            model.inference(&input_tensor);
            //***********

            my_timer.stop();
            auto diff = my_timer.elapsed();
            dag_timing.push_back(diff);
            std::cout << ": " << diff << " ns.\n";

            // Test that the answer stays the same through multiple invocations
            bool passed = true;
            for (size_t ix = 0; ix < num_outputs; ++ix)
            {
                bool same_value = (output_answers[ix] == output_tensors[0]->buffer()[ix]);

#if SUMMARY == 1
                std::cout << (same_value ? "pass: " : "FAIL: ")
                          << "baseline (first run), Model output " << ix
                          << ": " << (float)output_answers[ix]
                          << " ?= " << (float)output_tensors[0]->buffer()[ix]
                          << std::endl;
#endif
                passed &= same_value;
            }
            TEST_CHECK(passed);
        }
    }

    int num_th = 1;
#if PARALLEL == 1
    char const *env_nt(std::getenv("OMP_NUM_THREADS"));
    if (nullptr != env_nt)
    {
        num_th = atoi(std::getenv("OMP_NUM_THREADS"));
    }
#endif
    std::cout << "\nSUMMARY STATS:" << std::endl;
    std::cout << "Num Threads: " << num_th << std::endl;
    print_stats(small_timing, "SMaLL:tinyyolov2 Baseline");
    print_stats(layer_timing, "SMaLL:tinyyolov2 Layers  ");
    print_stats(dag_timing,   "SMaLL:tinyyolov2 DAGModel");

    // clean up
    for (auto filter : filter_buf_ptrs)
    {
        delete filter;
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"tinyyolov2_data_performance", test_tinyyolov2},
    {NULL, NULL}
};
