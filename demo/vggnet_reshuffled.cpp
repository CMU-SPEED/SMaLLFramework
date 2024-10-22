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
#include <iomanip>

#include <small.h>
#include <small/utils/Timer.hpp>
#include "utils.h"

/// @todo Which of these defines are needed?
#ifndef RUNS
#define RUNS 10
#endif

#define TRIALS 10
#define SUMMARY 0


#include<small/Layer.hpp>
#include<small/DepthwiseConv2DLayer.hpp>
#include<small/Conv2DLayer.hpp>
#include<small/MaxPool2DLayer.hpp>
#include<small/ReLULayer.hpp>


template <class T>
inline bool almost_equal(T v1, T v2, float rtol = 5e-03, float atol = 1e-05)
{
    float abs_diff = fabs((float)(v1) - (float)(v2));
    float diff_tolerance = (atol + rtol * fabs(v2));
    return (abs_diff <= diff_tolerance);
}

#define CORRECTNESS_CHECK(passing, calculated_output_dc, actual_output_dc, extent) \
    for (size_t ix = 0; ix < extent; ++ix)            \
    {                                                                      \
        if ((actual_output_dc[ix] != calculated_output_dc[ix]) &&          \
            !almost_equal(actual_output_dc[ix], calculated_output_dc[ix])) \
        {                                                                  \
            passing = false;                                               \
                                                                           \
            std::cout << "FAIL: Conv2D_out(" << ix << ")-->"               \
                      << std::setw(12) << std::setprecision(10)            \
                      << actual_output_dc[ix] << "(computed) != "          \
                      << std::setw(12) << std::setprecision(10)            \
                      << calculated_output_dc[ix]                          \
                      << std::endl;                                        \
        }                                                                  \
    }



#define TIME_LAYER 1


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
double layer_timers[3][20];
double min_layer_timers[3][20];
double avg_layer_timers[3][20];
int pool_layer_num = 0;

template <class BufferT>
inline void vgg_block(
    std::array<uint32_t, 2> const &in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size_conv,
    uint32_t stride_conv,
    uint8_t t_pad_conv,
    uint8_t b_pad_conv,
    uint8_t l_pad_conv,
    uint8_t r_pad_conv,
    uint32_t output_channels,
    uint32_t kernel_size,
    uint32_t stride,
    uint8_t t_pad,
    uint8_t b_pad,
    uint8_t l_pad,
    uint8_t r_pad,
    BufferT const &I,
    BufferT const &F_conv,
    BufferT &O_intermediate,
    BufferT &O
    )
{
    uint32_t i_w = in_dims[0];
    uint32_t i_h = in_dims[1];

    #if TIME_LAYER 
    small::Timer my_timer;
    my_timer.start();
    #endif
    
    small::Conv2D(kernel_size_conv, kernel_size_conv, stride_conv,
                           t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                  output_channels, input_channels,
                  i_h, i_w,
                  I, F_conv, O_intermediate);
    small::ReLUActivation(output_channels, i_h, i_w, O_intermediate, O_intermediate);

    /**/
    //calculate output dimensions of Conv
    uint32_t o_w_p2 = small::output_dim(i_w + l_pad_conv + r_pad_conv,
                                     stride_conv, kernel_size_conv);
    uint32_t o_h_p2 = small::output_dim(i_h + t_pad_conv + b_pad_conv,
                                        stride_conv, kernel_size_conv); 

    #if TIME_LAYER
    my_timer.stop();
    layer_timers[0][pool_layer_num-1] = my_timer.elapsed();
    #endif

    #if TIME_LAYER 
    // small::Timer my_timer;
    my_timer.start();
    #endif
    small::MaxPool2D(kernel_size, kernel_size, stride,
                           t_pad, b_pad, l_pad, r_pad,
                           output_channels,
                           o_h_p2, o_w_p2,
                           O_intermediate,O);

    #if TIME_LAYER
    my_timer.stop();
    layer_timers[0][pool_layer_num] = my_timer.elapsed();
    #endif



}

template <class BufferT>
inline void fused_ewise_vgg_block(
    std::array<uint32_t, 2> const &in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size_conv,
    uint32_t stride_conv,
    uint8_t t_pad_conv,
    uint8_t b_pad_conv,
    uint8_t l_pad_conv,
    uint8_t r_pad_conv,
    uint32_t output_channels,
    uint32_t kernel_size,
    uint32_t stride,
    uint8_t t_pad,
    uint8_t b_pad,
    uint8_t l_pad,
    uint8_t r_pad,
    BufferT const &I,
    BufferT const &F_conv,
    BufferT &O_intermediate,
    BufferT &O
    )
{
    /**/

    uint32_t i_w = in_dims[0];
    uint32_t i_h = in_dims[1];

    #if TIME_LAYER 
    small::Timer my_timer;
    my_timer.start();
    #endif
    small::Conv2D_ReLU(kernel_size_conv, kernel_size_conv, stride_conv,
                           t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
                  output_channels, input_channels,
                  i_h, i_w,
                  I, F_conv, O_intermediate);

    #if TIME_LAYER
    my_timer.stop();
    layer_timers[1][pool_layer_num-1] = my_timer.elapsed();
    #endif

    //Calculate output dimensions of Conv
    uint32_t o_w_p2 = small::output_dim(i_w + l_pad_conv + r_pad_conv,
                                     stride_conv, kernel_size_conv);
    uint32_t o_h_p2 = small::output_dim(i_h + t_pad_conv + b_pad_conv
                                        , stride_conv, kernel_size_conv);

     #if TIME_LAYER 
    // small::Timer my_timer;
    my_timer.start();
    #endif
    small::MaxPool2D(kernel_size, kernel_size, stride,
                           t_pad, b_pad, l_pad, r_pad,
                           output_channels,
                           o_h_p2, o_w_p2,
                           O_intermediate,O);
    
    #if TIME_LAYER
    my_timer.stop();
    layer_timers[1][pool_layer_num] = my_timer.elapsed();
    #endif
}

template <class BufferT>
inline void fused_vgg_block(
    std::array<uint32_t, 2> const &in_dims, uint32_t input_channels, // Input dimensions
    uint32_t kernel_size_conv,
    uint32_t stride_conv,
    uint8_t t_pad_conv,
    uint8_t b_pad_conv,
    uint8_t l_pad_conv,
    uint8_t r_pad_conv,
    uint32_t output_channels,
    uint32_t kernel_size,
    uint32_t stride,
    uint8_t t_pad,
    uint8_t b_pad,
    uint8_t l_pad,
    uint8_t r_pad,
    BufferT const &I,
    BufferT const &F_conv,
    BufferT &O_intermediate,
    BufferT &O
    )
{
//    uint32_t i_w = in_dims[0];
//     uint32_t i_h = in_dims[1];
//         small::Conv2D_ReLU(kernel_size_conv, kernel_size_conv, stride_conv,
//                            t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv,
//                   output_channels, input_channels,
//                   i_h, i_w,
//                   I, F_conv, O_intermediate);
    small::Conv2D_ReLU_Maxpool2D( kernel_size_conv, kernel_size_conv, stride_conv, 
                                            t_pad_conv, b_pad_conv, l_pad_conv, r_pad_conv, 
                                            kernel_size, kernel_size, stride,
                           t_pad, b_pad, l_pad, r_pad,
                           output_channels, input_channels,
                         in_dims[0], in_dims[1],
                         I,
                         F_conv,
                         O_intermediate,
                         O);
}

//****************************************************************************
template <class BufferT>
BufferT &
model_inference(uint32_t layer_num_total,
                uint16_t layer_params[30][10],
                uint32_t vgg_blocks,
                uint32_t *num_convs,
                std::vector<std::array<uint32_t, 2>> const &intermediate_dims,
                std::vector<BufferT *> const &filter_buf_ptrs,
                BufferT const &input_dc,
                BufferT       &inter_0_dc,
                BufferT       &inter_1_dc)
{
    auto layer_num = 0;
    auto impl = 0;
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

     #if TIME_LAYER
    my_timer.stop();
    layer_timers[impl][0] = my_timer.elapsed();
    #endif


    #if TIME_LAYER
        pool_layer_num = 2;
    //    my_timer.start();
    #endif
    vgg_block(intermediate_dims[1], REDUCTION_C(1), // Input dimensions
    REDUCTION_HW(1),
    STRIDE(1),
    PADDING(1),
    GROUP_C(1),
    REDUCTION_HW(2),
    STRIDE(2),
    PADDING(2),
    inter_0_dc,
    *filter_buf_ptrs[1],
    inter_1_dc,
    inter_0_dc);

    layer_num+=2;
     #if TIME_LAYER
    // my_timer.stop();
    // layer_timers[impl][1] = my_timer.elapsed();
    #endif

    for (int vgg_layer = 1; vgg_layer < 5; vgg_layer++)
    {
   
    #if TIME_LAYER
    my_timer.start();
    #endif
    small::Conv2D(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                  STRIDE(layer_num), PADDING(layer_num),
                  GROUP_C(layer_num), REDUCTION_C(layer_num),
                  I_HEIGHT(layer_num), I_WIDTH(layer_num),
                  inter_0_dc,
                  *filter_buf_ptrs[layer_num],
                  inter_1_dc);

    
    small::ReLUActivation(GROUP_C(0),
                          I_HEIGHT(layer_num), I_WIDTH(layer_num),
                          inter_1_dc,
                          inter_1_dc);

     #if TIME_LAYER
    my_timer.stop();
    layer_timers[impl][layer_num] = my_timer.elapsed();
    #endif
    layer_num++;


    //if num_convs[vgg_layer] > 2, do an additional conv

    if(num_convs[vgg_layer] == 3)
    {
       #if TIME_LAYER
       my_timer.start();
       #endif
       small::Conv2D(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                  STRIDE(layer_num), PADDING(layer_num),
                  GROUP_C(layer_num), REDUCTION_C(layer_num),
                  I_HEIGHT(layer_num), I_WIDTH(layer_num),
                  inter_1_dc,
                  *filter_buf_ptrs[layer_num],
                  inter_0_dc);

    

    small::ReLUActivation(GROUP_C(0),
                          I_HEIGHT(layer_num), I_WIDTH(layer_num),
                          inter_0_dc,
                          inter_0_dc);

        #if TIME_LAYER
        my_timer.stop();
        layer_timers[impl][layer_num] = my_timer.elapsed();
        #endif

        layer_num++; 


        //switch inter_1 and inter_0 ptrs
        std::swap(inter_1_dc, inter_0_dc);
    }


       #if TIME_LAYER
       pool_layer_num = layer_num + 1;
    //    my_timer.start();
       #endif

        vgg_block(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
            REDUCTION_HW(layer_num),
            STRIDE(layer_num),
            PADDING(layer_num),
            GROUP_C(layer_num),
            REDUCTION_HW(layer_num + 1),
            STRIDE(layer_num + 1),
            PADDING(layer_num + 1),
            inter_1_dc,
            *filter_buf_ptrs[layer_num],
            inter_0_dc,
            inter_1_dc);

        
        #if TIME_LAYER
        // my_timer.stop();
        // layer_timers[impl][layer_num] = my_timer.elapsed();
        #endif

        layer_num += 2;
        //switch inter_1 and inter_0 ptrs, so that we start with inter_0 for next iteration
        std::swap(inter_1_dc, inter_0_dc);
    }

    return inter_0_dc;
}



//****************************************************************************
template <class BufferT>
BufferT &
fused_ewise_model_inference(uint32_t layer_num_total,
                uint16_t layer_params[30][10],
                uint32_t vgg_blocks,
                uint32_t *num_convs,
                std::vector<std::array<uint32_t, 2>> const &intermediate_dims,
                std::vector<BufferT *> const &filter_buf_ptrs,
                BufferT const &input_dc,
                BufferT &inter_0_dc,
                BufferT       &inter_1_dc)
{
 
    auto layer_num = 0;
    auto impl = 1;
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

     #if TIME_LAYER
    my_timer.stop();
    layer_timers[impl][0] = my_timer.elapsed();
    #endif


    #if TIME_LAYER
    pool_layer_num = 2;
    //    my_timer.start();
    #endif
    fused_ewise_vgg_block(intermediate_dims[1], REDUCTION_C(1), // Input dimensions
    REDUCTION_HW(1),
    STRIDE(1),
    PADDING(1),
    GROUP_C(1),
    REDUCTION_HW(2),
    STRIDE(2),
    PADDING(2),
    inter_0_dc,
    *filter_buf_ptrs[1],
    inter_1_dc,
    inter_0_dc);

    layer_num+=2;
     #if TIME_LAYER
    // my_timer.stop();
    // layer_timers[impl][1] = my_timer.elapsed();
    #endif

    for (int vgg_layer = 1; vgg_layer < 5; vgg_layer++)
    {
    #if TIME_LAYER
    my_timer.start();
    #endif
    small::Conv2D_ReLU(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                  STRIDE(layer_num), PADDING(layer_num),
                  GROUP_C(layer_num), REDUCTION_C(layer_num),
                  I_HEIGHT(layer_num), I_WIDTH(layer_num),
                  inter_0_dc,
                  *filter_buf_ptrs[layer_num],
                  inter_1_dc);

     #if TIME_LAYER
    my_timer.stop();
    layer_timers[impl][layer_num] = my_timer.elapsed();
    #endif
    layer_num++;


    //if num_convs[vgg_layer] > 2, do an additional conv

    if(num_convs[vgg_layer] == 3)
    {
       #if TIME_LAYER
       my_timer.start();
       #endif
       small::Conv2D_ReLU(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                  STRIDE(layer_num), PADDING(layer_num),
                  GROUP_C(layer_num), REDUCTION_C(layer_num),
                  I_HEIGHT(layer_num), I_WIDTH(layer_num),
                  inter_1_dc,
                  *filter_buf_ptrs[layer_num],
                  inter_0_dc);


        #if TIME_LAYER
        my_timer.stop();
        layer_timers[impl][layer_num] = my_timer.elapsed();
        #endif

        layer_num++; 


        //switch inter_1 and inter_0 ptrs
        std::swap(inter_1_dc, inter_0_dc);
    }


       #if TIME_LAYER
       pool_layer_num = layer_num + 1;
    //    my_timer.start();
       #endif

        fused_ewise_vgg_block(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
            REDUCTION_HW(layer_num),
            STRIDE(layer_num),
            PADDING(layer_num),
            GROUP_C(layer_num),
            REDUCTION_HW(layer_num + 1),
            STRIDE(layer_num + 1),
            PADDING(layer_num + 1),
            inter_1_dc,
            *filter_buf_ptrs[layer_num],
            inter_0_dc,
            inter_1_dc);

        
        #if TIME_LAYER
        // my_timer.stop();
        // layer_timers[impl][layer_num] = my_timer.elapsed();
        #endif

        layer_num += 2;
        //switch inter_1 and inter_0 ptrs, so that we start with inter_0 for next iteration
        std::swap(inter_1_dc, inter_0_dc);
    }

    return inter_0_dc;
}


//****************************************************************************
template <class BufferT>
BufferT &
fused_model_inference(uint32_t layer_num_total,
                uint16_t layer_params[30][10],
                uint32_t vgg_blocks,
                uint32_t *num_convs,
                std::vector<std::array<uint32_t, 2>> const &intermediate_dims,
                std::vector<BufferT *> const &filter_buf_ptrs,
                BufferT const &input_dc,
                BufferT &inter_0_dc,
                BufferT &inter_1_dc,
                BufferT &inter_0_buffer_dc)
{
    
    auto layer_num = 0;
    auto impl = 2;
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

     #if TIME_LAYER
    my_timer.stop();
    layer_timers[impl][0] = my_timer.elapsed();
    #endif


    #if TIME_LAYER
       my_timer.start();
    #endif
    // Run this layer sinmgle threaded to save memory
    // char *num_threads = getenv("OMP_NUM_THREADS");
    // setenv("OMP_NUM_THREADS", "1", 1);
    fused_vgg_block(intermediate_dims[1], REDUCTION_C(1), // Input dimensions
    REDUCTION_HW(1),
    STRIDE(1),
    PADDING(1),
    GROUP_C(1),
    REDUCTION_HW(2),
    STRIDE(2),
    PADDING(2),
    inter_0_dc,
    *filter_buf_ptrs[1],
    inter_0_buffer_dc,
    inter_1_dc);

    layer_num+=2;
     #if TIME_LAYER
    my_timer.stop();
    layer_timers[impl][1] = my_timer.elapsed();
    #endif
    // setenv("OMP_NUM_THREADS", num_threads, 1);
    for (int vgg_layer = 1; vgg_layer < 5; vgg_layer++)
    {
    #if TIME_LAYER
    my_timer.start();
    #endif
    small::Conv2D_ReLU(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                  STRIDE(layer_num), PADDING(layer_num),
                  GROUP_C(layer_num), REDUCTION_C(layer_num),
                  I_HEIGHT(layer_num), I_WIDTH(layer_num),
                  inter_1_dc,
                  *filter_buf_ptrs[layer_num],
                  inter_0_dc);

     #if TIME_LAYER
    my_timer.stop();
    layer_timers[impl][layer_num] = my_timer.elapsed();
    #endif

    layer_num++;


    //if num_convs[vgg_layer] > 2, do an additional conv

    if(num_convs[vgg_layer] == 3)
    {
       #if TIME_LAYER
       my_timer.start();
       #endif
       small::Conv2D_ReLU(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                  STRIDE(layer_num), PADDING(layer_num),
                  GROUP_C(layer_num), REDUCTION_C(layer_num),
                  I_HEIGHT(layer_num), I_WIDTH(layer_num),
                  inter_0_dc,
                  *filter_buf_ptrs[layer_num],
                  inter_1_dc);


        #if TIME_LAYER
        my_timer.stop();
        layer_timers[impl][layer_num] = my_timer.elapsed();
        #endif

        layer_num++; 


        //switch inter_1 and inter_0 ptrs
        std::swap(inter_1_dc, inter_0_dc);
    }


       #if TIME_LAYER
       my_timer.start();
       #endif

        fused_vgg_block(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
            REDUCTION_HW(layer_num),
            STRIDE(layer_num),
            PADDING(layer_num),
            GROUP_C(layer_num),
            REDUCTION_HW(layer_num + 1),
            STRIDE(layer_num + 1),
            PADDING(layer_num + 1),
            inter_0_dc,
            *filter_buf_ptrs[layer_num],
            inter_0_buffer_dc,
            inter_1_dc);

        
        #if TIME_LAYER
        my_timer.stop();
        layer_timers[impl][layer_num] = my_timer.elapsed();
        #endif

        layer_num += 2;
        //switch inter_1 and inter_0 ptrs, so that we start with inter_0 for next iteration
        std::swap(inter_1_dc, inter_0_dc);
    }

    return inter_0_dc;
}

//****************************************************************************
//****************************************************************************
template <class BufferT>
void inference()
{
    uint32_t C_i = 3;
    uint32_t N = 224;
    uint32_t M = 224;
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

    // first pooling block
    //{224, 224, 3} -> {112, 112, 64}
    // conv1
    REDUCTION_C(layer_num) = C_i;
    GROUP_C(layer_num) = 64;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 3;
    STRIDE(layer_num) = 1;
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
    //conv 2
    REDUCTION_C(layer_num) = 64;
    GROUP_C(layer_num) = 64;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 3;
    STRIDE(layer_num) = 1;
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                        STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                        STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
    layer_num++; // 1
    intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
    inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_1 =
        (inter_dim > max_numel_inter_1)? inter_dim : max_numel_inter_1;

    //pooling layer
    REDUCTION_C(layer_num) = 1;
    GROUP_C(layer_num) = 1;
    GROUPS(layer_num) = GROUP_C(layer_num - 1);
    REDUCTION_HW(layer_num) = 3;
    STRIDE(layer_num) = 2;
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                        STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                        STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
    layer_num++; // 1
    intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
    inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_0 =
        (inter_dim > max_numel_inter_0)? inter_dim : max_numel_inter_0;
    
    //}
    // common set up for model architecture
    auto vgg_blocks = 5;

    uint32_t num_convs[] = {2, 2, 3, 3, 3};
    // dwise 1
    for (int vgg_layer = 1; vgg_layer < vgg_blocks; vgg_layer++)
    {
        int channel_multiplier = 2;
        if (GROUP_C(layer_num-1)*GROUPS(layer_num-1)>= 512)
        {
            channel_multiplier = 1;
        }

        REDUCTION_C(layer_num) = GROUP_C(layer_num-1)*GROUPS(layer_num-1);
        GROUP_C(layer_num) = GROUP_C(layer_num-1)*GROUPS(layer_num-1)*channel_multiplier;
        GROUPS(layer_num) = 1;
        REDUCTION_HW(layer_num) = 3;
        STRIDE(layer_num) = 1;
        small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                            STRIDE(layer_num), t_pad, b_pad);
        small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                            STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
        layer_num++; // 1
        intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));

        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_1 =
            (inter_dim > max_numel_inter_1)? inter_dim : max_numel_inter_1;
        //conv 2
        REDUCTION_C(layer_num) = GROUP_C(layer_num-1)*GROUPS(layer_num-1);
        GROUP_C(layer_num) = GROUP_C(layer_num-1)*GROUPS(layer_num-1);
        GROUPS(layer_num) = 1;
        REDUCTION_HW(layer_num) = 3;
        STRIDE(layer_num) = 1;
        small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                            STRIDE(layer_num), t_pad, b_pad);
        small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                            STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
        layer_num++; // 1
        intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_0 =
            (inter_dim > max_numel_inter_0)? inter_dim : max_numel_inter_0;


        if(num_convs[vgg_layer]==3)
        {
            //conv 3
            REDUCTION_C(layer_num) = GROUP_C(layer_num-1)*GROUPS(layer_num-1);
            GROUP_C(layer_num) = GROUP_C(layer_num-1)*GROUPS(layer_num-1);
            GROUPS(layer_num) = 1;
            REDUCTION_HW(layer_num) = 3;
            STRIDE(layer_num) = 1;
            small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                                STRIDE(layer_num), t_pad, b_pad);
            small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                                STRIDE(layer_num), l_pad, r_pad);
            SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
            layer_num++; // 1
            intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
            inter_dim = INPUT_NUMEL(layer_num);
            max_numel_inter_1 =
                (inter_dim > max_numel_inter_1)? inter_dim : max_numel_inter_1;
        }
        //pooling layer
        REDUCTION_C(layer_num) = 1;
        GROUP_C(layer_num) = 1;
        GROUPS(layer_num) = GROUP_C(layer_num - 1);
        REDUCTION_HW(layer_num) = 3;
        STRIDE(layer_num) = 2;
        small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                            STRIDE(layer_num), t_pad, b_pad);
        small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                            STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);
        layer_num++; // 1
        intermediate_dims.push_back(std::array<uint, 2>(OUTPUT_DIMS(layer_num)));
        inter_dim = INPUT_NUMEL(layer_num);
        //if there was a 3rd conv, then we need to update max_numel_inter_0
        if(num_convs[vgg_layer]==3)
        {
            max_numel_inter_0 =
                (inter_dim > max_numel_inter_0)? inter_dim : max_numel_inter_0;
        }
        else
        {
            max_numel_inter_1 =
                (inter_dim > max_numel_inter_1)? inter_dim : max_numel_inter_1;
        }
        // max_numel_inter_0 =
        //     (inter_dim > max_numel_inter_0)? inter_dim : max_numel_inter_0;

        //switch the max counters for 0 and 1 as needed, so that we start with max_numel_inter_1 at the top of the loop
        if(num_convs[vgg_layer]==3)
        {
            std::swap(max_numel_inter_0, max_numel_inter_1);
        }
    }

    auto layer_num_total = layer_num;  /// @todo is this (-1) right?

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
        printf("\b\b");
        // printf("input dims: %d %d ", I_HEIGHT(i+1), I_WIDTH(i+1));
        printf("\n");
    }
    // print the size of the 2 intermediate buffers
    printf("Max numel inter 0: %d\n", max_numel_inter_0);
    printf("Max numel inter 1: %d\n", max_numel_inter_1);
#endif

    //  Copy layer weights to temporaries
    std::vector<BufferT *> filter_buf_ptrs;

    for (auto l = 0; l < layer_num_total; l++)
    {
        uint32_t filter_dimensions =
            REDUCTION_HW(l) * REDUCTION_HW(l) * REDUCTION_C(l) *
            GROUP_C(l) * GROUPS(l);

        //SKIP POOLING LAYERS
        if (GROUP_C(l) == 1)
        {
            //HACK: putting a single value into the filter buffer so that the indexing matches up with layer numbers
            filter_dimensions = 0;
            // continue;
        }
        BufferT *filter_buf_ptr =
            new BufferT(filter_dimensions);
        init(*filter_buf_ptr, filter_dimensions);
        filter_buf_ptrs.push_back(filter_buf_ptr);
    }
    //print size of filter_buf_ptrs
    printf("Size of filter_buf_ptrs: %d\n", filter_buf_ptrs.size());

    uint32_t filter_dimensions = GROUP_C(layer_num_total-1) * num_classes;
    BufferT *filter_fc_dc_ptr =
        new BufferT(filter_dimensions);
    init(*filter_fc_dc_ptr, filter_dimensions);
    filter_buf_ptrs.push_back(filter_fc_dc_ptr);

    BufferT inter_0_dc(max_numel_inter_0);
    BufferT inter_0_unfused_dc(max_numel_inter_0);
    BufferT inter_0_ewise_dc(max_numel_inter_0);
    BufferT inter_1_dc(max_numel_inter_1);
    BufferT inter_0_buffer_dc(max_numel_inter_0);




    //Test implementations of vggnet_block by running just the first block and comparing outputs
    //======================================================
    //First layer is common to all 3 impl
    small::Conv2D_ReLU(REDUCTION_HW(0), REDUCTION_HW(0),
                       STRIDE(0), PADDING(0),
                       GROUP_C(0), REDUCTION_C(0),
                       I_HEIGHT(0), I_WIDTH(0),
                       input_dc,
                       *filter_buf_ptrs[0],
                       inter_0_dc);

    //======================================================    
    //unfused vgg_block output
    //======================================================
    vgg_block(intermediate_dims[1], REDUCTION_C(1), // Input dimensions
    REDUCTION_HW(1),
    STRIDE(1),
    PADDING(1),
    GROUP_C(1),
    REDUCTION_HW(2),
    STRIDE(2),
    PADDING(2),
    inter_0_dc,
    *filter_buf_ptrs[1],
    inter_1_dc,
    inter_0_unfused_dc);

    //======================================================
    // ewise fused vgg block output
    //======================================================
    fused_ewise_vgg_block(intermediate_dims[1], REDUCTION_C(1), // Input dimensions
    REDUCTION_HW(1),
    STRIDE(1),
    PADDING(1),
    GROUP_C(1),
    REDUCTION_HW(2),
    STRIDE(2),
    PADDING(2),
    inter_0_dc,
    *filter_buf_ptrs[1],
    inter_1_dc,
    inter_0_ewise_dc);
  
    //======================================================
    // fused vgg block output
    //======================================================
    fused_vgg_block(intermediate_dims[1], REDUCTION_C(1), // Input dimensions
    REDUCTION_HW(1),
    STRIDE(1),
    PADDING(1),
    GROUP_C(1),
    REDUCTION_HW(2),
    STRIDE(2),
    PADDING(2),
    inter_0_dc,
    *filter_buf_ptrs[1],
    inter_1_dc,
    inter_0_buffer_dc);

    //======================================================
    //check the 2 fused outputs against unfused
    //======================================================
    bool check = 1;
    printf("%d", intermediate_dims[3][0]*intermediate_dims[3][1]*REDUCTION_C(3));
    CORRECTNESS_CHECK(check, inter_0_unfused_dc, inter_0_ewise_dc, intermediate_dims[3][0]*intermediate_dims[3][1]*GROUP_C(2)); 
    assert(check);
    CORRECTNESS_CHECK(check, inter_0_unfused_dc, inter_0_buffer_dc, intermediate_dims[3][0]*intermediate_dims[3][1]*GROUP_C(2));
    assert(check);

   
    //======================================================
    // End-to-End models
    //======================================================

    //Check that end-to-end unfused model works
    auto &output_dc =
        model_inference(layer_num_total, layer_params, vgg_blocks, num_convs, 
                        intermediate_dims,
                        filter_buf_ptrs,
                        input_dc, inter_0_dc, inter_1_dc);


    auto output_ewise = fused_ewise_model_inference(layer_num_total, layer_params, vgg_blocks, num_convs, 
                        intermediate_dims,
                        filter_buf_ptrs,
                        input_dc, inter_0_dc, inter_1_dc);

    bool check_model = 1;
    CORRECTNESS_CHECK(check_model, output_dc, output_ewise, 7*7*512);
    assert(check_model);


    auto output_fused = fused_model_inference(layer_num_total, layer_params, vgg_blocks, num_convs, 
                        intermediate_dims,
                        filter_buf_ptrs,
                        input_dc, inter_0_dc, inter_1_dc, inter_0_buffer_dc);
    check_model = 1;
    CORRECTNESS_CHECK(check_model, output_dc, output_fused, 7*7*512);
    assert(check_model);

//======================================================
//Run all 3 models and compare timings
//======================================================
    small::Timer my_timer;
    double min_small = std::numeric_limits<double>::max();
    std::vector<double> small_timing;
    std::cout << "Performing timing "<< RUNS <<" runs unfused ...\n";
    for (int r = 0; r < RUNS; r++)
    {
        my_timer.start();

        for (int i = 0; i < TRIALS; i++)
        {
            model_inference(layer_num_total, layer_params, vgg_blocks, num_convs, 
                            intermediate_dims,
                            filter_buf_ptrs,
                            input_dc, inter_0_dc, inter_1_dc);
        }
        my_timer.stop();
        auto diff = my_timer.elapsed() / TRIALS;
        min_small = std::min<double>(min_small, diff);
        small_timing.push_back(diff);
    }

    std::cout << "Minimum time: " << min_small << " ns.\n";

    double min_small_fused_ewise = std::numeric_limits<double>::max();
    std::vector<double> small_fused_ewise_timing;
    std::cout << "Performing timing runs ewise layers fused...\n";
    for (int r = 0; r < RUNS; r++)
    {
        my_timer.start();
        for(int i = 0; i < TRIALS; i++)
        {
        fused_ewise_model_inference(layer_num_total, layer_params, vgg_blocks, num_convs, 
                              intermediate_dims,
                              filter_buf_ptrs,
                              input_dc, inter_0_dc, inter_1_dc);
        }
        my_timer.stop();
        auto diff = my_timer.elapsed() / TRIALS;
        min_small_fused_ewise = std::min<double>(min_small_fused_ewise, diff);
        small_fused_ewise_timing.push_back(diff);
    }

    std::cout << "Minimum time: " << min_small_fused_ewise << " ns.\n";

    double min_small_fused = std::numeric_limits<double>::max();
    std::vector<double> small_fused_timing;
    std::cout << "Performing timing runs fused...\n";
    for (int r = 0; r < RUNS; r++)
    {
        my_timer.start();
        for(int i = 0; i < TRIALS; i++)
        {
        fused_model_inference(layer_num_total, layer_params, vgg_blocks, num_convs, 
                              intermediate_dims,
                              filter_buf_ptrs,
                              input_dc, inter_0_dc, inter_1_dc, inter_0_buffer_dc);
        }
        my_timer.stop();
        auto diff = my_timer.elapsed() / TRIALS;
        min_small_fused = std::min<double>(min_small_fused, diff);
        small_fused_timing.push_back(diff);
    }

    std::cout << "Minimum time: " << min_small_fused << " ns.\n";
    

    #if TIME_LAYER
   
    std::cout << "Performing timing "<< RUNS <<" runs unfused ...\n";
    for (int r = 0; r < RUNS; r++)
    {
       
        int impl = 0;

        for (int i = 0; i < TRIALS; i++)
        {
            model_inference(layer_num_total, layer_params, vgg_blocks, num_convs, intermediate_dims,
                                        filter_buf_ptrs,
                                        input_dc, inter_0_dc, inter_1_dc);
            for (int timer = 0; timer < layer_num_total; timer++)
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
        for (int timer = 0; timer < layer_num_total; timer++)
        {
            avg_layer_timers[impl][timer] /= TRIALS;
            if (0 < r)
            {
                min_layer_timers[impl][timer] = std::min<double>(min_layer_timers[impl][timer], avg_layer_timers[impl][timer]);
            }
            else
            {
                min_layer_timers[impl][timer] = avg_layer_timers[impl][timer];
            }
        }
    }



    std::cout << "Performing timing runs ewise layers fused...\n";
    for (int r = 0; r < RUNS; r++)
    {
       
        int impl = 1;

        for (int i = 0; i < TRIALS; i++)
        {
            fused_ewise_model_inference(layer_num_total, layer_params, vgg_blocks, num_convs, intermediate_dims,
                                  filter_buf_ptrs,
                                  input_dc, inter_0_dc, inter_1_dc);
            for (int timer = 0; timer < layer_num_total; timer++)
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
        for (int timer = 0; timer < layer_num_total; timer++)
        {
            avg_layer_timers[impl][timer] /= TRIALS;
            if (0 < r)
            {
                min_layer_timers[impl][timer] = std::min<double>(min_layer_timers[impl][timer], avg_layer_timers[impl][timer]);
            }
            else
            {
                min_layer_timers[impl][timer] = avg_layer_timers[impl][timer];
            }
        }
      
        // small_fused_ewise_timing.push_back(diff);
    }


    std::cout << "Performing timing runs fused block...\n";
    for (int r = 0; r < RUNS; r++)
    {


        int impl = 2;
        for (int i = 0; i < TRIALS; i++) {
            fused_model_inference(layer_num_total, layer_params, vgg_blocks, num_convs, intermediate_dims,
                                  filter_buf_ptrs,
                                  input_dc, inter_0_dc, inter_1_dc, inter_0_buffer_dc);
            for (int timer = 0; timer < layer_num_total; timer++){
                if (0 < i) {
                    avg_layer_timers[impl][timer] += layer_timers[impl][timer];
                } else {
                    avg_layer_timers[impl][timer] = layer_timers[impl][timer];
                }
            }
        }
        for (int timer = 0; timer < layer_num_total; timer++)
        {
            avg_layer_timers[impl][timer] /= TRIALS;
            if (0 < r) {
                min_layer_timers[impl][timer] = std::min<double>(min_layer_timers[impl][timer], avg_layer_timers[impl][timer]);
            } else {
                min_layer_timers[impl][timer] = avg_layer_timers[impl][timer];

            }
        }
  
        // small_fused_timing.push_back(diff);
    }



    
    double sum_unfused = 0, sum_ewise = 0, sum_fused = 0;
    for(int layer = 0 ; layer < layer_num_total; layer++)
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

//     printf("sum , %f, %f, %f \n", mins_small, min_small_fused_ewise, min_small_fused);


//     int num_th = 1;
// #if PARALLEL == 1
//     char const *env_nt(std::getenv("OMP_NUM_THREADS"));
//     if (nullptr != env_nt)
//     {
//         num_th = atoi(std::getenv("OMP_NUM_THREADS"));
//     }
// #endif
//     std::cout << "Num Threads: " << num_th << std::endl;
// #if PARALLEL_DIST == ELEMENTAL
//     printf("ELEMENTAL\n");
// #else
//     printf("BLOCK\n");
// #endif
//     // print_stats(small_fused_timing, "\nSMaLL:mobilenet");



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
