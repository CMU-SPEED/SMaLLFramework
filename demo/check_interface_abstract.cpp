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
#include <stdint.h>
#include <stdexcept>
#include <type_traits>

#include <params.h>
// #ifdef FLOAT_W_ob 
// #undef FLOAT_W_ob
// #endif
// #define FLOAT_W_ob 1


#include <small/op_type.hpp>

// reference only supports UNROLL==1, undef and redef FLOAT_UNROLLL
#ifdef FLOAT_UNROLL
#undef FLOAT_UNROLL
#endif

#define FLOAT_UNROLL 1
// todo: add support for UNROLL in the macros

#if defined(SMALL_HAS_FLOAT_SUPPORT)
#include <small/platforms/reference/intrinsics_float.h>
// #include <small/platforms/reference/FloatBuffer.hpp>
#include <small/abstract_layer.hpp> /// @todo abstract_layer_float.hpp
#endif

// #if defined(SMALL_HAS_QUINT8_SUPPORT)
// #include <small/q_abstract_layer.hpp> /// @todo abstract_layer_quint8.hpp
// #endif

// #define RECORD_CALLS

#if defined(RECORD_CALLS)
#include <iostream>
#endif

// ================== Public API ====================
#include <small/utils.hpp>
#include <small/buffers.hpp>
#include "check_interface.h"

// #include <small/interface_abstract.hpp>

/// @todo Currently only works with FloatBuffer from reference

//****************************************************************************

extern "C++" {

template <typename BufferT=small::FloatBuffer>
void check_Conv2D(int kernel_height, int kernel_width, int stride,
                  uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                  int output_channels, int input_channels,
                  int input_height, int input_width,
                  BufferT const &input_buf,
                  BufferT const &filter_buf,
                  BufferT       &output_buf)
{    //Specific case for the first layer


    if (input_channels == 3)
    {
        if (stride == 1)
        {
            small::detail::abstract_layer<BufferT, 1, FLOAT_C_ob, 3, FLOAT_W_ob, 1, 1, small::OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {

            small::detail::abstract_layer<BufferT, 1, FLOAT_C_ob, 3, FLOAT_W_ob, 2, 1, small::OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            printf("This stride is unsupported, please change the interface.cpp file\n");
        }
   
    }
    else
    {
        if (stride == 1)
        {
            small::detail::abstract_layer<BufferT, 1, FLOAT_C_ob, FLOAT_C_ob, FLOAT_W_ob, 1, 1, small::OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        // else if (stride == 2)
        // {

        //     small::detail::abstract_layer<BufferT, 1, FLOAT_C_ob, FLOAT_C_ob, FLOAT_W_ob, 2, 1, small::OP_CONV, 2, 1>(
        //         1,               // Output Channel Grouping
        //         output_channels, // Output Channels per group
        //         input_channels,
        //         input_height, input_width,
        //         kernel_height, kernel_width,
        //         t_pad, l_pad, r_pad, b_pad,
        //         &input_buf, &filter_buf, &output_buf);
        // }
        
        else
        {
            printf("This stride is unsupported, please change the interface.cpp file\n");
        }
    }

}

//****************************************************************************
template <typename BufferT = small::FloatBuffer>
void check_PartialConv2D(int kernel_height, int kernel_width, int stride,
                         uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                         int output_channels, int input_channels,
                         int input_height, int input_width,
                         BufferT const &input_buf,
                         BufferT const &filter_buf,
                         BufferT &output_buf)
{
    // Specific case for the first layer
    if (input_channels == 3)
    {
        if (stride == 1)
        {
            small::detail::abstract_layer<BufferT, 1, FLOAT_C_ob, 3, FLOAT_W_ob, 1, 1, small::OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            small::detail::abstract_layer<BufferT, 1, FLOAT_C_ob, 3, FLOAT_W_ob, 2, 1, small::OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            printf("This stride is unsupported, please change the interface.cpp file\n");
        }
    }
    else
    {
        if (stride == 1)
        {
            small::detail::abstract_layer<BufferT, 1, FLOAT_C_ob, FLOAT_C_ob, FLOAT_W_ob, 1, 1, small::OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            small::detail::abstract_layer<BufferT, 1, FLOAT_C_ob, FLOAT_C_ob, FLOAT_W_ob, 2, 1, small::OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            printf("This stride is unsupported, please change the interface.cpp file\n");
        }
    }
}

//****************************************************************************
template <typename BufferT=small::FloatBuffer>
void check_MaxPool2D(int kernel_height, int kernel_width, int stride,
                     uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                     int input_channels,
                     int input_height, int input_width,
                     BufferT const &input_buf,
                     BufferT       &output_buf)
{
    printf("Maxpool stride: %d %d %d\n", stride, FLOAT_W_ob, FLOAT_C_ob);
    
    if (stride == 1)
    {
        small::detail::abstract_layer<BufferT, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, small::OP_MAX_POOL, 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_height, kernel_width,
            t_pad, l_pad, r_pad, b_pad,
            &input_buf, (BufferT const *)nullptr, &output_buf);
    }
    else if (stride == 2)
    {
        printf("stride 2\n");
        small::detail::abstract_layer<BufferT, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 2, 1, small::OP_MAX_POOL, 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_height, kernel_width,
            t_pad, l_pad, r_pad, b_pad,
            &input_buf, (BufferT const *)nullptr, &output_buf);
    }
    else
    {
        printf("This stride is unsupported, please change the interface.cpp file\n");
    }
}

//****************************************************************************
template <typename BufferT = small::FloatBuffer>
void check_DepthwiseConv2D(int kernel_height, int kernel_width, int stride,
                           uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                           int input_channels,
                           int input_height, int input_width,
                           BufferT const &input_buf,
                           BufferT const &filter_buf,
                           BufferT &output_buf)
{
    if (stride == 1)
    {
        small::detail::abstract_layer<BufferT, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, small::OP_CONV, 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_height, kernel_width,
            t_pad, l_pad, r_pad, b_pad,
            &input_buf, &filter_buf, &output_buf);
    }
    else if (stride == 2)
    {
        small::detail::abstract_layer<BufferT, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 2, 1, small::OP_CONV, 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_height, kernel_width,
            t_pad, l_pad, r_pad, b_pad,
            &input_buf, &filter_buf, &output_buf);
    }
    else
    {
        printf("This stride is unsupported, please change the interface.cpp file\n");
    }
}

// //****************************************************************************
template <typename BufferT=small::FloatBuffer>
void check_ReLUActivation(int input_channels,
                          int input_height, int input_width,
                          BufferT const &input_buf,
                          BufferT       &output_buf)
{
    // printf("Cob = %d FLOAT_W_ob = %d\n", FLOAT_C_ob, FLOAT_W_ob);
    small::detail::abstract_layer<BufferT, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, small::OP_RELU, 0, 1>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &input_buf, (BufferT const *)nullptr, &output_buf);
}

// //****************************************************************************
// template <typename BufferT=small::FloatBuffer>
// void check_Dense(int output_elements, int input_elements,
//                  BufferT const &input_buf,
//                  BufferT const &filter_buf,
//                  BufferT       &output_buf)
// {
//     small::detail::abstract_layer<BufferT, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, small::OP_CONV, 1, 1>(
//         output_elements, // Output Channel Grouping
//         1,              // Output Channels per group
//         1,
//         1, input_elements,
//         1, 1,
//         0, 0, 0, 0,
//         &input_buf, &filter_buf, &output_buf);
// }
auto check_DepthwiseConv2D_float = check_DepthwiseConv2D<small::FloatBuffer>;

auto check_Conv2D_float = check_Conv2D<small::FloatBuffer>;
// auto check_PartialConv2D_float = check_PartialConv2D<small::FloatBuffer>;
auto check_MaxPool2D_float = check_MaxPool2D<small::FloatBuffer>;
auto check_ReLUActivation_float = check_ReLUActivation<small::FloatBuffer>;
// auto check_Dense_float = check_Dense<small::FloatBuffer>;

} // extern "C++"

