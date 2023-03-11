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
// #include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

//****************************************************************************
// Recreate include order of small.h but hijack interface.hpp, intrinsics.h
// and interface_abstract.hpp (the contents of this file)

// ============ Implementation details ==============
// Platform specific includes.
// Use -I compile option to point to correct platform
#include <params.h>
#include <Buffer.hpp>

/// @todo this code currently does not support quantized.
#if defined(QUANTIZED)
#include <../quantized_reference/intrinsics.h>
using Buffer = small::QUInt8Buffer;
#else
#include <../reference/intrinsics.h>  // hardcode reference intrinsics ICK
using Buffer = small::FloatBuffer;
#endif

// ================== Public API ====================
#include <small/utils.hpp>
#include <small/buffers.hpp>
#include "check_interface.h"

/// This must come after platform-specific includes.
// #include <small/abstract_layer.hpp>

/// @todo push this if into the lower header files
#if defined(QUANTIZED)
#include <small/q_abstract_layer.hpp>
#include <small/q_interface_abstract.hpp> // template defs of interface.hpp
#else
#include <small/abstract_layer.hpp>
#include <small/interface_abstract.hpp>
#endif

/// @todo Currently only works with FloatBuffer from reference

//****************************************************************************
template <>
void check_Conv2D(int kernel_size, int stride,
                  uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                  int output_channels, int input_channels,
                  int input_height, int input_width,
                  Buffer const &input_buf,
                  Buffer const &filter_buf,
                  Buffer       &output_buf)
{
    //Specific case for the first layer
    if (input_channels == 3)
    {
        if (stride == 1)
        {
            small::detail::abstract_layer<Buffer, 1, C_ob, 3, W_ob, 1, 1, 'c', 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {

            small::detail::abstract_layer<Buffer, 1, C_ob, 3, W_ob, 2, 1, 'c', 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
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
            small::detail::abstract_layer<Buffer, 1, C_ob, C_ob, W_ob, 1, 1, 'c', 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {

            small::detail::abstract_layer<Buffer, 1, C_ob, C_ob, W_ob, 2, 1, 'c', 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
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
template <>
void check_PartialConv2D(int kernel_size, int stride,
                         uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                         int output_channels, int input_channels,
                         int input_height, int input_width,
                         Buffer const &input_buf,
                         Buffer const &filter_buf,
                         Buffer       &output_buf)
{
    // Specific case for the first layer
    if (input_channels == 3)
    {
        if (stride == 1)
        {
            small::detail::abstract_layer<Buffer, 1, C_ob, 3, W_ob, 1, 1, 'c', 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            small::detail::abstract_layer<Buffer, 1, C_ob, 3, W_ob, 2, 1, 'c', 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
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
            small::detail::abstract_layer<Buffer, 1, C_ob, C_ob, W_ob, 1, 1, 'c', 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            small::detail::abstract_layer<Buffer, 1, C_ob, C_ob, W_ob, 2, 1, 'c', 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
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
template <>
void check_Maxpool2D(int kernel_size, int stride,
                     uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                     int input_channels,
                     int input_height, int input_width,
                     Buffer const &input_buf,
                     Buffer       &output_buf)
{
    if (stride == 1)
    {
        small::detail::abstract_layer<Buffer, C_ob, 1, 1, W_ob, 1, 1, 'p', 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_size, kernel_size,
            t_pad, l_pad, r_pad, b_pad,
            &input_buf, (Buffer const *)nullptr, &output_buf);
    }
    else if (stride == 2)
    {
        small::detail::abstract_layer<Buffer, C_ob, 1, 1, W_ob, 2, 1, 'p', 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_size, kernel_size,
            t_pad, l_pad, r_pad, b_pad,
            &input_buf, (Buffer const *)nullptr, &output_buf);
    }
    else
    {
        printf("This stride is unsupported, please change the interface.cpp file\n");
    }
}

//****************************************************************************
template <>
void check_DepthwiseConv2D(int kernel_size, int stride,
                           uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                           int input_channels,
                           int input_height, int input_width,
                           Buffer const &input_buf,
                           Buffer const &filter_buf,
                           Buffer       &output_buf)
{
    if (stride == 1)
    {
        small::detail::abstract_layer<Buffer, C_ob, 1, 1, W_ob, 1, 1, 'c', 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_size, kernel_size,
            t_pad, l_pad, r_pad, b_pad,
            &input_buf, &filter_buf, &output_buf);
    }
    else if (stride == 2)
    {
        small::detail::abstract_layer<Buffer, C_ob, 1, 1, W_ob, 2, 1, 'c', 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_size, kernel_size,
            t_pad, l_pad, r_pad, b_pad,
            &input_buf, &filter_buf, &output_buf);
    }
    else
    {
        printf("This stride is unsupported, please change the interface.cpp file\n");
    }
}

//****************************************************************************
template <>
void check_ReLUActivation(int input_channels,
                          int input_height, int input_width,
                          Buffer const &input_buf,
                          Buffer       &output_buf)
{
    // printf("Cob = %d W_ob = %d\n", C_ob, W_ob);
    small::detail::abstract_layer<Buffer, C_ob, 1, 1, W_ob, 1, 1, 'a', 0, 1>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &input_buf, (Buffer const *)nullptr, &output_buf);
}

//****************************************************************************
template <>
void check_Dense(int output_elements, int input_elements,
                 Buffer const &input_buf,
                 Buffer const &filter_buf,
                 Buffer       &output_buf)
{
    small::detail::abstract_layer<Buffer, C_ob, 1, 1, W_ob, 1, 1, 'c', 1, 1>(
        output_elements, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        1, input_elements,
        1, 1,
        0, 0, 0, 0,
        &input_buf, &filter_buf, &output_buf);
}
