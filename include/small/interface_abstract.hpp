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

#pragma once

#include <math.h>
#include <stdint.h>
#include <stdexcept>
#include <type_traits>

#if defined(SMALL_HAS_FLOAT_SUPPORT)
#include <small/abstract_layer.hpp> /// @todo abstract_layer_float.hpp
#endif

#if defined(SMALL_HAS_QUINT8_SUPPORT)
#include <small/q_abstract_layer.hpp> /// @todo abstract_layer_quint8.hpp
#endif

// #define RECORD_CALLS

#if defined(RECORD_CALLS)
#include <iostream>
#endif

/// @todo Consider replacing exceptions with debug asserts

namespace small
{

//****************************************************************************
//****************************************************************************
template <class BufferT>
void Conv2D(int kernel_height, int kernel_width, int stride,
            uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
            int output_channels, int input_channels,
            int input_height, int input_width,
            BufferT const &input_buf,
            BufferT const &filter_buf,
            BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

// Other options for not defining the base template:
// 1. use "= delete;"
// 2. use the following:
// {
//     static_assert(sizeof(BufferT) == 0,
//                   "Only specializations of Conv2D can be used.");
// }

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void Conv2D<FloatBuffer>(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer const &filter_buf,
    FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Conv2D<float>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif
    // Use a different kernel if the input width is not a multiple of FLOAT_W_ob
    // if(((input_width % 5 == 0) || (input_width % 4 == 0)) && (input_width % FLOAT_W_ob != 0))
    // {
    //     if(input_width % 5 == 0)
    //     {
    //         config_Conv2D<5>(kernel_height, kernel_width, stride, t_pad, b_pad, l_pad, r_pad, output_channels, input_channels, input_height, input_width, input_buf, filter_buf, output_buf);
    //     }
    //     else if(input_width % 4 == 0)
    //     {
    //         config_Conv2D<4>(kernel_height, kernel_width, stride, t_pad, b_pad, l_pad, r_pad, output_channels, input_channels, input_height, input_width, input_buf, filter_buf, output_buf);

    //     }
    // }
    // else{

        /// @todo add an assert for invalid numbers of output channels
        ///       (layer classes should be responsible for padding filters).

        if (input_channels % FLOAT_C_ib == 0)
        {
            if (stride == 1)
            {
                detail::abstract_layer<
                    FloatBuffer, 1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob, 1, FLOAT_UNROLL, OP_CONV, 2, 1>(
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
                detail::abstract_layer<
                    FloatBuffer, 1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob, 2, FLOAT_UNROLL, OP_CONV, 2, 1>(
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
                throw std::invalid_argument(
                    "Conv2D<float> ERROR: stride unsupported.");
            }
        }

        // Specific case for the first layer
        else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
        {
            if (stride == 1)
            {
                detail::abstract_layer<
                    FloatBuffer, 1, FLOAT_C_ob, 3,
                    FLOAT_W_ob, 1, 1, OP_CONV, 2, 1>(
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
                detail::abstract_layer<
                    FloatBuffer, 1, FLOAT_C_ob, 3,
                    FLOAT_W_ob, 2, 1, OP_CONV, 2, 1>( // unroll?
                    1,                                     // Output Channel Grouping
                    output_channels,                       // Output Channels per group
                    input_channels,
                    input_height, input_width,
                    kernel_height, kernel_width,
                    t_pad, l_pad, r_pad, b_pad,
                    &input_buf, &filter_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D<float> ERROR: stride unsupported.");
            }
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D<float> ERROR: in_channels unsupported.");
        }

    // }
}


//Test a different kernel

template <int Kernel_W_ob>
void config_Conv2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer const &filter_buf,
    FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Conv2D<float>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif

    /// @todo add an assert for invalid numbers of output channels
    ///       (layer classes should be responsible for padding filters).
    if (input_channels % FLOAT_C_ib == 0)
    {
        if (stride == 1)
        {
            detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, FLOAT_C_ib,
                Kernel_W_ob, 1, FLOAT_UNROLL, OP_CONV, 2, 1>(
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
            detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, FLOAT_C_ib,
                Kernel_W_ob, 2, FLOAT_UNROLL, OP_CONV, 2, 1>(
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
            throw std::invalid_argument(
                "Conv2D<float> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
    {
        if (stride == 1)
        {
            detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 3,
                Kernel_W_ob, 1, 1, OP_CONV, 2, 1>(
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
            detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 3,
                Kernel_W_ob, 2, 1, OP_CONV, 2, 1>( // unroll?
                1,                                // Output Channel Grouping
                output_channels,                  // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D<float> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "Conv2D<float> ERROR: in_channels unsupported.");
    }
}

#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <>
void Conv2D<QUInt8Buffer>(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    QUInt8Buffer const &input_buf,
    QUInt8Buffer const &filter_buf,
    QUInt8Buffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Conv2D<quint8>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif

    /// @todo add an assert for invalid numbers of output channels
    ///       (layer classes should be responsible for padding filters).

    if (input_channels % QUINT8_C_ib == 0)
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer<
                QUInt8Buffer, 1, QUINT8_C_ob, QUINT8_C_ib,
                QUINT8_W_ob, 1, QUINT8_UNROLL, 'c', 2, 1>(
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
            quint8_detail::abstract_layer<
                QUInt8Buffer, 1, QUINT8_C_ob, QUINT8_C_ib,
                QUINT8_W_ob, 2, QUINT8_UNROLL, 'c', 2, 1>(
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
            throw std::invalid_argument(
                "Conv2D<quint8> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 3) && (input_channels < QUINT8_C_ib))
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer<
                QUInt8Buffer, 1, QUINT8_C_ob, 3,
                QUINT8_W_ob, 1, 1, 'c', 2, 1>(
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
            quint8_detail::abstract_layer<
                QUInt8Buffer, 1, QUINT8_C_ob, 3,
                QUINT8_W_ob, 2, 1, 'c', 2, 1>( // unroll?
                1,                             // Output Channel Grouping
                output_channels,               // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D<quint8> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "Conv2D<quint8> ERROR: in_channels unsupported.");
    }
}

#endif

//****************************************************************************
//****************************************************************************
/// @todo add support for rectangular kernels
template <class BufferT>
void PartialConv2D(int kernel_height, int kernel_width, int stride,
                   uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                   int output_channels, int input_channels,
                   int input_height, int input_width,
                   BufferT const &input_buf,
                   BufferT const &filter_buf,
                   BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void PartialConv2D<FloatBuffer>(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer const &filter_buf,
    FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "PartialConv2D<float>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (input_channels % FLOAT_C_ib == 0)
    {
        if (stride == 1)
        {
            detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob, 1, FLOAT_UNROLL, OP_CONV, 2, 0>(
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

            detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob, 2, FLOAT_UNROLL, OP_CONV, 2, 0>(
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
            throw std::invalid_argument(
                "PartialConv2D<float> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
    {
        if (stride == 1)
        {
            detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 3, FLOAT_W_ob, 1, 1, OP_CONV, 2, 0>(
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
            detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 3, FLOAT_W_ob, 2, 1, OP_CONV, 2, 0>(
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
            throw std::invalid_argument(
                "PartialConv2D<float> ERROR: stride unsupported.");
        }
    }

    /// @todo We need another specific case for input_channels==1 (maybe more)

    else
    {
        throw std::invalid_argument(
            "PartialConv2D<float> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <>
void PartialConv2D<QUInt8Buffer>(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    QUInt8Buffer const &input_buf,
    QUInt8Buffer const &filter_buf,
    QUInt8Buffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "PartialConv2D<quint8>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (input_channels % QUINT8_C_ib == 0)
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer<
                QUInt8Buffer, 1, QUINT8_C_ob, QUINT8_C_ib,
                QUINT8_W_ob, 1, QUINT8_UNROLL, 'c', 2, 0>(
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

            quint8_detail::abstract_layer<
                QUInt8Buffer, 1, QUINT8_C_ob, QUINT8_C_ib,
                QUINT8_W_ob, 2, QUINT8_UNROLL, 'c', 2, 0>(
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
            throw std::invalid_argument(
                "PartialConv2D<quint8> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 3) && (input_channels < QUINT8_C_ib))
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer<
                QUInt8Buffer, 1, QUINT8_C_ob, 3, QUINT8_W_ob, 1, 1, 'c', 2, 0>(
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
            quint8_detail::abstract_layer<
                QUInt8Buffer, 1, QUINT8_C_ob, 3, QUINT8_W_ob, 2, 1, 'c', 2, 0>(
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
            throw std::invalid_argument(
                "PartialConv2D<quint8> ERROR: stride unsupported.");
        }
    }

    /// @todo We need another specific case for input_channels==1 (maybe more)

    else
    {
        throw std::invalid_argument(
            "PartialConv2D<quint8> ERROR: in_channels unsupported.");
    }
}
#endif

//****************************************************************************
//****************************************************************************
template <class BufferT>
void MaxPool2D(int kernel_height, int kernel_width, int stride,
               uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
               int input_channels,
               int input_height, int input_width,
               BufferT const &input_buf,
               BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void MaxPool2D<FloatBuffer>(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "MaxPool2D<float>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        if (stride == 1)
        {
            detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MAX_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
        }
        else if (stride == 2)
        {
            detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 2, 1, OP_MAX_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "MaxPool2D<float> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "MaxPool2D<float> ERROR: in_channels unsupported.");
    }
}

template <int Kernel_W_ob>
void config_MaxPool2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "config_MaxPool2D<float>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        if (stride == 1)
        {
            detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, Kernel_W_ob, 1, 1, OP_MAX_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
        }
        else if (stride == 2)
        {
            detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, Kernel_W_ob, 2, 1, OP_MAX_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "config_MaxPool2D<float> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "config_MaxPool2D<float> ERROR: in_channels unsupported.");
    }
}

#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <>
void MaxPool2D<QUInt8Buffer>(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    QUInt8Buffer const &input_buf,
    QUInt8Buffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "MaxPool2D<quint8>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % QUINT8_C_ib == 0)
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer<
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, 'p', 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, (QUInt8Buffer *)nullptr, &output_buf);
        }
        else if (stride == 2)
        {
            quint8_detail::abstract_layer<
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 2, 1, 'p', 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, (QUInt8Buffer *)nullptr, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "MaxPool2D<quint8> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "MaxPool2D<quint8> ERROR: in_channels unsupported.");
    }
}
#endif

//****************************************************************************
//****************************************************************************
template <class BufferT>
void AveragePool2D(int kernel_height, int kernel_width, int stride,
                   uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                   int input_channels,
                   int input_height, int input_width,
                   BufferT const &input_buf,
                   BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void AveragePool2D<FloatBuffer>(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "AveragePool2D<float>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        if (stride == 1)
        {
            detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_AVERAGE_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
        }
        else if (stride == 2)
        {
            detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 2, 1, OP_AVERAGE_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
        }
        else if (stride == 3)
        {
            detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 3, 1, OP_AVERAGE_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                kernel_height, kernel_width,
                t_pad, l_pad, r_pad, b_pad,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "AveragePool2D<float> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "AveragePool2D<float> ERROR: in_channels unsupported.");
    }
}

#endif

//============================================================================

//****************************************************************************
//****************************************************************************
template <class BufferT>
void DepthwiseConv2D(int kernel_size, int stride,
                     uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                     int input_channels,
                     int input_height, int input_width,
                     BufferT const &input_buf,
                     BufferT const &filter_buf,
                     BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void DepthwiseConv2D<FloatBuffer>(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer const &filter_buf,
    FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "DepthwiseConv2D<float>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif
    // if ((input_width % FLOAT_W_ob != 0) && ((input_width % 5 == 0)||(input_width % 4 == 0)))
    // {
    //     if(input_width % 5 == 0)
    //     {
    //         config_DepthwiseConv2D<5>(kernel_height, kernel_width, stride, t_pad, b_pad, l_pad, r_pad, input_channels, input_height, input_width, input_buf, filter_buf, output_buf);
    //     }
    //     else if (input_width % 4 == 0)
    //     {
    //         config_DepthwiseConv2D<4>(kernel_height, kernel_width, stride, t_pad, b_pad, l_pad, r_pad, input_channels, input_height, input_width, input_buf, filter_buf, output_buf);
    //     }

    // }
    // else
    // {
        if (input_channels % FLOAT_C_ib == 0)
        {
            if (stride == 1)
            {
                detail::abstract_layer<
                    FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_CONV, 1, 1>(
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

                detail::abstract_layer<
                    FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 2, 1, OP_CONV, 1, 1>(
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
                throw std::invalid_argument(
                    "DepthwiseConv2D<float> ERROR: stride unsupported.");
            }
        }
        else
        {
            throw std::invalid_argument(
                "DepthwiseConv2D<float> ERROR: in_channels unsupported.");
        }
    // }
}

template <int Kernel_W_ob>
void config_DepthwiseConv2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer const &filter_buf,
    FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "DepthwiseConv2D<float>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (input_channels % FLOAT_C_ib == 0)
    {
        if (stride == 1)
        {
            detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, Kernel_W_ob, 1, 1, OP_CONV, 1, 1>(
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

            detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, Kernel_W_ob, 2, 1, OP_CONV, 1, 1>(
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
            throw std::invalid_argument(
                "DepthwiseConv2D<float> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "DepthwiseConv2D<float> ERROR: in_channels unsupported.");
    }
}



#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <>
void DepthwiseConv2D<QUInt8Buffer>(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    QUInt8Buffer const &input_buf,
    QUInt8Buffer const &filter_buf,
    QUInt8Buffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "DepthwiseConv2D<quint8>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (input_channels % QUINT8_C_ib == 0)
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer<
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, 'c', 1, 1>(
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

            quint8_detail::abstract_layer<
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 2, 1, 'c', 1, 1>(
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
            throw std::invalid_argument(
                "DepthwiseConv2D<quint8> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "DepthwiseConv2D<quint8> ERROR: in_channels unsupported.");
    }
}
#endif

//****************************************************************************
//****************************************************************************
template <class BufferT>
void PartialDepthwiseConv2D(int kernel_size, int stride,
                     uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                     int input_channels,
                     int input_height, int input_width,
                     BufferT const &input_buf,
                     BufferT const &filter_buf,
                     BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void PartialDepthwiseConv2D<FloatBuffer>(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer const &filter_buf,
    FloatBuffer       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "PartialDepthwiseConv2D<float>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (input_channels % FLOAT_C_ib == 0)
    {
        if (stride == 1)
        {
            detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1,
                FLOAT_W_ob, 1, 1, OP_CONV, 1, 0>(
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

            detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1,
                FLOAT_W_ob, 2, 1, OP_CONV, 1, 0>(
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
            throw std::invalid_argument(
                "PartialDepthwiseConv2D<float> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "PartialDepthwiseConv2D<float> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <>
void PartialDepthwiseConv2D<QUInt8Buffer>(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    QUInt8Buffer const &input_buf,
    QUInt8Buffer const &filter_buf,
    QUInt8Buffer       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "PartialDepthwiseConv2D<quint8>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (input_channels % QUINT8_C_ib == 0)
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer<
                QUInt8Buffer, QUINT8_C_ob, 1, 1,
                QUINT8_W_ob, 1, 1, 'c', 1, 0>(
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

            quint8_detail::abstract_layer<
                QUInt8Buffer, QUINT8_C_ob, 1, 1,
                QUINT8_W_ob, 2, 1, 'c', 1, 0>(
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
            throw std::invalid_argument(
                "PartialDepthwiseConv2D<quint8> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "PartialDepthwiseConv2D<quint8> ERROR: in_channels unsupported.");
    }
}
#endif

//****************************************************************************
//****************************************************************************
/// @todo In the original q_interface_abstract.hpp, why was this the only
///       layer with 'zero' param?
template <class BufferT>
void ReLUActivation(int input_channels,
                    int input_height, int input_width,
                    BufferT const &input_buf,
                    BufferT &output_buf) //, int zero = 0)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void ReLUActivation<FloatBuffer>(int input_channels,
                                 int input_height, int input_width,
                                 FloatBuffer const &input_buf,
                                 FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "ReLUActivation<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_RELU, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_buf, (FloatBuffer *)nullptr, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "ReLUActivation<float> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <>
void ReLUActivation<QUInt8Buffer>(int input_channels,
                                  int input_height, int input_width,
                                  QUInt8Buffer const &input_buf,
                                  QUInt8Buffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "ReLUActivation<quint8>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % QUINT8_C_ib == 0)
    {
        quint8_detail::abstract_layer<
            QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, 'a', 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_buf, (QUInt8Buffer *)nullptr, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "ReLUActivation<quint8> ERROR: in_channels unsupported.");
    }
}
#endif

//****************************************************************************
//****************************************************************************
template <class BufferT>
void LeakyReLUActivation(int input_channels,
                         int input_height, int input_width,
                         BufferT const &input_buf,
                         BufferT const &filter_buf,
                         BufferT &output_buf)
{
    // for (auto ix = 0; ix < input_channels*input_height*input_width; ++ix)
    // {
    //     output_buf[ix] = ((input_buf[ix] < 0) ?
    //                       (negative_slope*input_buf[ix]) :
    //                       input_buf[ix]);
    // }

    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void LeakyReLUActivation<FloatBuffer>(int input_channels,
                                      int input_height, int input_width,
                                      FloatBuffer const &input_buf,
                                      FloatBuffer const &filter_buf,
                                      FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "LeakyReLUActivation<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",slope:" << filter_buf.data()[0]
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_LEAKY_RELU, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_buf, &filter_buf, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "LeakyReLUActivation<float> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if 0 // defined(SMALL_HAS_QUINT8_SUPPORT)
template <>
void LeakyReLUActivation<QUInt8Buffer>(int input_channels,
                                       int input_height, int input_width,
                                       QUInt8Buffer const &input_buf,
                                       QUInt8Buffer const &filter_buf,
                                       QUInt8Buffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "LeakyReLUActivation<quint8>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",slope:" << filter_buf.data()[0]
              << ",I,O)\n";
#endif

#pragma message("WARNING: LeakyReLUActivation microkernels not implemented for QUInt8Buffer")

    if (input_channels % QUINT8_C_ib == 0)
    {
        quint8_detail::abstract_layer<
            QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, 'l', 0, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &input_buf, &filter_buf, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "LeakyReLUActivation<quint8> ERROR: in_channels unsupported.");
    }
}
#endif

//****************************************************************************
//****************************************************************************
template <class BufferT>
void Dropout(int input_channels,
             int input_height, int input_width,
             BufferT const &input_buf,
             BufferT const &filter_buf,
             BufferT       &output_buf)
{
    // for (auto ix = 0; ix < input_channels*input_height*input_width; ++ix)
    // {
    //     output_buf[ix] = ((input_buf[ix] < 0) ?
    //                       (negative_slope*input_buf[ix]) :
    //                       input_buf[ix]);
    // }

    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void Dropout<FloatBuffer>(int input_channels,
                          int input_height, int input_width,
                          FloatBuffer const &input_buf,
                          FloatBuffer const &filter_buf,
                          FloatBuffer       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Dropout<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",slope:" << filter_buf.data()[0]
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MUL, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_buf, &filter_buf, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "Dropout<float> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if 0 // defined(SMALL_HAS_QUINT8_SUPPORT)
template <>
void Dropout<QUInt8Buffer>(int input_channels,
                           int input_height, int input_width,
                           QUInt8Buffer const &input_buf,
                           QUInt8Buffer const &filter_buf,
                           QUInt8Buffer       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Dropout<quint8>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",slope:" << filter_buf.data()[0]
              << ",I,O)\n";
#endif

#pragma message("WARNING: Dropout microkernels not implemented for QUInt8Buffer")

    if (input_channels % QUINT8_C_ib == 0)
    {
        quint8_detail::abstract_layer<
            QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, 'm', 0, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &input_buf, &filter_buf, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "Dropout<quint8> ERROR: in_channels unsupported.");
    }
}
#endif

//****************************************************************************
//****************************************************************************
template <class BufferT>
void SoftMax(int input_channels,
             int input_height, int input_width,
             BufferT const &input_buf,
             BufferT       &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void SoftMax<FloatBuffer>(int input_channels,
                          int input_height, int input_width,
                          FloatBuffer const &input_buf,
                          FloatBuffer       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "SoftMax<float>(chans:" << input_channels
                << ",img:" << input_height << "x" << input_width
                << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        // SoftMax is a point wise exponent, global ADD, pointwise multiply

        // pointwise exponent
        detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EXP, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_buf, (FloatBuffer *)nullptr, &output_buf);


        // // global sum
        FloatBuffer softmax_norm_buf(1);
        detail::abstract_layer<
            FloatBuffer, 1, 1, FLOAT_C_ob, FLOAT_W_ob, 1, FLOAT_C_ob, OP_ADD, 3, 1>(
            1, // Output Channel Grouping
            1, // Output Channels per group
            input_channels,
            input_height, input_width,
            input_height, input_width,
            0, 0, 0, 0,
            &output_buf, (FloatBuffer *)nullptr, &softmax_norm_buf);

        std::cout << "Global sum: " << softmax_norm_buf.data()[0]
                  << std::endl;

        // //elementwise scaling
        softmax_norm_buf.data()[0] = 1.0/softmax_norm_buf.data()[0];
        detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MUL, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &output_buf, &softmax_norm_buf, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "SoftMax<float> ERROR: in_channels unsupported.");
    }
}
#endif

//****************************************************************************
//****************************************************************************
// nearest neighbor upsampling
template <typename BufferT>
void UpSample2D(int scale_factor,
                int input_channels,
                int input_height, int input_width,
                BufferT const &input_buf,
                BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void UpSample2D<FloatBuffer>(int scale_factor,
                             int input_channels,
                             int input_height, int input_width,
                             FloatBuffer const &input_buf,
                             FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "UpSample2D<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",scale:" << scale_factor
              << ",I,O)\n";
#endif

    if (scale_factor == 1)
    {
        output_buf = input_buf;
    }
    else if (scale_factor == 2)
    {
        if (input_channels % FLOAT_C_ib == 0)
        {
            detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 2, 1, OP_UPSAMPLE, 0, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Upsample<float> ERROR: in_channels unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "Upsample<float> ERROR: scale factor unsupported (only 1 or 2).");
    }
}
#endif

//============================================================================
#if 0 // defined(SMALL_HAS_QUINT8_SUPPORT)
template <>
void UpSample2D<QUInt8Buffer>(int scale_factor,
                             int input_channels,
                             int input_height, int input_width,
                             QUInt8Buffer const &input_buf,
                             QUInt8Buffer       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "UpSample2D<quint8>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",scale:" << scale_factor
              << ",I,O)\n";
#endif

#pragma message("WARNING: UpSample2D microkernels not implemented for QUInt8Buffer")

    if (scale_factor == 1)
    {
        output_buf = input_buf;
    }
    else if (scale_factor == 2)
    {
        if (input_channels % QUINT8_C_ib == 0)
        {
            quint8_detail::abstract_layer<
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 2, 1, 'u', 0, 1>(
                    input_channels, // Output Channel Grouping
                    1,              // Output Channels per group
                    1,
                    input_height, input_width,
                    1, 1,
                    0, 0, 0, 0,
                    &input_buf, (QUInt8Buffer *)nullptr, &output_buf);
        }
        else
        {
        throw std::invalid_argument(
            "Upsample<quint8> ERROR: in_channels unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "Upsample<quint8> ERROR: scale factor unsupported (only 1 or 2).");
    }
}
#endif

//****************************************************************************
//****************************************************************************
template <class BufferT>
void Accum(int input_channels,
           int input_height, int input_width,
           BufferT const &input_buf,
           BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void Accum<FloatBuffer>(int input_channels,
                        int input_height, int input_width,
                        FloatBuffer const &input_buf,
                        FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Accum<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_buf, (FloatBuffer *)NULL, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "Accum<float> ERROR: in_channels unsupported.");
    }
}
#endif

/// @todo Accum<QUInt8Buffer>(..) implementation

//****************************************************************************
//****************************************************************************
// init a buffer with bias values, 1 per channel
template <class BufferT>
void Bias(int num_channels,
          int output_height, int output_width,
          BufferT const &input_buf,
          BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void Bias<FloatBuffer>(int num_channels,
                       int output_height, int output_width,
                       FloatBuffer const &input_buf,
                       FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Bias<float>(chans:" << num_channels
              << ",img:" << output_height << "x" << output_width
              << ",I,O)\n";
#endif

    if (num_channels % FLOAT_C_ob == 0)
    {
        detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1,
            FLOAT_W_ob, std::numeric_limits<dim_t>::max(), 1, OP_UPSAMPLE, 0, 1>(
            num_channels, // Output Channel Grouping
            1,            // Output Channels per group
            1,
            output_height, output_width,
            1, 1,
            0, 0, 0, 0,
            &input_buf, (FloatBuffer *)nullptr, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "Bias<float> ERROR: in_channels unsupported.");
    }
}
#endif

/// @todo Bias<QUInt8Buffer>(..) implementation

//****************************************************************************
//****************************************************************************
// init a buffer with bias values, 1 per channel
template <class BufferT>
void PartialBias(int num_channels,
                 int output_height, int output_width,
                 BufferT const &input_buf,
                 BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void PartialBias<FloatBuffer>(int num_channels,
                              int output_height, int output_width,
                              FloatBuffer const &input_buf,
                              FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "PartialBias<float>(chans:" << num_channels
              << ",img:" << output_height << "x" << output_width
              << ",I,O)\n";
#endif

    if (num_channels % FLOAT_C_ob == 0)
    {
        detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1,
            FLOAT_W_ob, std::numeric_limits<dim_t>::max(), 1, OP_UPSAMPLE, 0, 0>(
                num_channels, // Output Channel Grouping
                1,            // Output Channels per group
                1,
                output_height, output_width,
                1, 1,
                0, 0, 0, 0,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "PartialBias<float> ERROR: in_channels unsupported.");
    }
}
#endif

/// @todo PartialBias<QUInt8Buffer>(..) implementation

//****************************************************************************
//****************************************************************************
template <class BufferT>
void Concat(uint32_t input0_channels,
            uint32_t input1_channels,
            uint32_t input_height, uint32_t input_width,
            BufferT const &input0_buf,
            BufferT const &input1_buf,
            BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Concat(inchans:" << input0_channels
              << "+" << input1_channels
              << ",img:" << input_height << "x" << input_width
              << ",I1,I2,O)\n";
#endif

    /// @todo check for valid channels values
    /// @todo Write abstract_layer implementation for this?

    // With tensor notation, the function should do the following:
    //    concat( buf1(C1/Cb, H, W, Cb), buf2(C2/Cb, H, W, Cb) )
    //                  ---> buf3((C1+C2)/Cb, H, W, Cb).
    //
    // Since channels is the slowest dimension, I believe that we can
    // just do 2 copies into a large buffer assuming that we just want
    // to concat 2 packed buffers in the channel dimension.
    size_t size0(input0_channels * input_height * input_width);
    std::copy(input0_buf.data(), input0_buf.data() + size0,
              output_buf.data());

    size_t size1(input1_channels * input_height * input_width);
    std::copy(input1_buf.data(), input1_buf.data() + size1,
              output_buf.data() + size0);
}

//****************************************************************************
//****************************************************************************
template <class BufferT>
void Dense(int output_elements, int input_elements,
           BufferT const &input_buf,
           BufferT const &filter_buf,
           BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void Dense<FloatBuffer>(int output_elements, int input_elements,
                        FloatBuffer const &input_buf,
                        FloatBuffer const &filter_buf,
                        FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Dense<float>(outxin:" << output_elements
              << "x" << input_elements << "I,F,O)\n";
#endif
    detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_CONV, 1, 1>(
        output_elements, // Output Channel Grouping
        1,               // Output Channels per group
        1,
        1, input_elements,
        1, 1,
        0, 0, 0, 0,
        &input_buf, &filter_buf, &output_buf);
}
#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <>
void Dense<QUInt8Buffer>(int output_elements, int input_elements,
                         QUInt8Buffer const &input_buf,
                         QUInt8Buffer const &filter_buf,
                         QUInt8Buffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Dense<quint8>(outxin:" << output_elements
              << "x" << input_elements << "I,F,O)\n";
#endif
    quint8_detail::abstract_layer<
        QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, 'c', 1, 1>(
        output_elements, // Output Channel Grouping
        1,               // Output Channels per group
        1,
        1, input_elements,
        1, 1,
        0, 0, 0, 0,
        &input_buf, &filter_buf, &output_buf);
}
#endif

} // small
