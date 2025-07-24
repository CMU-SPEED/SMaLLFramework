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
#include <small/float_detail/abstract_layer.hpp>    /// @todo abstract_layer_float.hpp?
#endif

#if defined(SMALL_HAS_DOUBLE_SUPPORT)
#include <small/double_detail/abstract_layer.hpp>    /// @todo abstract_layer_double.hpp?
#endif

#if defined(SMALL_HAS_QUINT8_SUPPORT)
#include <small/quint8_detail/q_abstract_layer.hpp> /// @todo abstract_layer_quint8.hpp?
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

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void Conv2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
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
            float_detail::abstract_layer<
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
            float_detail::abstract_layer<
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
            float_detail::abstract_layer<
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
            float_detail::abstract_layer<
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

    // Specific case for the first layer
    else if ((input_channels == 2) && (input_channels < FLOAT_C_ib))
    {
        if (stride == 1)
        {
            float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 2,
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
            float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 2,
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

    // Specific case for the first layer
    else if ((input_channels == 1) && (input_channels < FLOAT_C_ib))
    {
        if (stride == 1)
        {
            std::cout << "Debug: Conv2D<float> input_channels == 1, no filter needed.\n";
            float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 1,
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
            float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 1,
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

    /// @todo Do we need other specific cases for input_channels > 3?

    // everything else.
    else
    {
        throw std::invalid_argument(
            "Conv2D<float> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void Conv2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
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
                QUINT8_W_ob, 1, QUINT8_UNROLL, OP_CONV, 2, 1>(
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
                QUINT8_W_ob, 2, QUINT8_UNROLL, OP_CONV, 2, 1>(
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
                QUINT8_W_ob, 1, 1, OP_CONV, 2, 1>(
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
                QUINT8_W_ob, 2, 1, OP_CONV, 2, 1>( // unroll?
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

    // Specific case for the first layer
    else if ((input_channels == 2) && (input_channels < QUINT8_C_ib))
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer<
                QUInt8Buffer, 1, QUINT8_C_ob, 2,
                QUINT8_W_ob, 1, 1, OP_CONV, 2, 1>(
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
                QUInt8Buffer, 1, QUINT8_C_ob, 2,
                QUINT8_W_ob, 2, 1, OP_CONV, 2, 1>( // unroll?
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

    // Specific case for the first layer
    else if ((input_channels == 1) && (input_channels < QUINT8_C_ib))
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer<
                QUInt8Buffer, 1, QUINT8_C_ob, 1,
                QUINT8_W_ob, 1, 1, OP_CONV, 2, 1>(
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
                QUInt8Buffer, 1, QUINT8_C_ob, 1,
                QUINT8_W_ob, 2, 1, OP_CONV, 2, 1>( // unroll?
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

    /// @todo Do we need other specific cases for input_channels > 3?

    // everything else.
    else
    {
        throw std::invalid_argument(
            "Conv2D<quint8> ERROR: in_channels unsupported.");
    }
}

#endif

//****************************************************************************
//****************************************************************************

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void PartialConv2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
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
            float_detail::abstract_layer<
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

            float_detail::abstract_layer<
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
            float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 3,
                FLOAT_W_ob, 1, 1, OP_CONV, 2, 0>(
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
            float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 3,
                FLOAT_W_ob, 2, 1, OP_CONV, 2, 0>(
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
    else if ((input_channels == 2) && (input_channels < FLOAT_C_ib))
    {
        if (stride == 1)
        {
            float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 2,
                FLOAT_W_ob, 1, 1, OP_CONV, 2, 0>(
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
            float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 2,
                FLOAT_W_ob, 2, 1, OP_CONV, 2, 0>(
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
    else if ((input_channels == 1) && (input_channels < FLOAT_C_ib))
    {
        if (stride == 1)
        {
            float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 1,
                FLOAT_W_ob, 1, 1, OP_CONV, 2, 0>(
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
            float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 1,
                FLOAT_W_ob, 2, 1, OP_CONV, 2, 0>(
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

    /// @todo Do we need other specific cases for input_channels > 3?

    else
    {
        throw std::invalid_argument(
            "PartialConv2D<float> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void PartialConv2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
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
                QUINT8_W_ob, 1, QUINT8_UNROLL, OP_CONV, 2, 0>(
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
                QUINT8_W_ob, 2, QUINT8_UNROLL, OP_CONV, 2, 0>(
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
                QUInt8Buffer, 1, QUINT8_C_ob, 3,
                QUINT8_W_ob, 1, 1, OP_CONV, 2, 0>(
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
                QUINT8_W_ob, 2, 1, OP_CONV, 2, 0>(
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
    else if ((input_channels == 2) && (input_channels < QUINT8_C_ib))
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer<
                QUInt8Buffer, 1, QUINT8_C_ob, 2,
                QUINT8_W_ob, 1, 1, OP_CONV, 2, 0>(
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
                QUInt8Buffer, 1, QUINT8_C_ob, 2,
                QUINT8_W_ob, 2, 1, OP_CONV, 2, 0>(
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
    else if ((input_channels == 1) && (input_channels < QUINT8_C_ib))
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer<
                QUInt8Buffer, 1, QUINT8_C_ob, 1,
                QUINT8_W_ob, 1, 1, OP_CONV, 2, 0>(
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
                QUInt8Buffer, 1, QUINT8_C_ob, 1,
                QUINT8_W_ob, 2, 1, OP_CONV, 2, 0>(
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

    /// @todo Do we need other specific cases for input_channels > 3?

    // everything else.
    else
    {
        throw std::invalid_argument(
            "PartialConv2D<quint8> ERROR: in_channels unsupported.");
    }
}
#endif

//****************************************************************************
//****************************************************************************

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void MaxPool2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT &output_buf)
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
            float_detail::abstract_layer<
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
            float_detail::abstract_layer<
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

#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void MaxPool2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT &output_buf)
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
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, OP_MAX_POOL, 1, 1>(
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
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 2, 1, OP_MAX_POOL, 1, 1>(
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

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void AveragePool2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT &output_buf)
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
            float_detail::abstract_layer<
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
            float_detail::abstract_layer<
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
            float_detail::abstract_layer<
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
/// @todo AveragePool1D support for quint8
//============================================================================

//****************************************************************************
//****************************************************************************

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void DepthwiseConv2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
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
            float_detail::abstract_layer<
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

            float_detail::abstract_layer<
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
}
#endif

#if defined(SMALL_HAS_DOUBLE_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<DoubleBuffer, BufferT>::value, bool> = true>
void DepthwiseConv2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "DepthwiseConv2D<double>(k:"
              << kernel_height << "x" << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (input_channels % DOUBLE_C_ib == 0)
    {
        if (stride == 1)
        {
            double_detail::abstract_layer<
                DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_CONV, 1, 1>(
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

            double_detail::abstract_layer<
                DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 2, 1, OP_CONV, 1, 1>(
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
                "DepthwiseConv2D<double> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "DepthwiseConv2D<double> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void DepthwiseConv2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
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
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, OP_CONV, 1, 1>(
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
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 2, 1, OP_CONV, 1, 1>(
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

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void PartialDepthwiseConv2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT       &output_buf)
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
            float_detail::abstract_layer<
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

            float_detail::abstract_layer<
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
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void PartialDepthwiseConv2D(
    int kernel_height, int kernel_width, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT       &output_buf)
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
                QUINT8_W_ob, 1, 1, OP_CONV, 1, 0>(
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
                QUINT8_W_ob, 2, 1, OP_CONV, 1, 0>(
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
// Assumes that output channels = input channels, output groups splits input channels evenly
//****************************************************************************

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void GroupConv2D(
    int kernel_height, int kernel_width, int stride,
    int output_groups,

    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "GroupConv2D<float>(k:"
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
            float_detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_CONV, 2, 1>(
                    output_groups,  // Output Channel Grouping
                    input_channels, // Output Channels per group
                    input_channels,
                    input_height, input_width,
                    kernel_height, kernel_width,
                    t_pad, l_pad, r_pad, b_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {

            float_detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 2, 1, OP_CONV, 2, 1>(
                    output_groups,  // Output Channel Grouping
                    input_channels, // Output Channels per group
                    input_channels,
                    input_height, input_width,
                    kernel_height, kernel_width,
                    t_pad, l_pad, r_pad, b_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "GroupConv2D<float> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "GroupConv2D<float> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void GroupConv2D(
    int kernel_height, int kernel_width, int stride,
    int output_groups,

    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "GroupConv2D<quint8>(k:"
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
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, OP_CONV, 1, 1>(
                    output_groups, // Output Channel Grouping
                    input_channels,              // Output Channels per group
                    input_channels,
                    input_height, input_width,
                    kernel_height, kernel_width,
                    t_pad, l_pad, r_pad, b_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {

            quint8_detail::abstract_layer<
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 2, 1, OP_CONV, 1, 1>(
                    output_groups,  // Output Channel Grouping
                    input_channels, // Output Channels per group
                    input_channels,
                    input_height, input_width,
                    kernel_height, kernel_width,
                    t_pad, l_pad, r_pad, b_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "GroupConv2D<quint8> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "GroupConv2D<quint8> ERROR: in_channels unsupported.");
    }
}
#endif

//****************************************************************************
//****************************************************************************

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void PartialGroupConv2D(
    int kernel_height, int kernel_width, int stride,
    int output_groups,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "PartialGroupConv2D<float>(k:"
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
            float_detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1,
                FLOAT_W_ob, 1, 1, OP_CONV, 2, 0>(
                    output_groups,  // Output Channel Grouping
                    input_channels, // Output Channels per group
                    input_channels,
                    input_height, input_width,
                    kernel_height, kernel_width,
                    t_pad, l_pad, r_pad, b_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {

            float_detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1,
                FLOAT_W_ob, 2, 1, OP_CONV, 2, 0>(
                    output_groups,  // Output Channel Grouping
                    input_channels, // Output Channels per group
                    input_channels,
                    input_height, input_width,
                    kernel_height, kernel_width,
                    t_pad, l_pad, r_pad, b_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "PartialGroupConv2D<float> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "PartialGroupConv2D<float> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void PartialGroupConv2D(
    int kernel_height, int kernel_width, int stride,
    int output_groups,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "PartialGroupConv2D<quint8>(k:"
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
                QUINT8_W_ob, 1, 1, OP_CONV, 1, 0>(
                    output_groups, // Output Channel Grouping
                    input_channels,              // Output Channels per group
                    input_channels,
                    input_height, input_width,
                    kernel_height, kernel_width,
                    t_pad, l_pad, r_pad, b_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {

            quint8_detail::abstract_layer<
                QUInt8Buffer, QUINT8_C_ob, 1, 1,
                QUINT8_W_ob, 2, 1, OP_CONV, 1, 0>(
                    output_groups,  // Output Channel Grouping
                    input_channels, // Output Channels per group
                    input_channels,
                    input_height, input_width,
                    kernel_height, kernel_width,
                    t_pad, l_pad, r_pad, b_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "PartialGroupConv2D<quint8> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "PartialGroupConv2D<quint8> ERROR: in_channels unsupported.");
    }
}
#endif

//****************************************************************************
//****************************************************************************

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void ReLUActivation(int input_channels,
                    int input_height, int input_width,
                    BufferT const &input_buf,
                    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "ReLUActivation<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        float_detail::abstract_layer<
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
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void ReLUActivation(int input_channels,
                    int input_height, int input_width,
                    BufferT const &input_buf,
                    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "ReLUActivation<quint8>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % QUINT8_C_ib == 0)
    {
        quint8_detail::abstract_layer<
            QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, OP_RELU, 0, 1>(
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

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void LeakyReLUActivation(int input_channels,
                         int input_height, int input_width,
                         BufferT const &input_buf,
                         BufferT const &filter_buf,
                         BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "LeakyReLUActivation<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",slope:" << filter_buf.data()[0]
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        float_detail::abstract_layer<
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
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void LeakyReLUActivation(int input_channels,
                         int input_height, int input_width,
                         BufferT const &input_buf,
                         BufferT const &filter_buf,  // @todo quantized?
                         BufferT &output_buf)
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
            QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, OP_LEAKY_RELU, 0, 1>(
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

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void Dropout(int input_channels,
             int input_height, int input_width,
             BufferT const &input_buf,
             BufferT const &filter_buf,
             BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Dropout<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",slope:" << filter_buf.data()[0]
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 1>(
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
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void Dropout(int input_channels,
             int input_height, int input_width,
             BufferT const &input_buf,
             BufferT const &filter_buf,
             BufferT       &output_buf)
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
            QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 1>(
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

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void SoftMax(int input_channels,
             int input_height, int input_width,
             BufferT const &input_buf,
             BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "SoftMax<float>(chans:" << input_channels
                << ",img:" << input_height << "x" << input_width
                << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        // SoftMax is a point wise exponent + global ADD + pointwise multiply

        // point-wise exponent
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EXP, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_buf, (FloatBuffer *)nullptr, &output_buf);

        // global sum
        FloatBuffer softmax_norm_buf(1);
        float_detail::abstract_layer<
            FloatBuffer, 1, 1, FLOAT_C_ob, FLOAT_W_ob, 1, FLOAT_C_ob, OP_ADD, 3, 1>(
            1, // Output Channel Grouping
            1, // Output Channels per group
            input_channels,
            input_height, input_width,
            input_height, input_width,
            0, 0, 0, 0,
            &output_buf, (FloatBuffer *)nullptr, &softmax_norm_buf);

        // element-wise scaling
        softmax_norm_buf.data()[0] = 1.0/softmax_norm_buf.data()[0];
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 1>(
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

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void LogSoftMax(int input_channels,
                int input_height, int input_width,
                BufferT const &input_buf,
                BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "LogSoftMax<float>(chans:" << input_channels
                << ",img:" << input_height << "x" << input_width
                << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        // LogSoftMax is a point-wise ADD of input to a global ADD of point-wise exp

        // point-wise exponent
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EXP, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_buf, (FloatBuffer *)nullptr, &output_buf);

        // global sum
        FloatBuffer softmax_norm_buf(1);
        float_detail::abstract_layer<
            FloatBuffer, 1, 1, FLOAT_C_ob, FLOAT_W_ob, 1, FLOAT_C_ob, OP_ADD, 3, 1>(
            1, // Output Channel Grouping
            1, // Output Channels per group
            input_channels,
            input_height, input_width,
            input_height, input_width,
            0, 0, 0, 0,
            &output_buf, (FloatBuffer *)nullptr, &softmax_norm_buf);

        softmax_norm_buf.data()[0] = -std::log(softmax_norm_buf.data()[0]);

        // element-wise shift
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_ADD_SCALAR, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_buf, &softmax_norm_buf, &output_buf);

    }
    else
    {
        throw std::invalid_argument(
            "SoftMax<float> ERROR: in_channels unsupported.");
    }
}
#endif

//===========================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void SoftSign(int input_channels,
              int input_height, int input_width,
              BufferT const &input_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "SoftSign<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_SOFTSIGN, 0, 1>(
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
            "SoftSign<float> ERROR: in_channels unsupported.");
    }

}
#endif

//===========================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void SoftSign_3Pass(int input_channels,
              int input_height, int input_width,
              BufferT const &input_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "SoftSign<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ABS, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_buf, (FloatBuffer *)nullptr, &output_buf);

        FloatBuffer scalar_buf(1);
        scalar_buf.data()[0] = 1.0f;
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_ADD_SCALAR, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &output_buf, &scalar_buf, &output_buf);

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_DIV, 0, 0>(
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
            "SoftSign<float> ERROR: in_channels unsupported.");
    }

}
#endif

//****************************************************************************
//****************************************************************************
// nearest neighbor upsampling

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void UpSample2D(int scale_factor,
                             int input_channels,
                             int input_height, int input_width,
                             BufferT const &input_buf,
                             BufferT &output_buf)
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
            float_detail::abstract_layer<
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
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void UpSample2D(int scale_factor,
                int input_channels,
                int input_height, int input_width,
                BufferT const &input_buf,
                BufferT       &output_buf)
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
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 2, 1, OP_UPSAMPLE, 0, 1>(
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

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void Accum(int input_channels,
                        int input_height, int input_width,
                        BufferT const &input_buf,
           BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Accum<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        float_detail::abstract_layer<
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

#if defined(SMALL_HAS_DOUBLE_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<DoubleBuffer, BufferT>::value, bool> = true>
void Accum(int input_channels,
                        int input_height, int input_width,
                        BufferT const &input_buf,
           BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Accum<double>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % DOUBLE_C_ib == 0)
    {
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_buf, (DoubleBuffer *)NULL, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "Accum<double> ERROR: in_channels unsupported.");
    }
}
#endif

/// @todo Accum<QUInt8Buffer>(..) implementation

//****************************************************************************
//****************************************************************************
// init a buffer with bias values, 1 per channel

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void Bias(int num_channels,
                       int output_height, int output_width,
                       BufferT const &input_buf,
                       BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Bias<float>(chans:" << num_channels
              << ",img:" << output_height << "x" << output_width
              << ",I,O)\n";
#endif

    if (num_channels % FLOAT_C_ob == 0)
    {
        float_detail::abstract_layer<
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

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void PartialBias(int num_channels,
                 int output_height, int output_width,
                 BufferT const &input_buf,
                 BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "PartialBias<float>(chans:" << num_channels
              << ",img:" << output_height << "x" << output_width
              << ",I,O)\n";
#endif

    if (num_channels % FLOAT_C_ob == 0)
    {
        float_detail::abstract_layer<
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
    /// @todo Write abstract_layer_1D implementation for this?

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

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void Dense(int output_elements, int input_elements,
           BufferT const &input_buf,
           BufferT const &filter_buf,
           BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Dense<float>(out x in:" << output_elements
              << "x" << input_elements << "I,F,O)\n";
#endif
    float_detail::abstract_layer<
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
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void Dense(int output_elements, int input_elements,
           BufferT const &input_buf,
           BufferT const &filter_buf,
           QUInt8Buffer  &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Dense<quint8>(out x in:" << output_elements
              << "x" << input_elements << "I,F,O)\n";
#endif
    quint8_detail::abstract_layer<
        QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, OP_CONV, 1, 1>(
        output_elements, // Output Channel Grouping
        1,               // Output Channels per group
        1,
        1, input_elements,
        1, 1,
        0, 0, 0, 0,
        &input_buf, &filter_buf, &output_buf);
}
#endif

#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void Sub(int input_channels,
              int input_height, int input_width,
              BufferT const &input_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Sub<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if ((input_channels % FLOAT_C_ib == 0))
    {
        float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 1,
                FLOAT_W_ob, 1, 1, OP_SUB, 0, 0>(
                input_channels,               // Output Channel Grouping
                1, // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "Sub<float> ERROR: in_channels unsupported.");
    }

}
#endif

#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void Mul(int input_channels,
              int input_height, int input_width,
              BufferT const &input_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Mul<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if ((input_channels % FLOAT_C_ib == 0))
    {
        float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 1,
                FLOAT_W_ob, 1, 1, OP_MUL, 0, 0>(
                input_channels,               // Output Channel Grouping
                1, // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "Mul<float> ERROR: in_channels unsupported.");
    }

}
#endif

#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void MulScalar(int input_channels,
              int input_height, int input_width,
              BufferT const &input_buf,
              BufferT const &scalar_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "MulScalar<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if ((input_channels % FLOAT_C_ib == 0))
    {
        float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 1,
                FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
                input_channels,               // Output Channel Grouping
                1, // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &input_buf, &scalar_buf, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "MulScalar<float> ERROR: in_channels unsupported.");
    }

}
#endif

#if defined(SMALL_HAS_DOUBLE_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<DoubleBuffer, BufferT>::value, bool> = true>
void MulScalar(int input_channels,
              int input_height, int input_width,
              BufferT const &input_buf,
              BufferT const &scalar_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "MulScalar<double>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if ((input_channels % DOUBLE_C_ib == 0))
    {
        float_detail::abstract_layer<
                DoubleBuffer, 1, DOUBLE_C_ob, 1,
                DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
                input_channels,               // Output Channel Grouping
                1, // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &input_buf, &scalar_buf, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "MulScalar<double> ERROR: in_channels unsupported.");
    }

}
#endif

#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void Div(int input_channels,
              int input_height, int input_width,
              BufferT const &input_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Div<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if ((input_channels % FLOAT_C_ib == 0))
    {
        float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 1,
                FLOAT_W_ob, 1, 1, OP_DIV, 0, 0>(
                input_channels,               // Output Channel Grouping
                1, // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "Div<float> ERROR: in_channels unsupported.");
    }

}
#endif

#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void ConsToPrim(int input_channels,
              int input_height, int input_width,
              int dim, BufferT const &gamma,
              BufferT const &input_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Upwind<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif
    size_t size(input_channels * input_height * input_width);
    // a_W(0) = a_U(0)
    std::copy(input_buf.data(), input_buf.data() + size,
              output_buf.data());
    
    // a_W(1 - dim + 1) = a_U(1 - dim + 1) / a_U(0)
    FloatBuffer v2_buf(input_channels * input_height * input_width);
    init_zeros(v2_buf, v2_buf.size());
    for (int i = 0; i < dim; ++i) 
    {
        FloatBuffer v_buf(input_channels * input_height * input_width);
        std::copy(input_buf.data(), input_buf.data() + size, 
                v_buf.data());
        FloatBuffer input_G_buf(input_channels * input_height * input_width);
        std::copy(input_buf.data() + size * (i + 1), input_buf.data() + size * (i + 2), 
                  input_G_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_DIV, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_G_buf, (FloatBuffer *)nullptr, &v_buf);
        std::copy(v_buf.data(), v_buf.data() + size,
                  output_buf.data() + size * (i + 1));
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MUL, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &v_buf, (FloatBuffer *)nullptr, &v_buf);
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &v_buf, (FloatBuffer *)nullptr, &v2_buf);
    }
    // std::copy(v2_buf.data(), v2_buf.data() + size,
    //           output_buf.data() + size * (dim + 1));
    // a_W(dim + 2) = (a_U(dim + 2) - .5 * rho * v2) * (a_gamma - 1.0) 
    FloatBuffer rho_buf(input_channels * input_height * input_width);
    std::copy(input_buf.data(), input_buf.data() + size,
              rho_buf.data());
    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MUL, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &rho_buf, (FloatBuffer *)nullptr, &v2_buf);
    FloatBuffer scalar_buf(1);
    scalar_buf.data()[0] = 0.5f;
    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &v2_buf, &scalar_buf, &v2_buf);
    
    FloatBuffer output_E_buf(input_channels * input_height * input_width);
    std::copy(input_buf.data() + size * (dim + 1), input_buf.data() + size * (dim + 2),
              output_E_buf.data());
    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_SUB, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &v2_buf, (FloatBuffer *)nullptr, &output_E_buf);
    FloatBuffer gamma_buf(1);
    gamma_buf.data()[0] = gamma[0] - 1.0f;
    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &output_E_buf, &gamma_buf, &output_E_buf);
    
    std::copy(output_E_buf.data(), output_E_buf.data() + size,
              output_buf.data() + size * (dim + 1));
}

#endif 

#if defined(SMALL_HAS_DOUBLE_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<DoubleBuffer, BufferT>::value, bool> = true>
void ConsToPrim(int input_channels,
              int input_height, int input_width,
              int dim, BufferT const &gamma,
              BufferT const &input_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Upwind<double>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif
    size_t size(input_channels * input_height * input_width);
    // a_W(0) = a_U(0)
    std::copy(input_buf.data(), input_buf.data() + size,
              output_buf.data());
    
    // a_W(1 - dim + 1) = a_U(1 - dim + 1) / a_U(0)
    DoubleBuffer v2_buf(input_channels * input_height * input_width);
    init_zeros(v2_buf, v2_buf.size());
    for (int i = 0; i < dim; ++i) 
    {
        DoubleBuffer v_buf(input_channels * input_height * input_width);
        std::copy(input_buf.data(), input_buf.data() + size, 
                v_buf.data());
        DoubleBuffer input_G_buf(input_channels * input_height * input_width);
        std::copy(input_buf.data() + size * (i + 1), input_buf.data() + size * (i + 2), 
                  input_G_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_DIV, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_G_buf, (DoubleBuffer *)nullptr, &v_buf);
        std::copy(v_buf.data(), v_buf.data() + size,
                  output_buf.data() + size * (i + 1));
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_MUL, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &v_buf, (DoubleBuffer *)nullptr, &v_buf);
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &v_buf, (DoubleBuffer *)nullptr, &v2_buf);
    }
    // std::copy(v2_buf.data(), v2_buf.data() + size,
    //           output_buf.data() + size * (dim + 1));
    // a_W(dim + 2) = (a_U(dim + 2) - .5 * rho * v2) * (a_gamma - 1.0) 
    DoubleBuffer rho_buf(input_channels * input_height * input_width);
    std::copy(input_buf.data(), input_buf.data() + size,
              rho_buf.data());
    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_MUL, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &rho_buf, (DoubleBuffer *)nullptr, &v2_buf);
    DoubleBuffer scalar_buf(1);
    scalar_buf.data()[0] = 0.5;
    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &v2_buf, &scalar_buf, &v2_buf);
    
    DoubleBuffer output_E_buf(input_channels * input_height * input_width);
    std::copy(input_buf.data() + size * (dim + 1), input_buf.data() + size * (dim + 2),
              output_E_buf.data());
    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_SUB, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &v2_buf, (DoubleBuffer *)nullptr, &output_E_buf);
    DoubleBuffer gamma_buf(1);
    gamma_buf.data()[0] = gamma[0] - 1.0;
    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &output_E_buf, &gamma_buf, &output_E_buf);
    
    std::copy(output_E_buf.data(), output_E_buf.data() + size,
              output_buf.data() + size * (dim + 1));
}

#endif 

#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void Upwind(int input_channels,
              int input_height, int input_width,
              int dim, int dir, BufferT const &gamma,
              BufferT const &input_low_buf,
              BufferT const &input_high_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Upwind<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        FloatBuffer input_low_rho_buf(input_channels * input_height * input_width);
        FloatBuffer input_high_rho_buf(input_channels * input_height * input_width);
        FloatBuffer input_low_G_buf(input_channels * input_height * input_width * dim);
        FloatBuffer input_high_G_buf(input_channels * input_height * input_width * dim);
        FloatBuffer input_low_E_buf(input_channels * input_height * input_width);
        FloatBuffer input_high_E_buf(input_channels * input_height * input_width);
        FloatBuffer output_rho_buf(input_channels * input_height * input_width);
        FloatBuffer output_G_buf(input_channels * input_height * input_width * dim);
        FloatBuffer output_E_buf(input_channels * input_height * input_width);
        size_t size(input_channels * input_height * input_width);
        std::copy(input_low_buf.data(), input_low_buf.data() + size,
                input_low_rho_buf.data());
        std::copy(input_high_buf.data(), input_high_buf.data() + size,
                input_high_rho_buf.data());
        std::copy(input_low_buf.data() + size, input_low_buf.data() + size * (dim + 1),
                input_low_G_buf.data());
        std::copy(input_high_buf.data() + size, input_high_buf.data() + size * (dim + 1),
                input_high_G_buf.data());
        std::copy(input_low_buf.data() + size * (dim + 1), input_low_buf.data() + size * (dim + 2),
                input_low_E_buf.data());
        std::copy(input_high_buf.data() + size * (dim + 1), input_high_buf.data() + size * (dim + 2),
                input_high_E_buf.data());

        /* rhobar = (rho_low + rho_high) * 0.5 */
        FloatBuffer rhobar_buf(input_channels * input_height * input_width);
        std::copy(input_high_rho_buf.data(), input_high_rho_buf.data() + size,
                rhobar_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_low_rho_buf, (FloatBuffer *)nullptr, &rhobar_buf);

        FloatBuffer scalar_buf(1);
        scalar_buf.data()[0] = 0.5f;
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &rhobar_buf, &scalar_buf, &rhobar_buf);

        /* ubar = (G_low(dir) + G_high(dir)) * 0.5 */
        FloatBuffer ubar_buf(input_channels * input_height * input_width);
        FloatBuffer low_G_buf(input_channels * input_height * input_width);

        std::copy(input_high_G_buf.data() + size * dir, input_high_G_buf.data() + size * (dir + 1), 
                ubar_buf.data());
        
        std::copy(input_low_G_buf.data() + size * dir, input_low_G_buf.data() + size * (dir + 1),
                low_G_buf.data());
        
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &low_G_buf, (FloatBuffer *)nullptr, &ubar_buf);

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &ubar_buf, &scalar_buf, &ubar_buf);


        /* pbar = (E_low(dir) + E_high(dir)) * 0.5 */
        FloatBuffer pbar_buf(input_channels * input_height * input_width);
        std::copy(input_high_E_buf.data(), input_high_E_buf.data() + size,
                pbar_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_low_E_buf, (FloatBuffer *)nullptr, &pbar_buf);

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &pbar_buf, &scalar_buf, &pbar_buf);


        /* cbar = sqrt(gamma * pbar / rhobar) */
        FloatBuffer cbar_buf(input_channels * input_height * input_width);
        std::copy(rhobar_buf.data(), rhobar_buf.data() + size, 
            cbar_buf.data());

        std::copy(rhobar_buf.data(), rhobar_buf.data() + size, 
            output_rho_buf.data());
        
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_DIV, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &pbar_buf, (FloatBuffer *)nullptr, &cbar_buf);


        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &cbar_buf, &gamma, &cbar_buf);

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_SQRT, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &cbar_buf, (FloatBuffer *)nullptr, &cbar_buf);
        

        /* pstar = pbar + rhobar * cbar * udiff */
        FloatBuffer pstar_buf(input_channels * input_height * input_width);
        std::copy(rhobar_buf.data(), rhobar_buf.data() + size, 
                pstar_buf.data());

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MUL, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &cbar_buf, (FloatBuffer *)nullptr, &pstar_buf);
        
        FloatBuffer ustar_buf(input_channels * input_height * input_width);
        std::copy(pstar_buf.data(), pstar_buf.data() + size,
                ustar_buf.data());

        FloatBuffer udiff_buf(input_channels * input_height * input_width);
        FloatBuffer high_G_buf(input_channels * input_height * input_width);
        std::copy(input_high_G_buf.data() + size * dir, input_high_G_buf.data() + size * (dir + 1), 
                high_G_buf.data());
        std::copy(input_low_G_buf.data() + size * dir, input_low_G_buf.data() + size * (dir + 1), 
                udiff_buf.data());

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_SUB, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &high_G_buf, (FloatBuffer *)nullptr, &udiff_buf);

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &udiff_buf, &scalar_buf, &udiff_buf);
        
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MUL, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &udiff_buf, (FloatBuffer *)nullptr, &pstar_buf);
        
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &pbar_buf, (FloatBuffer *)nullptr, &pstar_buf);

        /* ustar = ubar + pdiff / (rho_bar * cbar) */
        

        FloatBuffer pdiff_buf(input_channels * input_height * input_width);
        std::copy(input_low_E_buf.data(), input_low_E_buf.data() + size, 
                pdiff_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_SUB, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_high_E_buf, (FloatBuffer *)nullptr, &pdiff_buf);
        
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &pdiff_buf, &scalar_buf, &pdiff_buf);
        
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_DIV, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &pdiff_buf, (FloatBuffer *)nullptr, &ustar_buf);

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &ubar_buf, (FloatBuffer *)nullptr, &ustar_buf);

        /* Blend rho*/

        FloatBuffer tmp_rho_buf(input_channels * input_height * input_width);
        std::copy(ustar_buf.data(), ustar_buf.data() + size,
                tmp_rho_buf.data());
        
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_SIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_low_rho_buf, (FloatBuffer *)nullptr, &tmp_rho_buf);

        std::copy(ustar_buf.data(), ustar_buf.data() + size,
                output_rho_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_NSIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_high_rho_buf, (FloatBuffer *)nullptr, &output_rho_buf);

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_rho_buf, (FloatBuffer *)nullptr, &output_rho_buf);
        

        FloatBuffer G_low_batch(input_channels * input_height * input_width);
        FloatBuffer G_high_batch(input_channels * input_height * input_width);
        FloatBuffer tmp_G_low_buf(input_channels * input_height * input_width);
        FloatBuffer tmp_G_high_buf(input_channels * input_height * input_width);
        for (int i = 0; i < dim; ++i) 
        {
            std::copy(input_low_G_buf.data() + i * size, input_low_G_buf.data() + (i + 1) * size, 
                    G_low_batch.data());
            std::copy(input_high_G_buf.data() + i * size, input_high_G_buf.data() + (i + 1) * size, 
                    G_high_batch.data());
            std::copy(ustar_buf.data(), ustar_buf.data() + size,
                    tmp_G_low_buf.data());
            std::copy(ustar_buf.data(), ustar_buf.data() + size,
                    tmp_G_high_buf.data());
            
            float_detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_SIGN, 0, 0>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &G_low_batch, (FloatBuffer *)nullptr, &tmp_G_low_buf);

            float_detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_NSIGN, 0, 0>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &G_high_batch, (FloatBuffer *)nullptr, &tmp_G_high_buf);

            float_detail::abstract_layer<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &tmp_G_low_buf, (FloatBuffer *)nullptr, &tmp_G_high_buf);
            
            std::copy(tmp_G_high_buf.data(), tmp_G_high_buf.data() + size, 
                output_G_buf.data() + i * size);

            
        }

        /* Blend E*/

        FloatBuffer tmp_E_buf(input_channels * input_height * input_width);
        std::copy(ustar_buf.data(), ustar_buf.data() + size,
                tmp_E_buf.data());
        
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_SIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_low_E_buf, (FloatBuffer *)nullptr, &tmp_E_buf);

        std::copy(ustar_buf.data(), ustar_buf.data() + size,
                output_E_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_NSIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_high_E_buf, (FloatBuffer *)nullptr, &output_E_buf);

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_E_buf, (FloatBuffer *)nullptr, &output_E_buf);

        
        // std::copy(ustar_buf.data(), ustar_buf.data() + size,
        //         output_E_buf.data());

        FloatBuffer outval_buf(input_channels * input_height * input_width); 
        std::copy(cbar_buf.data(), cbar_buf.data() + size, 
                outval_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MUL, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &cbar_buf, (FloatBuffer *)nullptr, &outval_buf);
        
        
        FloatBuffer tmp_outval_buf(input_channels * input_height * input_width);
        std::copy(pstar_buf.data(), pstar_buf.data() + size, 
                tmp_outval_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_SUB, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &output_E_buf, (FloatBuffer *)nullptr, &tmp_outval_buf);
        
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_DIV, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_outval_buf, (FloatBuffer *)nullptr, &outval_buf);

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &output_rho_buf, (FloatBuffer *)nullptr, &outval_buf);

        FloatBuffer mask_buf(input_channels * input_height * input_width);
        std::copy(ustar_buf.data(), ustar_buf.data() + size, 
                    mask_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_COND_NSIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &ubar_buf, (FloatBuffer *)nullptr, &mask_buf);    
        
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &cbar_buf, (FloatBuffer *)nullptr, &mask_buf);      
        
        FloatBuffer tmp_buf(input_channels * input_height * input_width); 
        std::copy(mask_buf.data(), mask_buf.data() + size, 
                    tmp_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_SIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &outval_buf, (FloatBuffer *)nullptr, &tmp_buf);
        
        std::copy(output_rho_buf.data(), output_rho_buf.data() + size, 
                    tmp_rho_buf.data());
        std::copy(mask_buf.data(), mask_buf.data() + size, 
                    output_rho_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_NSIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_rho_buf, (FloatBuffer *)nullptr, &output_rho_buf);

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_buf, (FloatBuffer *)nullptr, &output_rho_buf);


        std::copy(mask_buf.data(), mask_buf.data() + size, 
                    tmp_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_SIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &pstar_buf, (FloatBuffer *)nullptr, &tmp_buf);
        
        std::copy(output_E_buf.data(), output_E_buf.data() + size, 
                    tmp_E_buf.data());
        std::copy(mask_buf.data(), mask_buf.data() + size, 
                    output_E_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_NSIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_E_buf, (FloatBuffer *)nullptr, &output_E_buf);

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_buf, (FloatBuffer *)nullptr, &output_E_buf);

        
        std::copy(mask_buf.data(), mask_buf.data() + size, 
                    tmp_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_SIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &ustar_buf, (FloatBuffer *)nullptr, &tmp_buf);
        
        FloatBuffer tmp_G_buf(input_channels * input_height * input_width);
        std::copy(output_G_buf.data() + dir * size, output_G_buf.data() + (dir + 1) * size, 
                    tmp_G_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_NSIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_G_buf, (FloatBuffer *)nullptr, &mask_buf);

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_buf, (FloatBuffer *)nullptr, &mask_buf);

        std::copy(mask_buf.data(), mask_buf.data() + size, 
                    output_G_buf.data() + dir * size);

        std::copy(output_rho_buf.data(), output_rho_buf.data() + size, 
                    output_buf.data());
        std::copy(output_G_buf.data(), output_G_buf.data() + size * dim,
                    output_buf.data() + size);
        std::copy(output_E_buf.data(), output_E_buf.data() + size,
                    output_buf.data() + size * (dim + 1));

    }
    else
    {
        throw std::invalid_argument(
            "SoftSign<float> ERROR: in_channels unsupported.");
    }

}
#endif // SMALL_HAS_FLOAT_SUPPORT


#if defined(SMALL_HAS_DOUBLE_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<DoubleBuffer, BufferT>::value, bool> = true>
void Upwind(int input_channels,
              int input_height, int input_width,
              int dim, int dir, BufferT const &gamma,
              BufferT const &input_low_buf,
              BufferT const &input_high_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Upwind<double>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % DOUBLE_C_ib == 0)
    {
        DoubleBuffer input_low_rho_buf(input_channels * input_height * input_width);
        DoubleBuffer input_high_rho_buf(input_channels * input_height * input_width);
        DoubleBuffer input_low_G_buf(input_channels * input_height * input_width * dim);
        DoubleBuffer input_high_G_buf(input_channels * input_height * input_width * dim);
        DoubleBuffer input_low_E_buf(input_channels * input_height * input_width);
        DoubleBuffer input_high_E_buf(input_channels * input_height * input_width);
        DoubleBuffer output_rho_buf(input_channels * input_height * input_width);
        DoubleBuffer output_G_buf(input_channels * input_height * input_width * dim);
        DoubleBuffer output_E_buf(input_channels * input_height * input_width);
        size_t size(input_channels * input_height * input_width);
        std::copy(input_low_buf.data(), input_low_buf.data() + size,
                input_low_rho_buf.data());
        std::copy(input_high_buf.data(), input_high_buf.data() + size,
                input_high_rho_buf.data());
        std::copy(input_low_buf.data() + size, input_low_buf.data() + size * (dim + 1),
                input_low_G_buf.data());
        std::copy(input_high_buf.data() + size, input_high_buf.data() + size * (dim + 1),
                input_high_G_buf.data());
        std::copy(input_low_buf.data() + size * (dim + 1), input_low_buf.data() + size * (dim + 2),
                input_low_E_buf.data());
        std::copy(input_high_buf.data() + size * (dim + 1), input_high_buf.data() + size * (dim + 2),
                input_high_E_buf.data());

        /* rhobar = (rho_low + rho_high) * 0.5 */
        DoubleBuffer rhobar_buf(input_channels * input_height * input_width);
        std::copy(input_high_rho_buf.data(), input_high_rho_buf.data() + size,
                rhobar_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_low_rho_buf, (DoubleBuffer *)nullptr, &rhobar_buf);

        DoubleBuffer scalar_buf(1);
        scalar_buf.data()[0] = 0.5;
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &rhobar_buf, &scalar_buf, &rhobar_buf);
        

        /* ubar = (G_low(dir) + G_high(dir)) * 0.5 */
        DoubleBuffer ubar_buf(input_channels * input_height * input_width);
        DoubleBuffer low_G_buf(input_channels * input_height * input_width);

        std::copy(input_high_G_buf.data() + size * dir, input_high_G_buf.data() + size * (dir + 1), 
                ubar_buf.data());
        
        std::copy(input_low_G_buf.data() + size * dir, input_low_G_buf.data() + size * (dir + 1),
                low_G_buf.data());
        
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &low_G_buf, (DoubleBuffer *)nullptr, &ubar_buf);
        
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &ubar_buf, &scalar_buf, &ubar_buf);

        /* pbar = (E_low(dir) + E_high(dir)) * 0.5 */
        DoubleBuffer pbar_buf(input_channels * input_height * input_width);
        std::copy(input_high_E_buf.data(), input_high_E_buf.data() + size,
                pbar_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_low_E_buf, (DoubleBuffer *)nullptr, &pbar_buf);

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &pbar_buf, &scalar_buf, &pbar_buf);

        /* cbar = sqrt(gamma * pbar / rhobar) */
        DoubleBuffer cbar_buf(input_channels * input_height * input_width);
        std::copy(rhobar_buf.data(), rhobar_buf.data() + size, 
            cbar_buf.data());

        std::copy(rhobar_buf.data(), rhobar_buf.data() + size, 
            output_rho_buf.data());
        
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_DIV, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &pbar_buf, (DoubleBuffer *)nullptr, &cbar_buf);


        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &cbar_buf, &gamma, &cbar_buf);

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_SQRT, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &cbar_buf, (DoubleBuffer *)nullptr, &cbar_buf);
        

        /* pstar = pbar + rhobar * cbar * udiff */
        DoubleBuffer pstar_buf(input_channels * input_height * input_width);
        std::copy(rhobar_buf.data(), rhobar_buf.data() + size, 
                pstar_buf.data());

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_MUL, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &cbar_buf, (DoubleBuffer *)nullptr, &pstar_buf);
        
        DoubleBuffer ustar_buf(input_channels * input_height * input_width);
        std::copy(pstar_buf.data(), pstar_buf.data() + size,
                ustar_buf.data());

        DoubleBuffer udiff_buf(input_channels * input_height * input_width);
        std::copy(input_low_G_buf.data() + size * dir, input_low_G_buf.data() + size * (dir + 1), 
                udiff_buf.data());
        DoubleBuffer high_G_buf(input_channels * input_height * input_width);
        std::copy(input_high_G_buf.data() + size * dir, input_high_G_buf.data() + size * (dir + 1), 
                high_G_buf.data());
        

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_SUB, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &high_G_buf, (DoubleBuffer *)nullptr, &udiff_buf);
        
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &udiff_buf, &scalar_buf, &udiff_buf);
        
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_MUL, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &udiff_buf, (DoubleBuffer *)nullptr, &pstar_buf);
        
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &pbar_buf, (DoubleBuffer *)nullptr, &pstar_buf);

        /* ustar = ubar + pdiff / (rho_bar * cbar) */
        

        DoubleBuffer pdiff_buf(input_channels * input_height * input_width);
        std::copy(input_low_E_buf.data(), input_low_E_buf.data() + size, 
                pdiff_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_SUB, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_high_E_buf, (DoubleBuffer *)nullptr, &pdiff_buf);

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &pdiff_buf, &scalar_buf, &pdiff_buf);

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_DIV, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &pdiff_buf, (DoubleBuffer *)nullptr, &ustar_buf);

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &ubar_buf, (DoubleBuffer *)nullptr, &ustar_buf);

        /* Blend rho*/

        DoubleBuffer tmp_rho_buf(input_channels * input_height * input_width);
        std::copy(ustar_buf.data(), ustar_buf.data() + size,
                tmp_rho_buf.data());
        
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_SIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_low_rho_buf, (DoubleBuffer *)nullptr, &tmp_rho_buf);

        std::copy(ustar_buf.data(), ustar_buf.data() + size,
                output_rho_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_NSIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_high_rho_buf, (DoubleBuffer *)nullptr, &output_rho_buf);

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_rho_buf, (DoubleBuffer *)nullptr, &output_rho_buf);
        

        DoubleBuffer G_low_batch(input_channels * input_height * input_width);
        DoubleBuffer G_high_batch(input_channels * input_height * input_width);
        DoubleBuffer tmp_G_low_buf(input_channels * input_height * input_width);
        DoubleBuffer tmp_G_high_buf(input_channels * input_height * input_width);
        for (int i = 0; i < dim; ++i) 
        {
            std::copy(input_low_G_buf.data() + i * size, input_low_G_buf.data() + (i + 1) * size, 
                    G_low_batch.data());
            std::copy(input_high_G_buf.data() + i * size, input_high_G_buf.data() + (i + 1) * size, 
                    G_high_batch.data());
            std::copy(ustar_buf.data(), ustar_buf.data() + size,
                    tmp_G_low_buf.data());
            std::copy(ustar_buf.data(), ustar_buf.data() + size,
                    tmp_G_high_buf.data());
            
            double_detail::abstract_layer<
                DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_SIGN, 0, 0>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &G_low_batch, (DoubleBuffer *)nullptr, &tmp_G_low_buf);

            double_detail::abstract_layer<
                DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_NSIGN, 0, 0>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &G_high_batch, (DoubleBuffer *)nullptr, &tmp_G_high_buf);

            double_detail::abstract_layer<
                DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                input_height, input_width,
                1, 1,
                0, 0, 0, 0,
                &tmp_G_low_buf, (DoubleBuffer *)nullptr, &tmp_G_high_buf);
            
            std::copy(tmp_G_high_buf.data(), tmp_G_high_buf.data() + size, 
                output_G_buf.data() + i * size);

            
        }

        /* Blend E*/

        DoubleBuffer tmp_E_buf(input_channels * input_height * input_width);
        std::copy(ustar_buf.data(), ustar_buf.data() + size,
                tmp_E_buf.data());
        
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_SIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_low_E_buf, (DoubleBuffer *)nullptr, &tmp_E_buf);

        std::copy(ustar_buf.data(), ustar_buf.data() + size,
                output_E_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_NSIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &input_high_E_buf, (DoubleBuffer *)nullptr, &output_E_buf);

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_E_buf, (DoubleBuffer *)nullptr, &output_E_buf);

        
        // std::copy(ustar_buf.data(), ustar_buf.data() + size,
        //         output_E_buf.data());

        DoubleBuffer outval_buf(input_channels * input_height * input_width); 
        std::copy(cbar_buf.data(), cbar_buf.data() + size, 
                outval_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_MUL, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &cbar_buf, (DoubleBuffer *)nullptr, &outval_buf);
        
        
        DoubleBuffer tmp_outval_buf(input_channels * input_height * input_width);
        std::copy(pstar_buf.data(), pstar_buf.data() + size, 
                tmp_outval_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_SUB, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &output_E_buf, (DoubleBuffer *)nullptr, &tmp_outval_buf);
        
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_DIV, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_outval_buf, (DoubleBuffer *)nullptr, &outval_buf);

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &output_rho_buf, (DoubleBuffer *)nullptr, &outval_buf);

        DoubleBuffer mask_buf(input_channels * input_height * input_width);
        std::copy(ustar_buf.data(), ustar_buf.data() + size, 
                    mask_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_COND_NSIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &ubar_buf, (DoubleBuffer *)nullptr, &mask_buf);    
        
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &cbar_buf, (DoubleBuffer *)nullptr, &mask_buf);      
        
        DoubleBuffer tmp_buf(input_channels * input_height * input_width); 
        std::copy(mask_buf.data(), mask_buf.data() + size, 
                    tmp_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_SIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &outval_buf, (DoubleBuffer *)nullptr, &tmp_buf);
        
        std::copy(output_rho_buf.data(), output_rho_buf.data() + size, 
                    tmp_rho_buf.data());
        std::copy(mask_buf.data(), mask_buf.data() + size, 
                    output_rho_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_NSIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_rho_buf, (DoubleBuffer *)nullptr, &output_rho_buf);

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_buf, (DoubleBuffer *)nullptr, &output_rho_buf);


        std::copy(mask_buf.data(), mask_buf.data() + size, 
                    tmp_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_SIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &pstar_buf, (DoubleBuffer *)nullptr, &tmp_buf);
        
        std::copy(output_E_buf.data(), output_E_buf.data() + size, 
                    tmp_E_buf.data());
        std::copy(mask_buf.data(), mask_buf.data() + size, 
                    output_E_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_NSIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_E_buf, (DoubleBuffer *)nullptr, &output_E_buf);

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_buf, (DoubleBuffer *)nullptr, &output_E_buf);

        
        std::copy(mask_buf.data(), mask_buf.data() + size, 
                    tmp_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_SIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &ustar_buf, (DoubleBuffer *)nullptr, &tmp_buf);
        
        DoubleBuffer tmp_G_buf(input_channels * input_height * input_width);
        std::copy(output_G_buf.data() + dir * size, output_G_buf.data() + (dir + 1) * size, 
                    tmp_G_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_NSIGN, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_G_buf, (DoubleBuffer *)nullptr, &mask_buf);

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &tmp_buf, (DoubleBuffer *)nullptr, &mask_buf);

        std::copy(mask_buf.data(), mask_buf.data() + size, 
                    output_G_buf.data() + dir * size);

        std::copy(output_rho_buf.data(), output_rho_buf.data() + size, 
                    output_buf.data());
        std::copy(output_G_buf.data(), output_G_buf.data() + size * dim,
                    output_buf.data() + size);
        std::copy(output_E_buf.data(), output_E_buf.data() + size,
                    output_buf.data() + size * (dim + 1));

    }
    else
    {
        throw std::invalid_argument(
            "SoftSign<double> ERROR: in_channels unsupported.");
    }

}
#endif // SMALL_HAS_DOUBLE_SUPPORT

#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void GetFlux(int input_channels,
              int input_height, int input_width,
              int dim, int dir, BufferT const &gamma,
              BufferT const &input_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Upwind<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif
    size_t size(input_channels * input_height * input_width);
    // a_W(0) = a_U(0)
    FloatBuffer F0_buf(size);
    std::copy(input_buf.data(), input_buf.data() + size,
              F0_buf.data());
    FloatBuffer W_dir_buf(size);
    std::copy(input_buf.data() + size * (dir + 1), input_buf.data() + size * (dir + 2),
              W_dir_buf.data());
    
    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MUL, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &W_dir_buf, (FloatBuffer *)nullptr, &F0_buf);

    std::copy(F0_buf.data(), F0_buf.data() + size,
              output_buf.data());
    
    FloatBuffer W2_buf(size);
    init_zeros(W2_buf, size);
    for (int i = 0; i < dim; ++i)
    {
        FloatBuffer W_dim_buf(size);
        std::copy(input_buf.data() + size * (i + 1), input_buf.data() + size * (i + 2),
                  W_dim_buf.data());
        FloatBuffer Fd_buf(size);
        std::copy(input_buf.data() + size * (i + 1), input_buf.data() + size * (i + 2),
                  Fd_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MUL, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &F0_buf, (FloatBuffer *)nullptr, &Fd_buf);
        
        std::copy(Fd_buf.data(), Fd_buf.data() + size,
                  output_buf.data() + size * (i + 1));

        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MUL, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &W_dim_buf, (FloatBuffer *)nullptr, &W_dim_buf);
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels,
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &W_dim_buf, (FloatBuffer *)nullptr, &W2_buf);
    }
    // a_F(dir + 1) += a_W(dim + 2)
    FloatBuffer tmp_buf(size);
    std::copy(input_buf.data() + size * (dim + 1), input_buf.data() + size * (dim + 2),
              tmp_buf.data());
    FloatBuffer F_dir_buf(size);
    std::copy(output_buf.data() + size * (dir + 1), output_buf.data() + size * (dir + 2),
              F_dir_buf.data());
    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &tmp_buf, (FloatBuffer *)nullptr, &F_dir_buf);
    std::copy(F_dir_buf.data(), F_dir_buf.data() + size,
              output_buf.data() + size * (dir + 1));
    
    FloatBuffer scalar_buf(1);
    scalar_buf.data()[0] = 0.5f;
    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &W2_buf, &scalar_buf, &W2_buf);

    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MUL, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &F0_buf, (FloatBuffer *)nullptr, &W2_buf);
    
    
    std::copy(input_buf.data() + size * (dir + 1), input_buf.data() + size * (dir + 2),
              F_dir_buf.data());
    FloatBuffer gamma_buf(1);
    gamma_buf.data()[0] = gamma.data()[0] / (gamma.data()[0] - 1.0f);
    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &F_dir_buf, &gamma_buf, &F_dir_buf);
        
    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MUL, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &tmp_buf, (FloatBuffer *)nullptr, &F_dir_buf);
    

    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &F_dir_buf, (FloatBuffer *)nullptr, &W2_buf);
    std::copy(W2_buf.data(), W2_buf.data() + size,
              output_buf.data() + size * (dim + 1));
    scalar_buf.data()[0] = -1.0f;
    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels * (dim + 2),
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &output_buf, &scalar_buf, &output_buf);
}

#endif 

#if defined(SMALL_HAS_DOUBLE_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<DoubleBuffer, BufferT>::value, bool> = true>
void GetFlux(int input_channels,
              int input_height, int input_width,
              int dim, int dir, BufferT const &gamma,
              BufferT const &input_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Upwind<double>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif
    size_t size(input_channels * input_height * input_width);
    // a_W(0) = a_U(0)
    DoubleBuffer F0_buf(size);
    std::copy(input_buf.data(), input_buf.data() + size,
              F0_buf.data());
    DoubleBuffer W_dir_buf(size);
    std::copy(input_buf.data() + size * (dir + 1), input_buf.data() + size * (dir + 2),
              W_dir_buf.data());
    
    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_MUL, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &W_dir_buf, (DoubleBuffer *)nullptr, &F0_buf);

    std::copy(F0_buf.data(), F0_buf.data() + size,
              output_buf.data());
    
    DoubleBuffer W2_buf(size);
    init_zeros(W2_buf, size);
    for (int i = 0; i < dim; ++i)
    {
        DoubleBuffer W_dim_buf(size);
        std::copy(input_buf.data() + size * (i + 1), input_buf.data() + size * (i + 2),
                  W_dim_buf.data());
        DoubleBuffer Fd_buf(size);
        std::copy(input_buf.data() + size * (i + 1), input_buf.data() + size * (i + 2),
                  Fd_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_MUL, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &F0_buf, (DoubleBuffer *)nullptr, &Fd_buf);
        
        std::copy(Fd_buf.data(), Fd_buf.data() + size,
                  output_buf.data() + size * (i + 1));

        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_MUL, 0, 0>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &W_dim_buf, (DoubleBuffer *)nullptr, &W_dim_buf);
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels,
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &W_dim_buf, (DoubleBuffer *)nullptr, &W2_buf);
    }
    // a_F(dir + 1) += a_W(dim + 2)
    DoubleBuffer tmp_buf(size);
    std::copy(input_buf.data() + size * (dim + 1), input_buf.data() + size * (dim + 2),
              tmp_buf.data());
    DoubleBuffer F_dir_buf(size);
    std::copy(output_buf.data() + size * (dir + 1), output_buf.data() + size * (dir + 2),
              F_dir_buf.data());
    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &tmp_buf, (DoubleBuffer *)nullptr, &F_dir_buf);
    std::copy(F_dir_buf.data(), F_dir_buf.data() + size,
              output_buf.data() + size * (dir + 1));
    
    DoubleBuffer scalar_buf(1);
    scalar_buf.data()[0] = 0.5;
    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &W2_buf, &scalar_buf, &W2_buf);

    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_MUL, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &F0_buf, (DoubleBuffer *)nullptr, &W2_buf);
    
    
    std::copy(input_buf.data() + size * (dir + 1), input_buf.data() + size * (dir + 2),
              F_dir_buf.data());
    DoubleBuffer gamma_buf(1);
    gamma_buf.data()[0] = gamma.data()[0] / (gamma.data()[0] - 1.0);
    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &F_dir_buf, &gamma_buf, &F_dir_buf);
        
    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_MUL, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &tmp_buf, (DoubleBuffer *)nullptr, &F_dir_buf);
    

    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
        input_channels,
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &F_dir_buf, (DoubleBuffer *)nullptr, &W2_buf);
    std::copy(W2_buf.data(), W2_buf.data() + size,
              output_buf.data() + size * (dim + 1));
    scalar_buf.data()[0] = -1.0;
    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels * (dim + 2),
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &output_buf, &scalar_buf, &output_buf);
}

#endif

#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void WaveSpeedBound(int input_channels,
              int input_height, int input_width,
              int dim, BufferT const &gamma,
              BufferT const &input_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "WaveSpeedBound<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif
    size_t size(input_channels * input_height * input_width);
    FloatBuffer gamma_buf(1);
    gamma_buf.data()[0] = gamma.data()[0];
    std::copy(input_buf.data(), input_buf.data() + size,
              output_buf.data());
    FloatBuffer input_E_buf(size);
    std::copy(input_buf.data() + size * (dim + 1), input_buf.data() + size * (dim + 2),
              input_E_buf.data());
    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &input_E_buf, &gamma_buf, &input_E_buf);
    
    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_DIV, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &input_E_buf, (FloatBuffer *)nullptr, &output_buf);

    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_SQRT, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &output_buf, (FloatBuffer *)nullptr, &output_buf);
    
    FloatBuffer scalar_buf(1);
    scalar_buf.data()[0] = (float) dim;
    float_detail::abstract_layer<
        FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &output_buf, &scalar_buf, &output_buf);
    
    for (int dir = 1; dir <= dim; ++dir)
    {
        FloatBuffer w_dir_buf(size);
        std::copy(input_buf.data() + size * dir, input_buf.data() + size * (dir + 1),
                  w_dir_buf.data());
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ABS, 0, 0>(
            input_channels,
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &w_dir_buf, (FloatBuffer *)nullptr, &w_dir_buf);
        float_detail::abstract_layer<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels,
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &w_dir_buf, (FloatBuffer *)nullptr, &output_buf);
    }
}

#endif 




#if defined(SMALL_HAS_DOUBLE_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<DoubleBuffer, BufferT>::value, bool> = true>
void WaveSpeedBound(int input_channels,
              int input_height, int input_width,
              int dim, BufferT const &gamma,
              BufferT const &input_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "WaveSpeedBound<double>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif
    size_t size(input_channels * input_height * input_width);
    DoubleBuffer gamma_buf(1);
    gamma_buf.data()[0] = gamma.data()[0];
    std::copy(input_buf.data(), input_buf.data() + size,
              output_buf.data());
    DoubleBuffer input_E_buf(size);
    std::copy(input_buf.data() + size * (dim + 1), input_buf.data() + size * (dim + 2),
              input_E_buf.data());
    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &input_E_buf, &gamma_buf, &input_E_buf);
    
    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_DIV, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &input_E_buf, (DoubleBuffer *)nullptr, &output_buf);

    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_SQRT, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &output_buf, (DoubleBuffer *)nullptr, &output_buf);
    
    DoubleBuffer scalar_buf(1);
    scalar_buf.data()[0] = (double) dim;
    double_detail::abstract_layer<
        DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_EWISE_MUL_SCALAR, 0, 0>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        &output_buf, &scalar_buf, &output_buf);
    
    for (int dir = 1; dir <= dim; ++dir)
    {
        DoubleBuffer w_dir_buf(size);
        std::copy(input_buf.data() + size * dir, input_buf.data() + size * (dir + 1),
                  w_dir_buf.data());
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ABS, 0, 0>(
            input_channels,
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &w_dir_buf, (DoubleBuffer *)nullptr, &w_dir_buf);
        double_detail::abstract_layer<
            DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ADD, 0, 0>(
            input_channels,
            1,              // Output Channels per group
            1,
            input_height, input_width,
            1, 1,
            0, 0, 0, 0,
            &w_dir_buf, (DoubleBuffer *)nullptr, &output_buf);
    }
}

#endif 

#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void Stencil(int input_channels,
              int input_height, int input_width,
              int l_pad, int r_pad,
              BufferT const &input_buf,
              BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Stencil<float>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if ((input_channels % FLOAT_C_ib == 0))
    {
        std::cout << "Stencil<float> ERROR: in_channels == 1, no filter needed.\n";
        FloatBuffer filter_buf(2);
        filter_buf.data()[0] = 0.5f; // left
        filter_buf.data()[1] = 0.5f; // right
        float_detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, 1,
                FLOAT_W_ob, 1, 1, OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                input_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                1, 2,
                0, l_pad, r_pad, 0,
                &input_buf, &filter_buf, &output_buf);
    }
    else
    {
        throw std::invalid_argument(
            "Stencil<float> ERROR: in_channels unsupported.");
    }

}
#endif

#if defined(SMALL_HAS_DOUBLE_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<DoubleBuffer, BufferT>::value, bool> = true>
void Max(int input_channels,
             int input_height, int input_width,
             BufferT const &input_buf,
             BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "SoftMax<double>(chans:" << input_channels
                << ",img:" << input_height << "x" << input_width
                << ",I,O)\n";
#endif

    if (input_channels % DOUBLE_C_ib == 0)
    {
        // size_t size(input_channels * input_height * input_width);
        // DoubleBuffer tmp_buf(size);
        // double_detail::abstract_layer<
        //     DoubleBuffer, DOUBLE_C_ob, 1, 1, DOUBLE_W_ob, 1, 1, OP_ABS, 0, 0>(
        //     input_channels,
        //     1,              // Output Channels per group
        //     1,
        //     input_height, input_width,
        //     1, 1,
        //     0, 0, 0, 0,
        //     &input_buf, (DoubleBuffer *)nullptr, &tmp_buf);

        double_detail::abstract_layer<
            DoubleBuffer, 1, 1, DOUBLE_C_ob, DOUBLE_W_ob, 1, DOUBLE_C_ob, OP_MAX_POOL, 3, 1>(
            1, // Output Channel Grouping
            1, // Output Channels per group
            input_channels,
            input_height, input_width,
            input_height, input_width,
            0, 0, 0, 0,
            &input_buf, (DoubleBuffer *)nullptr, &output_buf);

    }
    else
    {
        throw std::invalid_argument(
            "SoftMax<float> ERROR: in_channels unsupported.");
    }
}
#endif

} // namespace: small
