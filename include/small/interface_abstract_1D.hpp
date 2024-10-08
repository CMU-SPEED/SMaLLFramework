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

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void Conv1D(
    int kernel_width, int stride,
    uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int batch_size, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Conv1D<float>(k:"
              << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)l_pad << "," << (int)r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",I,F,O)\n";
#endif

    /// @todo add an assert for invalid numbers of output channels
    ///       (layer classes should be responsible for padding filters).

    if (input_channels % FLOAT_C_ib == 0)
    {
        if (stride == 1)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob, 1, FLOAT_UNROLL, OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob, 2, FLOAT_UNROLL, OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv1D<float> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
    {
        if (stride == 1)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, 3,
                FLOAT_W_ob, 1, 1, OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, 3,
                FLOAT_W_ob, 2, 1, OP_CONV, 2, 1>( // unroll?
                1,                                     // Output Channel Grouping
                output_channels,                       // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv1D<float> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 2) && (input_channels < FLOAT_C_ib))
    {
        if (stride == 1)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, 2,
                FLOAT_W_ob, 1, 1, OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, 2,
                FLOAT_W_ob, 2, 1, OP_CONV, 2, 1>( // unroll?
                1,                                     // Output Channel Grouping
                output_channels,                       // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv1D<float> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 1) && (input_channels < FLOAT_C_ib))
    {
        if (stride == 1)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, 1,
                FLOAT_W_ob, 1, 1, OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, 1,
                FLOAT_W_ob, 2, 1, OP_CONV, 2, 1>( // unroll?
                1,                                     // Output Channel Grouping
                output_channels,                       // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv1D<float> ERROR: stride unsupported.");
        }
    }

    /// @todo Do we need other specific cases for input_channels > 3?

    // everything else.
    else
    {
        throw std::invalid_argument(
            "Conv1D<float> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void Conv1D(
    int kernel_width, int stride,
    uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int batch_size, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Conv1D<quint8>(k:"
              << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)l_pad << "," << (int)r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",I,F,O)\n";
#endif

    /// @todo add an assert for invalid numbers of output channels
    ///       (layer classes should be responsible for padding filters).

    if (input_channels % QUINT8_C_ib == 0)
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, QUINT8_C_ib,
                QUINT8_W_ob, 1, QUINT8_UNROLL, OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, QUINT8_C_ib,
                QUINT8_W_ob, 2, QUINT8_UNROLL, OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv1D<quint8> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 3) && (input_channels < QUINT8_C_ib))
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, 3,
                QUINT8_W_ob, 1, 1, OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, 3,
                QUINT8_W_ob, 2, 1, OP_CONV, 2, 1>( // unroll?
                1,                             // Output Channel Grouping
                output_channels,               // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv1D<quint8> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 2) && (input_channels < QUINT8_C_ib))
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, 2,
                QUINT8_W_ob, 1, 1, OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, 2,
                QUINT8_W_ob, 2, 1, OP_CONV, 2, 1>( // unroll?
                1,                             // Output Channel Grouping
                output_channels,               // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv1D<quint8> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 1) && (input_channels < QUINT8_C_ib))
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, 1,
                QUINT8_W_ob, 1, 1, OP_CONV, 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, 1,
                QUINT8_W_ob, 2, 1, OP_CONV, 2, 1>( // unroll?
                1,                             // Output Channel Grouping
                output_channels,               // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv1D<quint8> ERROR: stride unsupported.");
        }
    }

    /// @todo Do we need other specific cases for input_channels > 3?

    // everything else.
    else
    {
        throw std::invalid_argument(
            "Conv1D<quint8> ERROR: in_channels unsupported.");
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
void PartialConv1D(
    int kernel_width, int stride,
    uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int batch_size, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "PartialConv1D<float>(k:"
              << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)l_pad << "," << (int)r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (input_channels % FLOAT_C_ib == 0)
    {
        if (stride == 1)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob, 1, FLOAT_UNROLL, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {

            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob, 2, FLOAT_UNROLL, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "PartialConv1D<float> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
    {
        if (stride == 1)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, 3,
                FLOAT_W_ob, 1, 1, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, 3,
                FLOAT_W_ob, 2, 1, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "PartialConv1D<float> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 2) && (input_channels < FLOAT_C_ib))
    {
        if (stride == 1)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, 2,
                FLOAT_W_ob, 1, 1, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, 2,
                FLOAT_W_ob, 2, 1, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "PartialConv1D<float> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 1) && (input_channels < FLOAT_C_ib))
    {
        if (stride == 1)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, 1,
                FLOAT_W_ob, 1, 1, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            detail::abstract_layer_1D<
                FloatBuffer, 1, FLOAT_C_ob, 1,
                FLOAT_W_ob, 2, 1, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "PartialConv1D<float> ERROR: stride unsupported.");
        }
    }

    /// @todo Do we need other specific cases for input_channels > 3?

    // everything else.
    else
    {
        throw std::invalid_argument(
            "PartialConv1D<float> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void PartialConv1D(
    int kernel_width, int stride,
    uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int batch_size, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "PartialConv1D<quint8>(k:"
              << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)l_pad << "," << (int)r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (input_channels % QUINT8_C_ib == 0)
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, QUINT8_C_ib,
                QUINT8_W_ob, 1, QUINT8_UNROLL, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {

            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, QUINT8_C_ib,
                QUINT8_W_ob, 2, QUINT8_UNROLL, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "PartialConv1D<quint8> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 3) && (input_channels < QUINT8_C_ib))
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, 3,
                QUINT8_W_ob, 1, 1, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, 3,
                QUINT8_W_ob, 2, 1, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "PartialConv1D<quint8> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 2) && (input_channels < QUINT8_C_ib))
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, 2,
                QUINT8_W_ob, 1, 1, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, 2,
                QUINT8_W_ob, 2, 1, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "PartialConv1D<quint8> ERROR: stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 1) && (input_channels < QUINT8_C_ib))
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, 1,
                QUINT8_W_ob, 1, 1, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, 1, QUINT8_C_ob, 1,
                QUINT8_W_ob, 2, 1, OP_CONV, 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "PartialConv1D<quint8> ERROR: stride unsupported.");
        }
    }

    /// @todo Do we need other specific cases for input_channels > 3?

    // everything else.
    else
    {
        throw std::invalid_argument(
            "PartialConv1D<quint8> ERROR: in_channels unsupported.");
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
void MaxPool1D(
    int kernel_width, int stride,
    uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int batch_size, int input_width,
    BufferT const &input_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "MaxPool1D<float>(k:"
              << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        if (stride == 1)
        {
            detail::abstract_layer_1D<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_MAX_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
        }
        else if (stride == 2)
        {
            detail::abstract_layer_1D<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 2, 1, OP_MAX_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "MaxPool1D<float> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "MaxPool1D<float> ERROR: in_channels unsupported.");
    }
}

#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void MaxPool1D(
    int kernel_width, int stride,
    uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int batch_size, int input_width,
    BufferT const &input_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "MaxPool1D<quint8>(k:"
              << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % QUINT8_C_ib == 0)
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, OP_MAX_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, (QUInt8Buffer *)nullptr, &output_buf);
        }
        else if (stride == 2)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 2, 1, OP_MAX_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, (QUInt8Buffer *)nullptr, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "MaxPool1D<quint8> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "MaxPool1D<quint8> ERROR: in_channels unsupported.");
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
void AveragePool1D(
    int kernel_width, int stride,
    uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int batch_size, int input_width,
    BufferT const &input_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "AveragePool1D<float>(k:"
              << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        if (stride == 1)
        {
            detail::abstract_layer_1D<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_AVERAGE_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
        }
        else if (stride == 2)
        {
            detail::abstract_layer_1D<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 2, 1, OP_AVERAGE_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
        }
        else if (stride == 3)
        {
            detail::abstract_layer_1D<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 3, 1, OP_AVERAGE_POOL, 1, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                batch_size, input_width,
                kernel_width,
                l_pad, r_pad,
                &input_buf, (FloatBuffer *)nullptr, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "AveragePool1D<float> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "AveragePool1D<float> ERROR: in_channels unsupported.");
    }
}

#endif

//============================================================================
/// @todo AveragePool1D support for quint8
//============================================================================

//****************************************************************************
//****************************************************************************
// #if defined(SMALL_HAS_FLOAT_SUPPORT)
// template <class BufferT,
//           std::enable_if_t<
//               std::is_same<FloatBuffer, BufferT>::value, bool> = true>
// void DepthwiseConv1D(
//     int kernel_width, int stride,
//     uint8_t l_pad, uint8_t r_pad,
//     int input_channels,
//     int batch_size, int input_width,
//     BufferT const &input_buf,
//     BufferT const &filter_buf,
//     BufferT &output_buf)
// {
// #if defined(RECORD_CALLS)
//     std::cout << "DepthwiseConv1D<float>(k:"
//               << kernel_width
//               << ",s:" << stride
//               << ",pad:[" << (int)l_pad << "," << (int)r_pad
//               << "],chans:" << input_channels
//               << ",img:" << batch_size << "x" << input_width
//               << ",I,F,O)\n";
// #endif
//     if (input_channels % FLOAT_C_ib == 0)
//     {
//         if (stride == 1)
//         {
//             detail::abstract_layer_1D<
//                 FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_CONV, 1, 1>(
//                 input_channels, // Output Channel Grouping
//                 1,              // Output Channels per group
//                 1,
//                 batch_size, input_width,
//                 kernel_width,
//                 l_pad, r_pad,
//                 &input_buf, &filter_buf, &output_buf);
//         }
//         else if (stride == 2)
//         {

//             detail::abstract_layer_1D<
//                 FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 2, 1, OP_CONV, 1, 1>(
//                 input_channels, // Output Channel Grouping
//                 1,              // Output Channels per group
//                 1,
//                 batch_size, input_width,
//                 kernel_width,
//                 l_pad, r_pad,
//                 &input_buf, &filter_buf, &output_buf);
//         }
//         else
//         {
//             throw std::invalid_argument(
//                 "DepthwiseConv1D<float> ERROR: stride unsupported.");
//         }
//     }
//     else
//     {
//         throw std::invalid_argument(
//             "DepthwiseConv1D<float> ERROR: in_channels unsupported.");
//     }
// }
// #endif

// //============================================================================
// #if defined(SMALL_HAS_QUINT8_SUPPORT)
// template <class BufferT,
//           std::enable_if_t<
//               std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
// void DepthwiseConv1D(
//     int kernel_width, int stride,
//     uint8_t l_pad, uint8_t r_pad,
//     int input_channels,
//     int batch_size, int input_width,
//     BufferT const &input_buf,
//     BufferT const &filter_buf,
//     BufferT &output_buf)
// {
// #if defined(RECORD_CALLS)
//     std::cout << "DepthwiseConv1D<quint8>(k:"
//               << kernel_width
//               << ",s:" << stride
//               << ",pad:[" << (int)l_pad << "," << (int)r_pad
//               << "],chans:" << input_channels
//               << ",img:" << batch_size << "x" << input_width
//               << ",I,F,O)\n";
// #endif
//     if (input_channels % QUINT8_C_ib == 0)
//     {
//         if (stride == 1)
//         {
//             quint8_detail::abstract_layer_1D<
//                 QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, OP_CONV, 1, 1>(
//                 input_channels, // Output Channel Grouping
//                 1,              // Output Channels per group
//                 1,
//                 batch_size, input_width,
//                 kernel_width,
//                 l_pad, r_pad,
//                 &input_buf, &filter_buf, &output_buf);
//         }
//         else if (stride == 2)
//         {

//             quint8_detail::abstract_layer_1D<
//                 QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 2, 1, OP_CONV, 1, 1>(
//                 input_channels, // Output Channel Grouping
//                 1,              // Output Channels per group
//                 1,
//                 batch_size, input_width,
//                 kernel_width,
//                 l_pad, r_pad,
//                 &input_buf, &filter_buf, &output_buf);
//         }
//         else
//         {
//             throw std::invalid_argument(
//                 "DepthwiseConv1D<quint8> ERROR: stride unsupported.");
//         }
//     }
//     else
//     {
//         throw std::invalid_argument(
//             "DepthwiseConv1D<quint8> ERROR: in_channels unsupported.");
//     }
// }
// #endif

// //****************************************************************************
// //****************************************************************************

// //============================================================================
// #if defined(SMALL_HAS_FLOAT_SUPPORT)
// template <class BufferT,
//           std::enable_if_t<
//               std::is_same<FloatBuffer, BufferT>::value, bool> = true>
// void PartialDepthwiseConv1D(
//     int kernel_width, int stride,
//     uint8_t l_pad, uint8_t r_pad,
//     int input_channels,
//     int batch_size, int input_width,
//     BufferT const &input_buf,
//     BufferT const &filter_buf,
//     BufferT       &output_buf)
// {
// #if defined(RECORD_CALLS)
//     std::cout << "PartialDepthwiseConv1D<float>(k:"
//               << kernel_width
//               << ",s:" << stride
//               << ",pad:[" << (int)l_pad << "," << (int)r_pad
//               << "],chans:" << input_channels
//               << ",img:" << batch_size << "x" << input_width
//               << ",I,F,O)\n";
// #endif
//     if (input_channels % FLOAT_C_ib == 0)
//     {
//         if (stride == 1)
//         {
//             detail::abstract_layer_1D<
//                 FloatBuffer, FLOAT_C_ob, 1, 1,
//                 FLOAT_W_ob, 1, 1, OP_CONV, 1, 0>(
//                     input_channels, // Output Channel Grouping
//                     1,              // Output Channels per group
//                     1,
//                     batch_size, input_width,
//                     kernel_width,
//                     l_pad, r_pad,
//                     &input_buf, &filter_buf, &output_buf);
//         }
//         else if (stride == 2)
//         {
//             detail::abstract_layer_1D<
//                 FloatBuffer, FLOAT_C_ob, 1, 1,
//                 FLOAT_W_ob, 2, 1, OP_CONV, 1, 0>(
//                     input_channels, // Output Channel Grouping
//                     1,              // Output Channels per group
//                     1,
//                     batch_size, input_width,
//                     kernel_width,
//                     l_pad, r_pad,
//                     &input_buf, &filter_buf, &output_buf);
//         }
//         else
//         {
//             throw std::invalid_argument(
//                 "PartialDepthwiseConv1D<float> ERROR: stride unsupported.");
//         }
//     }
//     else
//     {
//         throw std::invalid_argument(
//             "PartialDepthwiseConv1D<float> ERROR: in_channels unsupported.");
//     }
// }
// #endif

// //============================================================================
// #if defined(SMALL_HAS_QUINT8_SUPPORT)
// template <class BufferT,
//           std::enable_if_t<
//               std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
// void PartialDepthwiseConv1D(
//     int kernel_width, int stride,
//     uint8_t l_pad, uint8_t r_pad,
//     int input_channels,
//     int batch_size, int input_width,
//     BufferT const &input_buf,
//     BufferT const &filter_buf,
//     BufferT       &output_buf)
// {
// #if defined(RECORD_CALLS)
//     std::cout << "PartialDepthwiseConv1D<quint8>(k:"
//               << kernel_width
//               << ",s:" << stride
//               << ",pad:[" << (int)l_pad << "," << (int)r_pad
//               << "],chans:" << input_channels
//               << ",img:" << batch_size << "x" << input_width
//               << ",I,F,O)\n";
// #endif
//     if (input_channels % QUINT8_C_ib == 0)
//     {
//         if (stride == 1)
//         {
//             quint8_detail::abstract_layer_1D<
//                 QUInt8Buffer, QUINT8_C_ob, 1, 1,
//                 QUINT8_W_ob, 1, 1, OP_CONV, 1, 0>(
//                     input_channels, // Output Channel Grouping
//                     1,              // Output Channels per group
//                     1,
//                     batch_size, input_width,
//                     kernel_width,
//                     l_pad, r_pad,
//                     &input_buf, &filter_buf, &output_buf);
//         }
//         else if (stride == 2)
//         {
//             quint8_detail::abstract_layer_1D<
//                 QUInt8Buffer, QUINT8_C_ob, 1, 1,
//                 QUINT8_W_ob, 2, 1, OP_CONV, 1, 0>(
//                     input_channels, // Output Channel Grouping
//                     1,              // Output Channels per group
//                     1,
//                     batch_size, input_width,
//                     kernel_width,
//                     l_pad, r_pad,
//                     &input_buf, &filter_buf, &output_buf);
//         }
//         else
//         {
//             throw std::invalid_argument(
//                 "PartialDepthwiseConv1D<quint8> ERROR: stride unsupported.");
//         }
//     }
//     else
//     {
//         throw std::invalid_argument(
//             "PartialDepthwiseConv1D<quint8> ERROR: in_channels unsupported.");
//     }
// }
// #endif

//****************************************************************************
//****************************************************************************

//****************************************************************************
// Assumes that output channels = input channels, output groups splits input channels evenly
//****************************************************************************

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void GroupConv1D(
    int kernel_width, int stride,
    int output_groups,
    uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int batch_size, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "GroupConv1D<float>(k:"
              << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (input_channels % FLOAT_C_ib == 0)
    {
        if (stride == 1)
        {
            detail::abstract_layer_1D<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_CONV, 2, 1>(
                    output_groups,  // Output Channel Grouping
                    input_channels, // Output Channels per group
                    input_channels,
                    batch_size, input_width,
                    kernel_width,
                    l_pad, r_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {

            detail::abstract_layer_1D<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 2, 1, OP_CONV, 2, 1>(
                    output_groups,  // Output Channel Grouping
                    input_channels, // Output Channels per group
                    input_channels,
                    batch_size, input_width,
                    kernel_width,
                    l_pad, r_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "GroupConv1D<float> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "GroupConv1D<float> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void GroupConv1D(
    int kernel_width, int stride,
    int output_groups,
    uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int batch_size, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "GroupConv1D<quint8>(k:"
              << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (input_channels % QUINT8_C_ib == 0)
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, OP_CONV, 1, 1>(
                    output_groups, // Output Channel Grouping
                    input_channels,              // Output Channels per group
                    input_channels,
                    batch_size, input_width,
                    kernel_width,
                    l_pad, r_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {

            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 2, 1, OP_CONV, 1, 1>(
                    output_groups,  // Output Channel Grouping
                    input_channels, // Output Channels per group
                    input_channels,
                    batch_size, input_width,
                    kernel_width,
                    l_pad, r_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "GroupConv1D<quint8> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "GroupConv1D<quint8> ERROR: in_channels unsupported.");
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
void PartialGroupConv1D(
    int kernel_width, int stride,
    int output_groups,
    uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int batch_size, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "PartialGroupConv1D<float>(k:"
              << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (input_channels % FLOAT_C_ib == 0)
    {
        if (stride == 1)
        {
            detail::abstract_layer_1D<
                FloatBuffer, FLOAT_C_ob, 1, 1,
                FLOAT_W_ob, 1, 1, OP_CONV, 2, 0>(
                    output_groups,  // Output Channel Grouping
                    input_channels, // Output Channels per group
                    input_channels,
                    batch_size, input_width,
                    kernel_width,
                    l_pad, r_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {

            detail::abstract_layer_1D<
                FloatBuffer, FLOAT_C_ob, 1, 1,
                FLOAT_W_ob, 2, 1, OP_CONV, 2, 0>(
                    output_groups,  // Output Channel Grouping
                    input_channels, // Output Channels per group
                    input_channels,
                    batch_size, input_width,
                    kernel_width,
                    l_pad, r_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "PartialGroupConv1D<float> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "PartialGroupConv1D<float> ERROR: in_channels unsupported.");
    }
}
#endif

//============================================================================
#if defined(SMALL_HAS_QUINT8_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<QUInt8Buffer, BufferT>::value, bool> = true>
void PartialGroupConv1D(
    int kernel_width, int stride,
    int output_groups,
    uint8_t l_pad, uint8_t r_pad,
    int input_channels,
    int batch_size, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "PartialGroupConv1D<quint8>(k:"
              << kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (input_channels % QUINT8_C_ib == 0)
    {
        if (stride == 1)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, QUINT8_C_ob, 1, 1,
                QUINT8_W_ob, 1, 1, OP_CONV, 1, 0>(
                    output_groups, // Output Channel Grouping
                    input_channels,              // Output Channels per group
                    input_channels,
                    batch_size, input_width,
                    kernel_width,
                    l_pad, r_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else if (stride == 2)
        {

            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, QUINT8_C_ob, 1, 1,
                QUINT8_W_ob, 2, 1, OP_CONV, 1, 0>(
                    output_groups,  // Output Channel Grouping
                    input_channels, // Output Channels per group
                    input_channels,
                    batch_size, input_width,
                    kernel_width,
                    l_pad, r_pad,
                    &input_buf, &filter_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "PartialGroupConv1D<quint8> ERROR: stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "PartialGroupConv1D<quint8> ERROR: in_channels unsupported.");
    }
}
#endif

//****************************************************************************
// ReLUActivation: 1D not necessary?
//****************************************************************************

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void ReLUActivation1D(int input_channels,
                      int batch_size, int input_width,
                      BufferT const &input_buf,
                      BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "ReLUActivation<float>(chans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % FLOAT_C_ib == 0)
    {
        detail::abstract_layer_1D<
            FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 1, 1, OP_RELU, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            batch_size, input_width,
            1, // F_w
            0, 0,
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
void ReLUActivation1D(int input_channels,
                      int batch_size, int input_width,
                      BufferT const &input_buf,
                      BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "ReLUActivation<quint8>(chans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",I,O)\n";
#endif

    if (input_channels % QUINT8_C_ib == 0)
    {
        quint8_detail::abstract_layer_1D<
            QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 1, 1, OP_RELU, 0, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            batch_size, input_width,
            1, // F_w
            0, 0,
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
// LeakyReLUActivation - 1D not necessary?
//****************************************************************************

//****************************************************************************
// Dropout - 1D not necessary?
//****************************************************************************

//****************************************************************************
// SoftMax - 1D not necessary?
//****************************************************************************

//****************************************************************************
//****************************************************************************

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
// only scales width dim of input
template <class BufferT,
          std::enable_if_t<
              std::is_same<FloatBuffer, BufferT>::value, bool> = true>
void UpSample1D(int scale_factor,
                int input_channels,
                int batch_size, int input_width,
                BufferT const &input_buf,
                BufferT &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "UpSample1D<float>(chans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
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
            detail::abstract_layer_1D<
                FloatBuffer, FLOAT_C_ob, 1, 1, FLOAT_W_ob, 2, 1, OP_UPSAMPLE, 0, 1>(
                input_channels, // Output Channel Grouping
                1,              // Output Channels per group
                1,
                batch_size, input_width,
                1,
                0, 0,
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
void UpSample1D(int scale_factor,
                int input_channels,
                int batch_size, int input_width,
                BufferT const &input_buf,
                BufferT       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "UpSample1D<quint8>(chans:" << input_channels
              << ",img:" << batch_size << "x" << input_width
              << ",scale:" << scale_factor
              << ",I,O)\n";
#endif

#pragma message("WARNING: UpSample1D microkernels not implemented for QUInt8Buffer")

    if (scale_factor == 1)
    {
        output_buf = input_buf;
    }
    else if (scale_factor == 2)
    {
        if (input_channels % QUINT8_C_ib == 0)
        {
            quint8_detail::abstract_layer_1D<
                QUInt8Buffer, QUINT8_C_ob, 1, 1, QUINT8_W_ob, 2, 1, OP_UPSAMPLE, 0, 1>(
                    input_channels, // Output Channel Grouping
                    1,              // Output Channels per group
                    1,
                    batch_size, input_width,
                    1,
                    0, 0,
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
// Accum - 1D not necessary
//****************************************************************************

//****************************************************************************
// Bias - 1D not necessary
//****************************************************************************

//****************************************************************************
// PartialBias - 1D not necessary
//****************************************************************************

//****************************************************************************
// Dense - no 1D equivalent
//****************************************************************************

} // small
