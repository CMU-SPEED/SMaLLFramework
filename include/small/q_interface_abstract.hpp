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
// #include <assert.h>
// #include <omp.h>
// #include <stdio.h>
// #include <stdlib.h>
#include <stdint.h>
#include <stdexcept>

// #include <small/abstract_layer.hpp>

int clip(int n, int upper, int lower = 0)
{
    n = (n > lower) * n + !(n > lower) * lower;
    return (n < upper) * n + !(n < upper) * upper;
}

// Quantization Functions
template <typename Q_T, typename T>
void Quantize(int num_elements, T *tensor_ptr, Q_T *quant_tensor_ptr)
{
    float scale_inv = (1.0 / quant_tensor_ptr->scale);
    uint64_t max_val = (1 << quant_tensor_ptr->b) - 1;
    int quant_val = rint(quant_tensor_ptr->zero + (0.0 * scale_inv));
    for (int i = 0; i < num_elements; i++)
    {
        int quant_val = rint(quant_tensor_ptr->zero + (tensor_ptr[i] * scale_inv));
        quant_tensor_ptr->tensor[i] = (quant_val < max_val) ? quant_val : max_val;
    }
}

template <typename Q_T, typename T>
void DeQuantize(int num_elements, T *tensor_ptr, Q_T *quant_tensor_ptr)
{
    for (int i = 0; i < num_elements; i++)
    {
        tensor_ptr[i] = (T)(quant_tensor_ptr->scale * (quant_tensor_ptr->tensor[i] - quant_tensor_ptr->zero));
    }
}

template <typename Q_T, typename T>
void DebugDeQuantize(int num_elements, T *tensor_ptr, Q_T *quant_tensor_ptr)
{
    for (int i = 0; i < num_elements; i++)
    {
        tensor_ptr[i] = (T)(quant_tensor_ptr->scale * ((T)(quant_tensor_ptr->tensor[i] - quant_tensor_ptr->zero)));
    }
}

//****************************************************************************
template <typename OperandT, typename Q_type>
void Conv2D(int layer_num,
            int kernel_size, int stride, /// @todo dim_t?
            uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
            int output_channels, int input_channels,
            int input_height, int input_width,
            Q_type const *input_ptr,
            Q_type const *filter_ptr,
            Q_type *output_ptr)
{
    /// @todo do we need another specific case for input_channels==1?

    // Specific case for the first layer
    if (input_channels == 3)
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, Q_type, int32_t,1, C_ob, 3, W_ob, 1, 1, 'c', 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
                t_pad, l_pad, r_pad, b_pad,
                input_ptr, filter_ptr, output_ptr);
        }
        else if (stride == 2)
        {
            abstract_layer<OperandT, Q_type, int32_t,1, C_ob, 3, W_ob, 2, 1, 'c', 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
                t_pad, l_pad, r_pad, b_pad,
                input_ptr, filter_ptr, output_ptr);
        }
        else
        {
            // printf("This stride is unsupported, please change the interface.cpp file\n");
            throw std::invalid_argument("Conv2D ERROR: stride unsupported.");
        }
    }
    else
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, Q_type, int32_t,1, C_ob, C_ob, W_ob, 1, UNROLL, 'c', 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
                t_pad, l_pad, r_pad, b_pad,
                input_ptr, filter_ptr, output_ptr);
        }
        else if (stride == 2)
        {
            abstract_layer<OperandT, Q_type, int32_t,1, C_ob, C_ob, W_ob, 2, UNROLL, 'c', 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
                t_pad, l_pad, r_pad, b_pad,
                input_ptr, filter_ptr, output_ptr);
        }
        else
        {
            // printf("This stride is unsupported, please change the interface.cpp file\n");
            throw std::invalid_argument("Conv2D ERROR: stride unsupported.");
        }
    }
}

//****************************************************************************
template <typename OperandT, typename Q_type>
void PartialConv2D(int layer_num,
                   int kernel_size, int stride,
                   uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                   int output_channels, int input_channels,
                   int input_height, int input_width,
                   Q_type const *input_ptr,
                   Q_type const *filter_ptr,
                   Q_type *output_ptr)
{
    // Specific case for the first layer
    if (input_channels == 3)
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, Q_type, int32_t,1, C_ob, 3, W_ob, 1, 1, 'c', 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
                t_pad, l_pad, r_pad, b_pad,
                input_ptr, filter_ptr, output_ptr);
        }
        else if (stride == 2)
        {

            abstract_layer<OperandT, Q_type, int32_t,1, C_ob, 3, W_ob, 2, 1, 'c', 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
                t_pad, l_pad, r_pad, b_pad,
                input_ptr, filter_ptr, output_ptr);
        }
        else
        {
            // printf("This stride is unsupported, please change the interface.cpp file\n");
        }
    }
    else
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, Q_type, int32_t,1, C_ob, C_ob, W_ob, 1, UNROLL, 'c', 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
                t_pad, l_pad, r_pad, b_pad,
                input_ptr, filter_ptr, output_ptr);
        }
        else if (stride == 2)
        {

            abstract_layer<OperandT, Q_type, int32_t,1, C_ob, C_ob, W_ob, 2, UNROLL, 'c', 2, 0>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size, kernel_size,
                t_pad, l_pad, r_pad, b_pad,
                input_ptr, filter_ptr, output_ptr);
        }
        else
        {
            // printf("This stride is unsupported, please change the interface.cpp file\n");
        }
    }
}

//****************************************************************************
template <typename OperandT, typename Q_type>
void Maxpool2D(int layer_num,
               int kernel_size, int stride,
               uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
               int input_channels,
               int input_height, int input_width,
               Q_type const *input_ptr,
               Q_type *output_ptr)
{
    if (stride == 1)
    {
        abstract_layer<OperandT, Q_type, int32_t,C_ob, 1, 1, W_ob, 1, 1, 'p', 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_size, kernel_size,
            t_pad, l_pad, r_pad, b_pad,
            input_ptr, NULL, output_ptr);
    }
    else if (stride == 2)
    {

        abstract_layer<OperandT, Q_type, int32_t,C_ob, 1, 1, W_ob, 2, 1, 'p', 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_size, kernel_size,
            t_pad, l_pad, r_pad, b_pad,
            input_ptr, NULL, output_ptr);
    }
    else
    {
        // printf("This stride is unsupported, please change the interface.cpp file\n");
        throw std::invalid_argument("Maxpool2D ERROR: stride unsupported.");
    }
}

//****************************************************************************
template <typename OperandT, typename Q_type>
void DepthwiseConv2D(int layer_num,
                     int kernel_size, int stride,
                     uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                     int input_channels,
                     int input_height, int input_width,
                     Q_type const *input_ptr,
                     Q_type const *filter_ptr,
                     Q_type *output_ptr)
{
    if (stride == 1)
    {

        abstract_layer<OperandT, Q_type, int32_t,C_ob, 1, 1, W_ob, 1, 1, 'c', 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_size, kernel_size,
            t_pad, l_pad, r_pad, b_pad,
            input_ptr, filter_ptr, output_ptr);
    }
    else if (stride == 2)
    {

        abstract_layer<OperandT, Q_type, int32_t,C_ob, 1, 1, W_ob, 2, 1, 'c', 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_size, kernel_size,
            t_pad, l_pad, r_pad, b_pad,
            input_ptr, filter_ptr, output_ptr);
    }
    else
    {
        // printf("This stride is unsupported, please change the interface.cpp file\n");
        throw std::invalid_argument("DepthwiseConv2D ERROR: stride unsupported.");
    }
}

//****************************************************************************
template <typename OperandT, typename Q_type>
void ReLUActivation(int layer_num,
                    int input_channels,
                    int input_height, int input_width,
                    Q_type const *input_ptr,
                    Q_type *output_ptr,
                    int zero = 0)
{
    abstract_layer<OperandT, Q_type, int32_t,C_ob, 1, 1, W_ob, 1, 1, 'a', 0, 1>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        input_ptr, NULL, output_ptr, zero);
}

// template <typename OperandT, typename Q_type>
// void QuantReLUActivation(int layer_num,
//                     int input_channels,
//                     int input_height, int input_width,
//                     Q_type const *input_ptr,
//                     Q_type *output_ptr,
//                     int zero )
// {
//     abstract_layer<OperandT, Q_type, int32_t,C_ob, 1, 1, W_ob, 1, 1, 'a', 0, 1>(
//         input_channels, // Output Channel Grouping
//         1,              // Output Channels per group
//         1,
//         input_height, input_width,
//         1, 1,
//         0, 0, 0, 0,
//         input_ptr, NULL, output_ptr);
// }
//****************************************************************************
template <typename OperandT, typename Q_type>
void Dense(int layer_num,
           int output_elements, int input_elements,
           Q_type const *input_ptr,
           Q_type const *filter_ptr,
           Q_type *output_ptr)
{
    abstract_layer<OperandT, Q_type, int32_t,C_ob, 1, 1, W_ob, 1, 1, 'c', 1, 1>(
        output_elements, // Output Channel Grouping
        1,               // Output Channels per group
        1,
        1, input_elements,
        1, 1,
        0, 0, 0, 0,
        input_ptr, filter_ptr, output_ptr);
}

//****************************************************************************
template <typename OperandT, typename Q_type>
void Conv2D_rect(int layer_num,
                 int kernel_size_h, int kernel_size_w, int stride,
                 uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                 int output_channels, int input_channels,
                 int input_height, int input_width,
                 Q_type const *input_ptr,
                 Q_type const *filter_ptr,
                 Q_type *output_ptr)
{
    // Specific case for the first layer
    if (input_channels == 3)
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, Q_type, int32_t,1, C_ob, 3, W_ob, 1, 1, 'c', 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size_h, kernel_size_w,
                t_pad, l_pad, r_pad, b_pad,
                input_ptr, filter_ptr, output_ptr);
        }
        else if (stride == 2)
        {
            abstract_layer<OperandT, Q_type, int32_t,1, C_ob, 3, W_ob, 2, 1, 'c', 2, 1>( // unroll?
                1,                                                       // Output Channel Grouping
                output_channels,                                         // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size_h, kernel_size_w,
                t_pad, l_pad, r_pad, b_pad,
                input_ptr, filter_ptr, output_ptr);
        }
        else
        {
            // printf("This stride is unsupported, please change the interface.cpp file\n");
            throw std::invalid_argument("Conv2D_rect ERROR: stride unsupported.");
        }
    }
    else
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, Q_type, int32_t,1, C_ob, C_ob, W_ob, 1, UNROLL, 'c', 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size_h, kernel_size_w,
                t_pad, l_pad, r_pad, b_pad,
                input_ptr, filter_ptr, output_ptr);
        }
        else if (stride == 2)
        {

            abstract_layer<OperandT, Q_type, int32_t,1, C_ob, C_ob, W_ob, 2, UNROLL, 'c', 2, 1>(
                1,               // Output Channel Grouping
                output_channels, // Output Channels per group
                input_channels,
                input_height, input_width,
                kernel_size_h, kernel_size_w,
                t_pad, l_pad, r_pad, b_pad,
                input_ptr, filter_ptr, output_ptr);
        }
        else
        {
            // printf("This stride is unsupported, please change the interface.cpp file\n");
            throw std::invalid_argument("Conv2D_rect ERROR: stride unsupported.");
        }
    }
}

//****************************************************************************
template <typename OperandT, typename Q_type>
void MaxPool2D_rect(int layer_num,
                    int kernel_size_h, int kernel_size_w, int stride,
                    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                    int input_channels,
                    int input_height, int input_width,
                    Q_type const *input_ptr,
                    Q_type *output_ptr)
{
    if (stride == 1)
    {
        abstract_layer<OperandT, Q_type, int32_t,C_ob, 1, 1, W_ob, 1, 1, 'p', 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_size_h, kernel_size_w,
            t_pad, l_pad, r_pad, b_pad,
            input_ptr, NULL, output_ptr);
    }
    else if (stride == 2)
    {
        abstract_layer<OperandT, Q_type, int32_t,C_ob, 1, 1, W_ob, 2, 1, 'p', 1, 1>(
            input_channels, // Output Channel Grouping
            1,              // Output Channels per group
            1,
            input_height, input_width,
            kernel_size_h, kernel_size_w,
            t_pad, l_pad, r_pad, b_pad,
            input_ptr, NULL, output_ptr);
    }
    else
    {
        // printf("This stride is unsupported, please change the interface.cpp file\n");
        throw std::invalid_argument("MaxPool2D_rect ERROR: stride unsupported.");
    }
}

inline void CALC_PADDING(uint32_t I_dim,
                         uint32_t K_dim,
                         uint16_t stride,
                         uint8_t &padding_front,
                         uint8_t &padding_back)
{
    uint32_t padding;
    if (I_dim % stride == 0)
    {
        padding = (K_dim > stride) ? K_dim - stride : 0;
    }
    else
    {
        padding = (K_dim > (I_dim % stride)) ? (K_dim - (I_dim % stride)) : 0;
    }
    padding_front = padding / 2;
    padding_back = padding - padding_front;
}
