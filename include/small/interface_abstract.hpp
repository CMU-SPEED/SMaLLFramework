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
//#include <stdlib.h>
#include <stdint.h>
#include <stdexcept>

// #include <small/abstract_layer.hpp>

//#define RECORD_CALLS

#if defined(RECORD_CALLS)
#include <iostream>
#endif

int clip(int n, int upper, int lower=0)
{
    n = (n > lower) * n + !(n > lower) * lower;
    return (n < upper) * n + !(n < upper) * upper;
}

//Quantization Functions
// template<typename Q_T, typename T>
// void Quantize(int num_elements, T * tensor_ptr, Q_T * quant_tensor_ptr)
// {
//     float scale_inv = (1.0 / quant_tensor_ptr->scale);
//     uint64_t max_val = (1 << quant_tensor_ptr->b) - 1;
//     printf("%lu max scale inverse: %f \n", max_val, scale_inv);
//     int quant_val = rint(quant_tensor_ptr->zero + (0.0 * scale_inv));
//     printf("%f %d  \t %u\n", 0.0, quant_val);
//     for (int i = 0; i < num_elements; i++)
//     {
//         int quant_val = rint(quant_tensor_ptr->zero + (tensor_ptr[i] * scale_inv));
//         quant_tensor_ptr->tensor[i] = (quant_val< max_val)?quant_val:max_val;
//         printf("%f %d  \t %u\n", tensor_ptr[i], quant_val, quant_tensor_ptr->tensor[i]);
//     }
// }

// template <typename Q_T, typename T>
// void DeQuantize(int num_elements, T *tensor_ptr, Q_T *quant_tensor_ptr)
// {
//     for (int i = 0; i < num_elements; i++)
//     {
//         tensor_ptr[i] = (T)(quant_tensor_ptr->scale*(quant_tensor_ptr->tensor[i] - quant_tensor_ptr->zero));
//         // printf("%f\n", tensor_ptr[i]);
//     }
// }

template <typename Q_T, typename T>
void DebugDeQuantize(int num_elements, T *tensor_ptr, Q_T *quant_tensor_ptr)
{
    for (int i = 0; i < num_elements; i++)
    {
        printf("%d\t", quant_tensor_ptr->tensor[i]);
        tensor_ptr[i] = (T)(quant_tensor_ptr->scale * ((T)(quant_tensor_ptr->tensor[i] - quant_tensor_ptr->zero)));
        printf("%f\n", tensor_ptr[i]);
    }
}

//****************************************************************************
template <typename OperandT>
void Conv2D(int layer_num,
            int kernel_size, int stride,  /// @todo dim_t?
            uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
            int output_channels, int input_channels,
            int input_height, int input_width,
            OperandT const *input_ptr,
            OperandT const *filter_ptr,
            OperandT       *output_ptr)
{
#if defined(RECORD_CALLS)
    std::cout << "Conv2D(k:" << kernel_size << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif

    /// @todo do we need another specific case for input_channels==1?

    //Specific case for the first layer
    if (input_channels == 3)
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, 1, C_ob, 3, W_ob, 1, 1, 'c', 2, 1>(
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
            abstract_layer<OperandT, 1, C_ob, 3, W_ob, 2, 1, 'c', 2, 1>(
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
            //printf("This stride is unsupported, please change the interface.cpp file\n");
            throw std::invalid_argument("Conv2D ERROR: stride unsupported.");
        }
    }
    else
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, 1, C_ob, C_ob, W_ob, 1, UNROLL, 'c', 2, 1>(
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
            abstract_layer<OperandT, 1, C_ob, C_ob, W_ob, 2, UNROLL, 'c', 2, 1>(
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
            //printf("This stride is unsupported, please change the interface.cpp file\n");
            throw std::invalid_argument("Conv2D ERROR: stride unsupported.");
        }
    }
}

//****************************************************************************
template <typename OperandT>
void PartialConv2D(int layer_num,
                   int kernel_size, int stride,
                   uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                   int output_channels, int input_channels,
                   int input_height, int input_width,
                   OperandT const *input_ptr,
                   OperandT const *filter_ptr,
                   OperandT       *output_ptr)
{
#if defined(RECORD_CALLS)
    std::cout << "PartialConv2D(k:" << kernel_size << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif
    /// @todo We need another specific case for input_channels==1 (maybe more)

    // Specific case for the first layer
    if (input_channels == 3)
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, 1, C_ob, 3, W_ob, 1, 1, 'c', 2, 0>(
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

            abstract_layer<OperandT, 1, C_ob, 3, W_ob, 2, 1, 'c', 2, 0>(
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
            //printf("This stride is unsupported, please change the interface.cpp file\n");
        }
    }
    else
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, 1, C_ob, C_ob, W_ob, 1, UNROLL, 'c', 2, 0>(
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

            abstract_layer<OperandT, 1, C_ob, C_ob, W_ob, 2, UNROLL, 'c', 2, 0>(
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
            //printf("This stride is unsupported, please change the interface.cpp file\n");
        }
    }
}

//****************************************************************************
template <typename OperandT>
void Maxpool2D(int layer_num,
               int kernel_size, int stride,
               uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
               int input_channels,
               int input_height, int input_width,
               OperandT const *input_ptr,
               OperandT       *output_ptr)
{
#if defined(RECORD_CALLS)
    std::cout << "MaxPool2D(k:" << kernel_size << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    if (stride == 1)
    {
        abstract_layer<OperandT, C_ob, 1, 1, W_ob, 1, 1, 'p', 1, 1>(
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

        abstract_layer<OperandT, C_ob, 1, 1, W_ob, 2, 1, 'p', 1, 1>(
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
        //printf("This stride is unsupported, please change the interface.cpp file\n");
        throw std::invalid_argument("Maxpool2D ERROR: stride unsupported.");
    }
}

//****************************************************************************
template <typename OperandT>
void DepthwiseConv2D(int layer_num,
                     int kernel_size, int stride,
                     uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                     int input_channels,
                     int input_height, int input_width,
                     OperandT const *input_ptr,
                     OperandT const *filter_ptr,
                     OperandT       *output_ptr)
{
#if defined(RECORD_CALLS)
    std::cout << "DepthwiseConv2D(k:" << kernel_size << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif
    if (stride == 1)
    {

        abstract_layer<OperandT, C_ob, 1, 1, W_ob, 1, 1, 'c', 1, 1>(
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

        abstract_layer<OperandT, C_ob, 1, 1, W_ob, 2, 1, 'c', 1, 1>(
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
        //printf("This stride is unsupported, please change the interface.cpp file\n");
        throw std::invalid_argument("DepthwiseConv2D ERROR: stride unsupported.");
    }
}

//****************************************************************************
template <typename OperandT>
void ReLUActivation(int layer_num,
                    int input_channels,
                    int input_height, int input_width,
                    OperandT const *input_ptr,
                    OperandT       *output_ptr,
                    int zero = 0)
{
#if defined(RECORD_CALLS)
    std::cout << "ReLUActivation(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif
    abstract_layer<OperandT, C_ob, 1, 1, W_ob, 1, 1, 'a', 0, 1>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        input_ptr, NULL, output_ptr, zero);
}

// template <typename OperandT>
// void QuantReLUActivation(int layer_num,
//                     int input_channels,
//                     int input_height, int input_width,
//                     OperandT const *input_ptr,
//                     OperandT *output_ptr,
//                     int zero )
// {
//     abstract_layer<OperandT, C_ob, 1, 1, W_ob, 1, 1, 'a', 0, 1>(
//         input_channels, // Output Channel Grouping
//         1,              // Output Channels per group
//         1,
//         input_height, input_width,
//         1, 1,
//         0, 0, 0, 0,
//         input_ptr, NULL, output_ptr);
// }
//****************************************************************************
template <typename OperandT>
void Dense(int layer_num,
           int output_elements, int input_elements,
           OperandT const *input_ptr,
           OperandT const *filter_ptr,
           OperandT       *output_ptr)
{
#if defined(RECORD_CALLS)
    std::cout << "Dense(I,F,O)\n";
#endif
    abstract_layer<OperandT, C_ob, 1, 1, W_ob, 1, 1, 'c', 1, 1>(
        output_elements, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        1, input_elements,
        1, 1,
        0, 0, 0, 0,
        input_ptr, filter_ptr, output_ptr);
}

//****************************************************************************
template <typename OperandT>
void Conv2D_rect(int layer_num,
                 int kernel_size_h, int kernel_size_w, int stride,
                 uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                 int output_channels, int input_channels,
                 int input_height, int input_width,
                 OperandT const *input_ptr,
                 OperandT const *filter_ptr,
                 OperandT       *output_ptr)
{
#if defined(RECORD_CALLS)
    std::cout << "Conv2D_rect(k:" << kernel_size_h << "x" << kernel_size_w
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif

    // Specific case for the first layer
    if (input_channels == 3)
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, 1, C_ob, 3, W_ob, 1, 1, 'c', 2, 1>(
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
            abstract_layer<OperandT, 1, C_ob, 3, W_ob, 2, 1, 'c', 2, 1>(  // unroll?
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
            //printf("This stride is unsupported, please change the interface.cpp file\n");
            throw std::invalid_argument("Conv2D_rect ERROR: stride unsupported.");
        }
    }
    else
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, 1, C_ob, C_ob, W_ob, 1, UNROLL, 'c', 2, 1>(
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

            abstract_layer<OperandT, 1, C_ob, C_ob, W_ob, 2, UNROLL, 'c', 2, 1>(
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
            //printf("This stride is unsupported, please change the interface.cpp file\n");
            throw std::invalid_argument("Conv2D_rect ERROR: stride unsupported.");
        }
    }
}

//****************************************************************************
template <typename OperandT>
void MaxPool2D_rect(int layer_num,
                    int kernel_size_h, int kernel_size_w, int stride,
                    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                    int input_channels,
                    int input_height, int input_width,
                    OperandT const *input_ptr,
                    OperandT       *output_ptr)
{
#if defined(RECORD_CALLS)
    std::cout << "MaxPool2D_rect(k:" << kernel_size_h << "x" << kernel_size_w
              << ",s:" << stride
              << ",pad:[" << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad
              << "],chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif
    if (stride == 1)
    {
        abstract_layer<OperandT, C_ob, 1, 1, W_ob, 1, 1, 'p', 1, 1>(
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
        abstract_layer<OperandT, C_ob, 1, 1, W_ob, 2, 1, 'p', 1, 1>(
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
        //printf("This stride is unsupported, please change the interface.cpp file\n");
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
