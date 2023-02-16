/*
 * SMaLL Framework
 *
 * Copyright 2022 Carnegie Mellon University and Authors.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DMxx-xxxx
 */

#include <math.h>
#include <assert.h>
// #include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

//#include "../config.h"
// #include "check_interface.h"
#include <params.h>  //platform specific params

typedef uint32_t index_t;
typedef uint32_t dim_t;
//typedef OperandT operand_t;

#include <../reference/intrinsics.h>  // hardcode reference intrinsics ICK
#include <small/abstract_layer.hpp>

//****************************************************************************
template <typename OperandT>
void check_Conv2D(int layer_num,
                  int kernel_size, int stride,
                  uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                  int output_channels, int input_channels,
                  int input_height, int input_width,
                  OperandT const *input_ptr,
                  OperandT const *filter_ptr,
                  OperandT *output_ptr)
{
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
            printf("This stride is unsupported, please change the interface.cpp file\n");
        }
    }
    else
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, 1, C_ob, C_ob, W_ob, 1, 1, 'c', 2, 1>(
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

            abstract_layer<OperandT, 1, C_ob, C_ob, W_ob, 2, 1, 'c', 2, 1>(
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
            printf("This stride is unsupported, please change the interface.cpp file\n");
        }
    }
}

//****************************************************************************
template <typename OperandT>
void check_PartialConv2D(int layer_num,
                         int kernel_size, int stride,
                         uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                         int output_channels, int input_channels,
                         int input_height, int input_width,
                         OperandT const *input_ptr,
                         OperandT const *filter_ptr,
                         OperandT       *output_ptr)
{
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
            printf("This stride is unsupported, please change the interface.cpp file\n");
        }
    }
    else
    {
        if (stride == 1)
        {
            abstract_layer<OperandT, 1, C_ob, C_ob, W_ob, 1, 1, 'c', 2, 0>(
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

            abstract_layer<OperandT, 1, C_ob, C_ob, W_ob, 2, 1, 'c', 2, 0>(
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
            printf("This stride is unsupported, please change the interface.cpp file\n");
        }
    }
}

//****************************************************************************
template <typename OperandT>
void check_Maxpool2D(int layer_num,
                     int kernel_size, int stride,
                     uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                     int input_channels,
                     int input_height, int input_width,
                     OperandT const *input_ptr,
                     OperandT       *output_ptr)
{
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
        printf("This stride is unsupported, please change the interface.cpp file\n");
    }
}

//****************************************************************************
template <typename OperandT>
void check_DepthwiseConv2D(int layer_num,
                           int kernel_size, int stride,
                           uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                           int input_channels,
                           int input_height, int input_width,
                           OperandT const *input_ptr,
                           OperandT const *filter_ptr,
                           OperandT       *output_ptr)
{
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
        printf("This stride is unsupported, please change the interface.cpp file\n");
    }
}

//****************************************************************************
template <typename OperandT>
void check_ReLUActivation(int layer_num,
                          int input_channels,
                          int input_height, int input_width,
                          OperandT const *input_ptr,
                          OperandT       *output_ptr)
{
    // printf("Cob = %d W_ob = %d\n", C_ob, W_ob);
    abstract_layer<OperandT, C_ob, 1, 1, W_ob, 1, 1, 'a', 0, 1>(
        input_channels, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        input_height, input_width,
        1, 1,
        0, 0, 0, 0,
        input_ptr, NULL, output_ptr);

}

//****************************************************************************
template <typename OperandT>
void check_Dense(int layer_num,
                 int output_elements, int input_elements,
                 OperandT const *input_ptr,
                 OperandT const *filter_ptr,
                 OperandT       *output_ptr)
{
    abstract_layer<OperandT, C_ob, 1, 1, W_ob, 1, 1, 'c', 1, 1>(
        output_elements, // Output Channel Grouping
        1,              // Output Channels per group
        1,
        1, input_elements,
        1, 1,
        0, 0, 0, 0,
        input_ptr, filter_ptr, output_ptr);
}
