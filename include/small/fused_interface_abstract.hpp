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

#define SMALL_HAS_FLOAT_SUPPORT 1

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
    // Convolution Fused Layers
    //****************************************************************************
    //****************************************************************************

    //****************************************************************************
    //****************************************************************************
    template <class BufferT>
    void Conv2D_ReLU(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
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
    void Conv2D_ReLU<FloatBuffer>(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
        int output_channels, int input_channels,
        int input_height, int input_width,
        FloatBuffer const &input_buf,
        FloatBuffer const &filter_buf,
        FloatBuffer &output_buf)
    {
#if defined(RECORD_CALLS)
        std::cout << "Conv2D_ReLU<float>(k:"
                  << conv_kernel_height << "x" << conv_kernel_width
                  << ",s:" << stride
                  << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
                  << "," << (int)conv_l_pad << "," << (int)conv_r_pad
                  << "],ochans:" << output_channels
                  << ",ichans:" << input_channels
                  << ",img:" << input_height << "x" << input_width
                  << ",I,F,O)\n";
#endif


        small::Mapping<FloatBuffer> convolution_params;
        convolution_params.G = 1;
        convolution_params.K = output_channels;
        convolution_params.F_c = input_channels;
        convolution_params.F_h = conv_kernel_height;
        convolution_params.F_w = conv_kernel_width;
        convolution_params.pad_top = conv_t_pad;
        convolution_params.pad_left = conv_l_pad;
        convolution_params.pad_right = conv_r_pad;
        convolution_params.pad_bottom = conv_b_pad;
        convolution_params.F = &filter_buf;

        small::Mapping<FloatBuffer> relu_params;
        relu_params.G = input_channels;
        relu_params.K = 1;
        relu_params.F_c = 1;
        relu_params.F_h = 1;
        relu_params.F_w = 1;
        relu_params.pad_top = 0;
        relu_params.pad_left = 0;
        relu_params.pad_right = 0;
        relu_params.pad_bottom = 0;
        relu_params.F = NULL;

        if (input_channels % FLOAT_C_ib == 0)
        {
            if (conv_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,
                    OP_NONE, OP_RELU>(
                    &convolution_params,
                    input_height, input_width,
                    &input_buf, &output_buf);

                // detail::fused_abstract_layer<
                //     FloatBuffer,
                //     1, FLOAT_C_ob, FLOAT_C_ib,
                //     FLOAT_W_ob,
                //     1,
                //     FLOAT_UNROLL,
                //     OP_CONV, 2,
                //     1,

                //     FLOAƒT_C_ob, 1, 1,
                //     FLOAT_W_ob,
                //     1,
                //     FLOAT_UNROLL,
                //     OP_RELU, 1,
                //     1,

                //     OP_NONE, OP_NONE,
                //     0, 0>(
                //     &convolution_params,
                //     &relu_params,
                //     input_height, input_width,
                //     &input_buf, &output_buf, &output_buf);
            }
            else if (conv_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,
                    OP_NONE, OP_RELU>(
                    &convolution_params,
                    input_height, input_width,
                    &input_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }

        if (input_channels < FLOAT_C_ib && input_channels == 3)
        {
            if (conv_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,
                    OP_NONE, OP_RELU>(
                    &convolution_params,
                    input_height, input_width,
                    &input_buf, &output_buf);
            }
            else if (conv_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,
                    OP_NONE, OP_RELU>(
                    &convolution_params,
                    input_height, input_width,
                    &input_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
    }

#endif

    template <class BufferT>
    void Conv2D_Bias_ReLU(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
        int output_channels, int input_channels,
        int input_height, int input_width,
        BufferT const &input_buf,
        BufferT const &filter_buf,
        BufferT const &bias_buf,
        BufferT &output_buf)
    {
        BufferT::unimplemented_function();
    }
//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    template <>
    void Conv2D_Bias_ReLU<FloatBuffer>(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
        int output_channels, int input_channels,
        int input_height, int input_width,
        FloatBuffer const &input_buf,
        FloatBuffer const &filter_buf,
        FloatBuffer const &bias_buf,
        FloatBuffer &output_buf)
    {
#if defined(RECORD_CALLS)
        std::cout << "Conv2D_ReLU<float>(k:"
                  << conv_kernel_height << "x" << conv_kernel_width
                  << ",s:" << stride
                  << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
                  << "," << (int)conv_l_pad << "," << (int)conv_r_pad
                  << "],ochans:" << output_channels
                  << ",ichans:" << input_channels
                  << ",img:" << input_height << "x" << input_width
                  << ",I,F,O)\n";
#endif

        /// @todo add an assert for invalid numbers of output channels
        ///       (layer classes should be responsible for padding filters).


        small::Mapping<FloatBuffer> convolution_params;
        convolution_params.G = 1;
        convolution_params.K = output_channels;
        convolution_params.F_c = input_channels;
        convolution_params.F_h = conv_kernel_height;
        convolution_params.F_w = conv_kernel_width;
        convolution_params.pad_top = conv_t_pad;
        convolution_params.pad_left = conv_l_pad;
        convolution_params.pad_right = conv_r_pad;
        convolution_params.pad_bottom = conv_b_pad;
        convolution_params.F = &filter_buf;
        convolution_params.F_before = &bias_buf;

        if (input_channels % FLOAT_C_ib == 0)
        {
            if (conv_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max()>(
                    &convolution_params,
                    input_height, input_width,
                    &input_buf, &output_buf);
            }
            else if (conv_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max()>(
                    &convolution_params,
                    input_height, input_width,
                    &input_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
        else if(input_channels == 3 && input_channels < FLOAT_C_ob)
        {
            if (conv_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max()>(
                    &convolution_params,
                    input_height, input_width,
                    &input_buf, &output_buf);
            }
            else if (conv_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max()>(
                    &convolution_params,
                    input_height, input_width,
                    &input_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
    }

#endif



//****************************************************************************
//****************************************************************************
template <class BufferT>
void Conv2D_Bias_ReLU_Maxpool2D(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

    int pool_kernel_height, int pool_kernel_width, int pool_stride,
    uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT const &bias_buf,
    BufferT &inter_output_buf,
    BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void Conv2D_Bias_ReLU_Maxpool2D<FloatBuffer>(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

    int pool_kernel_height, int pool_kernel_width, int pool_stride,
    uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,

    int output_channels, int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer const &filter_buf,
    FloatBuffer const &bias_buf,
    FloatBuffer &inter_output_buf,
    FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Conv2D_Bias_ReLU_Maxpool2D<float>(k:"
              << conv_kernel_height << "x" << conv_kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
              << "," << (int)conv_l_pad << "," << (int)conv_r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif

    /// @todo add an assert for invalid numbers of output channels
    ///       (layer classes should be responsible for padding filters).

    auto conv_output_height = small::output_dim_new((input_height + conv_t_pad + conv_b_pad),
                                                    conv_stride, conv_kernel_height);
    auto conv_output_width = small::output_dim_new((input_width + conv_l_pad + conv_r_pad),
                                                   conv_stride, conv_kernel_width);

    small::Mapping<FloatBuffer> convolution_params;
    convolution_params.G = 1;
    convolution_params.K = output_channels;
    convolution_params.F_c = input_channels;
    convolution_params.F_h = conv_kernel_height;
    convolution_params.F_w = conv_kernel_width;
    convolution_params.pad_top = conv_t_pad;
    convolution_params.pad_left = conv_l_pad;
    convolution_params.pad_right = conv_r_pad;
    convolution_params.pad_bottom = conv_b_pad;
    convolution_params.F = &filter_buf;
    convolution_params.F_before = &bias_buf;

    small::Mapping<FloatBuffer> max_pooling_params;
    max_pooling_params.G = input_channels;
    max_pooling_params.K = 1;
    max_pooling_params.F_c = 1;
    max_pooling_params.F_h = pool_kernel_height;
    max_pooling_params.F_w = pool_kernel_width;
    max_pooling_params.pad_top = pool_t_pad;
    max_pooling_params.pad_left = pool_l_pad;
    max_pooling_params.pad_right = pool_r_pad;
    max_pooling_params.pad_bottom = pool_b_pad;
    max_pooling_params.F = NULL;

    if (input_channels % FLOAT_C_ib == 0)
    {
        if (conv_stride == 1 && pool_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 1 && pool_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 2 && pool_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 2 && pool_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
        }
    }
    // Specific case for the first layer
    else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
    {
        if (conv_stride == 1 && pool_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, 3,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1,
                OP_NONE, OP_NONE,
                0, 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 1 && pool_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, 3,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1,
                OP_NONE, OP_NONE,
                0, 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 2 && pool_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, 3,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1,
                OP_NONE, OP_NONE,
                0, 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 2 && pool_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, 3,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1,
                OP_NONE, OP_NONE,
                0, 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
        }

        // small::MaxPool2D(pool_kernel_height, pool_kernel_width, pool_stride,
        //                  pool_t_pad, pool_b_pad, pool_l_pad, pool_r_pad,
        //                  output_channels,
        //                  conv_output_height, conv_output_width,
        //                  inter_output_buf, output_buf);
        
    }
    else
    {
        throw std::invalid_argument(
            "Conv2D_ReLU<float> ERROR: in_channels unsupported.");
    }


}

#endif

//****************************************************************************
//****************************************************************************
template <class BufferT>
void Conv2D_Maxpool2D(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

    int pool_kernel_height, int pool_kernel_width, int pool_stride,
    uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &inter_output_buf,
    BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    template <>
    void Conv2D_Maxpool2D<FloatBuffer>(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

        int pool_kernel_height, int pool_kernel_width, int pool_stride,
        uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,

        int output_channels, int input_channels,
        int input_height, int input_width,
        FloatBuffer const &input_buf,
        FloatBuffer const &filter_buf,
        FloatBuffer &inter_output_buf,
        FloatBuffer &output_buf)
    {
#if defined(RECORD_CALLS)
        std::cout << "Conv2D_Maxpool2D<float>(k:"
                  << conv_kernel_height << "x" << conv_kernel_width
                  << ",s:" << stride
                  << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
                  << "," << (int)conv_l_pad << "," << (int)conv_r_pad
                  << "],ochans:" << output_channels
                  << ",ichans:" << input_channels
                  << ",img:" << input_height << "x" << input_width
                  << ",I,F,O)\n";
#endif

        /// @todo add an assert for invalid numbers of output channels
        ///       (layer classes should be responsible for padding filters).

        auto conv_output_height = small::output_dim_new((input_height + conv_t_pad + conv_b_pad),
                                                        conv_stride, conv_kernel_height);
        auto conv_output_width = small::output_dim_new((input_width + conv_l_pad + conv_r_pad),
                                                       conv_stride, conv_kernel_width);

        small::Mapping<FloatBuffer> convolution_params;
        convolution_params.G = 1;
        convolution_params.K = output_channels;
        convolution_params.F_c = input_channels;
        convolution_params.F_h = conv_kernel_height;
        convolution_params.F_w = conv_kernel_width;
        convolution_params.pad_top = conv_t_pad;
        convolution_params.pad_left = conv_l_pad;
        convolution_params.pad_right = conv_r_pad;
        convolution_params.pad_bottom = conv_b_pad;
        convolution_params.F = &filter_buf;

        small::Mapping<FloatBuffer> max_pooling_params;
        max_pooling_params.G = input_channels;
        max_pooling_params.K = 1;
        max_pooling_params.F_c = 1;
        max_pooling_params.F_h = pool_kernel_height;
        max_pooling_params.F_w = pool_kernel_width;
        max_pooling_params.pad_top = pool_t_pad;
        max_pooling_params.pad_left = pool_l_pad;
        max_pooling_params.pad_right = pool_r_pad;
        max_pooling_params.pad_bottom = pool_b_pad;
        max_pooling_params.F = NULL;

        if (input_channels % FLOAT_C_ib == 0)
        {
            if (conv_stride == 1 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_MAX_POOL, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &max_pooling_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 1 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_MAX_POOL, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &max_pooling_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_MAX_POOL, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &max_pooling_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_MAX_POOL, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &max_pooling_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }

        // Specific case for the first layer
        else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
        {
            if (conv_stride == 1 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_MAX_POOL, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &max_pooling_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 1 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_MAX_POOL, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &max_pooling_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_MAX_POOL, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &max_pooling_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_MAX_POOL, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &max_pooling_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D_ReLU<float> ERROR: in_channels unsupported.");
        }

        // small::MaxPool2D(pool_kernel_height, pool_kernel_width, pool_stride,
        //                  pool_t_pad, pool_b_pad, pool_l_pad, pool_r_pad,
        //                  output_channels,
        //                  conv_output_height, conv_output_width,
        //                  inter_output_buf, output_buf);
    }

#endif

    //****************************************************************************
    //****************************************************************************
    template <class BufferT>
    void Conv2D_ReLU_Maxpool2D(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

        int pool_kernel_height, int pool_kernel_width, int pool_stride,
        uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,
        int output_channels, int input_channels,
        int input_height, int input_width,
        BufferT const &input_buf,
        BufferT const &filter_buf,
        BufferT &inter_output_buf,
        BufferT &output_buf)
    {
        BufferT::unimplemented_function();
    }

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void Conv2D_ReLU_Maxpool2D<FloatBuffer>(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

    int pool_kernel_height, int pool_kernel_width, int pool_stride,
    uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,

    int output_channels, int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer const &filter_buf,
    FloatBuffer &inter_output_buf,
    FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Conv2D_ReLU_Maxpool2D<float>(k:"
              << conv_kernel_height << "x" << conv_kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
              << "," << (int)conv_l_pad << "," << (int)conv_r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif

    /// @todo add an assert for invalid numbers of output channels
    ///       (layer classes should be responsible for padding filters).

    auto conv_output_height = small::output_dim_new((input_height + conv_t_pad + conv_b_pad),
                                                    conv_stride, conv_kernel_height);
    auto conv_output_width = small::output_dim_new((input_width + conv_l_pad + conv_r_pad),
                                                   conv_stride, conv_kernel_width);


    small::Mapping<FloatBuffer> convolution_params;
    convolution_params.G = 1;
    convolution_params.K = output_channels;
    convolution_params.F_c = input_channels;
    convolution_params.F_h = conv_kernel_height;
    convolution_params.F_w = conv_kernel_width;
    convolution_params.pad_top = conv_t_pad;
    convolution_params.pad_left = conv_l_pad;
    convolution_params.pad_right = conv_r_pad;
    convolution_params.pad_bottom = conv_b_pad;
    convolution_params.F = &filter_buf;

    small::Mapping<FloatBuffer> max_pooling_params;
    max_pooling_params.G = input_channels;
    max_pooling_params.K = 1;
    max_pooling_params.F_c = 1;
    max_pooling_params.F_h = pool_kernel_height;
    max_pooling_params.F_w = pool_kernel_width;
    max_pooling_params.pad_top = pool_t_pad;
    max_pooling_params.pad_left = pool_l_pad;
    max_pooling_params.pad_right = pool_r_pad;
    max_pooling_params.pad_bottom = pool_b_pad;
    max_pooling_params.F = NULL;

    if (input_channels % FLOAT_C_ib == 0)
    {
        if (conv_stride == 1 && pool_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_NONE, OP_RELU,
                0, 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 1 && pool_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_NONE, OP_RELU,
                0, 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 2 && pool_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_NONE, OP_RELU,
                0, 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 2 && pool_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_NONE, OP_RELU,
                0, 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
    {
        if (conv_stride == 1 && pool_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, 3,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_NONE, OP_RELU,
                0, 1,
                OP_NONE, OP_NONE,
                0, 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 1 && pool_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, 3,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_NONE, OP_RELU,
                0, 1,
                OP_NONE, OP_NONE,
                0, 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 2 && pool_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, 3,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_NONE, OP_RELU,
                0, 1,
                OP_NONE, OP_NONE,
                0, 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 2 && pool_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, 3,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_MAX_POOL, 1,
                1,

                OP_NONE, OP_RELU,
                0, 1,
                OP_NONE, OP_NONE,
                0, 1>(
                &convolution_params,
                &max_pooling_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "Conv2D_ReLU<float> ERROR: in_channels unsupported.");
    }

    // small::MaxPool2D(pool_kernel_height, pool_kernel_width, pool_stride,
    //                  pool_t_pad, pool_b_pad, pool_l_pad, pool_r_pad,
    //                  output_channels,
    //                  conv_output_height, conv_output_width,
    //                  inter_output_buf, output_buf);
}

#endif

//****************************************************************************
//****************************************************************************
// Depthwise Convolution Fused Layers
//****************************************************************************
//****************************************************************************

//****************************************************************************
//****************************************************************************
template <class BufferT>
void DepthwiseConv2D_ReLU(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
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
void DepthwiseConv2D_ReLU<FloatBuffer>(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
    int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer const &filter_buf,
    FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "DepthwiseConv2D_ReLU<float>(k:"
              << conv_kernel_height << "x" << conv_kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
              << "," << (int)conv_l_pad << "," << (int)conv_r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif



    small::Mapping<FloatBuffer> dwise_convolution_params;
    dwise_convolution_params.G = input_channels;
    dwise_convolution_params.K = 1;
    dwise_convolution_params.F_c = 1;
    dwise_convolution_params.F_h = conv_kernel_height;
    dwise_convolution_params.F_w = conv_kernel_width;
    dwise_convolution_params.pad_top = conv_t_pad;
    dwise_convolution_params.pad_left = conv_l_pad;
    dwise_convolution_params.pad_right = conv_r_pad;
    dwise_convolution_params.pad_bottom = conv_b_pad;
    dwise_convolution_params.F = &filter_buf;

    if (input_channels % FLOAT_C_ib == 0)
    {
        if (conv_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer, 
                FLOAT_C_ob, 1, 1, 
                FLOAT_W_ob, 
                1, 
                FLOAT_UNROLL, 
                OP_CONV, 1, 
                1, 
                OP_NONE, OP_RELU>(
                &dwise_convolution_params,
                input_height, input_width,
                &input_buf, &output_buf);
        }
        else if (conv_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer, 
                FLOAT_C_ob, 1, 1, 
                FLOAT_W_ob, 
                2, 
                FLOAT_UNROLL, 
                OP_CONV, 1, 
                1, 
                OP_NONE, OP_RELU>(
                &dwise_convolution_params,
                input_height, input_width,
                &input_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "DepthwiseConv2D_ReLU<float> ERROR: conv_stride unsupported.");
        }
    }
}

#endif

template <class BufferT>
void DepthwiseConv2D_Bias_ReLU(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
    int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT const &bias_buf,
    BufferT &output_buf)
{
    BufferT::unimplemented_function();
}
//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void DepthwiseConv2D_Bias_ReLU<FloatBuffer>(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
    int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer const &filter_buf,
    FloatBuffer const &bias_buf,
    FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "DepthwiseConv2D_ReLU<float>(k:"
              << conv_kernel_height << "x" << conv_kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
              << "," << (int)conv_l_pad << "," << (int)conv_r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif

    /// @todo add an assert for invalid numbers of output channels
    ///       (layer classes should be responsible for padding filters).



    small::Mapping<FloatBuffer> dwise_convolution_params;
    dwise_convolution_params.G = input_channels;
    dwise_convolution_params.K = 1;
    dwise_convolution_params.F_c = 1;
    dwise_convolution_params.F_h = conv_kernel_height;
    dwise_convolution_params.F_w = conv_kernel_width;
    dwise_convolution_params.pad_top = conv_t_pad;
    dwise_convolution_params.pad_left = conv_l_pad;
    dwise_convolution_params.pad_right = conv_r_pad;
    dwise_convolution_params.pad_bottom = conv_b_pad;
    dwise_convolution_params.F = &filter_buf;
    dwise_convolution_params.F_before = &bias_buf;

    if (input_channels % FLOAT_C_ib == 0)
    {
        if (conv_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer, 
                FLOAT_C_ob, 1, 1, 
                FLOAT_W_ob, 
                1, 
                FLOAT_UNROLL, 
                OP_CONV, 1, 
                1, 
                OP_UPSAMPLE, OP_RELU, 
                std::numeric_limits<dim_t>::max()>(
                &dwise_convolution_params,
                input_height, input_width,
                &input_buf, &output_buf);
        }
        else if (conv_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer, 
                FLOAT_C_ob, 1, 1, 
                FLOAT_W_ob, 
                2, 
                FLOAT_UNROLL, 
                OP_CONV, 1, 
                1, 
                OP_UPSAMPLE, OP_RELU, 
                std::numeric_limits<dim_t>::max()>(
                &dwise_convolution_params,
                input_height, input_width,
                &input_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "DepthwiseConv2D_ReLU<float> ERROR: conv_stride unsupported.");
        }
    }
}

#endif

//****************************************************************************
//****************************************************************************
template <class BufferT>
void Conv2D_DepthwiseConv2D(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

    int pool_kernel_height, int pool_kernel_width, int pool_stride,
    uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT &inter_output_buf,
    BufferT const &dwise_filter_buf,
    BufferT &output_buf)
{
    BufferT::unimplemented_function();
}

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    template <>
    void Conv2D_DepthwiseConv2D<FloatBuffer>(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

        int pool_kernel_height, int pool_kernel_width, int pool_stride,
        uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,

        int output_channels, int input_channels,
        int input_height, int input_width,
        FloatBuffer const &input_buf,
        FloatBuffer const &filter_buf,
        FloatBuffer &inter_output_buf,
        FloatBuffer const &dwise_filter_buf,
        FloatBuffer &output_buf)
    {
#if defined(RECORD_CALLS)
        std::cout << "Conv2D_DepthwiseConv2D<float>(k:"
                  << conv_kernel_height << "x" << conv_kernel_width
                  << ",s:" << stride
                  << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
                  << "," << (int)conv_l_pad << "," << (int)conv_r_pad
                  << "],ochans:" << output_channels
                  << ",ichans:" << input_channels
                  << ",img:" << input_height << "x" << input_width
                  << ",I,F,O)\n";
#endif

        /// @todo add an assert for invalid numbers of output channels
        ///       (layer classes should be responsible for padding filters).

        auto conv_output_height = small::output_dim_new((input_height + conv_t_pad + conv_b_pad),
                                                        conv_stride, conv_kernel_height);
        auto conv_output_width = small::output_dim_new((input_width + conv_l_pad + conv_r_pad),
                                                       conv_stride, conv_kernel_width);

        small::Mapping<FloatBuffer> convolution_params;
        convolution_params.G = 1;
        convolution_params.K = output_channels;
        convolution_params.F_c = input_channels;
        convolution_params.F_h = conv_kernel_height;
        convolution_params.F_w = conv_kernel_width;
        convolution_params.pad_top = conv_t_pad;
        convolution_params.pad_left = conv_l_pad;
        convolution_params.pad_right = conv_r_pad;
        convolution_params.pad_bottom = conv_b_pad;
        convolution_params.F = &filter_buf;

        small::Mapping<FloatBuffer> dwise_convolution_params;
        dwise_convolution_params.G = output_channels;
        dwise_convolution_params.K = 1;
        dwise_convolution_params.F_c = 1;
        dwise_convolution_params.F_h = pool_kernel_height;
        dwise_convolution_params.F_w = pool_kernel_width;
        dwise_convolution_params.pad_top = pool_t_pad;
        dwise_convolution_params.pad_left = pool_l_pad;
        dwise_convolution_params.pad_right = pool_r_pad;
        dwise_convolution_params.pad_bottom = pool_b_pad;
        dwise_convolution_params.F = &dwise_filter_buf;

        if (input_channels % FLOAT_C_ib == 0)
        {
            if (conv_stride == 1 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 1 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
        // Specific case for the first layer
        else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
        {
            if (conv_stride == 1 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 1 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D_ReLU<float> ERROR: in_channels unsupported.");
        }

        // }
    }

#endif

    //****************************************************************************
    //****************************************************************************
    template <class BufferT>
    void Conv2D_ReLU_DepthwiseConv2D_ReLU(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

        int pool_kernel_height, int pool_kernel_width, int pool_stride,
        uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,
        int output_channels, int input_channels,
        int input_height, int input_width,
        BufferT const &input_buf,
        BufferT const &filter_buf,
        BufferT &inter_output_buf,
        BufferT const &dwise_filter_buf,
        BufferT &output_buf)
    {
        BufferT::unimplemented_function();
    }

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    template <>
    void Conv2D_ReLU_DepthwiseConv2D_ReLU<FloatBuffer>(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

        int pool_kernel_height, int pool_kernel_width, int pool_stride,
        uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,

        int output_channels, int input_channels,
        int input_height, int input_width,
        FloatBuffer const &input_buf,
        FloatBuffer const &filter_buf,
        FloatBuffer &inter_output_buf,
        FloatBuffer const &dwise_filter_buf,
        FloatBuffer &output_buf)
    {
#if defined(RECORD_CALLS)
        std::cout << "Conv2D_ReLU_DepthwiseConv2D_ReLU<float>(k:"
                  << conv_kernel_height << "x" << conv_kernel_width
                  << ",s:" << stride
                  << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
                  << "," << (int)conv_l_pad << "," << (int)conv_r_pad
                  << "],ochans:" << output_channels
                  << ",ichans:" << input_channels
                  << ",img:" << input_height << "x" << input_width
                  << ",I,F,O)\n";
#endif

        /// @todo add an assert for invalid numbers of output channels
        ///       (layer classes should be responsible for padding filters).

        auto conv_output_height = small::output_dim_new((input_height + conv_t_pad + conv_b_pad),
                                                        conv_stride, conv_kernel_height);
        auto conv_output_width = small::output_dim_new((input_width + conv_l_pad + conv_r_pad),
                                                       conv_stride, conv_kernel_width);

        small::Mapping<FloatBuffer> convolution_params;
        convolution_params.G = 1;
        convolution_params.K = output_channels;
        convolution_params.F_c = input_channels;
        convolution_params.F_h = conv_kernel_height;
        convolution_params.F_w = conv_kernel_width;
        convolution_params.pad_top = conv_t_pad;
        convolution_params.pad_left = conv_l_pad;
        convolution_params.pad_right = conv_r_pad;
        convolution_params.pad_bottom = conv_b_pad;
        convolution_params.F = &filter_buf;

        small::Mapping<FloatBuffer> dwise_convolution_params;
        dwise_convolution_params.G = output_channels;
        dwise_convolution_params.K = 1;
        dwise_convolution_params.F_c = 1;
        dwise_convolution_params.F_h = pool_kernel_height;
        dwise_convolution_params.F_w = pool_kernel_width;
        dwise_convolution_params.pad_top = pool_t_pad;
        dwise_convolution_params.pad_left = pool_l_pad;
        dwise_convolution_params.pad_right = pool_r_pad;
        dwise_convolution_params.pad_bottom = pool_b_pad;
        dwise_convolution_params.F = &dwise_filter_buf;

        if (input_channels % FLOAT_C_ib == 0)
        {
            if (conv_stride == 1 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 1 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
        // Specific case for the first layer
        else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
        {
            if (conv_stride == 1 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 1 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    FLOAT_C_ob, 1, 1,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 1,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &dwise_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D_ReLU<float> ERROR: in_channels unsupported.");
        }


        // }
    }

#endif

//****************************************************************************
//****************************************************************************
template <class BufferT>
void Conv2D_Bias_ReLU_DepthwiseConv2D_Bias_ReLU(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

    int pool_kernel_height, int pool_kernel_width, int pool_stride,
    uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    BufferT const &input_buf,
    BufferT const &filter_buf,
    BufferT const &bias_buf,
    BufferT &inter_output_buf,
    BufferT const &dwise_filter_buf,
    BufferT const &dwise_bias_buf,
    BufferT &output_buf)
{
    BufferT::unimplemented_function();
}


//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void Conv2D_Bias_ReLU_DepthwiseConv2D_Bias_ReLU<FloatBuffer>(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

    int pool_kernel_height, int pool_kernel_width, int pool_stride,
    uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,

    int output_channels, int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer const &filter_buf,
    FloatBuffer const &bias_buf,
    FloatBuffer &inter_output_buf,
    FloatBuffer const &dwise_filter_buf,
    FloatBuffer const &dwise_bias_buf,
    FloatBuffer &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "Conv2D_Bias_ReLU_DepthwiseConv2D_Bias_ReLU<float>(k:"
              << conv_kernel_height << "x" << conv_kernel_width
              << ",s:" << stride
              << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
              << "," << (int)conv_l_pad << "," << (int)conv_r_pad
              << "],ochans:" << output_channels
              << ",ichans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,F,O)\n";
#endif

    /// @todo add an assert for invalid numbers of output channels
    ///       (layer classes should be responsible for padding filters).

    // auto conv_output_height = small::output_dim_new((input_height + conv_t_pad + conv_b_pad),
    //                                                 conv_stride, conv_kernel_height);
    // auto conv_output_width = small::output_dim_new((input_width + conv_l_pad + conv_r_pad),
    //                                                conv_stride, conv_kernel_width);

    small::Mapping<FloatBuffer> convolution_params;
    convolution_params.G = 1;
    convolution_params.K = output_channels;
    convolution_params.F_c = input_channels;
    convolution_params.F_h = conv_kernel_height;
    convolution_params.F_w = conv_kernel_width;
    convolution_params.pad_top = conv_t_pad;
    convolution_params.pad_left = conv_l_pad;
    convolution_params.pad_right = conv_r_pad;
    convolution_params.pad_bottom = conv_b_pad;
    convolution_params.F = &filter_buf;
    convolution_params.F_before = &bias_buf;

    small::Mapping<FloatBuffer> dwise_convolution_params;
    dwise_convolution_params.G = output_channels;
    dwise_convolution_params.K = 1;
    dwise_convolution_params.F_c = 1;
    dwise_convolution_params.F_h = pool_kernel_height;
    dwise_convolution_params.F_w = pool_kernel_width;
    dwise_convolution_params.pad_top = pool_t_pad;
    dwise_convolution_params.pad_left = pool_l_pad;
    dwise_convolution_params.pad_right = pool_r_pad;
    dwise_convolution_params.pad_bottom = pool_b_pad;
    dwise_convolution_params.F = &dwise_filter_buf;
    dwise_convolution_params.F_before = &dwise_bias_buf;

    if (input_channels % FLOAT_C_ib == 0)
    {
        if (conv_stride == 1 && pool_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1,
                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1
                >(
                &convolution_params,
                &dwise_convolution_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 1 && pool_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1,
                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1
                >(
                &convolution_params,
                &dwise_convolution_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 2 && pool_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1,
                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1>(
                &convolution_params,
                &dwise_convolution_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 2 && pool_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1,
                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1>(
                &convolution_params,
                &dwise_convolution_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
        }
    }

    // Specific case for the first layer
    else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
    {
        if (conv_stride == 1 && pool_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, 3,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1,
                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1>(
                &convolution_params,
                &dwise_convolution_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 1 && pool_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, 3,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1,
                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1>(
                &convolution_params,
                &dwise_convolution_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 2 && pool_stride == 1)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, 3,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                1,
                FLOAT_UNROLL,
                OP_CONV, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1,
                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1>(
                &convolution_params,
                &dwise_convolution_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else if (conv_stride == 2 && pool_stride == 2)
        {

            detail::fused_abstract_layer<
                FloatBuffer,
                1, FLOAT_C_ob, 3,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 2,
                1,

                FLOAT_C_ob, 1, 1,
                FLOAT_W_ob,
                2,
                FLOAT_UNROLL,
                OP_CONV, 1,
                1,

                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1,
                OP_UPSAMPLE, OP_RELU,
                std::numeric_limits<dim_t>::max(), 1>(
                &convolution_params,
                &dwise_convolution_params,
                input_height, input_width,
                &input_buf, &inter_output_buf, &output_buf);
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
        }
    }
    else
    {
        throw std::invalid_argument(
            "Conv2D_ReLU<float> ERROR: in_channels unsupported.");
    }

}

#endif

//****************************************************************************
// Fused Operations with Group Convolutions (Only supporting group ==  C/C_ob and K = C_ob, F_c = C_ob)
//****************************************************************************

//****************************************************************************
//****************************************************************************
template <class BufferT>
void GroupConv2D_ReLU(
    int conv_kernel_height, int conv_kernel_width, int conv_stride,
    int output_groups,
    uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
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
    void GroupConv2D_ReLU<FloatBuffer>(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        int output_groups,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
        int input_channels,
        int input_height, int input_width,
        FloatBuffer const &input_buf,
        FloatBuffer const &filter_buf,
        FloatBuffer &output_buf)
    {
#if defined(RECORD_CALLS)
        std::cout << "GroupConv2D_ReLU<float>(k:"
                  << conv_kernel_height << "x" << conv_kernel_width
                  << ",s:" << stride
                  << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
                  << "," << (int)conv_l_pad << "," << (int)conv_r_pad
                  << "],ochans:" << output_channels
                  << ",ichans:" << input_channels
                  << ",img:" << input_height << "x" << input_width
                  << ",I,F,O)\n";
#endif

        small::Mapping<FloatBuffer> group_convolution_params;
        group_convolution_params.G = output_groups;
        group_convolution_params.K = FLOAT_C_ob;
        group_convolution_params.F_c = FLOAT_C_ib;
        group_convolution_params.F_h = conv_kernel_height;
        group_convolution_params.F_w = conv_kernel_width;
        group_convolution_params.pad_top = conv_t_pad;
        group_convolution_params.pad_left = conv_l_pad;
        group_convolution_params.pad_right = conv_r_pad;
        group_convolution_params.pad_bottom = conv_b_pad;
        group_convolution_params.F = &filter_buf;

        if (input_channels % FLOAT_C_ib == 0)
        {
            if (conv_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,
                    OP_NONE, OP_RELU>(
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &output_buf);
            }
            else if (conv_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,
                    OP_NONE, OP_RELU>(
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "GroupConv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
    }

#endif

    template <class BufferT>
    void GroupConv2D_Bias_ReLU(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        int output_groups,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
        int input_channels,
        int input_height, int input_width,
        BufferT const &input_buf,
        BufferT const &filter_buf,
        BufferT const &bias_buf,
        BufferT &output_buf)
    {
        BufferT::unimplemented_function();
    }
//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    template <>
    void GroupConv2D_Bias_ReLU<FloatBuffer>(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        int output_groups,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,
        int input_channels,
        int input_height, int input_width,
        FloatBuffer const &input_buf,
        FloatBuffer const &filter_buf,
        FloatBuffer const &bias_buf,
        FloatBuffer &output_buf)
    {
#if defined(RECORD_CALLS)
        std::cout << "GroupConv2D_ReLU<float>(k:"
                  << conv_kernel_height << "x" << conv_kernel_width
                  << ",s:" << stride
                  << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
                  << "," << (int)conv_l_pad << "," << (int)conv_r_pad
                  << "],ochans:" << output_channels
                  << ",ichans:" << input_channels
                  << ",img:" << input_height << "x" << input_width
                  << ",I,F,O)\n";
#endif

        /// @todo add an assert for invalid numbers of output channels
        ///       (layer classes should be responsible for padding filters).

        small::Mapping<FloatBuffer> group_convolution_params;
        group_convolution_params.G = output_groups;
        group_convolution_params.K = FLOAT_C_ob;
        group_convolution_params.F_c = FLOAT_C_ib;
        group_convolution_params.F_h = conv_kernel_height;
        group_convolution_params.F_w = conv_kernel_width;
        group_convolution_params.pad_top = conv_t_pad;
        group_convolution_params.pad_left = conv_l_pad;
        group_convolution_params.pad_right = conv_r_pad;
        group_convolution_params.pad_bottom = conv_b_pad;
        group_convolution_params.F = &filter_buf;
        group_convolution_params.F_before = &bias_buf;

        if (input_channels % FLOAT_C_ib == 0)
        {
            if (conv_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max()>(
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &output_buf);
            }
            else if (conv_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max()>(
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "GroupConv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
    }

#endif

    //****************************************************************************
    //****************************************************************************
    template <class BufferT>
    void Conv2D_GroupConv(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

        int pool_kernel_height, int pool_kernel_width, int pool_stride,
        int output_groups,
        uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,
        int output_channels, int input_channels,
        int input_height, int input_width,
        BufferT const &input_buf,
        BufferT const &filter_buf,
        BufferT &inter_output_buf,
        BufferT const &dwise_filter_buf,
        BufferT &output_buf)
    {
        BufferT::unimplemented_function();
    }

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    template <>
    void Conv2D_GroupConv<FloatBuffer>(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

        int pool_kernel_height, int pool_kernel_width, int pool_stride,
        int output_groups,
        uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,

        int output_channels, int input_channels,
        int input_height, int input_width,
        FloatBuffer const &input_buf,
        FloatBuffer const &filter_buf,
        FloatBuffer &inter_output_buf,
        FloatBuffer const &dwise_filter_buf,
        FloatBuffer &output_buf)
    {
#if defined(RECORD_CALLS)
        std::cout << "Conv2D_GroupConv<float>(k:"
                  << conv_kernel_height << "x" << conv_kernel_width
                  << ",s:" << stride
                  << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
                  << "," << (int)conv_l_pad << "," << (int)conv_r_pad
                  << "],ochans:" << output_channels
                  << ",ichans:" << input_channels
                  << ",img:" << input_height << "x" << input_width
                  << ",I,F,O)\n";
#endif

        /// @todo add an assert for invalid numbers of output channels
        ///       (layer classes should be responsible for padding filters).

        auto conv_output_height = small::output_dim_new((input_height + conv_t_pad + conv_b_pad),
                                                        conv_stride, conv_kernel_height);
        auto conv_output_width = small::output_dim_new((input_width + conv_l_pad + conv_r_pad),
                                                       conv_stride, conv_kernel_width);

        small::Mapping<FloatBuffer> convolution_params;
        convolution_params.G = 1;
        convolution_params.K = output_channels;
        convolution_params.F_c = input_channels;
        convolution_params.F_h = conv_kernel_height;
        convolution_params.F_w = conv_kernel_width;
        convolution_params.pad_top = conv_t_pad;
        convolution_params.pad_left = conv_l_pad;
        convolution_params.pad_right = conv_r_pad;
        convolution_params.pad_bottom = conv_b_pad;
        convolution_params.F = &filter_buf;

        small::Mapping<FloatBuffer> group_convolution_params;
        group_convolution_params.G = output_groups;
        group_convolution_params.K = FLOAT_C_ob;
        group_convolution_params.F_c = FLOAT_C_ib;
        group_convolution_params.F_h = pool_kernel_height;
        group_convolution_params.F_w = pool_kernel_width;
        group_convolution_params.pad_top = pool_t_pad;
        group_convolution_params.pad_left = pool_l_pad;
        group_convolution_params.pad_right = pool_r_pad;
        group_convolution_params.pad_bottom = pool_b_pad;
        group_convolution_params.F = &dwise_filter_buf;

        if (input_channels % FLOAT_C_ib == 0)
        {
            if (conv_stride == 1 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 1 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
        // Specific case for the first layer
        else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
        {
            if (conv_stride == 1 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 1 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_NONE,
                    0, 1,
                    OP_NONE, OP_NONE,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D_ReLU<float> ERROR: in_channels unsupported.");
        }

        // }
    }

#endif

    //****************************************************************************
    //****************************************************************************

    template <class BufferT>
    void Conv2D_ReLU_GroupConv2D_ReLU(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

        int pool_kernel_height, int pool_kernel_width, int pool_stride,
        int output_groups,
        uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,
        int output_channels, int input_channels,
        int input_height, int input_width,
        BufferT const &input_buf,
        BufferT const &filter_buf,
        BufferT &inter_output_buf,
        BufferT const &dwise_filter_buf,
        BufferT &output_buf)
    {
        BufferT::unimplemented_function();
    }

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    template <>
    void Conv2D_ReLU_GroupConv2D_ReLU<FloatBuffer>(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

        int pool_kernel_height, int pool_kernel_width, int pool_stride,
        int output_groups,
        uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,

        int output_channels, int input_channels,
        int input_height, int input_width,
        FloatBuffer const &input_buf,
        FloatBuffer const &filter_buf,
        FloatBuffer &inter_output_buf,
        FloatBuffer const &dwise_filter_buf,
        FloatBuffer &output_buf)
    {
#if defined(RECORD_CALLS)
        std::cout << "Conv2D_ReLU_GroupConv2D_ReLU<float>(k:"
                  << conv_kernel_height << "x" << conv_kernel_width
                  << ",s:" << stride
                  << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
                  << "," << (int)conv_l_pad << "," << (int)conv_r_pad
                  << "],ochans:" << output_channels
                  << ",ichans:" << input_channels
                  << ",img:" << input_height << "x" << input_width
                  << ",I,F,O)\n";
#endif

        /// @todo add an assert for invalid numbers of output channels
        ///       (layer classes should be responsible for padding filters).

        auto conv_output_height = small::output_dim_new((input_height + conv_t_pad + conv_b_pad),
                                                        conv_stride, conv_kernel_height);
        auto conv_output_width = small::output_dim_new((input_width + conv_l_pad + conv_r_pad),
                                                       conv_stride, conv_kernel_width);

        small::Mapping<FloatBuffer> convolution_params;
        convolution_params.G = 1;
        convolution_params.K = output_channels;
        convolution_params.F_c = input_channels;
        convolution_params.F_h = conv_kernel_height;
        convolution_params.F_w = conv_kernel_width;
        convolution_params.pad_top = conv_t_pad;
        convolution_params.pad_left = conv_l_pad;
        convolution_params.pad_right = conv_r_pad;
        convolution_params.pad_bottom = conv_b_pad;
        convolution_params.F = &filter_buf;

        small::Mapping<FloatBuffer> group_convolution_params;
        group_convolution_params.G = output_channels / FLOAT_C_ob;
        group_convolution_params.K = FLOAT_C_ob;
        group_convolution_params.F_c = FLOAT_C_ob;
        group_convolution_params.F_h = pool_kernel_height;
        group_convolution_params.F_w = pool_kernel_width;
        group_convolution_params.pad_top = pool_t_pad;
        group_convolution_params.pad_left = pool_l_pad;
        group_convolution_params.pad_right = pool_r_pad;
        group_convolution_params.pad_bottom = pool_b_pad;
        group_convolution_params.F = &dwise_filter_buf;

        if (input_channels % FLOAT_C_ib == 0)
        {
            if (conv_stride == 1 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 1 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
        // Specific case for the first layer
        else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
        {
            if (conv_stride == 1 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 1 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_NONE, OP_RELU,
                    0, 1,
                    OP_NONE, OP_RELU,
                    0, 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D_ReLU<float> ERROR: in_channels unsupported.");
        }

        // }
    }

#endif

    //****************************************************************************
    //****************************************************************************
    template <class BufferT>
    void Conv2D_Bias_ReLU_GroupConv2D_Bias_ReLU(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

        int pool_kernel_height, int pool_kernel_width, int pool_stride,
        int output_groups,
        uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,
        int output_channels, int input_channels,
        int input_height, int input_width,
        BufferT const &input_buf,
        BufferT const &filter_buf,
        BufferT const &bias_buf,
        BufferT &inter_output_buf,
        BufferT const &dwise_filter_buf,
        BufferT const &dwise_bias_buf,
        BufferT &output_buf)
    {
        BufferT::unimplemented_function();
    }

//============================================================================
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    template <>
    void Conv2D_Bias_ReLU_GroupConv2D_Bias_ReLU<FloatBuffer>(
        int conv_kernel_height, int conv_kernel_width, int conv_stride,
        uint8_t conv_t_pad, uint8_t conv_b_pad, uint8_t conv_l_pad, uint8_t conv_r_pad,

        int pool_kernel_height, int pool_kernel_width, int pool_stride,
        int output_groups,
        uint8_t pool_t_pad, uint8_t pool_b_pad, uint8_t pool_l_pad, uint8_t pool_r_pad,

        int output_channels, int input_channels,
        int input_height, int input_width,
        FloatBuffer const &input_buf,
        FloatBuffer const &filter_buf,
        FloatBuffer const &bias_buf,
        FloatBuffer &inter_output_buf,
        FloatBuffer const &dwise_filter_buf,
        FloatBuffer const &dwise_bias_buf,
        FloatBuffer &output_buf)
    {
#if defined(RECORD_CALLS)
        std::cout << "Conv2D_Bias_ReLU_GroupConv2D_Bias_ReLU<float>(k:"
                  << conv_kernel_height << "x" << conv_kernel_width
                  << ",s:" << stride
                  << ",pad:[" << (int)conv_t_pad << "," << (int)conv_b_pad
                  << "," << (int)conv_l_pad << "," << (int)conv_r_pad
                  << "],ochans:" << output_channels
                  << ",ichans:" << input_channels
                  << ",img:" << input_height << "x" << input_width
                  << ",I,F,O)\n";
#endif

        /// @todo add an assert for invalid numbers of output channels
        ///       (layer classes should be responsible for padding filters).

        // auto conv_output_height = small::output_dim_new((input_height + conv_t_pad + conv_b_pad),
        //                                                 conv_stride, conv_kernel_height);
        // auto conv_output_width = small::output_dim_new((input_width + conv_l_pad + conv_r_pad),
        //                                                conv_stride, conv_kernel_width);

        small::Mapping<FloatBuffer> convolution_params;
        convolution_params.G = 1;
        convolution_params.K = output_channels;
        convolution_params.F_c = input_channels;
        convolution_params.F_h = conv_kernel_height;
        convolution_params.F_w = conv_kernel_width;
        convolution_params.pad_top = conv_t_pad;
        convolution_params.pad_left = conv_l_pad;
        convolution_params.pad_right = conv_r_pad;
        convolution_params.pad_bottom = conv_b_pad;
        convolution_params.F = &filter_buf;
        convolution_params.F_before = &bias_buf;

        small::Mapping<FloatBuffer> group_convolution_params;
        group_convolution_params.G = output_groups;
        group_convolution_params.K = FLOAT_C_ob;
        group_convolution_params.F_c = FLOAT_C_ib;
        group_convolution_params.F_h = pool_kernel_height;
        group_convolution_params.F_w = pool_kernel_width;
        group_convolution_params.pad_top = pool_t_pad;
        group_convolution_params.pad_left = pool_l_pad;
        group_convolution_params.pad_right = pool_r_pad;
        group_convolution_params.pad_bottom = pool_b_pad;
        group_convolution_params.F = &dwise_filter_buf;
        group_convolution_params.F_before = &dwise_bias_buf;

        if (input_channels % FLOAT_C_ib == 0)
        {
            if (conv_stride == 1 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 1 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }

        // Specific case for the first layer
        else if ((input_channels == 3) && (input_channels < FLOAT_C_ib))
        {
            if (conv_stride == 1 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 1 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 1)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    1,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else if (conv_stride == 2 && pool_stride == 2)
            {

                detail::fused_abstract_layer<
                    FloatBuffer,
                    1, FLOAT_C_ob, 3,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    1, FLOAT_C_ob, FLOAT_C_ib,
                    FLOAT_W_ob,
                    2,
                    FLOAT_UNROLL,
                    OP_CONV, 2,
                    1,

                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1,
                    OP_UPSAMPLE, OP_RELU,
                    std::numeric_limits<dim_t>::max(), 1>(
                    &convolution_params,
                    &group_convolution_params,
                    input_height, input_width,
                    &input_buf, &inter_output_buf, &output_buf);
            }
            else
            {
                throw std::invalid_argument(
                    "Conv2D_ReLU<float> ERROR: conv_stride unsupported.");
            }
        }
        else
        {
            throw std::invalid_argument(
                "Conv2D_ReLU<float> ERROR: in_channels unsupported.");
        }
    }

#endif

}
