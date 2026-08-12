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

#include <small.h>
#include <small/buffers.hpp>
#include <small/Layer.hpp>
#include <small/Conv2DLayer.hpp>

namespace small
{
//****************************************************************************
template <typename BufferT>
class FusedConv2DReLULayer : public Layer<BufferT>
{
public:
    // take ownership of Conv2DLayer buffers, don't need ReLULayer object
    FusedConv2DReLULayer(Conv2DLayer<BufferT> const &conv2d_layer);

    // take ownership of Conv2DLayer buffers, don't need ReLULayer object
    FusedConv2DReLULayer(Conv2DLayer<BufferT> &&conv2d_layer);

    virtual ~FusedConv2DReLULayer()
    {
    }

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const;

private:
    Conv2DLayer<BufferT> m_conv2d_layer;
};

//****************************************************************************
// copy ctor
template <class BufferT>
FusedConv2DReLULayer<BufferT>::FusedConv2DReLULayer(
    Conv2DLayer<BufferT> const &conv2d_layer)
    : m_conv2d_layer(conv2d_layer)
{
    this->set_output_shape(m_conv2d_layer.output_shape());
}


//****************************************************************************
// move ctor
template <class BufferT>
FusedConv2DReLULayer<BufferT>::FusedConv2DReLULayer(
    Conv2DLayer<BufferT> &&conv2d_layer)
    : m_conv2d_layer(std::move(conv2d_layer))
{
    this->set_output_shape(m_conv2d_layer.output_shape());
}


//****************************************************************************
template <class BufferT>
void FusedConv2DReLULayer<BufferT>::compute_output(
    std::vector<Tensor<BufferT> const *> input,
    Tensor<BufferT>*                     output) const
{
    auto& output_shape(this->output_shape());

    // CHECK buffer sizes

    if (m_conv2d_layer.m_packed_bias.size() == 0)
    {
        //std::cerr << "calling Conv2D_ReLU\n";
        small::Conv2D_ReLU(m_conv2d_layer.m_kernel_height,
                           m_conv2d_layer.m_kernel_width,
                           m_conv2d_layer.m_stride,
                           m_conv2d_layer.m_t_pad,
                           m_conv2d_layer.m_b_pad,
                           m_conv2d_layer.m_l_pad,
                           m_conv2d_layer.m_r_pad,
                           output_shape[CHANNEL],
                           m_conv2d_layer.m_input_shape[CHANNEL],
                           m_conv2d_layer.m_input_shape[HEIGHT],
                           m_conv2d_layer.m_input_shape[WIDTH],
                           input[0]->buffer(),
                           m_conv2d_layer.m_packed_filters,
                           output->buffer());
    }
    else
    {
        //std::cerr << "calling Conv2D_Bias_ReLU, bias.size()="
        //          << m_conv2d_layer.m_packed_bias.size() << std::endl;
        small::Conv2D_Bias_ReLU(m_conv2d_layer.m_kernel_height,
                                m_conv2d_layer.m_kernel_width,
                                m_conv2d_layer.m_stride,
                                m_conv2d_layer.m_t_pad,
                                m_conv2d_layer.m_b_pad,
                                m_conv2d_layer.m_l_pad,
                                m_conv2d_layer.m_r_pad,
                                output_shape[CHANNEL],
                                m_conv2d_layer.m_input_shape[CHANNEL],
                                m_conv2d_layer.m_input_shape[HEIGHT],
                                m_conv2d_layer.m_input_shape[WIDTH],
                                input[0]->buffer(),
                                m_conv2d_layer.m_packed_filters,
                                m_conv2d_layer.m_packed_bias,
                                output->buffer());
    }

    output->set_shape(output_shape);
}

} //ns small
