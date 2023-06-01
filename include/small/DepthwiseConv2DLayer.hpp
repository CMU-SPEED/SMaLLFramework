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

namespace small
{

//****************************************************************************
template <typename BufferT>
class DepthwiseConv2DLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    //DepthwiseConv2DLayer () delete;

    /// @param[in] filters  Unpacked set of filters with dimensions packed
    ///                     in the following order:
    ///                     {in_chans, out_chans, kern_h, kern_w}
    ///
    DepthwiseConv2DLayer(shape_type const &input_shape,
                         uint32_t          kernel_size,
                         uint32_t          stride,
                         PaddingEnum       padding_type,
                         BufferT    const &filters,      /// @todo support move
                         bool              filters_are_packed = true,
                         ActivationType    activation_type = NONE,
                         float             leaky_slope = 1.e-2)
        : Layer<BufferT>(),
          m_input_shape(input_shape),
          m_kernel_size(kernel_size),
          m_stride(stride),
          m_activation_type(activation_type),
          m_t_pad(0), m_b_pad(0), m_l_pad(0), m_r_pad(0),
          m_leaky_slope(1),
          m_packed_filters(input_shape[CHANNEL]*kernel_size*kernel_size)
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "DWConv(batches:" << m_input_shape[BATCH]
                  << ",k:" << kernel_size
                  << ",s:" << stride
                  << ",p:" << ((padding_type == PADDING_V) ? "'v'" : "'f'")
                  << ",chans:" << m_input_shape[CHANNEL]
                  << ",img:" << m_input_shape[HEIGHT]
                  << "x" << m_input_shape[WIDTH]
                  << "), filters.size=" << filters.size() << std::endl;
#endif

        if (filters.size() < m_input_shape[CHANNEL]*kernel_size*kernel_size)
        {
            throw std::invalid_argument(
                "DepthwiseConv2DLayer::ctor ERROR: "
                "filters buffer too small.");
        }

        /// @todo is there a clean way to make these const members, or
        ///       will image size get moved to compute_output and all of
        ///       this moves to compute output?
        shape_type output_shape;
        output_shape[BATCH] = m_input_shape[BATCH];
        output_shape[CHANNEL] = m_input_shape[CHANNEL];
        small::compute_padding_output_dim(m_input_shape[HEIGHT], kernel_size,
                                          stride, padding_type,
                                          m_t_pad, m_b_pad,
                                          output_shape[HEIGHT]);
        small::compute_padding_output_dim(m_input_shape[WIDTH], kernel_size,
                                          stride, padding_type,
                                          m_l_pad, m_r_pad,
                                          output_shape[WIDTH]);

#if defined(DEBUG_LAYERS)
        std::cerr << "DW padding: " << (int)m_t_pad << "," << (int)m_b_pad
                  << "," << (int)m_l_pad << "," << (int)m_r_pad << std::endl;

        if (activation_type == RELU)
        {
            std::cerr << "ReLU(batches:" << output_shape[BATCH]
                      << ",chans:" << output_shape[CHANNEL]
                      << ",img:" << output_shape[HEIGHT]
                      << "x" << output_shape[WIDTH]
                      << ")" << std::endl;
        }
        else if (activation_type == LEAKY)
        {
            std::cerr << "LeakyReLU(batches:" << output_shape[BATCH]
                      << ",chans:" << output_shape[CHANNEL]
                      << ",slope:" << leaky_slope
                      << ",img:" << output_shape[HEIGHT]
                      << "x" << output_shape[WIDTH]
                      << ")" << std::endl;
        }
#endif

        this->set_output_shapes({output_shape});

        m_leaky_slope[0] = leaky_slope;
        // Pack the filter buffers for SMaLL use
        if (!filters_are_packed)
        {
            small::pack_buffer(filters,
                               FILTER_DW,
                               output_shape[CHANNEL], 1U,
                               m_kernel_size, m_kernel_size,
                               C_ib, C_ob,
                               m_packed_filters);
        }
        else
        {
            std::copy(filters.data(),
                      filters.data() + m_packed_filters.size(),
                      m_packed_filters.data());
        }
    }

    virtual ~DepthwiseConv2DLayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        std::vector<Tensor<BufferT>*>        output) const
    {
        if ((input.size() != 1) || (input[0]->shape() != m_input_shape))
        {
            throw std::invalid_argument(
                "DepthwiseConv2DLayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        if ((output.size() != 1) || (output[0]->capacity() < this->output_size(0)))
        {
            throw std::invalid_argument(
                "DepthwiseConv2DLayer::compute_output() ERROR: "
                "insufficient output buffer space.");
        }

        DepthwiseConv2D(m_kernel_size, m_stride,
                        m_t_pad, m_b_pad, m_l_pad, m_r_pad,
                        m_input_shape[CHANNEL],
                        m_input_shape[HEIGHT], m_input_shape[WIDTH],
                        input[0]->buffer(),
                        m_packed_filters,
                        output[0]->buffer());


        auto& output_shape(this->output_shape(0));
        output[0]->set_shape(output_shape);

        if (m_activation_type == RELU)
        {
            small::ReLUActivation(output_shape[CHANNEL],
                                  output_shape[HEIGHT], output_shape[WIDTH],
                                  output[0]->buffer(),
                                  output[0]->buffer());
        }
        else if (m_activation_type == LEAKY)
        {
            small::LeakyReLUActivation(output_shape[CHANNEL],
                                       output_shape[HEIGHT], output_shape[WIDTH],
                                       output[0]->buffer(),
                                       m_leaky_slope,
                                       output[0]->buffer());
        }
    }

private:
    shape_type const m_input_shape;

    uint32_t   const m_kernel_size;
    uint32_t   const m_stride;

    ActivationType const m_activation_type;

    /// @todo: how to make const?
    uint8_t          m_t_pad, m_b_pad, m_l_pad, m_r_pad;

    BufferT          m_leaky_slope;
    BufferT          m_packed_filters;
};

}
