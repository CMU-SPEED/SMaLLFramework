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
#include <small/Conv2DLayer.hpp>  // for detail::initialize_conv2d_buffers

namespace small
{

//****************************************************************************
template <typename BufferT>
class PartialConv2DLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    //PartialConv2DLayer () delete;

    /// @param[in] filters  Unpacked set of filters with dimensions packed
    ///                     in the following order:
    ///                     {in_chans, out_chans, kern_h, kern_w}
    ///
    PartialConv2DLayer(shape_type const &input_shape,    //pred.output_shape()
                       uint32_t          kernel_size,
                       uint32_t          stride,
                       PaddingEnum       padding_type,
                       uint32_t          num_output_channels,
                       BufferT    const &filters,
                       bool              buffers_are_packed = true,
                       ActivationType    activation_type = NONE,
                       float             leaky_slope = 1.e-2)
        : Layer<BufferT>(),
          m_input_shape(input_shape),
          m_kernel_size(kernel_size),
          m_stride(stride),
          m_activation_type(activation_type),
          m_t_pad(0), m_b_pad(0), m_l_pad(0), m_r_pad(0),
          m_leaky_slope(1),  /// @note Allocating 1-element buffer
          m_packed_filters(num_output_channels*input_shape[CHANNEL]*
                           kernel_size*kernel_size),
          m_packed_bias()
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "PartialConv2D(batches:" << m_input_shape[BATCH]
                  << ",k:" << kernel_size
                  << ",s:" << stride
                  << ",p:" << ((padding_type == PADDING_V) ? "'v'" : "'f'")
                  << ",ichans:" << m_input_shape[CHANNEL]
                  << ",ochans:" << num_output_channels
                  << ",img:" << m_input_shape[HEIGHT]
                  << "x" << m_input_shape[WIDTH]
                  << "), filter.size=" << filters.size() << std::endl;
#endif
        if (((input_shape[CHANNEL] % BufferT::C_ib) != 0) &&
            (input_shape[CHANNEL] != 3))
        {
            throw std::invalid_argument(
                "PartialConv2DLayer::ctor ERROR: "
                "invalid number of input channels.");
        }
        if ((num_output_channels % BufferT::C_ob) != 0)
        {
            throw std::invalid_argument(
                "PartialConv2DLayer::ctor ERROR: "
                "invalid number of output channels.");
        }

        m_leaky_slope[0] = leaky_slope;
        compute_padding_output_shape(input_shape,
                                     kernel_size, kernel_size,
                                     stride,
                                     padding_type,
                                     num_output_channels);

        detail::initialize_conv2d_buffers(
            num_output_channels,
            num_output_channels,
            m_input_shape[CHANNEL],
            m_kernel_size, m_kernel_size,
            filters,
            BufferT(),  // empty bias
            BufferT(), BufferT(), BufferT(), BufferT(), 0.f, // no BN
            buffers_are_packed,
            m_packed_filters,
            m_packed_bias);

#if defined(DEBUG_LAYERS)
        auto &output_shape = this->output_shape(0);
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
    }

    PartialConv2DLayer(shape_type const &input_shape,    //pred.output_shape()
                       uint32_t          kernel_size,
                       uint32_t          stride,
                       PaddingEnum       padding_type,
                       uint32_t          num_output_channels,
                       BufferT    const &filters,
                       BufferT    const &bias,
                       bool              buffers_are_packed = true,
                       ActivationType    activation_type = NONE,
                       float             leaky_slope = 1.e-2)
        : Layer<BufferT>(),
          m_input_shape(input_shape),
          m_kernel_size(kernel_size),
          m_stride(stride),
          m_activation_type(activation_type),
          m_t_pad(0), m_b_pad(0), m_l_pad(0), m_r_pad(0),
          m_leaky_slope(1),  /// @note Allocating 1-element buffer
          m_packed_filters(num_output_channels*m_input_shape[CHANNEL]*
                           kernel_size*kernel_size)
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "PartialConv2D(batches:" << m_input_shape[BATCH]
                  << ",k:" << kernel_size
                  << ",s:" << stride
                  << ",p:" << ((padding_type == PADDING_V) ? "'v'" : "'f'")
                  << ",ichans:" << m_input_shape[CHANNEL]
                  << ",ochans:" << num_output_channels
                  << ",img:" << m_input_shape[HEIGHT]
                  << "x" << m_input_shape[WIDTH]
                  << "),filters.size=" << filters.size()
                  << ",bias.size=" << bias.size() << std::endl;
#endif
        if (((input_shape[CHANNEL] % BufferT::C_ib) != 0) &&
            (input_shape[CHANNEL] != 3))
        {
            throw std::invalid_argument(
                "PartialConv2DLayer::ctor ERROR: "
                "invalid number of input channels.");
        }
        if ((num_output_channels % BufferT::C_ob) != 0)
        {
            throw std::invalid_argument(
                "PartialConv2DLayer::ctor ERROR: "
                "invalid number of output channels.");
        }

        m_leaky_slope[0] = leaky_slope;
        compute_padding_output_shape(input_shape,
                                     kernel_size, kernel_size,
                                     stride,
                                     padding_type,
                                     num_output_channels);

        detail::initialize_conv2d_buffers(
            num_output_channels,
            num_output_channels,
            m_input_shape[CHANNEL],
            m_kernel_size, m_kernel_size,
            filters,
            bias,
            BufferT(), BufferT(), BufferT(), BufferT(), 0.f, // no BN
            buffers_are_packed,
            m_packed_filters,
            m_packed_bias);


#if defined(DEBUG_LAYERS)
        auto &output_shape = this->output_shape(0);
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

    }

    PartialConv2DLayer(shape_type const &input_shape,    //pred.output_shape()
                       uint32_t          kernel_size,
                       uint32_t          stride,
                       PaddingEnum       padding_type,
                       uint32_t          num_output_channels,
                       BufferT    const &filters,
                       BufferT    const &bn_weight,            // gamma
                       BufferT    const &bn_bias,              // beta
                       BufferT    const &bn_running_mean,      // mu_hat
                       BufferT    const &bn_running_variance,  // sigma_hat^2
                       float      const &bn_eps = 1.e-5,       // float?
                       bool              buffers_are_packed = true,
                       ActivationType    activation_type = NONE,
                       float             leaky_slope = 1.e-2)
        : Layer<BufferT>(),
          m_input_shape(input_shape),
          m_kernel_size(kernel_size),
          m_stride(stride),
          m_activation_type(activation_type),
          m_t_pad(0), m_b_pad(0), m_l_pad(0), m_r_pad(0),
          m_leaky_slope(1),  /// @note Allocating 1-element buffer
          m_packed_filters(num_output_channels*m_input_shape[CHANNEL]*
                           kernel_size*kernel_size)
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "PartialConv2D(batches:" << m_input_shape[BATCH]
                  << ",k:" << kernel_size
                  << ",s:" << stride
                  << ",p:" << ((padding_type == PADDING_V) ? "'v'" : "'f'")
                  << ",ichans:" << m_input_shape[CHANNEL]
                  << ",ochans:" << num_output_channels
                  << ",img:" << m_input_shape[HEIGHT]
                  << "x" << m_input_shape[WIDTH]
                  << "), filters.size=" << filters.size()
                  << ",bn.sizes(weight,bias,run_var,run_avg)=("
                  << bn_weight.size()
                  << "," << bn_bias.size()
                  << "," << bn_running_variance.size()
                  << "," << bn_running_mean.size()
                  << "),bn_eps:" << bn_eps << std::endl;
#endif
        if (((input_shape[CHANNEL] % BufferT::C_ib) != 0) &&
            (input_shape[CHANNEL] != 3))
        {
            throw std::invalid_argument(
                "PartialConv2DLayer::ctor ERROR: "
                "invalid number of input channels.");
        }
        if ((num_output_channels % BufferT::C_ob) != 0)
        {
            throw std::invalid_argument(
                "PartialConv2DLayer::ctor ERROR: "
                "invalid number of output channels.");
        }

        m_leaky_slope[0] = leaky_slope;
        compute_padding_output_shape(input_shape,
                                     kernel_size, kernel_size,
                                     stride,
                                     padding_type,
                                     num_output_channels);

        detail::initialize_conv2d_buffers(
            num_output_channels,
            num_output_channels,
            m_input_shape[CHANNEL],
            m_kernel_size, m_kernel_size,
            filters,
            BufferT(), // no bias
            bn_weight, bn_bias,
            bn_running_mean, bn_running_variance, bn_eps,
            buffers_are_packed,
            m_packed_filters,
            m_packed_bias);


#if defined(DEBUG_LAYERS)
        auto &output_shape = this->output_shape(0);
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
                      << ",slope:" << m_leaky_slope[0]
                      << ",img:" << output_shape[HEIGHT]
                      << "x" << output_shape[WIDTH]
                      << ")" << std::endl;
        }
#endif
    }

    virtual ~PartialConv2DLayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        std::vector<Tensor<BufferT>*>        output) const
    {
        if ((input.size() != 1) || (input[0]->shape() != m_input_shape))
        {
            throw std::invalid_argument(
                "PartialConv2DLayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        if ((output.size() != 1) || (output[0]->capacity() < this->output_size(0)))
        {
            throw std::invalid_argument(
                "PartialConv2DLayer::compute_output() ERROR: "
                "insufficient output buffer space.");
        }

        auto& output_shape(this->output_shape(0));

        PartialConv2D(m_kernel_size,
                      m_stride,
                      m_t_pad, m_b_pad, m_l_pad, m_r_pad,
                      output_shape[CHANNEL],
                      m_input_shape[CHANNEL],
                      m_input_shape[HEIGHT], m_input_shape[WIDTH],
                      input[0]->buffer(),
                      m_packed_filters,
                      output[0]->buffer());

        // HACK: placeholder for bias term
        if (m_packed_bias.size() == output_shape[CHANNEL])
        {
            for (size_t Co = 0; Co < output_shape[CHANNEL]; ++Co)
            {
                for (size_t h = 0; h < output_shape[HEIGHT]; ++h)
                {
                    for (size_t w = 0; w < output_shape[WIDTH]; ++w)
                    {
                        size_t idx = packed_buffer_index(output_shape[CHANNEL],
                                                         output_shape[HEIGHT],
                                                         output_shape[WIDTH],
                                                         BufferT::C_ob,
                                                         Co, h, w);
                        output[0]->buffer()[idx] += m_packed_bias[Co];
                    }
                }
            }
        }

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

    //************************************************************************
    void compute_padding_output_shape(shape_type const &input_shape,
                                      uint32_t          kernel_height,
                                      uint32_t          kernel_width,
                                      uint32_t          stride,
                                      PaddingEnum       padding_type,
                                      uint32_t          num_output_channels)
    {
        shape_type output_shape;

        /// @todo is there a clean way to make these const members, or
        ///       will image size get moved to compute_output and all of
        ///       this moves to compute output?
        output_shape[BATCH] = input_shape[BATCH];
        output_shape[CHANNEL] = num_output_channels;
        small::compute_padding_output_dim(input_shape[HEIGHT], kernel_height,
                                          stride, padding_type,
                                          m_t_pad, m_b_pad,
                                          output_shape[HEIGHT]);
        small::compute_padding_output_dim(input_shape[WIDTH], kernel_width,
                                          stride, padding_type,
                                          m_l_pad, m_r_pad,
                                          output_shape[WIDTH]);

#if defined(DEBUG_LAYERS)
        std::cerr << "PartialConv2D padding: "
                  << (int)m_t_pad << "," << (int)m_b_pad
                  << "," << (int)m_l_pad << "," << (int)m_r_pad << std::endl;
#endif

        this->set_output_shapes({output_shape});
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
    BufferT          m_packed_bias;
};

}
