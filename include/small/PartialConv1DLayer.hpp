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
#include <small/Conv1DLayer.hpp>  // for detail::initialize_conv1d_buffers

namespace small
{
//****************************************************************************
template <typename BufferT>
class PartialConv1DLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    //PartialConv1DLayer () delete;

    // No bias, no batch normalization
    PartialConv1DLayer(shape_type const &input_shape,    //pred.output_shape()
                       uint32_t          kernel_width,
                       uint32_t          stride,
                       PaddingEnum       padding_type,
                       uint32_t          num_logical_output_channels,
                       BufferT    const &filters,
                       bool              buffers_are_packed = true,
                       ActivationType    activation_type = NONE,
                       float             leaky_slope = 1.e-2);

    // With bias term
    PartialConv1DLayer(shape_type const &input_shape,    //pred.output_shape()
                       uint32_t          kernel_width,
                       uint32_t          stride,
                       PaddingEnum       padding_type,
                       uint32_t          num_logical_output_channels,
                       BufferT    const &filters,
                       BufferT    const &bias,
                       bool              buffers_are_packed = true,
                       ActivationType    activation_type = NONE,
                       float             leaky_slope = 1.e-2);

    // With fused batch normalization
    PartialConv1DLayer(shape_type const &input_shape,    //pred.output_shape()
                       uint32_t          kernel_width,
                       uint32_t          stride,
                       PaddingEnum       padding_type,
                       uint32_t          num_logical_output_channels,
                       BufferT    const &filters,
                       BufferT    const &bn_weight,            // gamma
                       BufferT    const &bn_bias,              // beta
                       BufferT    const &bn_running_mean,      // mu_hat
                       BufferT    const &bn_running_variance,  // sigma_hat^2
                       float      const &bn_eps = 1.e-3,       // float?
                       bool              buffers_are_packed = true,
                       ActivationType    activation_type = NONE,
                       float             leaky_slope = 1.e-2);

    // With bias and fused batch normalization
    PartialConv1DLayer(shape_type const &input_shape,    //pred.output_shape()
                       uint32_t          kernel_width,
                       uint32_t          stride,
                       PaddingEnum       padding_type,
                       uint32_t          num_logical_output_channels,
                       BufferT    const &filters,
                       BufferT    const &bias,
                       BufferT    const &bn_weight,            // gamma
                       BufferT    const &bn_bias,              // beta
                       BufferT    const &bn_running_mean,      // mu_hat
                       BufferT    const &bn_running_variance,  // sigma_hat^2
                       float      const &bn_eps = 1.e-3,       // float?
                       bool              buffers_are_packed = true,
                       ActivationType    activation_type = NONE,
                       float             leaky_slope = 1.e-2);

    virtual ~PartialConv1DLayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const;

    BufferT const &get_packed_filters() const { return m_packed_filters; }
    BufferT const &get_packed_bias()    const { return m_packed_bias; }

private:
    void initialize(
        shape_type const &input_shape,
        uint32_t          kernel_width,
        uint32_t          stride,
        PaddingEnum       padding_type,
        uint32_t          num_logical_output_channels,
        BufferT    const &filters,
        BufferT    const &bias,
        BufferT    const &bn_weight,            // gamma
        BufferT    const &bn_bias,              // beta
        BufferT    const &bn_running_mean,      // mu_hat
        BufferT    const &bn_running_variance,  // sigma_hat^2
        float      const &bn_eps,               // float?
        bool              buffers_are_packed,
        ActivationType    activation_type,
        float             leaky_slope);

    void compute_padding_output_shape(
        shape_type const &input_shape,
        uint32_t          kernel_width,
        uint32_t          stride,
        PaddingEnum       padding_type,
        uint32_t          num_output_channels,
        uint32_t          num_logical_output_channels);

private:
    shape_type const m_input_shape;

    uint32_t   const m_kernel_width;
    uint32_t   const m_stride;

    ActivationType const m_activation_type;

    /// @todo: how to make const?
    uint8_t          m_l_pad, m_r_pad;

    BufferT          m_leaky_slope;
    BufferT          m_packed_filters;
    BufferT          m_packed_bias;
};

//****************************************************************************

//****************************************************************************
template <class BufferT>
void PartialConv1DLayer<BufferT>::initialize(
    shape_type const &input_shape,
    uint32_t          kernel_width,
    uint32_t          stride,
    PaddingEnum       padding_type,
    uint32_t          num_logical_output_channels,
    BufferT    const &filters,
    BufferT    const &bias,
    BufferT    const &bn_weight,            // gamma
    BufferT    const &bn_bias,              // beta
    BufferT    const &bn_running_mean,      // mu_hat
    BufferT    const &bn_running_variance,  // sigma_hat^2
    float      const &bn_eps,               // float?
    bool              buffers_are_packed,
    ActivationType    activation_type,
    float             leaky_slope)
{
    if (((input_shape[CHANNEL] % BufferT::C_ib) != 0) &&
        (input_shape[CHANNEL] != 3) &&
        (input_shape[CHANNEL] != 2) &&
        (input_shape[CHANNEL] != 1))
    {
        throw std::invalid_argument(
            "PartialConv1DLayer::ctor ERROR: invalid number of input channels.");
    }

    if ((stride != 1) && (stride != 2))
    {
        throw std::invalid_argument(
            "PartialConv1DLayer::ctor ERROR: invalid stride.");
    }

    // Deal with odd numbers of output channels by padding unpacked filters
    uint32_t num_output_channels{num_logical_output_channels};
    if ((num_output_channels % BufferT::C_ob) != 0)
    {
        if (buffers_are_packed)
        {
            throw std::invalid_argument(
                "PartialConv1DLayer::ctor ERROR: "
                "invalid number of output channels for packed weights.");
        }

        // set to next integer multiple of blocking factor (for this platform).
        num_output_channels +=
            (BufferT::C_ob - (num_output_channels % BufferT::C_ob));
    }

    m_leaky_slope[0] = leaky_slope;
    compute_padding_output_shape(input_shape,
                                 kernel_width,
                                 stride,
                                 padding_type,
                                 num_output_channels,
                                 num_logical_output_channels);

    detail::initialize_conv1d_buffers(
        num_output_channels,
        num_logical_output_channels,
        input_shape[CHANNEL],
        kernel_width,
        filters,
        bias,
        bn_weight, bn_bias,
        bn_running_mean, bn_running_variance, bn_eps,
        buffers_are_packed,
        m_packed_filters,
        m_packed_bias);

#if defined(DEBUG_LAYERS)
    auto &output_shape = this->output_shape();
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
    else if (activation_type == SOFTMAX)
    {
        std::cerr << "SoftMax(batches:" << output_shape[BATCH]
                  << ",chans(logical):" << output_shape[CHANNEL]
                  << "(" << num_logical_output_channels << ")"
                  << ",img:" << output_shape[HEIGHT]
                  << "x" << output_shape[WIDTH]
                  << ")" << std::endl;
    }
#endif
}

//****************************************************************************
template <class BufferT>
PartialConv1DLayer<BufferT>::PartialConv1DLayer(
    shape_type const &input_shape,
    uint32_t          kernel_width,
    uint32_t          stride,
    PaddingEnum       padding_type,
    uint32_t          num_logical_output_channels,
    BufferT    const &filters,
    bool              buffers_are_packed,
    ActivationType    activation_type,
    float             leaky_slope)
    : Layer<BufferT>(num_logical_output_channels),
      m_input_shape(input_shape),
      m_kernel_width(kernel_width),
      m_stride(stride),
      m_activation_type(activation_type),
      m_l_pad(0), m_r_pad(0),
      m_leaky_slope(1),  /// @note Allocating 1-element buffer
      m_packed_filters(),
      m_packed_bias()
{
#if defined(DEBUG_LAYERS)
    std::cerr << "PartialConv1D(batches:" << m_input_shape[BATCH]
              << ",k:" << m_kernel_width
              << ",s:" << m_stride
              << ",p:" << ((padding_type == PADDING_V) ? "'v'" : "'f'")
              << ",ichans:" << m_input_shape[CHANNEL]
              << ",ochans:" << num_logical_output_channels
              << ",img:" << m_input_shape[HEIGHT]
              << "x" << m_input_shape[WIDTH]
              << "), filters.size=" << filters.size() << std::endl;
#endif

    initialize(input_shape, kernel_width, stride,
               padding_type, num_logical_output_channels,
               filters, BufferT(), // no bias
               BufferT(), BufferT(), BufferT(), BufferT(), 0.f, // no BN
               buffers_are_packed, activation_type, leaky_slope);
}

//****************************************************************************
template <class BufferT>
PartialConv1DLayer<BufferT>::PartialConv1DLayer(
    shape_type const &input_shape,
    uint32_t          kernel_width,
    uint32_t          stride,
    PaddingEnum       padding_type,
    uint32_t          num_logical_output_channels,
    BufferT    const &filters,
    BufferT    const &bias,
    bool              buffers_are_packed,
    ActivationType    activation_type,
    float             leaky_slope)
    : Layer<BufferT>(num_logical_output_channels),
      m_input_shape(input_shape),
      m_kernel_width(kernel_width),
      m_stride(stride),
      m_activation_type(activation_type),
      m_l_pad(0), m_r_pad(0),
      m_leaky_slope(1),  /// @note Allocating 1-element buffer
      m_packed_filters(),
      m_packed_bias()
{
#if defined(DEBUG_LAYERS)
    std::cerr << "PartialConv1D(batches:" << m_input_shape[BATCH]
              << ",k:" << m_kernel_width
              << ",s:" << m_stride
              << ",p:" << ((padding_type == PADDING_V) ? "'v'" : "'f'")
              << ",ichans:" << m_input_shape[CHANNEL]
              << ",ochans:" << num_logical_output_channels
              << ",img:" << m_input_shape[HEIGHT]
              << "x" << m_input_shape[WIDTH]
              << "),filters.size=" << filters.size()
              << ",bias.size=" << bias.size() << std::endl;
#endif

    initialize(input_shape, kernel_width, stride,
               padding_type, num_logical_output_channels,
               filters, bias,
               BufferT(), BufferT(), BufferT(), BufferT(), 0.f, // no BN
               buffers_are_packed, activation_type, leaky_slope);
}

//****************************************************************************
template <class BufferT>
PartialConv1DLayer<BufferT>::PartialConv1DLayer(
    shape_type const &input_shape,
    uint32_t          kernel_width,
    uint32_t          stride,
    PaddingEnum       padding_type,
    uint32_t          num_logical_output_channels,
    BufferT    const &filters,
    BufferT    const &bn_weight,            // gamma
    BufferT    const &bn_bias,              // beta
    BufferT    const &bn_running_mean,      // mu_hat
    BufferT    const &bn_running_variance,  // sigma_hat^2
    float      const &bn_eps,               // float?
    bool              buffers_are_packed,
    ActivationType    activation_type,
    float             leaky_slope)
    : Layer<BufferT>(),
      m_input_shape(input_shape),
      m_kernel_width(kernel_width),
      m_stride(stride),
      m_activation_type(activation_type),
      m_l_pad(0), m_r_pad(0),
      m_leaky_slope(1),  /// @note Allocating 1-element buffer
      m_packed_filters(),
      m_packed_bias()
{
#if defined(DEBUG_LAYERS)
    std::cerr << "PartialConv1D(batches:" << m_input_shape[BATCH]
              << ",k:" << m_kernel_width
              << ",s:" << m_stride
              << ",p:" << ((padding_type == PADDING_V) ? "'v'" : "'f'")
              << ",ichans:" << m_input_shape[CHANNEL]
              << ",ochans:" << num_logical_output_channels
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

    initialize(input_shape, kernel_width, stride,
               padding_type, num_logical_output_channels,
               filters, BufferT(), // no bias
               bn_weight, bn_bias,
               bn_running_mean, bn_running_variance, bn_eps,
               buffers_are_packed, activation_type, leaky_slope);
}

//****************************************************************************
template <class BufferT>
PartialConv1DLayer<BufferT>::PartialConv1DLayer(
    shape_type const &input_shape,
    uint32_t          kernel_width,
    uint32_t          stride,
    PaddingEnum       padding_type,
    uint32_t          num_logical_output_channels,
    BufferT    const &filters,
    BufferT    const &bias,
    BufferT    const &bn_weight,            // gamma
    BufferT    const &bn_bias,              // beta
    BufferT    const &bn_running_mean,      // mu_hat
    BufferT    const &bn_running_variance,  // sigma_hat^2
    float      const &bn_eps,               // float?
    bool              buffers_are_packed,
    ActivationType    activation_type,
    float             leaky_slope)
    : Layer<BufferT>(),
      m_input_shape(input_shape),
      m_kernel_width(kernel_width),
      m_stride(stride),
      m_activation_type(activation_type),
      m_l_pad(0), m_r_pad(0),
      m_leaky_slope(1),  /// @note Allocating 1-element buffer
      m_packed_filters(),
      m_packed_bias()
{
#if defined(DEBUG_LAYERS)
    std::cerr << "PartialConv1D(batches:" << m_input_shape[BATCH]
              << ",k:" << m_kernel_width
              << ",s:" << m_stride
              << ",p:" << ((padding_type == PADDING_V) ? "'v'" : "'f'")
              << ",ichans:" << m_input_shape[CHANNEL]
              << ",ochans:" << num_logical_output_channels
              << ",img:" << m_input_shape[HEIGHT]
              << "x" << m_input_shape[WIDTH]
              << "), filters.size=" << filters.size()
              << ",bias.size=" << bias.size()
              << ",bn.sizes(weight,bias,run_var,run_avg)=("
              << bn_weight.size()
              << "," << bn_bias.size()
              << "," << bn_running_variance.size()
              << "," << bn_running_mean.size()
              << "),bn_eps:" << bn_eps << std::endl;
#endif

    initialize(input_shape, kernel_width, stride,
               padding_type, num_logical_output_channels,
               filters, bias,
               bn_weight, bn_bias,
               bn_running_mean, bn_running_variance, bn_eps,
               buffers_are_packed, activation_type, leaky_slope);
}

//****************************************************************************
template <class BufferT>
void PartialConv1DLayer<BufferT>::compute_output(
    std::vector<Tensor<BufferT> const *> input,
    Tensor<BufferT>*                     output) const
{
    if ((input.size() != 1) || (input[0]->shape() != m_input_shape))
    {
        throw std::invalid_argument(
            "PartialConv1DLayer::compute_output() ERROR: "
            "incorrect input buffer shape.");
    }

    if (output->capacity() < this->output_size())
    {
        throw std::invalid_argument(
            "PartialConv1DLayer::compute_output() ERROR: "
            "insufficient output buffer space.");
    }

    auto& output_shape(this->output_shape());

    small::PartialConv1D(m_kernel_width, m_stride,
                         m_l_pad, m_r_pad,
                         output_shape[CHANNEL],
                         m_input_shape[CHANNEL],
                         m_input_shape[HEIGHT],  /// @todo BATCH?
                         m_input_shape[WIDTH],
                         input[0]->buffer(),
                         m_packed_filters,
                         output->buffer());

    if (m_packed_bias.size() == output_shape[CHANNEL])
    {
        small::PartialBias(output_shape[CHANNEL],
                           output_shape[HEIGHT], output_shape[WIDTH],
                           m_packed_bias,
                           output->buffer());
    }

    output->set_shape(output_shape);

    if (m_activation_type == RELU)
    {
        small::ReLUActivation(output_shape[CHANNEL], /// @todo 1D?
                              output_shape[HEIGHT],  /// @todo BATCH?
                              output_shape[WIDTH],
                              output->buffer(),
                              output->buffer());
    }
    else if (m_activation_type == LEAKY)
    {
        small::LeakyReLUActivation(output_shape[CHANNEL], /// @todo 1D?
                                   output_shape[HEIGHT],  /// @todo BATCH?
                                   output_shape[WIDTH],
                                   output->buffer(),
                                   m_leaky_slope,
                                   output->buffer());
    }
    else if (m_activation_type == SOFTMAX)
    {
        small::SoftMax(output_shape[CHANNEL],
                       this->logical_output_channels(),
                       output_shape[HEIGHT],  /// @todo BATCH?
                       output_shape[WIDTH],
                       output->buffer(),
                       output->buffer());
    }
}

//****************************************************************************
template <class BufferT>
void PartialConv1DLayer<BufferT>::compute_padding_output_shape(
    shape_type const &input_shape,
    uint32_t          kernel_width,
    uint32_t          stride,
    PaddingEnum       padding_type,
    uint32_t          num_output_channels,
    uint32_t          num_logical_output_channels)
{
    /// @todo is there a clean way to make these const members, or
    ///       will image size get moved to compute_output and all of
    ///       this moves to compute output?
    size_t W;
    small::compute_padding_output_dim(input_shape[WIDTH], kernel_width,
                                      stride, padding_type,
                                      m_l_pad, m_r_pad,
                                      W);

#if defined(DEBUG_LAYERS)
    std::cerr << "PartialConv1D padding: "
              << (int)m_l_pad << "," << (int)m_r_pad << std::endl;
#endif

    this->set_output_shape(
        {input_shape[BATCH],
         num_output_channels,
         input_shape[HEIGHT],  // batch?
         W,
         num_logical_output_channels});

}

}
