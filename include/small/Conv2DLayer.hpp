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

#include <cmath>
#include <small.h>
#include <small/buffers.hpp>
#include <small/Layer.hpp>

namespace small
{
//****************************************************************************
template <typename BufferT>
class Conv2DLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    //Conv2DLayer () delete;

    // No bias, no batch normalization
    Conv2DLayer(shape_type const &input_shape,    //pred.output_shape()
                uint32_t          kernel_height,
                uint32_t          kernel_width,
                uint32_t          stride,
                PaddingEnum       padding_type,
                uint32_t          num_output_channels,
                BufferT    const &filters,
                bool              buffers_are_packed = true,
                ActivationType    activation_type = NONE,
                float             leaky_slope = 1.e-2);

    // With bias term
    Conv2DLayer(shape_type const &input_shape,    //pred.output_shape()
                uint32_t          kernel_height,
                uint32_t          kernel_width,
                uint32_t          stride,
                PaddingEnum       padding_type,
                uint32_t          num_output_channels,
                BufferT    const &filters,
                BufferT    const &bias,
                bool              buffers_are_packed = true,
                ActivationType    activation_type = NONE,
                float             leaky_slope = 1.e-2);

    // With fused batch normalization
    Conv2DLayer(shape_type const &input_shape,    //pred.output_shape()
                uint32_t          kernel_height,
                uint32_t          kernel_width,
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
                float             leaky_slope = 1.e-2);

    virtual ~Conv2DLayer() {}

    inline uint32_t get_effective_output_channels() const
    {
        return m_effective_output_channels;
    }

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const;

private:
    void compute_padding_output_shape(shape_type const &input_shape,
                                      uint32_t          kernel_height,
                                      uint32_t          kernel_width,
                                      uint32_t          stride,
                                      PaddingEnum       padding_type,
                                      uint32_t          num_output_channels);

    shape_type const m_input_shape;

    uint32_t   const m_kernel_height, m_kernel_width;
    uint32_t   const m_stride;
    uint32_t   const m_effective_output_channels;

    ActivationType const m_activation_type;

    /// @todo: how to make const?
    uint8_t          m_t_pad, m_b_pad, m_l_pad, m_r_pad;

    BufferT          m_leaky_slope;
    BufferT          m_packed_filters;
    BufferT          m_packed_bias;
};

//****************************************************************************

namespace detail
{
    //************************************************************************
    template <class BufferT>
    void initialize_conv2d_buffers(
        uint32_t          num_output_channels,
        uint32_t          num_effective_output_channels,
        uint32_t          num_input_channels,
        uint32_t          kernel_height,
        uint32_t          kernel_width,
        BufferT    const &filters,
        BufferT    const &bias,
        BufferT    const &bn_weight,            // gamma
        BufferT    const &bn_bias,              // beta
        BufferT    const &bn_running_mean,      // mu_hat
        BufferT    const &bn_running_variance,  // sigma_hat^2
        float      const &bn_eps,               // float?
        bool              buffers_are_packed,
        BufferT          &packed_filters,
        BufferT          &packed_bias)
    {
        // ============ Filter weights ===========
        if (filters.size() !=   /// @todo consider allowing larger filter buffers??
            num_effective_output_channels*num_input_channels*
            kernel_height*kernel_width)
        {
            throw std::invalid_argument(
                "*Conv2DLayer::ctor ERROR: filters buffer is incorrect size.");
        }

        // Allocate packed filters if necessary
        if (packed_filters.size() == 0)
        {
            BufferT filters(num_input_channels*kernel_height*kernel_width*
                            num_output_channels);
            small::init_zeros(filters, filters.size()); // optional
            packed_filters = std::move(filters);
        }
        else if (packed_filters.size() !=
                 num_input_channels*kernel_height*kernel_width*
                 num_output_channels)
        {
            throw std::invalid_argument(
                "*Conv2DLayer::ctor ERROR: "
                "packed filters buffer incorrect size.");
        }

        if (!buffers_are_packed)
        {
            if (num_output_channels != num_effective_output_channels)
            {
                // pad out the unpacked
                size_t unpacked_idx = 0;
                for (size_t co = 0; co < num_effective_output_channels; ++co)
                    for (size_t ci = 0; ci < num_input_channels; ++ci)
                        for (size_t h = 0; h < kernel_height; ++h)
                            for (size_t w = 0; w < kernel_width; ++w)
                            {
                                size_t packed_idx = small::packed_weight_index(
                                    num_output_channels, num_input_channels,
                                    kernel_height, kernel_width,
                                    BufferT::C_ob, BufferT::C_ib,
                                    co, ci, h, w);
                                //std::cerr << "unpacked-->packed: " << unpacked_idx
                                //          << "-->" << packed_idx << std::endl;
                                packed_filters[packed_idx] = filters[unpacked_idx++];
                            }
            }
            else
            {
                // Pack the filter buffers for SMaLL use
                small::pack_buffer(filters,
                                   FILTER_CONV,
                                   num_output_channels, num_input_channels,
                                   kernel_height, kernel_width,
                                   BufferT::C_ib, BufferT::C_ob,
                                   packed_filters);
            }
        }
        else
        {
            if (num_output_channels != num_effective_output_channels)
            {
                throw std::invalid_argument(
                    "Conv2DLayer::ctor error: invalid number of output channels.");
            }
            std::copy(filters.data(),
                      filters.data() + packed_filters.size(),
                      packed_filters.data());
        }

        // ============ Bias term ===========
        if (bias.size() > 0)
        {
            if (bias.size() != num_effective_output_channels)
            {
                throw std::invalid_argument(
                    "*Conv2DLayer::ctor ERROR: "
                    "bias buffer incorrect size.");
            }

            // if (!buffers_are_packed)
            BufferT local_bias(num_output_channels);
            small::init_zeros(local_bias, local_bias.size());  // optional
            std::copy(bias.data(),
                      bias.data() + num_effective_output_channels,
                      local_bias.data());
            packed_bias = std::move(local_bias);
        }

        // ============ BN terms ===========
        // Note: it is all or nothing for 4 buffers
        if ((bn_weight.size() > 0) ||
            (bn_bias.size() > 0) ||
            (bn_running_mean.size() > 0) ||
            (bn_running_variance.size() > 0))
        {
            if ((bn_weight.size() != num_effective_output_channels) ||
                (bn_bias.size() != num_effective_output_channels) ||
                (bn_running_mean.size() != num_effective_output_channels) ||
                (bn_running_variance.size() != num_effective_output_channels))
            {
                throw std::invalid_argument(
                    "*Conv2DLayer::ctor ERROR: "
                    "BN buffers incorrect size.");
            }

            // Fuse the BN parameters with packed filters and bias
            /* ----------------------------------------------------------------
             * From: https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
             *
             * # prepare filters
             * w_conv = conv.weight.clone().view(conv.out_channels, -1)
             * w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
             *
             * fusedconv.weight.copy_(
             *     torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
             *
             * # prepare spatial bias
             * if conv.bias is not None:
             *     b_conv = conv.bias
             * else:
             *     b_conv = torch.zeros( conv.weight.size(0) )
             *
             * b_bn = bn.bias -
             *        bn.weight.mul(bn.running_mean).div(
             *            torch.sqrt(bn.running_var + bn.eps))
             *
             * fusedconv.bias.copy_( torch.matmul(w_bn, b_conv) + b_bn )
             */
            bool no_bias = false;
            if (packed_bias.size() == 0)
            {
                packed_bias = std::move(BufferT(num_output_channels));
                small::init_zeros(packed_bias, packed_bias.size());
                no_bias = true;
            }

            for (size_t ochan = 0; ochan < num_effective_output_channels; ++ochan)
            {
                // compute scaling factor for filters of this output channel
                float filter_scale =
                    bn_weight[ochan]/std::sqrt(bn_running_variance[ochan] + bn_eps);

                /// @todo REVISIT: this does not look like python code above
                if (no_bias)
                {
                    packed_bias[ochan] =
                        bn_bias[ochan] - bn_running_mean[ochan]*filter_scale;
                }
                else
                {
                    packed_bias[ochan] = filter_scale*packed_bias[ochan] +
                        bn_bias[ochan] - bn_running_mean[ochan]*filter_scale;
                }
                //std::cerr << ": packed_bias(" << ochan << ") = "
                //          << packed_bias[ochan]
                //          << std::endl;

                for (size_t ichan = 0; ichan < num_input_channels; ++ichan)
                {
                    for (size_t fh = 0; fh < kernel_height; ++fh)
                    {
                        for (size_t fw = 0; fw < kernel_width; ++fw)
                        {
                            size_t packed_index =
                                packed_weight_index(num_output_channels,
                                                    num_input_channels,
                                                    kernel_height,
                                                    kernel_width,
                                                    BufferT::C_ob,
                                                    BufferT::C_ib,
                                                    ochan, ichan, fh, fw);
                            //std::cerr << "packed_index = " << packed_index
                            //          << std::endl;
                            packed_filters[packed_index] *= filter_scale;
                        }
                    }
                }
            }
        }
#if defined(DEBUG_LAYERS)
        std::cerr << "*Conv2D: ochans:" << num_output_channels
                  << ",effective_ochans:" << num_effective_output_channels
                  << std::endl;
        std::cerr << "*Conv2D: packed_bias.size():    "
                  << packed_bias.size() << std::endl;
        std::cerr << "*Conv2D: packed_filters.size(): "
                  << packed_filters.size() << std::endl;
#endif
    }

} // detail

//****************************************************************************
/// @param[in] filters  Unpacked set of filters with dimensions packed
///                     in the following order:
///                     {in_chans, out_chans, kern_h, kern_w}
///
template <class BufferT>
Conv2DLayer<BufferT>::Conv2DLayer(
    shape_type const &input_shape,
    uint32_t          kernel_height,
    uint32_t          kernel_width,
    uint32_t          stride,
    PaddingEnum       padding_type,
    uint32_t          num_output_channels,
    BufferT    const &filters,
    bool              buffers_are_packed,
    ActivationType    activation_type,
    float             leaky_slope)
    : Layer<BufferT>(),
      m_input_shape(input_shape),
      m_kernel_height(kernel_height),
      m_kernel_width(kernel_width),
      m_stride(stride),
      m_effective_output_channels(num_output_channels),
      m_activation_type(activation_type),
      m_t_pad(0), m_b_pad(0), m_l_pad(0), m_r_pad(0),
      m_leaky_slope(1),  /// @note Allocating 1-element buffer
      m_packed_filters(),
      m_packed_bias()
{
#if defined(DEBUG_LAYERS)
    std::cerr << "Conv2D(batches:" << m_input_shape[BATCH]
              << ",k:" << m_kernel_height << "x" << m_kernel_width
              << ",s:" << m_stride
              << ",p:" << ((padding_type == PADDING_V) ? "'v'" : "'f'")
              << ",ichans:" << m_input_shape[CHANNEL]
              << ",ochans:" << num_output_channels
              << ",img:" << m_input_shape[HEIGHT]
              << "x" << m_input_shape[WIDTH]
              << "), filters.size=" << filters.size() << std::endl;
#endif
    if (((input_shape[CHANNEL] % BufferT::C_ib) != 0) &&
        (input_shape[CHANNEL] != 3))
    {
        throw std::invalid_argument(
            "Conv2DLayer::ctor ERROR: invalid number of input channels.");
    }

    // Deal with odd numbers of output channels by padding unpacked filters
    if ((num_output_channels % BufferT::C_ob) != 0)
    {
        if (buffers_are_packed)
        {
            throw std::invalid_argument(
                "Conv2DLayer::ctor ERROR: invalid number of output channels.");
        }

        // set to next integer multiple of blocking factor (for this platform).
        num_output_channels +=
            (BufferT::C_ob - (num_output_channels % BufferT::C_ob));
    }

    m_leaky_slope[0] = leaky_slope;
    compute_padding_output_shape(input_shape,
                                 m_kernel_height, m_kernel_width,
                                 m_stride,
                                 padding_type,
                                 num_output_channels);

    detail::initialize_conv2d_buffers(
        num_output_channels,
        m_effective_output_channels,
        m_input_shape[CHANNEL],
        m_kernel_height, m_kernel_width,
        filters,
        BufferT(),  // empty bias
        BufferT(), BufferT(), BufferT(), BufferT(), 0.f, // no BN
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
        std::cerr << "Softmax(batches:" << output_shape[BATCH]
                  << ",chans:" << output_shape[CHANNEL]
                  << ",img:" << output_shape[HEIGHT]
                  << "x" << output_shape[WIDTH]
                  << ")" << std::endl;
    }
#endif
}

//****************************************************************************
template <class BufferT>
Conv2DLayer<BufferT>::Conv2DLayer(
    shape_type const &input_shape,
    uint32_t          kernel_height,
    uint32_t          kernel_width,
    uint32_t          stride,
    PaddingEnum       padding_type,
    uint32_t          num_output_channels,
    BufferT    const &filters,
    BufferT    const &bias,
    bool              buffers_are_packed,
    ActivationType    activation_type,
    float             leaky_slope)
    : Layer<BufferT>(),
      m_input_shape(input_shape),
      m_kernel_height(kernel_height),
      m_kernel_width(kernel_width),
      m_stride(stride),
      m_effective_output_channels(num_output_channels),
      m_activation_type(activation_type),
      m_t_pad(0), m_b_pad(0), m_l_pad(0), m_r_pad(0),
      m_leaky_slope(1),  /// @note Allocating 1-element buffer
      m_packed_filters(),
      m_packed_bias()
{
#if defined(DEBUG_LAYERS)
    std::cerr << "Conv2D(batches:" << m_input_shape[BATCH]
              << ",k:" << m_kernel_height << "x" << m_kernel_width
              << ",s:" << m_stride
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
            "Conv2DLayer::ctor ERROR: invalid number of input channels.");
    }

    // Deal with odd numbers of output channels by padding unpacked filters
    if ((num_output_channels % BufferT::C_ob) != 0)
    {
        if (buffers_are_packed)
        {
            throw std::invalid_argument(
                "Conv2DLayer::ctor ERROR: invalid number of output channels.");
        }

        // set to next integer multiple of blocking factor (for this platform).
        num_output_channels +=
            (BufferT::C_ob - (num_output_channels % BufferT::C_ob));
    }

    m_leaky_slope[0] = leaky_slope;
    compute_padding_output_shape(input_shape,
                                 m_kernel_height, m_kernel_width,
                                 m_stride,
                                 padding_type,
                                 num_output_channels);

    detail::initialize_conv2d_buffers(
        num_output_channels,
        m_effective_output_channels,
        m_input_shape[CHANNEL],
        m_kernel_height, m_kernel_width,
        filters,
        bias,
        BufferT(), BufferT(), BufferT(), BufferT(), 0.f, // no BN
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
        std::cerr << "Softmax(batches:" << output_shape[BATCH]
                  << ",chans:" << output_shape[CHANNEL]
                  << ",img:" << output_shape[HEIGHT]
                  << "x" << output_shape[WIDTH]
                  << ")" << std::endl;
    }
#endif

}

//****************************************************************************
template <class BufferT>
Conv2DLayer<BufferT>::Conv2DLayer(
    shape_type const &input_shape,
    uint32_t          kernel_height,
    uint32_t          kernel_width,
    uint32_t          stride,
    PaddingEnum       padding_type,
    uint32_t          num_output_channels,
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
      m_kernel_height(kernel_height),
      m_kernel_width(kernel_width),
      m_stride(stride),
      m_effective_output_channels(num_output_channels),
      m_activation_type(activation_type),
      m_t_pad(0), m_b_pad(0), m_l_pad(0), m_r_pad(0),
      m_leaky_slope(1),  /// @note Allocating 1-element buffer
      m_packed_filters(),
      m_packed_bias()
{
#if defined(DEBUG_LAYERS)
    std::cerr << "Conv2D(batches:" << m_input_shape[BATCH]
              << ",k:" << m_kernel_height << "x" << m_kernel_width
              << ",s:" << m_stride
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
            "Conv2DLayer::ctor ERROR: invalid number of input channels.");
    }

    // Deal with odd numbers of output channels by padding unpacked filters
    if ((num_output_channels % BufferT::C_ob) != 0)
    {
        if (buffers_are_packed)
        {
            throw std::invalid_argument(
                "Conv2DLayer::ctor ERROR: invalid number of output channels.");
        }

        // set to next integer multiple of blocking factor (for this platform).
        num_output_channels +=
            (BufferT::C_ob - (num_output_channels % BufferT::C_ob));
    }

    m_leaky_slope[0] = leaky_slope;
    compute_padding_output_shape(m_input_shape,
                                 m_kernel_height, m_kernel_width,
                                 m_stride,
                                 padding_type,
                                 num_output_channels);

    detail::initialize_conv2d_buffers(
        num_output_channels,
        m_effective_output_channels,
        m_input_shape[CHANNEL],
        m_kernel_height, m_kernel_width,
        filters,
        BufferT(), // no bias
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
                  << ",slope:" << m_leaky_slope[0]
                  << ",img:" << output_shape[HEIGHT]
                  << "x" << output_shape[WIDTH]
                  << ")" << std::endl;
    }
    else if (activation_type == SOFTMAX)
    {
        std::cerr << "Softmax(batches:" << output_shape[BATCH]
                  << ",chans:" << output_shape[CHANNEL]
                  << ",img:" << output_shape[HEIGHT]
                  << "x" << output_shape[WIDTH]
                  << ")" << std::endl;
    }
#endif
}

//****************************************************************************
template <class BufferT>
void Conv2DLayer<BufferT>::compute_output(
    std::vector<Tensor<BufferT> const *> input,
    Tensor<BufferT>*                     output) const
{
    if ((input.size() != 1) || (input[0]->shape() != m_input_shape))
    {
        throw std::invalid_argument(
            "Conv2DLayer::compute_output() ERROR: "
            "incorrect input buffer shape.");
    }

    if (output->capacity() < this->output_size())
    {
        throw std::invalid_argument(
            "Conv2DLayer::compute_output() ERROR: "
            "insufficient output buffer space.");
    }

    auto& output_shape(this->output_shape());

    if (m_packed_bias.size() == output_shape[CHANNEL])
    {
        small::Bias(output_shape[CHANNEL],
                    output_shape[HEIGHT],
                    output_shape[WIDTH],
                    m_packed_bias, output->buffer());
        small::PartialConv2D(m_kernel_height, m_kernel_width, m_stride,
                             m_t_pad, m_b_pad, m_l_pad, m_r_pad,
                             output_shape[CHANNEL],
                             m_input_shape[CHANNEL],
                             m_input_shape[HEIGHT],
                             m_input_shape[WIDTH],
                             input[0]->buffer(),
                             m_packed_filters,
                             output->buffer());
    }
    else
    {
        small::Conv2D(m_kernel_height, m_kernel_width, m_stride,
                      m_t_pad, m_b_pad, m_l_pad, m_r_pad,
                      output_shape[CHANNEL],
                      m_input_shape[CHANNEL],
                      m_input_shape[HEIGHT], m_input_shape[WIDTH],
                      input[0]->buffer(),
                      m_packed_filters,
                      output->buffer());
    }

    output->set_shape(output_shape);

    if (m_activation_type == RELU)
    {
        small::ReLUActivation(output_shape[CHANNEL],
                              output_shape[HEIGHT], output_shape[WIDTH],
                              output->buffer(),
                              output->buffer());
    }
    else if (m_activation_type == LEAKY)
    {
        small::LeakyReLUActivation(output_shape[CHANNEL],
                                   output_shape[HEIGHT], output_shape[WIDTH],
                                   output->buffer(),
                                   m_leaky_slope,
                                   output->buffer());
    }
    else if (m_activation_type == SOFTMAX)
    {
        small::SoftMax(output_shape[CHANNEL],
                       output_shape[HEIGHT], output_shape[WIDTH],
                       output->buffer(),
                       output->buffer());
    }
}

//****************************************************************************
template <class BufferT>
void Conv2DLayer<BufferT>::compute_padding_output_shape(
    shape_type const &input_shape,
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
    std::cerr << "Conv2D padding: "
              << (int)m_t_pad << "," << (int)m_b_pad
              << "," << (int)m_l_pad << "," << (int)m_r_pad << std::endl;
#endif

    this->set_output_shape(output_shape);
}

}
