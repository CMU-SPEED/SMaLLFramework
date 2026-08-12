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
class DenseLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    //DenseLayer () delete;

    // No bias, no batch normalization
    DenseLayer(shape_type const &input_shape,    //pred.output_shape()
               uint32_t          num_output_channels,
               BufferT    const &filters,
               bool              buffers_are_packed = true);

    // With bias term
    DenseLayer(shape_type const &input_shape,    //pred.output_shape()
               uint32_t          num_output_channels,
               BufferT    const &filters,
               BufferT    const &bias,
               bool              buffers_are_packed = true);

    virtual ~DenseLayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const;

    BufferT const &get_packed_filters() const { return m_packed_filters; }
    BufferT const &get_packed_bias()    const { return m_packed_bias; }

private:
    void initialize(shape_type const &input_shape,
                    uint32_t          num_logical_output_channels,
                    BufferT    const &filters,
                    BufferT    const &bias,
                    bool              buffers_are_packed);

    shape_type const m_input_shape;

    BufferT          m_packed_filters;
    BufferT          m_packed_bias;
};

//****************************************************************************

namespace detail
{
    //************************************************************************
    template <class BufferT>
    void initialize_dense_buffers(
        uint32_t          num_output_channels,
        uint32_t          num_logical_output_channels,
        uint32_t          num_input_channels,
        BufferT    const &filters,
        BufferT    const &bias,
        bool              buffers_are_packed,
        BufferT          &packed_filters,
        BufferT          &packed_bias)
    {
        // ============ Filter weights ===========
        if (filters.size() !=   /// @todo consider allowing larger filter buffers??
            num_logical_output_channels*num_input_channels)
        {
            throw std::invalid_argument(
                "*DenseLayer::ctor ERROR: filters buffer is incorrect size.");
        }

        // Allocate packed filters if necessary
        if (packed_filters.size() == 0)
        {
            BufferT filters(num_input_channels*num_output_channels);
            small::init_zeros(filters, filters.size()); // optional
            packed_filters = std::move(filters);
        }
        else if (packed_filters.size() !=
                 num_input_channels*num_output_channels)
        {
            throw std::invalid_argument(
                "*DenseLayer::ctor ERROR: "
                "packed filters buffer incorrect size.");
        }

        if (!buffers_are_packed)
        {
            if (num_output_channels != num_logical_output_channels)
            {
                // pad out the unpacked
                size_t unpacked_idx = 0;
                for (size_t co = 0; co < num_logical_output_channels; ++co)
                {
                    for (size_t ci = 0; ci < num_input_channels; ++ci)
                    {
                        size_t packed_idx = small::packed_weight_index(
                            num_output_channels, num_input_channels,
                            1U, 1U,
                            BufferT::C_ob, BufferT::C_ib,
                            co, ci, 0UL, 0UL);
                        //std::cerr << "unpacked-->packed: " << unpacked_idx
                        //          << "-->" << packed_idx
                        //          << ", val = " << filters[unpacked_idx] << std::endl;
                        packed_filters[packed_idx] = filters[unpacked_idx++];
                    }
                }
            }
            else
            {
                // Pack the filter buffers for SMaLL use
                small::pack_buffer(filters,
                                   FILTER_CONV,
                                   num_output_channels, num_input_channels,
                                   1U, 1U,
                                   BufferT::C_ib, BufferT::C_ob,
                                   packed_filters);
            }
        }
        else
        {
            if (num_output_channels != num_logical_output_channels)
            {
                throw std::invalid_argument(
                    "DenseLayer::ctor error: invalid number of output channels.");
            }
            std::copy(filters.data(),
                      filters.data() + packed_filters.size(),
                      packed_filters.data());
        }

        // ============ Bias term ===========
        if (bias.size() > 0)
        {
            if (bias.size() != num_logical_output_channels)
            {
                throw std::invalid_argument(
                    "*DenseLayer::ctor ERROR: bias buffer incorrect size.");
            }

            // if (!buffers_are_packed)
            BufferT local_bias(num_output_channels);
            small::init_zeros(local_bias, local_bias.size());  // optional
            std::copy(bias.data(),
                      bias.data() + num_logical_output_channels,
                      local_bias.data());
            packed_bias = std::move(local_bias);
        }

#if defined(DEBUG_LAYERS)
        std::cerr << "*Dense: ochans:" << num_output_channels
                  << ",logical_ochans:" << num_logical_output_channels
                  << std::endl;
        std::cerr << "*Dense: packed_bias.size():    "
                  << packed_bias.size() << std::endl;
        std::cerr << "*Dense: packed_filters.size(): "
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
DenseLayer<BufferT>::DenseLayer(
    shape_type const &input_shape,         // todo: assert NCHW = (1, C_i, 1, 1)
    uint32_t          num_logical_output_channels, // assumes NCHW = (1, C_o, 1, 1)
    BufferT    const &filters,
    bool              buffers_are_packed)
    : Layer<BufferT>(),
      m_input_shape(input_shape),
      m_packed_filters(),
      m_packed_bias()
{
#if defined(DEBUG_LAYERS)
    std::cerr << "Dense(batches:" << m_input_shape[BATCH]
              << ",ichans/lchans:" << m_input_shape[CHANNEL]
              << "/" << m_input_shape[L_CHAN]
              << ",ochans:" << num_logical_output_channels
              << ",img:" << m_input_shape[HEIGHT]
              << "x" << m_input_shape[WIDTH]
              << "), filters.size=" << filters.size() << std::endl;
#endif

    initialize(input_shape, num_logical_output_channels,
               filters, BufferT(), buffers_are_packed); // no bias
}

//****************************************************************************
template <class BufferT>
DenseLayer<BufferT>::DenseLayer(
    shape_type const &input_shape,         // todo: assert NCHW = (1, C_i, 1, 1)
    uint32_t          num_logical_output_channels, // assumes NCHW = (1, C_o, 1, 1)
    BufferT    const &filters,
    BufferT    const &bias,
    bool              buffers_are_packed)
    : Layer<BufferT>(),
      m_input_shape(input_shape),
      m_packed_filters(),
      m_packed_bias()
{
#if defined(DEBUG_LAYERS)
    std::cerr << "Dense(batches:" << m_input_shape[BATCH]
              << ",ichans/lchans:" << m_input_shape[CHANNEL]
              << "/" << m_input_shape[L_CHAN]
              << ",ochans:" << num_logical_output_channels
              << ",img:" << m_input_shape[HEIGHT]
              << "x" << m_input_shape[WIDTH]
              << "),filters.size=" << filters.size()
              << ",bias.size=" << bias.size() << std::endl;
#endif

    initialize(input_shape, num_logical_output_channels,
               filters, bias, buffers_are_packed);
}

//****************************************************************************
template <class BufferT>
void DenseLayer<BufferT>::initialize(shape_type const &input_shape,
                                     uint32_t          num_logical_output_channels,
                                     BufferT    const &filters,
                                     BufferT    const &bias,
                                     bool              buffers_are_packed)
{
    if (((input_shape[CHANNEL] % BufferT::C_ib) != 0) &&
        (input_shape[CHANNEL] != 3) &&  // all of the cases that Conv2D supports
        (input_shape[CHANNEL] != 2) &&
        (input_shape[CHANNEL] != 1))
    {
        throw std::invalid_argument(
            "DenseLayer::ctor ERROR: invalid number of input channels.");
    }

    // Deal with odd numbers of output channels by padding unpacked filters
    uint32_t num_output_channels{num_logical_output_channels};
    if ((num_logical_output_channels % BufferT::C_ob) != 0)
    {
        if (buffers_are_packed)
        {
            throw std::invalid_argument(
                "DenseLayer::ctor ERROR: invalid number of output channels for packed weights.");
        }

        // set to next integer multiple of blocking factor (for this platform).
        num_output_channels +=
            (BufferT::C_ob - (num_output_channels % BufferT::C_ob));
    }

    this->set_output_shape(
        {input_shape[BATCH],  num_output_channels,
         input_shape[HEIGHT], input_shape[WIDTH],
         num_logical_output_channels});

    detail::initialize_dense_buffers(
        num_output_channels,
        num_logical_output_channels,
        m_input_shape[CHANNEL],
        filters,
        bias,
        buffers_are_packed,
        m_packed_filters,
        m_packed_bias);
}

//****************************************************************************
template <class BufferT>
void DenseLayer<BufferT>::compute_output(
    std::vector<Tensor<BufferT> const *> input,
    Tensor<BufferT>*                     output) const
{
    if ((input.size() != 1) || (input[0]->shape() != m_input_shape))
    {
        throw std::invalid_argument(
            "DenseLayer::compute_output() ERROR: "
            "incorrect input buffer shape.");
    }

    if (output->capacity() < this->output_size())
    {
        throw std::invalid_argument(
            "DenseLayer::compute_output() ERROR: "
            "insufficient output buffer space.");
    }

    auto& output_shape(this->output_shape());

    if (m_packed_bias.size() == output_shape[CHANNEL])
    {
        small::Bias(output_shape[CHANNEL],
                    output_shape[HEIGHT],
                    output_shape[WIDTH],
                    m_packed_bias, output->buffer());
        /// @todo Consider implementing PartialDense
        small::PartialConv2D(1, 1, 1,
                             0U, 0U, 0U, 0U,
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
        small::Dense(output_shape[CHANNEL]*output_shape[HEIGHT]*output_shape[WIDTH],
                     m_input_shape[CHANNEL]*m_input_shape[HEIGHT]*m_input_shape[WIDTH],
                     input[0]->buffer(),
                     m_packed_filters,
                     output->buffer());
   }

    output->set_shape(output_shape);
}

}
