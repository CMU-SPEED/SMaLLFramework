//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2025 by The SMaLL Contributors, All Rights Reserved.
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
class MaxPool1DLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    MaxPool1DLayer(shape_type const &input_shape,
                   uint32_t          kernel_width,
                   uint32_t          stride,
                   PaddingEnum       padding_type)
        : Layer<BufferT>(),
          m_input_shape(input_shape),
          m_kernel_width(kernel_width),
          m_stride(stride),
          m_l_pad(0), m_r_pad(0)
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "MaxPool1D(batches:" << m_input_shape[BATCH]
                  << ",k:" << kernel_width
                  << ",s:" << stride
                  << ",p:" << ((padding_type == PADDING_V) ? "'v'" : "'f'")
                  << ",chans/lchans:" << m_input_shape[CHANNEL]
                  << "/" << m_input_shape[L_CHAN]
                  << ",img:" << m_input_shape[HEIGHT]
                  << "x" << m_input_shape[WIDTH] << ")"
                  << std::endl;
#endif
        /// @todo assert m_input_shape[HEIGHT] == 1 or support batches?
        if (m_input_shape[HEIGHT] != 1U)
        {
            throw std::invalid_argument(
                "MaxPool1DLayer ERROR: image height must be 1.");
        }

        /// @todo is there a clean way to make these const members, or
        ///       will image size get moved to compute_output() and all of
        ///       this moves to compute_output()?
        size_t W;
        small::compute_padding_output_dim(m_input_shape[WIDTH], kernel_width,
                                          stride, padding_type,
                                          m_l_pad, m_r_pad,
                                          W);
#if defined(DEBUG_LAYERS)
        std::cerr << "MaxPool1D padding: "
                  << (int)m_l_pad << "," << (int)m_r_pad << std::endl;
#endif

        this->set_output_shape(
            {m_input_shape[BATCH],
             m_input_shape[CHANNEL],
             1U, // m_input_shape[HEIGHT],
             W,
             m_input_shape[L_CHAN]});
    }

    virtual ~MaxPool1DLayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const
    {
        if ((input.size() != 1) || (input[0]->shape() != m_input_shape))
        {
            throw std::invalid_argument(
                "MaxPool1DLayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        if (output->capacity() < this->output_size())
        {
            throw std::invalid_argument(
                "MaxPool1DLayer::compute_output ERROR: "
                "insufficient output buffer space.");
        }

        MaxPool1D(m_kernel_width,
                  m_stride,
                  m_l_pad, m_r_pad,
                  m_input_shape[CHANNEL],
                  m_input_shape[HEIGHT], // batch
                  m_input_shape[WIDTH],
                  input[0]->buffer(),
                  output->buffer());

        output->set_shape(this->output_shape());
    }

private:
    shape_type const m_input_shape;

    uint32_t   const m_kernel_width;
    uint32_t   const m_stride;

    /// @todo: how to make const?
    uint8_t          m_l_pad, m_r_pad;
};

}
