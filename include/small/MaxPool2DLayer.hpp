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
class MaxPool2DLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;
    typedef typename Tensor<BufferT>::shape_type shape_type;

    MaxPool2DLayer(Layer<BufferT> const &predecessor,
                   uint32_t    kernel_height,
                   uint32_t    kernel_width,
                   uint32_t    stride,
                   PaddingEnum padding_type)
        : Layer<BufferT>(&predecessor),
          m_kernel_height(kernel_height),
          m_kernel_width(kernel_width),
          m_stride(stride),
          m_t_pad(0), m_b_pad(0), m_l_pad(0), m_r_pad(0),
          m_input_shape(predecessor.output_buffer_shape()),
          m_input_buffer_size(predecessor.output_buffer_size())
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "MaxPool2D(k:" << kernel_height << "x" << kernel_width
                  << ",s:" << stride
                  << ",p:" << ((padding_type == PADDING_V) ? "'v'" : "'f'")
                  << ",chans:" << m_input_shape[0]
                  << ",img:" << m_input_shape[1] << "x" << m_input_shape[1]
                  << std::endl;
#endif

        /// @todo is there a clean way to make these const members, or
        ///       will image size get moved to compute_output() and all of
        ///       this moves to compute_output()?
        m_output_shape[0] = m_input_shape[0];
        small::compute_padding_output_dim(m_input_shape[1], kernel_height,
                                          stride, padding_type,
                                          m_t_pad, m_b_pad,
                                          m_output_shape[1]);
        small::compute_padding_output_dim(m_input_shape[2], kernel_width,
                                          stride, padding_type,
                                          m_l_pad, m_r_pad,
                                          m_output_shape[2]);
        // std::cerr << "MaxPool2D padding: "
        //           << (int)m_t_pad << "," << (int)m_b_pad
        //          << "," << (int)m_l_pad << "," << (int)m_r_pad << std::endl;
        m_output_buffer_size =
            m_output_shape[0]*m_output_shape[1]*m_output_shape[2];
    }

    virtual ~MaxPool2DLayer() {}

    virtual size_t output_buffer_size() const { return m_output_buffer_size; }
    virtual shape_type output_buffer_shape() const { return m_output_shape; }

    // The input buffer is already packed for SMaLL computation ('dc')
    // The output buffer will be packed for SMaLL computation ('dc')
    virtual void compute_output(Tensor<BufferT> const &input,
                                Tensor<BufferT>       &output) const
    {
        // assert(input.shape() == m_input_shape)
        // assert(output.capacity() >= m_output_buffer_size);

        if (input.shape() != m_input_shape)
        {
            std::cerr << "MaxPool2DLayer ERROR: incorrect input buffer shape. "
                      << "Expected: " << m_input_shape[0]
                      << "," << m_input_shape[1]
                      << "," << m_input_shape[2]
                      << ", received: " << input.shape()[0]
                      << "," << input.shape()[1]
                      << "," << input.shape()[2] << std::endl;
            throw std::invalid_argument(
                "MaxPool2DLayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        if (output.capacity() < m_output_buffer_size)
        {
            std::cerr << "MaxPool2DLayer ERROR: output buffer size = "
                      << output.capacity()
                      << ", required size = " << m_output_buffer_size
                      << ": got " << output.capacity() << std::endl;
            throw std::invalid_argument(
                "MaxPool2DLayer::compute_output ERROR: "
                "insufficient output buffer space.");
        }

        if (m_kernel_width == m_kernel_height)
        {
            MaxPool2D(m_kernel_width,
                      m_stride,
                      m_t_pad, m_b_pad, m_l_pad, m_r_pad,
                      m_input_shape[0],
                      m_input_shape[1], m_input_shape[2],
                      input.buffer(),
                      output.buffer());
        }
        else
        {
            MaxPool2D_rect(m_kernel_height, m_kernel_width,
                           m_stride,
                           m_t_pad, m_b_pad, m_l_pad, m_r_pad,
                           m_input_shape[0],
                           m_input_shape[1], m_input_shape[2],
                           input.buffer(),
                           output.buffer());
        }
        output.set_shape(m_output_shape);
    }

private:
    uint32_t   const m_kernel_height, m_kernel_width;
    uint32_t   const m_stride;

    /// @todo: how to make const?
    uint8_t          m_t_pad, m_b_pad, m_l_pad, m_r_pad;

    shape_type const m_input_shape;
    size_t     const m_input_buffer_size;

    /// @todo: is it worth it to make const?
    shape_type       m_output_shape;
    size_t           m_output_buffer_size;
};

}
