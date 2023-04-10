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

    MaxPool2DLayer(uint32_t    kernel_height,
                   uint32_t    kernel_width,
                   uint32_t    stride,
                   PaddingEnum padding_type,
                   uint32_t    num_channels,
                   uint32_t    input_height,
                   uint32_t    input_width)
        : Layer<BufferT>(),
          m_kernel_height(kernel_height),
          m_kernel_width(kernel_width),
          m_stride(stride),
          m_num_channels(num_channels),
          m_input_height(input_height),
          m_input_width(input_width),
          m_input_buffer_size(num_channels*input_height*input_width),
          m_t_pad(0), m_b_pad(0), m_l_pad(0), m_r_pad(0),
          m_output_buffer_size(0)
    {
        // std::cerr << "*MaxPool2D(k:" << kernel_size
        //           << ",s:" << stride
        //           << ",'v'"
        //           << ",chans:" << num_channels
        //           << ",img:" << input_size
        //           << std::endl;

        /// @todo is there a clean way to make these const members, or
        ///       will image size get moved to compute_output() and all of
        ///       this moves to compute_output()?
        small::compute_padding_output_dim(input_height, kernel_height,
                                          stride, padding_type,
                                          m_t_pad, m_b_pad,
                                          m_output_height);
        small::compute_padding_output_dim(input_width, kernel_width,
                                          stride, padding_type,
                                          m_l_pad, m_r_pad,
                                          m_output_width);
        // std::cerr << "MaxPool2D padding: " << (int)m_t_pad << "," << (int)m_b_pad
        //          << "," << (int)m_l_pad << "," << (int)m_r_pad << std::endl;
        m_output_buffer_size = num_channels*m_output_height*m_output_width;
    }

    virtual ~MaxPool2DLayer() {}

    virtual size_t  input_buffer_size() const { return  m_input_buffer_size; }
    virtual size_t output_buffer_size() const { return m_output_buffer_size; }

    // The input buffer is already packed for SMaLL computation ('dc')
    // The output buffer will be packed for SMaLL computation ('dc')
    virtual void compute_output(BufferT const &input_dc,
                                BufferT       &output_dc) const
    {
        // assert(input_dc.size() >= m_input_buffer_size);
        // assert(output.size()   >= m_output_buffer_size);

        if (input_dc.size() < m_input_buffer_size)
        {
            std::cerr << "MaxPool2DLayer ERROR: input buffer size = " << input_dc.size()
                      << ", required size = " << m_input_buffer_size
                      << ": " << m_input_height << "x" << m_input_width
                      << "x" << m_num_channels << std::endl;
            throw std::invalid_argument(
                "MaxPool2DLayer::compute_output ERROR: "
                "insufficient input buffer space.");
        }

        if (output_dc.size() < m_output_buffer_size)
        {
            std::cerr << "MaxPool2DLayer ERROR: output buffer size = " << output_dc.size()
                      << ", required size = " << m_output_buffer_size
                      << ": " << m_input_height << "x" << m_input_width
                      << "x" << m_num_channels << std::endl;
            throw std::invalid_argument(
                "MaxPool2DLayer::compute_output ERROR: "
                "insufficient output buffer space.");
        }

        if (m_kernel_width == m_kernel_height)
        {
            MaxPool2D(m_kernel_width,
                      m_stride,
                      m_t_pad, m_b_pad, m_l_pad, m_r_pad,
                      m_num_channels,
                      m_input_height, m_input_width,
                      input_dc,
                      output_dc);
        }
        else
        {
            MaxPool2D_rect(m_kernel_height, m_kernel_width,
                           m_stride,
                           m_t_pad, m_b_pad, m_l_pad, m_r_pad,
                           m_num_channels,
                           m_input_height, m_input_width,
                           input_dc,
                           output_dc);
        }
    }

private:
    uint32_t const m_kernel_height, m_kernel_width;
    uint32_t const m_stride;
    uint32_t const m_num_channels;
    uint32_t const m_input_height, m_input_width;
    size_t   const m_input_buffer_size;

    /// @todo: how to make const?
    uint8_t  m_t_pad, m_b_pad, m_l_pad, m_r_pad;
    uint32_t m_output_height, m_output_width;
    size_t   m_output_buffer_size;
};

}
