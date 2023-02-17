//------------------------------------------------------------------------------
// MaxPool2D.hpp - Scott's attempt at OO design for SMaLL
//------------------------------------------------------------------------------

// SMaLL Framework, (c) 2023
// by The SMaLL Framework Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DMxx-xxxx

//------------------------------------------------------------------------------

#pragma once

#include <small.h>
#include <small/buffers.hpp>
#include <small/Layer.hpp>

namespace small
{
enum PaddingEnum
{
    PADDING_V,
    PADDING_F
};

//****************************************************************************
template <typename ScalarT=float>
class MaxPool2D : public Layer<ScalarT>
{
public:
    typedef ScalarT data_type;
    typedef typename Layer<ScalarT>::buffer_type buffer_type;

    MaxPool2D(uint32_t    kernel_height,
              uint32_t    kernel_width,
              uint32_t    stride,
              PaddingEnum padding_type,
              uint32_t    input_channels,
              uint32_t    input_height,
              uint32_t    input_width)
        : Layer<ScalarT>(),
          m_kernel_height(kernel_height),
          m_kernel_width(kernel_width),
          m_stride(stride),
          m_input_channels(num_channels),
          m_input_height(input_height),
          m_input_width(input_width),
          m_output_height(
              compute_output_dim(input_height, kernel_height,
                                 stride,
                                 ((padding_type == PADDING_V) ? 'v' : 'f'))),
          m_output_width(
              compute_output_dim(input_width, kernel_width,
                                 stride,
                                 ((padding_type == PADDING_V) ? 'v' : 'f'))),
          m_input_buffer_size(input_channels*input_height*input_width),
          m_output_buffer_size(input_channels*m_output_height*m_output_width)
    {
    }

    virtual size_t  input_buffer_size() const { return  m_input_buffer_size; }
    virtual size_t output_buffer_size() const { return m_output_buffer_size; }

    virtual void compute_output(buffer_type const &input_dc,
                                buffer_type       &output_dc)
    {
        // assert(input.size() == input_width*input_height);
        // assert(output.size()== input_width*input_height);
        if (m_kernel_width == m_kernel_height)
        {
            MaxPool2D(0,
                      m_kernel_width, m_stride,
                      m_t_pad, m_b_pad, m_l_pad, m_r_pad,
                      m_num_channels,
                      m_input_height, m_input_width,
                      input_dc.data(),
                      output_dc.data());
        }
        else
        {
        }
    }

private:
    uint32_t const m_kernel_height, m_kernel_width;
    uint32_t const m_stride;
    uint8_t  const m_t_pad, m_b_pad, m_l_pad, m_r_pad;
    uint32_t const m_input_channels;
    uint32_t const m_input_height, m_input_width;
    uint32_t const m_output_height, m_output_width;
    size_t   const m_input_buffer_size;
    size_t   const m_output_buffer_size;
};

}
