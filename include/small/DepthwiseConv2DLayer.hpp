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
    DepthwiseConv2DLayer(uint32_t       kernel_size,
                         uint32_t       stride,
                         PaddingEnum    padding_type,
                         uint32_t       num_channels,
                         uint32_t       input_height,
                         uint32_t       input_width,
                         BufferT const &filters)
        : Layer<BufferT>(),
          m_kernel_size(kernel_size),
          m_stride(stride),
          m_padding_type(padding_type),
          m_num_channels(num_channels),
          m_input_height(input_height),
          m_input_width(input_width),
          m_input_buffer_size(num_channels*input_height*input_width),
          m_t_pad(0), m_b_pad(0), m_l_pad(0), m_r_pad(0),
          m_output_buffer_size(0),
          m_packed_filters(num_channels*kernel_size*kernel_size)
    {
        if (filters.size() < num_channels*kernel_size*kernel_size)
        {
            throw std::invalid_argument(
                "DepthwiseConv2DLayer::ctor ERROR: "
                "filters buffer too small.");
        }

        /// @todo is there a clean way to make these const members, or
        ///       will image size get moved to compute_output and all of
        ///       this moves to compute output?
        small::compute_padding_output_dim(input_height, kernel_size,
                                          stride, padding_type,
                                          m_t_pad, m_b_pad,
                                          m_output_height);
        small::compute_padding_output_dim(input_width, kernel_size,
                                          stride, padding_type,
                                          m_l_pad, m_r_pad,
                                          m_output_width);

        m_output_buffer_size = num_channels*m_output_height*m_output_width;

        // Pack the filter buffers for SMaLL use
        small::pack_buffer(filters,
                           FILTER_DW,
                           m_num_channels, 1U, m_kernel_size, m_kernel_size,
                           C_ib, C_ob,
                           m_packed_filters);
    }

    virtual size_t  input_buffer_size() const { return  m_input_buffer_size; }
    virtual size_t output_buffer_size() const { return m_output_buffer_size; }

#if 0
    virtual std::array<size_t, 3UL> compute_output_dimensions(
        std::array<size_t, 3UL> const &input_dimensions) const
    {
        if (input_dimensions[0]  != m_num_channels)
        {
            throw std::invalid_argument(
                "DepthwiseConv2DLayer::compute_output_dimensions ERROR: "
                "mismatched input channels.");
        }

        uint8_t t_pad, b_pad, l_pad, r_pad;
        size_t  Ho, Wo;
        small::compute_padding_output_dim(input_dimensions[1],
                                          m_kernel_size, m_stride,
                                          m_padding_type,
                                          t_pad, b_pad, Ho);

        small::compute_padding_output_dim(input_dimensions[2],
                                          m_kernel_size, m_stride,
                                          m_padding_type,
                                          l_pad, r_pad, Wo);
        return std::array<size_t, 3UL>{m_num_channels, Ho, Wo};
    }
#endif

    // The input buffer is already packed for SMaLL computation ('dc')
    // The output buffer will be packed for SMaLL computation ('dc')
    virtual void compute_output(BufferT const &input_dc,
                                BufferT       &output_dc) const
    {
        // assert(input_dc.size() >= m_input_buffer_size);
        // assert(output.size()   >= m_output_buffer_size);

        if (input_dc.size() < m_input_buffer_size)
        {
            throw std::invalid_argument(
                "DepthwiseConv2DLayer::compute_output() ERROR: "
                "insufficient input buffer space.");
        }

        if (output_dc.size() < m_output_buffer_size)
        {
            throw std::invalid_argument(
                "DepthwiseConv2DLayer::compute_output() ERROR: "
                "insufficient output buffer space.");
        }

        DepthwiseConv2D(m_kernel_size, m_stride,
                        m_t_pad, m_b_pad, m_l_pad, m_r_pad,
                        m_num_channels,
                        m_input_height, m_input_width,
                        input_dc,
                        m_packed_filters,
                        output_dc);
    }

private:
    uint32_t              const m_kernel_size;
    uint32_t              const m_stride;
    small::PaddingEnum    const m_padding_type;
    uint32_t              const m_num_channels;
    uint32_t const m_input_height, m_input_width;
    size_t   const m_input_buffer_size;

    /// @todo: how to make const?
    uint8_t  m_t_pad, m_b_pad, m_l_pad, m_r_pad;
    uint32_t m_output_height, m_output_width;
    size_t   m_output_buffer_size;

    BufferT  m_packed_filters;
};

}
