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
    typedef typename Tensor<BufferT>::shape_type shape_type;

    //DepthwiseConv2DLayer () delete;

    /// @param[in] filters  Unpacked set of filters with dimensions packed
    ///                     in the following order:
    ///                     {in_chans, out_chans, kern_h, kern_w}
    ///
    DepthwiseConv2DLayer(Layer<BufferT> const &predecessor,
                         uint32_t       kernel_size,
                         uint32_t       stride,
                         PaddingEnum    padding_type,
                         BufferT const &filters,      /// @todo support move
                         bool filters_are_packed = true)
        : Layer<BufferT>(&predecessor),
          m_kernel_size(kernel_size),
          m_stride(stride),
          m_t_pad(0), m_b_pad(0), m_l_pad(0), m_r_pad(0),
          m_input_shape(predecessor.output_buffer_shape()),
          m_input_buffer_size(predecessor.output_buffer_size())
    {
        // std::cerr << "DW(k:" << kernel_size
        //           << ",s:" << stride
        //           << ",p:" << ((padding_type == PADDING_V) ? "'v'" : "'f'")
        //           << ",chans:" << m_input_shape[0]
        //           << ",img:" << m_input_shape[1] << "x" << m_input_shape[1]
        //           << "), filters.size=" << filters.size() << std::endl;

        if (filters.size() < m_input_shape[0]*kernel_size*kernel_size)
        {
            throw std::invalid_argument(
                "DepthwiseConv2DLayer::ctor ERROR: "
                "filters buffer too small.");
        }

        /// @todo is there a clean way to make these const members, or
        ///       will image size get moved to compute_output and all of
        ///       this moves to compute output?
        m_output_shape[0] = m_input_shape[0];
        small::compute_padding_output_dim(m_input_shape[1], kernel_size,
                                          stride, padding_type,
                                          m_t_pad, m_b_pad,
                                          m_output_shape[1]);
        small::compute_padding_output_dim(m_input_shape[2], kernel_size,
                                          stride, padding_type,
                                          m_l_pad, m_r_pad,
                                          m_output_shape[2]);
        // std::cerr << "DW padding: " << (int)m_t_pad << "," << (int)m_b_pad
        //           << "," << (int)m_l_pad << "," << (int)m_r_pad << std::endl;
        m_output_buffer_size =
            m_output_shape[0]*m_output_shape[1]*m_output_shape[2];

        // Pack the filter buffers for SMaLL use
        BufferT packed_filters(m_output_shape[0]*kernel_size*kernel_size);
        if (!filters_are_packed)
        {
            small::pack_buffer(filters,
                               FILTER_DW,
                               m_input_shape[0], 1U,
                               m_kernel_size, m_kernel_size,
                               C_ib, C_ob,
                               packed_filters);
        }
        else
        {
            std::copy(filters.data(),
                      filters.data() + packed_filters.size(),
                      packed_filters.data());
        }
        m_packed_filters = std::move(packed_filters);
    }

    virtual ~DepthwiseConv2DLayer() {}

    virtual size_t output_buffer_size() const { return m_output_buffer_size; }
    virtual shape_type output_buffer_shape() const { return m_output_shape; }

    // The input buffer is already packed for SMaLL computation ('dc')
    // The output buffer will be packed for SMaLL computation ('dc')
    virtual void compute_output(Tensor<BufferT> const &input,
                                Tensor<BufferT>       &output) const
    {
        // assert(input.shape() != m_input_shape);
        // assert(output.capacity() >= m_output_buffer_size);

        if (input.shape() != m_input_shape)
        {
            throw std::invalid_argument(
                "DepthwiseConv2DLayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        if (output.capacity() < m_output_buffer_size)
        {
            throw std::invalid_argument(
                "DepthwiseConv2DLayer::compute_output() ERROR: "
                "insufficient output buffer space.");
        }

        DepthwiseConv2D(m_kernel_size, m_stride,
                        m_t_pad, m_b_pad, m_l_pad, m_r_pad,
                        m_input_shape[0], // channels
                        m_input_shape[1], m_input_shape[2],
                        input.buffer(),
                        m_packed_filters,
                        output.buffer());
        output.set_shape(m_output_shape);
    }

private:
    uint32_t   const m_kernel_size;
    uint32_t   const m_stride;

    /// @todo: how to make const?
    uint8_t          m_t_pad, m_b_pad, m_l_pad, m_r_pad;

    shape_type const m_input_shape;
    size_t     const m_input_buffer_size;

    /// @todo: how to make const?
    shape_type       m_output_shape;
    size_t           m_output_buffer_size;

    BufferT          m_packed_filters;
};

}
