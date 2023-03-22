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
class ReLULayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    ReLULayer(uint32_t num_channels,
              uint32_t input_height,
              uint32_t input_width)
        : Layer<BufferT>(),
          m_num_channels(num_channels),
          m_input_height(input_height),
          m_input_width(input_width),
          m_buffer_size(num_channels*input_height*input_width)
    {
    }

    virtual size_t  input_buffer_size() const { return m_buffer_size; }
    virtual size_t output_buffer_size() const { return m_buffer_size; }

    virtual void compute_output(BufferT const &input_dc,
                                BufferT       &output_dc) const
    {
        // assert(input_dc.size() >= m_buffer_size);
        // assert(output.size()   >= m_buffer_size);
        small::ReLUActivation(m_num_channels,
                              m_input_height, m_input_width,
                              input_dc,
                              output_dc);
    }

private:
    uint32_t const m_num_channels;
    uint32_t const m_input_height;
    uint32_t const m_input_width;
    size_t const   m_buffer_size;
};

}
