//------------------------------------------------------------------------------
// ReLU.hpp - Scotts attempt at OO design for SMaLL
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

//****************************************************************************
template <typename ScalarT=float>
class ReLU : public Layer<ScalarT>
{
public:
    typedef ScalarT data_type;
    typedef typename Layer<ScalarT>::buffer_type buffer_type;

    ReLU(uint32_t input_height, uint32_t input_width,
         uint32_t num_channels)
        : Layer<ScalarT>(),
          m_input_height(input_height),
          m_input_width(input_width),
          m_num_channels(num_channels),
          m_buffer_size(num_channels*input_height*input_width)
    {
    }

    virtual size_t  input_buffer_size() const { return m_buffer_size; }
    virtual size_t output_buffer_size() const { return m_buffer_size; }

    virtual void compute_output(buffer_type const &input_dc,
                                buffer_type       &output_dc)
    {
        // assert(input.size() == input_width*input_height);
        // assert(output.size()== input_width*input_height);
        small::ReLUActivation(m_num_channels,
                              m_input_height, m_input_width,
                              input_dc.data(),
                              output_dc.data());
    }

private:
    uint32_t const m_input_height;
    uint32_t const m_input_width;
    uint32_t const m_num_channels;
    size_t const   m_buffer_size;
};

}
