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
class InputLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;
    typedef typename Tensor<BufferT>::shape_type shape_type;

    InputLayer(shape_type const &shape) // {C, H, W}
        : Layer<BufferT>(nullptr),      // has no predecessor
          m_shape(shape),
          m_buffer_size(shape[0]*shape[1]*shape[2])
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "Input(chans:" << shape[0]
                  << ",img:" << shape[1] << "x" << shape[2]
                  << ")" << std::endl;
#endif
    }

    virtual ~InputLayer() {}

    virtual size_t output_buffer_size() const { return m_buffer_size; }
    virtual shape_type output_buffer_shape() const { return m_shape; }

    virtual void compute_output(Tensor<BufferT> const &input,
                                Tensor<BufferT>       &output) const
    {
        ///@todo Do nothing or throw? or pack the input into the output?
    }

private:
    shape_type   m_shape;
    size_t const m_buffer_size;
};

}
