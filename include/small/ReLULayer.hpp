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
    typedef typename Tensor<BufferT>::shape_type shape_type;

    ReLULayer(Layer<BufferT> const &predecessor)
        : Layer<BufferT>(&predecessor),
          m_shape(predecessor.output_buffer_shape()),
          m_buffer_size(predecessor.output_buffer_size())
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "ReLU(batches:" << m_shape[BATCH]
                  << ",chans:" << m_shape[CHANNEL]
                  << ",img:" << m_shape[HEIGHT] << "x" << m_shape[WIDTH]
                  << ")" << std::endl;
#endif
    }

    virtual ~ReLULayer() {}

    virtual size_t output_buffer_size() const { return m_buffer_size; }
    virtual shape_type output_buffer_shape() const { return m_shape; }

    virtual void compute_output(Tensor<BufferT> const &input,
                                Tensor<BufferT>       &output) const
    {
        // assert(input.shape()  == m_shape);
        // assert(output.capacity() >= m_buffer_size);

        if (input.shape() != m_shape)
        {
            throw std::invalid_argument(
                "ReLULayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        if (output.capacity() < m_buffer_size)
        {
            throw std::invalid_argument(
                "ReLULayer::compute_output() ERROR: "
                "insufficient output buffer space.");
        }

        small::ReLUActivation(m_shape[CHANNEL],
                              m_shape[HEIGHT], m_shape[WIDTH],
                              input.buffer(),
                              output.buffer());
        output.set_shape(m_shape);
    }

private:
    shape_type const m_shape;
    size_t     const m_buffer_size;
};

}
