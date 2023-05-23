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
        : Layer<BufferT>(predecessor.output_shape())
    {
#if defined(DEBUG_LAYERS)
        auto const &output_shape(Layer<BufferT>::output_shape());
        std::cerr << "ReLU(batches:" << output_shape[BATCH]
                  << ",chans:" << output_shape[CHANNEL]
                  << ",img:" << output_shape[HEIGHT]
                  << "x" << output_shape[WIDTH]
                  << ")" << std::endl;
#endif
    }

    virtual ~ReLULayer() {}

    virtual void compute_output(Tensor<BufferT> const &input,
                                Tensor<BufferT>       &output) const
    {
        auto const &output_shape(Layer<BufferT>::output_shape());

        if (input.shape() != output_shape)
        {
            throw std::invalid_argument(
                "ReLULayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        if (output.capacity() < Layer<BufferT>::output_size())
        {
            throw std::invalid_argument(
                "ReLULayer::compute_output() ERROR: "
                "insufficient output buffer space.");
        }

        small::ReLUActivation(output_shape[CHANNEL],
                              output_shape[HEIGHT], output_shape[WIDTH],
                              input.buffer(),
                              output.buffer());
        output.set_shape(Layer<BufferT>::output_shape());
    }
};

}
