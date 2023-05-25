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

    ReLULayer(shape_type const &input_shape)
        : Layer<BufferT>({input_shape})       // input_shape == output_shape
    {
#if defined(DEBUG_LAYERS)
        auto const &output_shape(this->output_shape(0));
        std::cerr << "ReLU(batches:" << output_shape[BATCH]
                  << ",chans:" << output_shape[CHANNEL]
                  << ",img:" << output_shape[HEIGHT]
                  << "x" << output_shape[WIDTH]
                  << ")" << std::endl;
#endif
    }

    virtual ~ReLULayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT>*> const &input,
        std::vector<Tensor<BufferT>*>       &output) const
    {
        // assert input.size()==1, output.size()==1

        if ((input.size() != 1) || (input[0]->shape() != this->output_shape(0)))
        {
            throw std::invalid_argument(
                "ReLULayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        if ((output.size() != 1) || output[0]->capacity() < this->output_size(0))
        {
            throw std::invalid_argument(
                "ReLULayer::compute_output() ERROR: "
                "insufficient output buffer space.");
        }

        auto const &output_shape(this->output_shape(0));

        small::ReLUActivation(output_shape[CHANNEL],
                              output_shape[HEIGHT], output_shape[WIDTH],
                              input[0]->buffer(),
                              output[0]->buffer());

        output[0]->set_shape(output_shape);
    }
};

}
