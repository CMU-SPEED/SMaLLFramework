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

    ReLULayer(shape_type const &shape)
        : Layer<BufferT>(shape)       // input_shape == output_shape
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "ReLU(batches:" << shape[BATCH]
                  << ",chans/lchans:" << shape[CHANNEL]
                  << "/" << shape[L_CHAN]
                  << ",img:" << shape[HEIGHT]
                  << "x" << shape[WIDTH]
                  << ")" << std::endl;
#endif
        if (((shape[CHANNEL] % BufferT::C_ib) != 0) ||
            ((shape[CHANNEL] % BufferT::C_ob) != 0))
        {
            throw std::invalid_argument(
                "ReLULayer::ctor ERROR: invalid number of channels.");
        }
    }

    virtual ~ReLULayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const
    {
        if ((input.size() != 1) || (input[0]->shape() != this->output_shape()))
        {
            throw std::invalid_argument(
                "ReLULayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        if (output->capacity() < this->output_size())
        {
            throw std::invalid_argument(
                "ReLULayer::compute_output() ERROR: "
                "insufficient output buffer space.");
        }

        auto const &output_shape(this->output_shape());

        small::ReLUActivation(output_shape[CHANNEL],
                              output_shape[HEIGHT], output_shape[WIDTH],
                              input[0]->buffer(),
                              output->buffer());

        output->set_shape(output_shape);
    }
};

}
