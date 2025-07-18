//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2025 by The SMaLL Contributors, All Rights Reserved.
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
class LogSoftMaxLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    // ctor for actual == logical channels
    LogSoftMaxLayer(shape_type const &input_shape)
        : Layer<BufferT>(input_shape)       // input_shape == output_shape
    {
#if defined(DEBUG_LAYERS)
        auto const &output_shape(this->output_shape());
        std::cerr << "LogSoftMax(batches:" << output_shape[BATCH]
                  << ",chans/logical:" << output_shape[CHANNEL]
                  << "/" << this->logical_output_channels()
                  << ",img:" << output_shape[HEIGHT]
                  << "x" << output_shape[WIDTH]
                  << ")" << std::endl;
#endif
        if (((input_shape[CHANNEL] % BufferT::C_ib) != 0) ||
            ((input_shape[CHANNEL] % BufferT::C_ob) != 0))
        {
            throw std::invalid_argument(
                "LogSoftMaxLayer::ctor ERROR: invalid number of channels.");
        }
    }

    // ctor for actual != logical channels
    LogSoftMaxLayer(shape_type const &input_shape,
                    uint32_t          num_logical_channels)
        : Layer<BufferT>(input_shape, num_logical_channels)       // input_shape == output_shape
    {
#if defined(DEBUG_LAYERS)
        auto const &output_shape(this->output_shape());
        std::cerr << "LogSoftMax(batches:" << output_shape[BATCH]
                  << ",chans/logical:" << output_shape[CHANNEL]
                  << "/" << this->logical_output_channels()
                  << ",img:" << output_shape[HEIGHT]
                  << "x" << output_shape[WIDTH]
                  << ")" << std::endl;
#endif
        if (((input_shape[CHANNEL] % BufferT::C_ib) != 0) ||
            ((input_shape[CHANNEL] % BufferT::C_ob) != 0))
        {
            throw std::invalid_argument(
                "LogSoftMaxLayer::ctor ERROR: invalid number of channels.");
        }
        if ((num_logical_channels < 1) ||
            (num_logical_channels > input_shape[CHANNEL]))
        {
            throw std::invalid_argument(
                "LogSoftMaxLayer::ctor ERROR: invalid number of logical channels.");
        }
    }

    virtual ~LogSoftMaxLayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const
    {
        if ((input.size() != 1) || (input[0]->shape() != this->output_shape()))
        {
            throw std::invalid_argument(
                "LogSoftMaxLayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        if (output->capacity() < this->output_size())
        {
            throw std::invalid_argument(
                "LogSoftMaxLayer::compute_output() ERROR: "
                "insufficient output buffer space.");
        }

        auto const &output_shape(this->output_shape());

        small::LogSoftMax(output_shape[CHANNEL],
                          this->logical_output_channels(),
                          output_shape[HEIGHT], output_shape[WIDTH],
                          input[0]->buffer(),
                          output->buffer());

        output->set_shape(output_shape);
    }
};

}
