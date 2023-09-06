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
class LeakyReLULayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    LeakyReLULayer(shape_type const &input_shape,
                   float             leaky_slope = 0.01f)  /// @todo use value_type?
        : Layer<BufferT>(input_shape),          // input_shape == output_shape
          m_leaky_slope(1)
    {
#if defined(DEBUG_LAYERS)
        auto const &output_shape(this->output_shape());
        std::cerr << "LeakyReLU(batches:" << output_shape[BATCH]
                  << ",chans:" << output_shape[CHANNEL]
                  << ",leaky_slope:" << leaky_slope
                  << ",img:" << output_shape[HEIGHT]
                  << "x" << output_shape[WIDTH]
                  << ")" << std::endl;
#endif
        /// @todo Do any type conversion between float and value_type here.
        m_leaky_slope[0] = leaky_slope;
    }

    virtual ~LeakyReLULayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const
    {
        if ((input.size() != 1) || (input[0]->shape() != this->output_shape()))
        {
            throw std::invalid_argument(
                "LeakyReLULayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        if (output->capacity() < this->output_size())
        {
            throw std::invalid_argument(
                "LeakyReLULayer::compute_output() ERROR: "
                "insufficient output buffer space.");
        }

        auto const &output_shape(this->output_shape());

        small::LeakyReLUActivation(output_shape[CHANNEL],
                                   output_shape[HEIGHT], output_shape[WIDTH],
                                   input[0]->buffer(),
                                   m_leaky_slope,
                                   output->buffer());

        output->set_shape(this->output_shape());
    }

private:
    BufferT    m_leaky_slope;      /// @todo should this be value_type?
};

}
