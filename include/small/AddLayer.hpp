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
class AddLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    AddLayer(shape_type const &input1_shape,
             shape_type const &input2_shape)
        : Layer<BufferT>(input1_shape)
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "Add(batches:" << input1_shape[BATCH]
                  << ",chans:" << input1_shape[CHANNEL]
                  << ",img:" << input1_shape[HEIGHT]
                  << "x" << input1_shape[WIDTH]
                  << ")" << std::endl;
#endif

        if (input1_shape != input2_shape)
        {
            throw std::invalid_argument(
                "AddLayer ctor ERROR: "
                "predecessors do not have same output shape.");
        }
    }

    virtual ~AddLayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const
    {
        auto const &output_shape(this->output_shape());

        // AddLayer is a special case where the output buffer
        // must have the correct shape when this function is called.
        // No need to check capacity directly or set shape at end.
        if ((input.size()  != 1) || (input[0]->shape()  != output_shape) ||
            (output->shape() != output_shape))
        {
            throw std::invalid_argument(
                "AddLayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        /// out += in
        small::Accum(output_shape[CHANNEL],
                     output_shape[HEIGHT], output_shape[WIDTH],
                     input[0]->buffer(),
                     output->buffer());

        // No need to reset the shape of the output buffer.
        //output.set_shape(this->output_shape());
    }

private:
};

}
