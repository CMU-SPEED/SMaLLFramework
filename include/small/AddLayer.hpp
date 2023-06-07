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
             shape_type const &input2_shape,
             std::vector<int> const &parents_idxs)
        : Layer<BufferT>()
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

        this->set_output_shapes({input1_shape});
        this->set_parents_idxs(parents_idxs);
    }

    virtual ~AddLayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        std::vector<Tensor<BufferT>*>        output) const
    {
        auto const &output_shape(this->output_shape());

        // AddLayer is a special case where the output buffer
        // must have the correct shape when this function is called.
        // No need to check capacity directly or set shape at end.
        if ((input.size()  != 1) || (input[0]->shape()  != output_shape) ||
            (output.size() != 1) || (output[0]->shape() != output_shape))
        {
            throw std::invalid_argument(
                "DepthwiseConv2DLayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        /// out += in
        small::Accum(output_shape[CHANNEL],
                     output_shape[HEIGHT], output_shape[WIDTH],
                     input[0]->buffer(),
                     output[0]->buffer());

        // No need to reset the shape of the output buffer.
        //output.set_shape(this->output_shape());
    }
};

}
