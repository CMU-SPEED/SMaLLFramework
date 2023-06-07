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
class RouteLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    RouteLayer(shape_type const &input1_shape,
             shape_type const &input2_shape)
        : Layer<BufferT>()
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "Add(batches:" << input1_shape[BATCH]
                  << ",chans:" << input1_shape[CHANNEL]
                  << ",img:" << input1_shape[HEIGHT]
                  << "x" << input1_shape[WIDTH]
                  << ")" << std::endl;
#endif

        // H and W must match
        if (input1_shape[HEIGHT] != input2_shape[HEIGHT] ||
            input1_shape[WIDTH] != input2_shape[WIDTH])
        {
            throw std::invalid_argument(
                "RouteLayer ctor ERROR: "
                "predecessors do not have same output shape.");
        }

        // only concat along channel dimension
        shape_type output_shape = input1_shape;
        output_shape[CHANNEL] = input1_shape[CHANNEL] + input2_shape[CHANNEL];

        this->set_output_shapes({output_shape});
    }

    // accept single input shape
    RouteLayer(shape_type const &input_shape) {
        this->set_output_shapes({input_shape});
    }

    virtual ~RouteLayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        std::vector<Tensor<BufferT>*>        output) const
    {
        ///@todo Do nothing or throw? or concat here?
    }
};

}
