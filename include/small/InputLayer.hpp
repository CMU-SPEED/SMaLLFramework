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

    InputLayer(shape_type const &shape) // {B, C, H, W}
        : Layer<BufferT>(shape)
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "Input(batches:" << shape[BATCH]
                  << ",chans:" << shape[CHANNEL]
                  << ",img:" << shape[HEIGHT] << "x" << shape[WIDTH]
                  << ")" << std::endl;
#endif
    }

    virtual ~InputLayer() {}

    virtual void compute_output(Tensor<BufferT> const &input,
                                Tensor<BufferT>       &output) const
    {
        ///@todo Do nothing or throw? or pack the input into the output?
    }
};

}
