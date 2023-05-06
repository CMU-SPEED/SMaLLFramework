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

//#define DEBUG_LAYERS

#include<vector>
#include<small.h>
#include<small/Tensor.hpp>

namespace small
{
//****************************************************************************
template <typename BufferT>
class Layer
{
public:
    typedef typename Tensor<BufferT>::shape_type shape_type;
    Layer(Layer<BufferT> const *predecessor) : m_predecessor(predecessor) {};

    virtual ~Layer() {}

    virtual size_t output_buffer_size() const = 0;
    virtual shape_type output_buffer_shape() const = 0;

    virtual void compute_output(Tensor<BufferT> const &input,
                                Tensor<BufferT>       &output) const = 0;

protected:
    Layer<BufferT> const *m_predecessor;
};

}
