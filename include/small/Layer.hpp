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

// Currently only for Conv2DLayer
enum ActivationType {
    NONE    = 0,  // aka LINEAR
    RELU    = 1,
    LEAKY   = 2,
    SOFTMAX = 3
};

//****************************************************************************
template <typename BufferT>
class Layer
{
public:
    Layer() {}

    Layer(shape_type const &output_shape)
    {
        set_output_shape(output_shape);
    }


    virtual ~Layer() {}

    inline size_t output_size() const
    {
        return m_output_size;
    }

    inline shape_type const &output_shape() const
    {
        return m_output_shape;
    }

    /// @todo Revisit this interface, recently switched to taking copies
    ///       so that I could call with initializer lists; i.e.,
    ///          compute_output({&input_tensor}, {&output_tensor});
    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const = 0;

protected:
    inline void set_output_shape(shape_type const &output_shape)
    {
        m_output_shape = output_shape;
        m_output_size = (output_shape[0]*output_shape[1]*
                         output_shape[2]*output_shape[3]);
    }

    shape_type m_output_shape;
    size_t     m_output_size;
};

}
