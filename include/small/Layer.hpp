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
/**
 * "In-place activation functions that can used in DenseLayer,
 * Conv2D/Conv1DLayer, DepthwiseConv2DLayer, and PartialConv2DLayer
 *
 * @note LogSoftMax CANNOT be added here (it's not in-place)
 *
 * @todo Consider removing activations from within other layer classes
 */
enum ActivationType {
    NONE    = 0,  // aka LINEAR/Identity
    RELU    = 1,
    LEAKY   = 2,
    SOFTMAX = 3
};

//****************************************************************************
template <typename BufferT>
class Layer
{
public:
    Layer() = delete;

    Layer(uint32_t logical_output_channels) :
        m_logical_output_channels(logical_output_channels)
    {
    }

    Layer(shape_type const &output_shape) :
        m_logical_output_channels(output_shape[CHANNEL])
    {
        set_output_shape(output_shape);
    }

    Layer(shape_type const &output_shape,
          uint32_t          logical_output_channels) :
        m_logical_output_channels(logical_output_channels)
    {
        set_output_shape(output_shape);
    }


    virtual ~Layer() {}

    inline size_t output_size() const
    {
        return m_output_size;
    }

    uint32_t logical_output_channels() const
    {
        return m_logical_output_channels;
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
        if (output_shape[CHANNEL] < m_logical_output_channels)
        {
            throw std::invalid_argument(
                "Layer::set_output_shape() ERROR: "
                "output buffer channels not large enough for logical channels.");
        }

        m_output_shape = output_shape;
        m_output_size = (output_shape[0]*output_shape[1]*
                         output_shape[2]*output_shape[3]);
    }

    shape_type     m_output_shape;
    size_t         m_output_size;
    uint32_t const m_logical_output_channels;
};

}
