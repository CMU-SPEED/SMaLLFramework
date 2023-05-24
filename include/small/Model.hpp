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

#include <vector>
#include <small.h>
#include <small/Tensor.hpp>
#include <small/Layer.hpp>

namespace small
{

//****************************************************************************
template <typename BufferT>
class Model
{
public:
    typedef typename Tensor<BufferT>::shape_type shape_type;

    Model() = delete;

    // Assume one input layer with a single shape for now
    Model(shape_type const &input_shape)
        : m_input_shape(input_shape)
    {
    }

    virtual ~Model()
    {
        std::cerr << "Model::dtor: deleting " << m_layers.size() << " layers.\n";
        for (auto layer : m_layers)
        {
            delete layer;
        }
    }

    // Models can have multiple input and output buffers
    /// @todo consider a weak_ptr to the output buffers.
    virtual void inference(std::vector<Tensor<BufferT>*> const &input,
                           std::vector<Tensor<BufferT>*>       &output) const = 0;

protected:
    shape_type const             m_input_shape;

    std::vector<Layer<BufferT>*> m_layers;
};

}
