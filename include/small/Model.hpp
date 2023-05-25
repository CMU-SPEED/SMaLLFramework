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

// Some possible requirements
// - stores the input shape
// - manages/owns all of the layers in a model
// - pure virtual functions for
//      - create model [NO]
//      - inference [YES]
//      - post process output [DON'T THINK SO]
// - Should this class be subclassed for specific models
//      - building model occurs during subclass constructor.
// - support multiple input nodes??
// - support multiple outputs??
//      - Must the output be stored in a BufferT?
// - Manages the DAG of computation [POSSIBLE]
//      - automatic assignment of reusable buffers?
//      - automatic computation of buffer high water marks?
// - Where should .cfg file parsing be done.
//      - is there a general purpose Model subclass that can parse any input file

namespace small
{

//****************************************************************************
template <typename BufferT>
class Model
{
public:
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

    shape_type const &get_input_shape() const { return m_input_shape; }

    // Models can have multiple input and output buffers
    /// @note Ownership of output buffers is NOT transferred to the caller.
    /// @todo Consider returning a vector of smart pointers (weak_ptr?) to the
    ///       output buffers.
    virtual std::vector<Tensor<BufferT>*>
        inference(std::vector<Tensor<BufferT> const *> input) = 0;

protected:
    shape_type const             m_input_shape;

    std::vector<Layer<BufferT>*> m_layers;
};

}
