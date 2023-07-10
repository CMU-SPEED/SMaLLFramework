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
// - stores the (single) input shape
// - support multiple input nodes?? [NO, not until we have specific need.]
// - support multiple outputs?? [YES]
//      - Must the output be stored in one BufferT or multiple? [MULTIPLE]
// - manages/owns all of the layers in a model
// - pure virtual functions for
//      - create model [NO]
//      - inference [YES]
//      - post process output [DON'T THINK SO...PUT IN DERIVED CLASSES]
// - Should this class be subclassed for specific models
//      - building model occurs during subclass constructor.
// - Manages the DAG of computation and produces/manages a schedule [POSSIBLE]
//      - automatic assignment of reusable buffers?
//      - automatic computation of buffer high water marks?
//      - schedule for sequential processing only or support parallel
//        layer computation?
// - Where should .cfg file parsing be done. [IN DERIVED CLASSES]
//      - is there a general purpose Model subclass that can parse any input file

namespace small
{

//****************************************************************************
template <typename BufferT>
class Model
{
public:
    Model() = delete;

    // Model() {}

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

    // Models can have multiple output buffers
    // Assume one input layer with a single shape for now
    /// @note Ownership of output buffers is NOT transferred to the caller.
    /// @todo Consider returning a vector of smart pointers (weak_ptr?) to the
    ///       output buffers.
    virtual std::vector<Tensor<BufferT>*> inference(
        Tensor<BufferT> const *input) = 0;

    // For debugging purposes
    size_t get_num_layers() const { return m_layers.size(); }

    Layer<BufferT>* get_layer(size_t idx)
    {
        if (idx < m_layers.size())
            return m_layers[idx];
        else
            throw std::invalid_argument("ERROR: invalid layer index.");
    }

protected:
    shape_type              m_input_shape;

    std::vector<Layer<BufferT>*> m_layers;
};

}
