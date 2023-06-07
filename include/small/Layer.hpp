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
    LEAKY   = 2
};

//****************************************************************************
template <typename BufferT>
class Layer
{
public:
    Layer() {}

    Layer(shape_type const &output_shape) 
        : m_parent_idxs({-1}) // default to prev layer. @todo: need to consider input layer
    {
        set_output_shapes({output_shape});
    }

    Layer(shape_type const &output_shape, 
          std::vector<int> const &parents_idxs)
        : m_parent_idxs(parents_idxs)
    {
        set_output_shapes({output_shape});
    }


    Layer(std::vector<shape_type> const &output_shapes)
        : m_parent_idxs({-1})
    {
        if (output_shapes.size() == 0)
        {
            throw std::invalid_argument(
                "Layer::ctor ERROR: output_shapes empty.");
        }

        set_output_shapes(output_shapes);
    }

    Layer(std::vector<shape_type> const &output_shapes, 
          std::vector<int> const &parents_idxs)
        : m_parent_idxs(parents_idxs)
    {
        if (output_shapes.size() == 0)
        {
            throw std::invalid_argument(
                "Layer::ctor ERROR: output_shapes empty.");
        }

        set_output_shapes(output_shapes);
    }

    virtual ~Layer() {}

    inline size_t get_num_outputs() const { return m_output_sizes.size(); }

    inline std::vector<int> parents() const { return m_parent_idxs; }

    inline size_t output_size(size_t idx = 0) const
    {
        return m_output_sizes[idx];
    }

    inline shape_type const &output_shape(size_t idx = 0) const
    {
        return m_output_shapes[idx];
    }

    /// @todo Revisit this interface, recently switched to taking copies
    ///       so that I could call with initializer lists; i.e.,
    ///          compute_output({&input_tensor}, {&output_tensor});
    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        std::vector<Tensor<BufferT>*>        output) const = 0;

protected:
    inline void set_output_shapes(std::vector<shape_type> const &output_shapes)
    {
        m_output_shapes.clear();
        m_output_sizes.clear();
        for (auto& output_shape : output_shapes)
        {
            m_output_shapes.push_back(output_shape);
            m_output_sizes.push_back(output_shape[0]*output_shape[1]*
                                     output_shape[2]*output_shape[3]);
        }
    }

    inline void set_parents_idxs(std::vector<int> const &parents_idxs)
    {
        m_parent_idxs.clear();
        for(auto& parent_idx : parents_idxs) {
            m_parent_idxs.push_back(parent_idx);
        }
    }

    std::vector<shape_type> m_output_shapes;
    std::vector<size_t>     m_output_sizes;
    std::vector<int>        m_parent_idxs;
};

}
