//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2026 by The SMaLL Contributors, All Rights Reserved.
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
#include <typeinfo>

#include <small.h>
#include <small/Tensor.hpp>
#include <small/Layer.hpp>

// Some possible requirements
// - has the same interface as small::Layer
// - must be initialized with at least two primitive layers that are not
//   themselves SequentialLayer's.
// - single sequence of layers where the first layer may have 1+ input tensors
//   and one output tensor
// - stores a set of layers that may or shall be fused?  Those layers may be
//   replaced by a single fused layer
// - should it support unfused operation using intermediate activation buffers?


namespace small
{

//****************************************************************************
template <typename BufferT>
class SequentialLayer : public Layer<BufferT>
{
public:
    SequentialLayer(std::vector<Layer<BufferT>*> layers);

    virtual ~SequentialLayer()
    {
        for (auto layer : m_layers)
        {
            delete layer;
        }
        for (auto tensor : m_tensors)
        {
            delete tensor;
        }
    }

    void fuse_layers();
    void allocate_tensors();

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const;

private:
    std::vector<Layer<BufferT>*>          m_layers;
    mutable std::vector<Tensor<BufferT>*> m_tensors;

};

//****************************************************************************
template<class BufferT>
SequentialLayer<BufferT>::SequentialLayer(
    std::vector<Layer<BufferT>*> layers)
    : Layer<BufferT>(),
      m_layers(layers)
{
    if (layers.size() < 2)
    {
        throw std::invalid_argument(
            "SequentialLayer::ctor ERROR: "
            "invalid number of layers (must be >1).");
    }

    for (auto &&layer : m_layers)
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "SequentialLayer type: " << typeid(*layer).name()
                  << std::endl;
#endif
        if (typeid(*layer) == typeid(*this))
        {
            throw std::invalid_argument(
                "SequentialLayer::ctor ERROR: "
                "invalid layer type (cannot be SequentialLayer).");
        }
    }

    this->set_output_shape(m_layers.back()->output_shape());
}


//****************************************************************************
template<class BufferT>
void SequentialLayer<BufferT>::fuse_layers()
{
#if defined(DEBUG_LAYERS)
    std::cerr << "SequentialLayer: fusing layers.\n";
#endif
}

//****************************************************************************
template<class BufferT>
void SequentialLayer<BufferT>::allocate_tensors()
{
#if defined(DEBUG_LAYERS)
    std::cerr << "SequentialLayer: allocating tensors.\n";
#endif
}

//****************************************************************************
template<class BufferT>
void SequentialLayer<BufferT>::compute_output(
    std::vector<Tensor<BufferT> const *> input,
    Tensor<BufferT>*                     output) const
{
    // ERROR CHECKING NEEDED...sizes bw each pair of consecutive layers

    for (size_t ix = 0; ix < m_layers.size() - 1; ++ix)
    {
        m_layers[ix]->compute_output(input, m_tensors[ix]);
        input = {m_tensors[ix]};
    }

    m_layers.back()->compute_output(input, output);
    output->set_shape(this->output_shape());
}

} // ns small
