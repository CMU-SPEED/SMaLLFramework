//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2025 by The SMaLL Contributors, All Rights Reserved.
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
class UpSample1DLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    UpSample1DLayer(shape_type const &input_shape,
                    uint32_t scale_factor)
        : Layer<BufferT>(),
          m_input_shape(input_shape),
          m_scale_factor(scale_factor)
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "UpSample1D(batches:" << m_input_shape[BATCH]
                  << ",scale:" << scale_factor
                  << ",ichans/lchans:" << m_input_shape[CHANNEL]
                  << "/" << m_input_shape[L_CHAN]
                  << ",img:" << m_input_shape[HEIGHT]
                  << "x" << m_input_shape[WIDTH]
                  << std::endl;
#endif
        /// @todo assert m_input_shape[HEIGHT] == 1 or support batches?
        if (m_input_shape[HEIGHT] != 1U)
        {
            throw std::invalid_argument(
                "UpSample1DLayer ERROR: image height must be 1.");
        }

        if ((scale_factor != 1)  && (scale_factor !=2))
        {
            throw std::invalid_argument(
                "UpSample1DLayer ERROR: unsupported scale factor.");
        }

        /// @todo is there a clean way to make these const members, or
        ///       will image size get moved to compute_output() and all of
        ///       this moves to compute_output()?

        this->set_output_shape(
            {m_input_shape[BATCH],
             m_input_shape[CHANNEL],
             1U, //m_input_shape[HEIGHT]*scale_factor,
             m_input_shape[WIDTH]*scale_factor,
             m_input_shape[L_CHAN]});
    }

    virtual ~UpSample1DLayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const
    {
        if ((input.size() != 1) || (input[0]->shape() != m_input_shape))
        {
            throw std::invalid_argument(
                "UpSample1DLayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        if (output->capacity() < this->output_size())
        {
            throw std::invalid_argument(
                "UpSample1DLayer::compute_output ERROR: "
                "insufficient output buffer space.");
        }

        UpSample1D(m_scale_factor,
                   m_input_shape[CHANNEL],
                   m_input_shape[HEIGHT], // BATCH
                   m_input_shape[WIDTH],
                   input[0]->buffer(),
                   output->buffer());

        output->set_shape(this->output_shape());
    }

private:
    shape_type const m_input_shape;

    /// @todo support separate x,y scale factors
    uint32_t const m_scale_factor;
};

}
