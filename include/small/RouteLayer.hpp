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
class RouteLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    // accept single input shape
    RouteLayer(shape_type const &input0_shape)
        : Layer<BufferT>(input0_shape),
          m_num_inputs(1U),
          m_input0_shape(input0_shape),
          m_input1_shape({0,0,0,0})
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "Route: (batches:" << input0_shape[BATCH]
                  << ",chans:" << input0_shape[CHANNEL]
                  << ",img:" << input0_shape[HEIGHT]
                  << "x" << input0_shape[WIDTH]
                  << ")" << std::endl;
#endif
    }

    // two input shapes
    RouteLayer(shape_type const &input0_shape,
               shape_type const &input1_shape)
        : Layer<BufferT>(),
          m_num_inputs(2U),
          m_input0_shape(input0_shape),
          m_input1_shape(input1_shape)
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "Route: (batches:" << input0_shape[BATCH]
                  << ",chans:" << input0_shape[CHANNEL]
                  << ",img:" << input0_shape[HEIGHT]
                  << "x" << input0_shape[WIDTH]
                  << ")+"
                  << "(batches:" << input1_shape[BATCH]
                  << ",chans:" << input1_shape[CHANNEL]
                  << ",img:" << input1_shape[HEIGHT]
                  << "x" << input1_shape[WIDTH]
                  << ")" << std::endl;
#endif

        // B and H and W must match
        if (input1_shape[BATCH]  != input0_shape[BATCH] ||
            input1_shape[HEIGHT] != input0_shape[HEIGHT] ||
            input1_shape[WIDTH]  != input0_shape[WIDTH])
        {
            throw std::invalid_argument(
                "RouteLayer ctor ERROR: "
                "predecessors do not have same output shape.");
        }

        // only concat along channel dimension
        shape_type output_shape = input0_shape;
        output_shape[CHANNEL] = input0_shape[CHANNEL] + input1_shape[CHANNEL];

        this->set_output_shape(output_shape);
    }

    virtual ~RouteLayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const
    {
        if (input.size() != m_num_inputs)
        {
            throw std::invalid_argument(
                "RouteLayer::compute_output() ERROR: "
                "incorrect number of input buffers.");
        }

        if (input[0]->shape() != m_input0_shape)
        {
            throw std::invalid_argument(
                "RouteLayer::compute_output() ERROR: "
                "incorrect input[0] buffer shape.");
        }

        if ((m_num_inputs == 2) && (input[1]->shape() != m_input1_shape))
        {
            throw std::invalid_argument(
                "RouteLayer::compute_output() ERROR: "
                "incorrect input[1] buffer shape.");
        }

        if (output->capacity() < this->output_size())
        {
            throw std::invalid_argument(
                "RouteLayer::compute_output() ERROR: "
                "insufficient output buffer space.");
        }

        auto& output_shape(this->output_shape());

        /// @todo

        if (1 == input.size())
        {
            if (input[0] != output)
            {
                auto &input_buffer = input[0]->buffer();

                /// @todo Do we need microkernel for this operation?
                std::copy(input_buffer.data(),
                          input_buffer.data() + this->output_size(),
                          output->buffer().data());

                output->set_shape(output_shape);
            }

            // else do nothing
        }
        else if (2 == input.size())
        {
            if (input[1]->shape() != m_input1_shape)
            {
                throw std::invalid_argument(
                    "RouteLayer::compute_output() ERROR: "
                    "incorrect input[1] buffer shape.");
            }

            Concat(m_input0_shape[CHANNEL],
                   m_input1_shape[CHANNEL],
                   m_input0_shape[WIDTH],
                   m_input0_shape[HEIGHT],
                   input[0]->buffer(), input[1]->buffer(),
                   output->buffer());

            output->set_shape(output_shape);
        }
    }

private:
    uint32_t const   m_num_inputs;
    shape_type const m_input0_shape;
    shape_type const m_input1_shape;
};

}
