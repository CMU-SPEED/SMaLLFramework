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
#include <small/Model.hpp>
#include <small/InputLayer.hpp>
#include <small/Conv2DLayer.hpp>
#include <small/ReLULayer.hpp>

namespace small
{

//****************************************************************************
template <typename BufferT>
class AutoencoderTiny : public Model<BufferT>
{
public:
    typedef typename Tensor<BufferT>::shape_type shape_type;

    AutoencoderTiny() = delete;

    // Assume one input layer with a single shape for now
    AutoencoderTiny(shape_type            const &input_shape,
                    size_t                       dimension_reduction,
                    std::vector<BufferT*> const &filters,
                    bool                         filters_are_packed = false)
        : Model<BufferT>(input_shape)
    {
        create_model_and_buffers(dimension_reduction, filters, filters_are_packed);
    }

    virtual ~AutoencoderTiny()
    {
        for (auto buffer : m_buffers_0)
        {
            delete buffer;
        }
        for (auto buffer : m_buffers_1)
        {
            delete buffer;
        }
    }

    /// @todo consider returning a vector of weak pointers to internal buffers for
    ///       the outputs
    virtual void inference(std::vector<Tensor<BufferT>*> const &input_tensors,
                           std::vector<Tensor<BufferT>*>       &output_tensors) //weakptr?
    {
        // assert(input_tensors.size() == 1);
        // assert(input_tensors[0]->size() is correct);
        // assert(output_tensors.size() == 0);
        // assert(m_buffers.size() == 2);

        size_t layer_num = 0;  // skip input layer
        // Conv2D + ReLU
        this->m_layers[layer_num++]->compute_output(input_tensors, m_buffers_0);

        while (layer_num < this->m_layers.size())
        {
            // Conv2D + ReLU
            this->m_layers[layer_num++]->compute_output(m_buffers_0, m_buffers_1);

            m_buffers_0.swap(m_buffers_1);
        }

        output_tensors = m_buffers_0;
    }

private:
    std::vector<Tensor<BufferT>*> m_buffers_0;
    std::vector<Tensor<BufferT>*> m_buffers_1;

    void create_model_and_buffers(
        size_t                       dimension_reduction,
        std::vector<BufferT*> const &filters,
        bool                         filters_are_packed)
    {
        uint32_t kernel_size = 1;
        uint32_t stride = 1;
        uint32_t output_channels = 128;

        /// @todo assert dimension_reduction is a multiple of "16"

        //std::vector<shape_type> buffer_shapes;
        //buffer_shapes.push_back(this->get_input_shape());

        size_t max_elt_0 = 0UL;
        size_t max_elt_1 = 0UL;

        Layer<BufferT> *prev = nullptr;
        shape_type prev_shape(this->m_input_shape);

        for (auto ix = 0U; ix < filters.size(); ++ix)
        {
            /// @todo Support "dimension_reduction == 8;"
            if (ix == 4)
                output_channels = dimension_reduction;
            else
                output_channels = 128;

            prev = new small::Conv2DLayer<BufferT>(prev_shape,
                                                   kernel_size, kernel_size,
                                                   stride, small::PADDING_V,
                                                   output_channels,
                                                   *filters[ix], filters_are_packed,
                                                   RELU);
            this->m_layers.push_back(prev);
            prev_shape = prev->output_shape(0);

            if (ix == 0)
            {
                max_elt_0 = std::max<size_t>(max_elt_0, prev->output_size(0));
            }
            else
            {
                max_elt_1 = std::max<size_t>(max_elt_1, prev->output_size(0));
                max_elt_0 = std::max<size_t>(max_elt_0, max_elt_1);  // for the swap
            }
        }

        m_buffers_0.push_back(new Tensor<BufferT>(max_elt_0));
        m_buffers_1.push_back(new Tensor<BufferT>(max_elt_1));
    }
};

}
