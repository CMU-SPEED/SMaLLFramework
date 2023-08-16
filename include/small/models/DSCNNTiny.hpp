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
#include <small/Conv2DLayer.hpp>
#include <small/DepthwiseConv2DLayer.hpp>
#include <small/MaxPool2DLayer.hpp>
#include <small/ReLULayer.hpp>

//****************************************************************************
/* RECORD_CALLS:

   Conv2D_rect(k:3x1,s:2,pad:[1,1,0,0],ochans:64,ichans:3,img:49x10,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)

   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:64,ichans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)

   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:64,ichans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)

   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:64,ichans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)

   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:64,ichans:64,img:25x5,I,F,O)
   ReLUActivation(chans:64,img:25x5,I,O)

   MaxPool2D_rect(k:25x5,s:1,pad:[0,0,0,0],chans:64,img:25x5,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:16,ichans:64,img:1x1,I,F,O)
*/

namespace small
{

//****************************************************************************
template <typename BufferT>
class DSCNNTiny : public Model<BufferT>
{
public:
    DSCNNTiny() = delete;

    // Assume one input layer with a single shape for now
    DSCNNTiny(shape_type            const &input_shape,
                    std::vector<BufferT*> const &filters,
                    bool                         filters_are_packed = false)
        : Model<BufferT>(input_shape),
          m_buffer_0(nullptr),
          m_buffer_1(nullptr)
    {
        create_model_and_buffers(filters, filters_are_packed);
    }

    virtual ~DSCNNTiny()
    {
        delete m_buffer_0;
        delete m_buffer_1;
    }

    /// @todo Consider returning a vector of smart pointers (weak_ptr?) to the
    ///       output buffers.
    virtual std::vector<Tensor<BufferT>*>
        inference(Tensor<BufferT> const *input_tensor)
    {
        // assert(input_tensors[0]->size() is correct);

        size_t layer_num = 0;
        this->m_layers[layer_num++]->compute_output({input_tensor},
                                                    m_buffer_0); // Conv2D+ReLU

        for (auto ix = 0U; ix < 4; ++ix)
        {
            this->m_layers[layer_num++]->compute_output({m_buffer_0},
                                                        m_buffer_1); // DWConv+ReLU
            this->m_layers[layer_num++]->compute_output({m_buffer_1},
                                                        m_buffer_0); // Conv2D+ReLU
        }

        this->m_layers[layer_num++]->compute_output({m_buffer_0},
                                                    m_buffer_1); // MaxPool2D
        this->m_layers[layer_num++]->compute_output({m_buffer_1},
                                                    m_buffer_0); // Conv2D

        return {m_buffer_0};
    }

private:
    Tensor<BufferT> *m_buffer_0;
    Tensor<BufferT> *m_buffer_1;

    void create_model_and_buffers(
        std::vector<BufferT*> const &filters,
        bool                         filters_are_packed)
    {
        // settings for first layer
        uint32_t kernel_height = 3, kernel_width = 1;
        uint32_t stride = 2;
        uint32_t input_height = 49, input_width = 10;
        uint32_t input_channels = 3, output_channels = 64;
        size_t   filter_num = 0;

        size_t max_elt_0 = 0UL;
        size_t max_elt_1 = 0UL;

        small::shape_type input_shape(
            {1UL, input_channels, input_height, input_width});

        small::Layer<BufferT> *prev =
            new small::Conv2DLayer<BufferT>(input_shape,
                                            kernel_height, kernel_width,
                                            stride, small::PADDING_F,
                                            output_channels,
                                            *filters[filter_num],
                                            filters_are_packed,
                                            RELU);
        this->m_layers.push_back(prev);
        max_elt_0 = std::max<size_t>(max_elt_0, prev->output_size());

        stride = 1;
        uint32_t num_channels = 64;
        uint32_t kernel_size = 3;

        // Four blocks of dw/relu/conv/relu
        for (auto ix = 0U; ix < 4; ++ix)
        {
            ++filter_num;
            kernel_size = 3;
            prev = new small::DepthwiseConv2DLayer<BufferT>(
                prev->output_shape(),
                kernel_size, kernel_size, stride, small::PADDING_F,
                *filters[filter_num], filters_are_packed,
                RELU);
            this->m_layers.push_back(prev);
            max_elt_1 = std::max<size_t>(max_elt_1, prev->output_size());

            ++filter_num;
            kernel_size = 1;
            prev = new small::Conv2DLayer<BufferT>(prev->output_shape(),
                                                   kernel_size, kernel_size,
                                                   stride, small::PADDING_V,
                                                   num_channels,
                                                   *filters[filter_num],
                                                   filters_are_packed,
                                                   RELU);
            this->m_layers.push_back(prev);
            max_elt_0 = std::max<size_t>(max_elt_0, prev->output_size());
        }

        prev = new small::MaxPool2DLayer<BufferT>(prev->output_shape(),
                                                  prev->output_shape()[HEIGHT],
                                                  prev->output_shape()[WIDTH],
                                                  stride, small::PADDING_V);
        this->m_layers.push_back(prev);
        max_elt_1 = std::max<size_t>(max_elt_1, prev->output_size());

        ++filter_num;
        kernel_size = 1;
        output_channels = 16;

        prev = new small::Conv2DLayer<BufferT>(prev->output_shape(),
                                               kernel_size, kernel_size,
                                               stride, small::PADDING_V,
                                               output_channels,
                                               *filters[filter_num],
                                               filters_are_packed);
        this->m_layers.push_back(prev);
        max_elt_0 = std::max<size_t>(max_elt_0, prev->output_size());


        m_buffer_0 = new Tensor<BufferT>(max_elt_0);
        m_buffer_1 = new Tensor<BufferT>(max_elt_1);
    }
};

}
