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
#include <small/DepthwiseConv2DLayer.hpp>
#include <small/Conv2DLayer.hpp>
#include <small/MaxPool2DLayer.hpp>
#include <small/ReLULayer.hpp>

//****************************************************************************
/* This is the runtime recording:

   Conv2D(k:3,s:2,pad:[0,1,0,1],ochans:32,ichans:3,img:96x96,I,F,O)
   ReLUActivation(chans:32,img:48x48,I,O)
0
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:32,img:48x48,I,F,O)     s1
   ReLUActivation(chans:32,img:48x48,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:64,ichans:32,img:48x48,I,F,O)   2x
   ReLUActivation(chans:64,img:48x48,I,O)
1
   DepthwiseConv2D(k:3,s:2,pad:[0,1,0,1],chans:64,img:48x48,I,F,O)     s2
   ReLUActivation(chans:64,img:24x24,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:64,img:24x24,I,F,O)  2x
   ReLUActivation(chans:128,img:24x24,I,O)
2
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:128,img:24x24,I,F,O)    s1
   ReLUActivation(chans:128,img:24x24,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:24x24,I,F,O) 1x
   ReLUActivation(chans:128,img:24x24,I,O)
3
   DepthwiseConv2D(k:3,s:2,pad:[0,1,0,1],chans:128,img:24x24,I,F,O)    s2
   ReLUActivation(chans:128,img:12x12,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:256,ichans:128,img:12x12,I,F,O) 2x
   ReLUActivation(chans:256,img:12x12,I,O)
4
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:256,img:12x12,I,F,O)    s1
   ReLUActivation(chans:256,img:12x12,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:256,ichans:256,img:12x12,I,F,O) 1x
   ReLUActivation(chans:256,img:12x12,I,O)
5
   DepthwiseConv2D(k:3,s:2,pad:[0,1,0,1],chans:256,img:12x12,I,F,O)    s2
   ReLUActivation(chans:256,img:6x6,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:256,img:6x6,I,F,O)   2x
   ReLUActivation(chans:512,img:6x6,I,O)
6
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:512,img:6x6,I,F,O)      s1
   ReLUActivation(chans:512,img:6x6,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:512,img:6x6,I,F,O)   1x
   ReLUActivation(chans:512,img:6x6,I,O)
7
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:512,img:6x6,I,F,O)      s1
   ReLUActivation(chans:512,img:6x6,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:512,img:6x6,I,F,O)   1x
   ReLUActivation(chans:512,img:6x6,I,O)
8
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:512,img:6x6,I,F,O)      s1
   ReLUActivation(chans:512,img:6x6,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:512,img:6x6,I,F,O)   1x
   ReLUActivation(chans:512,img:6x6,I,O)
9
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:512,img:6x6,I,F,O)      s1
   ReLUActivation(chans:512,img:6x6,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:512,img:6x6,I,F,O)   1x
   ReLUActivation(chans:512,img:6x6,I,O)
10
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:512,img:6x6,I,F,O)      s1
   ReLUActivation(chans:512,img:6x6,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:512,img:6x6,I,F,O)   1x
   ReLUActivation(chans:512,img:6x6,I,O)
11
   DepthwiseConv2D(k:3,s:2,pad:[0,1,0,1],chans:512,img:6x6,I,F,O)      s2
   ReLUActivation(chans:512,img:3x3,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:1024,ichans:512,img:3x3,I,F,O)  2x
   ReLUActivation(chans:1024,img:3x3,I,O)
12
   DepthwiseConv2D(k:3,s:1,pad:[1,1,1,1],chans:1024,img:3x3,I,F,O)     s1
   ReLUActivation(chans:1024,img:3x3,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:1024,ichans:1024,img:3x3,I,F,O) 1x
   ReLUActivation(chans:1024,img:3x3,I,O)
FINAL
   MaxPool2D(k:3,s:1,pad:[0,0,0,0],chans:1024,img:3x3,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:16,ichans:1024,img:1x1,I,F,O)
*/

namespace small
{

//****************************************************************************
template <typename BufferT>
class MobileNetTiny : public Model<BufferT>
{
public:
    MobileNetTiny() = delete;

    // Expecting single input_shape = {1UL,  3UL, 96UL, 96UL}
    // Hardcoding num_classes = 16:   {1UL, 16UL,  1UL,  1UL}
    MobileNetTiny(shape_type            const &input_shape,
                  std::vector<BufferT*> const &filters,
                  bool                         filters_are_packed = false)
        : Model<BufferT>(input_shape),
          m_buffer_0(nullptr),
          m_buffer_1(nullptr)
    {
        create_model_and_buffers(filters, filters_are_packed);
    }

    virtual ~MobileNetTiny()
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

        for (auto ix = 0U; ix < 13; ++ix)
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
        // settings for first layers
        uint32_t kernel_size = 3;
        uint32_t stride = 2;
        uint32_t input_size = 96;
        uint32_t input_channels = 3;
        uint32_t output_channels = 32;
        uint32_t num_classes = 16;
        size_t   filter_num = 0;

        size_t max_elt_0 = 0UL;
        size_t max_elt_1 = 0UL;

        small::shape_type input_shape(
            {1UL, input_channels, input_size, input_size});

        small::Layer<BufferT> *prev =
            new small::Conv2DLayer<BufferT>(this->m_input_shape,
                                            kernel_size, kernel_size,
                                            stride, small::PADDING_F,
                                            output_channels,
                                            *filters[filter_num++],
                                            filters_are_packed,
                                            RELU);
        max_elt_0 = std::max<size_t>(max_elt_0, prev->output_size());
        this->m_layers.push_back(prev);

        size_t   const num_blocks{13};
        uint32_t const block_strides[]   = {1,2,1,2,1,2,1,1,1,1,1,2,1};
        uint32_t const channel_multiplier[] = {2,2,1,2,1,2,1,1,1,1,1,2,1};
        for (auto block_num = 0U; block_num < num_blocks; ++block_num)
        {
            kernel_size = 3;

            prev = new small::DepthwiseConv2DLayer<BufferT>(
                prev->output_shape(),
                kernel_size, kernel_size, block_strides[block_num],
                small::PADDING_F,
                *filters[filter_num++],
                filters_are_packed,
                RELU);
            max_elt_1 = std::max<size_t>(max_elt_1, prev->output_size());
            this->m_layers.push_back(prev);

            // =======================================================

            kernel_size = 1;
            stride = 1;
            output_channels =
                prev->output_shape()[small::CHANNEL]*channel_multiplier[block_num];

            prev = new small::Conv2DLayer<BufferT>(prev->output_shape(),
                                                   kernel_size, kernel_size,
                                                   stride, small::PADDING_V,
                                                   output_channels,
                                                   *filters[filter_num++],
                                                   filters_are_packed,
                                                   RELU);
            max_elt_0 = std::max<size_t>(max_elt_0, prev->output_size());
            this->m_layers.push_back(prev);
        }

        kernel_size = 3;
        stride = 1;

        prev = new small::MaxPool2DLayer<BufferT>(prev->output_shape(),
                                                  kernel_size, kernel_size,
                                                  stride, small::PADDING_V);
        max_elt_1 = std::max<size_t>(max_elt_1, prev->output_size());
        this->m_layers.push_back(prev);

        kernel_size = 1;
        prev = new small::Conv2DLayer<BufferT>(prev->output_shape(),
                                               kernel_size, kernel_size, stride,
                                               small::PADDING_V,
                                               num_classes,
                                               *filters[filter_num++],
                                               filters_are_packed);
        max_elt_0 = std::max<size_t>(max_elt_0, prev->output_size());
        this->m_layers.push_back(prev);

        m_buffer_0 = new Tensor<BufferT>(max_elt_0);
        m_buffer_1 = new Tensor<BufferT>(max_elt_1);

    }
};

}
