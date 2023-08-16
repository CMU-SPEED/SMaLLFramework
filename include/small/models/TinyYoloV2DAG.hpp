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
#include <small/DAGModel.hpp>
#include <small/Conv2DLayer.hpp>
#include <small/MaxPool2DLayer.hpp>
#include <small/ReLULayer.hpp>

//****************************************************************************

//****************************************************************************
/* RECORD_CALLS:

   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:16,ichans:3,img:416x416,I,F,O)
   ReLUActivation(chans:16,img:416x416,I,O)
   MaxPool2D(k:2,s:2,pad:[0,0,0,0],chans:16,img:416x416,I,O)
   layer 2
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:32,ichans:16,img:208x208,I,F,O)
   ReLUActivation(chans:32,img:208x208,I,O)
   MaxPool2D(k:2,s:2,pad:[0,0,0,0],chans:32,img:208x208,I,O)
   layer 4
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:64,ichans:32,img:104x104,I,F,O)
   ReLUActivation(chans:64,img:104x104,I,O)
   MaxPool2D(k:2,s:2,pad:[0,0,0,0],chans:64,img:104x104,I,O)
   layer 6
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:128,ichans:64,img:52x52,I,F,O)
   ReLUActivation(chans:128,img:52x52,I,O)
   MaxPool2D(k:2,s:2,pad:[0,0,0,0],chans:128,img:52x52,I,O)
   layer 8
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:256,ichans:128,img:26x26,I,F,O)
   ReLUActivation(chans:256,img:26x26,I,O)
   MaxPool2D(k:2,s:2,pad:[0,0,0,0],chans:256,img:26x26,I,O)
   layer 10
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:512,ichans:256,img:13x13,I,F,O)
   ReLUActivation(chans:512,img:13x13,I,O)
   MaxPool2D(k:2,s:1,pad:[0,1,0,1],chans:512,img:13x13,I,O)

   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:1024,ichans:512,img:13x13,I,F,O)
   ReLUActivation(chans:1024,img:13x13,I,O)

   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:1024,ichans:1024,img:13x13,I,F,O)
   ReLUActivation(chans:1024,img:13x13,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:16,ichans:1024,img:13x13,I,F,O)
   ReLUActivation(chans:16,img:13x13,I,O)

*/

namespace small
{

//****************************************************************************
template <typename BufferT>
class TinyYoloV2DAG : public DAGModel<BufferT>
{
public:
    TinyYoloV2DAG() = delete;

    // Assume one input layer with a single shape for now
    TinyYoloV2DAG(shape_type            const &input_shape,
                  std::vector<BufferT*> const &filters,
                  bool                         filters_are_packed = false)
        : DAGModel<BufferT>(input_shape),
          m_num_yolo_blocks(6U)
    {
        size_t max_buffer_size =
            create_model_and_buffers(filters, filters_are_packed);
        this->initializeDAG(max_buffer_size);
    }

    virtual ~TinyYoloV2DAG()
    {
    }

private:
    uint32_t const m_num_yolo_blocks;

    size_t create_model_and_buffers(
        std::vector<BufferT*> const &filters,
        bool                         filters_are_packed)
    {
        // settings for first layer
        uint32_t kernel_size = 3U;
        uint32_t stride = 1U;
        uint32_t output_channels = 16U;
        uint32_t num_classes = 16U;
        size_t filter_num = 0U;
        size_t layer_idx = 0UL;
        size_t max_buffer_size(0UL);

        shape_type input_shape = this->m_input_shape;

        uint32_t const layer_strides[] = {2,2,2,2,2,1};
        for (size_t yolo_block = 0; yolo_block < m_num_yolo_blocks; ++yolo_block)
        {
            kernel_size = 3;
            stride = 1;

            this->m_layers.push_back(
                new small::Conv2DLayer<BufferT>(
                    input_shape,
                    kernel_size, kernel_size,
                    stride, small::PADDING_F,
                    output_channels,
                    *filters[filter_num], filters_are_packed,
                    RELU));
            max_buffer_size =
                std::max<size_t>(max_buffer_size,
                                 this->m_layers.back()->output_size());
            ++filter_num;

            // =================================================
            this->m_graph.add_vertex(layer_idx);
            if (layer_idx > 0) this->m_graph.add_edge(layer_idx-1, layer_idx);
            ++layer_idx;
            // =================================================

            kernel_size = 2;
            stride = layer_strides[yolo_block];
            this->m_layers.push_back(
                new small::MaxPool2DLayer<BufferT>(
                    this->m_layers.back()->output_shape(),
                    kernel_size, kernel_size,
                    stride,
                    small::PADDING_F)); /// @todo check
            max_buffer_size =
                std::max<size_t>(max_buffer_size,
                                 this->m_layers.back()->output_size());

            // =================================================
            this->m_graph.add_vertex(layer_idx);
            this->m_graph.add_edge(layer_idx-1, layer_idx);
            ++layer_idx;
            // =================================================

            input_shape = this->m_layers.back()->output_shape();
            output_channels = 2*output_channels;
        }

        // Final convolution layers
        //=====================================
        kernel_size = 3;
        stride = 1;

        this->m_layers.push_back(
            new small::Conv2DLayer<BufferT>(
                this->m_layers.back()->output_shape(),
                kernel_size, kernel_size,
                stride, small::PADDING_F,
                output_channels,
                *filters[filter_num],
                filters_are_packed,
                RELU));

        // =================================================
        this->m_graph.add_vertex(layer_idx);
        this->m_graph.add_edge(layer_idx-1, layer_idx);
        ++layer_idx;
        // =================================================

        max_buffer_size =
            std::max<size_t>(max_buffer_size,
                             this->m_layers.back()->output_size());
        ++filter_num;

        //=====================================

        this->m_layers.push_back(
            new small::Conv2DLayer<BufferT>(
                this->m_layers.back()->output_shape(),
                kernel_size, kernel_size,
                stride, small::PADDING_F,
                output_channels,
                *filters[filter_num],
                filters_are_packed,
                RELU));

        // =================================================
        this->m_graph.add_vertex(layer_idx);
        this->m_graph.add_edge(layer_idx-1, layer_idx);
        ++layer_idx;
        // =================================================

        max_buffer_size =
            std::max<size_t>(max_buffer_size,
                             this->m_layers.back()->output_size());
        ++filter_num;

        //=====================================
        kernel_size = 1;
        output_channels = num_classes;

        this->m_layers.push_back(
            new small::Conv2DLayer<BufferT>(
                this->m_layers.back()->output_shape(),
                kernel_size, kernel_size,
                stride, small::PADDING_F,
                output_channels,
                *filters[filter_num],
                filters_are_packed,
                RELU));

        // =================================================
        this->m_graph.add_vertex(layer_idx);
        this->m_graph.add_edge(layer_idx-1, layer_idx);
        ++layer_idx;
        // =================================================

        max_buffer_size =
            std::max<size_t>(max_buffer_size,
                             this->m_layers.back()->output_size());
        ++filter_num;

        //===============================

        std::cerr << "TinyYoloV2DAG internal buffer sizes: "
                  << max_buffer_size << std::endl;

        return max_buffer_size;
    }
};

}
