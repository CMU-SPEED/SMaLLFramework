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
class TinyYoloV2 : public Model<BufferT>
{
public:
    TinyYoloV2() = delete;

    // Assume one input layer with a single shape for now
    TinyYoloV2(shape_type            const &input_shape,
               std::vector<BufferT*> const &filters,
               bool                         filters_are_packed = false)
        : Model<BufferT>(input_shape),
          m_num_yolo_blocks(6U),
          m_buffer_0(nullptr),
          m_buffer_1(nullptr)
    {
        create_model_and_buffers(filters, filters_are_packed);
    }

    virtual ~TinyYoloV2()
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

        // yolo_block = 0
        this->m_layers[layer_num++]->compute_output({input_tensor},
                                                    m_buffer_1); // Conv2D+ReLU
        // std::cout << "ReLU(Y): " << (*m_buffer_1).buffer()[0]
        //           << "\t" << (*m_buffer_1).buffer()[1]
        //           << "\t" << (*m_buffer_1).buffer()[2]
        //           << "\t" << (*m_buffer_1).buffer()[3]
        //           << std::endl;
        this->m_layers[layer_num++]->compute_output({m_buffer_1},
                                                    m_buffer_0); // MaxPool2D

        for (size_t yolo_block = 1; yolo_block < m_num_yolo_blocks; ++yolo_block)
        {
            this->m_layers[layer_num++]->compute_output({m_buffer_0},
                                                        m_buffer_1); // Conv2D+ReLU
            // std::cout << "ReLU(Y): " << (*m_buffer_1).buffer()[0]
            //           << "\t" << (*m_buffer_1).buffer()[1]
            //           << "\t" << (*m_buffer_1).buffer()[2]
            //           << "\t" << (*m_buffer_1).buffer()[3]
            //           << std::endl;
            this->m_layers[layer_num++]->compute_output({m_buffer_1},
                                                        m_buffer_0); // MaxPool2D
        }

        for (size_t conv_block = 0; conv_block < 3; ++conv_block)
        {
            this->m_layers[layer_num++]->compute_output({m_buffer_0},
                                                        m_buffer_1); // Conv2D+ReLU
            // std::cout << "ReLU(C): " << (*m_buffer_1).buffer()[0]
            //           << "\t" << (*m_buffer_1).buffer()[1]
            //           << "\t" << (*m_buffer_1).buffer()[2]
            //           << "\t" << (*m_buffer_1).buffer()[3]
            //           << std::endl;
            m_buffer_0->swap(*m_buffer_1);
        }
        return {m_buffer_0};
    }

private:
    uint32_t const m_num_yolo_blocks;
    Tensor<BufferT> *m_buffer_0;
    Tensor<BufferT> *m_buffer_1;

    void create_model_and_buffers(
        std::vector<BufferT*> const &filters,
        bool                         filters_are_packed)
    {
        // settings for first layer
        uint32_t kernel_size = 3U;
        uint32_t stride = 1U;
        uint32_t output_channels = 16U;
        uint32_t num_classes = 16U;
        size_t filter_num = 0U;

        size_t max_elt_0 = 0UL;
        size_t max_elt_1 = 0UL;

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
            max_elt_1 = std::max<size_t>(max_elt_1,
                                         this->m_layers.back()->output_size());
            ++filter_num;

            kernel_size = 2;
            stride = layer_strides[yolo_block];
            this->m_layers.push_back(
                new small::MaxPool2DLayer<BufferT>(
                    this->m_layers.back()->output_shape(),
                    kernel_size, kernel_size,
                    stride,
                    small::PADDING_F)); /// @todo check
            max_elt_0 = std::max<size_t>(max_elt_0,
                                         this->m_layers.back()->output_size());

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
        max_elt_1 = std::max<size_t>(max_elt_1,
                                     this->m_layers.back()->output_size());
        ++filter_num;

        max_elt_0 = std::max<size_t>(max_elt_0, max_elt_1);  // for the swap

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
        max_elt_1 = std::max<size_t>(max_elt_1,
                                     this->m_layers.back()->output_size());
        ++filter_num;

        max_elt_0 = std::max<size_t>(max_elt_0, max_elt_1);  // for the swap

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
        max_elt_1 = std::max<size_t>(max_elt_1,
                                     this->m_layers.back()->output_size());
        ++filter_num;

        max_elt_0 = std::max<size_t>(max_elt_0, max_elt_1);  // for the swap

        //===============================

        std::cerr << "TinyYoloV2 internal buffer sizes: "
                  << max_elt_0 << ", " << max_elt_1 << std::endl;
        m_buffer_0 = new Tensor<BufferT>(max_elt_0);
        m_buffer_1 = new Tensor<BufferT>(max_elt_1);
    }
};

}
