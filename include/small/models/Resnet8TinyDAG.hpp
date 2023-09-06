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
#include <small/PartialConv2DLayer.hpp>
#include <small/Conv2DLayer.hpp>
#include <small/MaxPool2DLayer.hpp>
#include <small/ReLULayer.hpp>
#include <small/AddLayer.hpp>

/* From https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/keras_model.py

   Resnet V1 (EEMBC)
#define model
def resnet_v1_eembc():
    # Resnet parameters
    input_shape=[32,32,3] # default size for cifar10
    num_classes=10 # default class number for cifar10
    num_filters = 16 # this should be 64 for an official resnet model

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x) # uncomment this for official resnet model


    # First stack

            # Weight layers
            y = Conv2D(num_filters,
                       kernel_size=3,
                       strides=1,
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(x)
            y = BatchNormalization()(y)

            y = Activation('relu')(y)

            y = Conv2D(num_filters,
                       kernel_size=3,
                       strides=1,
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(y)
            y = BatchNormalization()(y)


    #=====================================================================
    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)
    #=====================================================================


    # Second stack

            # Weight layers
            num_filters = 32 # Filters need to be double for each stack
            y = Conv2D(num_filters,
                          kernel_size=3,
                          strides=2,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(x)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(num_filters,
                          kernel_size=3,
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(y)
            y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)


    #=====================================================================
    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)
    #=====================================================================


    # Third stack

            # Weight layers
            num_filters = 64
            y = Conv2D(num_filters,
                       kernel_size=3,
                       strides=2,
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(x)
            y = BatchNormalization()(y)

            y = Activation('relu')(y)

            y = Conv2D(num_filters,
                       kernel_size=3,
                       strides=1,
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(y)
            y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
               kernel_size=1,
               strides=2,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(x)


    #=====================================================================
    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)
    #=====================================================================


    # Fourth stack.
    # While the paper uses four stacks, for cifar10 that leads to a large increase in complexity for minor benefits
    # Uncomment to use it

#    # Weight layers
#    num_filters = 128
#    y = Conv2D(num_filters,
#                  kernel_size=3,
#                  strides=2,
#                  padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=l2(1e-4))(x)
#    y = BatchNormalization()(y)
#    y = Activation('relu')(y)
#    y = Conv2D(num_filters,
#                  kernel_size=3,
#                  strides=1,
#                  padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=l2(1e-4))(y)
#    y = BatchNormalization()(y)
#
#    # Adjust for change in dimension due to stride in identity
#    x = Conv2D(num_filters,
#                  kernel_size=1,
#                  strides=2,
#                  padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=l2(1e-4))(x)
#
#    # Overall residual, connect weight layer and identity paths
#    x = tf.keras.layers.add([x, y])
#    x = Activation('relu')(x)


    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
 */

//****************************************************************************
/* RECORD_CALLS:

   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:16,ichans:3,img:32x32,I,F,O)
   ReLUActivation(chans:16,img:32x32,I,O)

   ** First Stack **
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:16,ichans:16,img:32x32,I,F,O)
   ReLUActivation(chans:16,img:32x32,I,O)
   PartialConv2D(k:3,s:1,pad:[1,1,1,1],ochans:16,ichans:16,img:32x32,I,F,O)
   ReLUActivation(chans:16,img:32x32,I,O)

   ** Second Stack **
   Conv2D(k:3,s:2,pad:[0,1,0,1],ochans:32,ichans:16,img:32x32,I,F,O)
   ReLUActivation(chans:32,img:16x16,I,O)

?  Conv2D(k:1,s:2,pad:[0,0,0,0],ochans:32,ichans:16,img:32x32,I,F,O)

   PartialConv2D(k:3,s:1,pad:[1,1,1,1],ochans:32,ichans:32,img:16x16,I,F,O)
   ReLUActivation(chans:32,img:16x16,I,O)

   ** Third Stack **
   Conv2D(k:3,s:2,pad:[0,1,0,1],ochans:64,ichans:32,img:16x16,I,F,O)
   ReLUActivation(chans:64,img:8x8,I,O)

?  Conv2D(k:1,s:2,pad:[0,0,0,0],ochans:64,ichans:32,img:16x16,I,F,O)

   PartialConv2D(k:3,s:1,pad:[1,1,1,1],ochans:64,ichans:64,img:8x8,I,F,O)
   ReLUActivation(chans:64,img:8x8,I,O)

   ** Final Classification Layer ** (Keras Model: AveragePooling2D + Dense)
   MaxPool2D(k:8,s:1,pad:[0,0,0,0],chans:64,img:8x8,I,O)
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:16,ichans:64,img:1x1,I,F,O)

*/

namespace small
{

//****************************************************************************
template <typename BufferT>
class Resnet8TinyDAG : public DAGModel<BufferT>
{
public:
    Resnet8TinyDAG() = delete;

    // Assume one input layer with a single shape for now
    Resnet8TinyDAG(shape_type            const &input_shape,
                   std::vector<BufferT*> const &filters,
                   bool                         filters_are_packed = false)
        : DAGModel<BufferT>(input_shape)
    {
        size_t max_buffer_size =
            create_model_and_buffers(filters, filters_are_packed);
        this->initializeDAG(max_buffer_size);
    }

    virtual ~Resnet8TinyDAG()
    {
    }

private:
    size_t create_model_and_buffers(
        std::vector<BufferT*> const &filters,
        bool                         filters_are_packed)
    {
        // settings for first layers
        uint32_t stride = 1;
        uint32_t input_channels = 3;
        uint32_t output_channels = 16;
        uint32_t image_size = 32;
        uint32_t num_classes = 16;  /// @todo should be 10
        size_t   filter_num = 0;

        size_t layer_idx = 0UL;
        size_t resid_idx = 0UL;
        size_t max_buffer_size = 0UL;

        small::shape_type input_shape(
            {1UL, input_channels, image_size, image_size});

        small::Layer<BufferT> *prev =
            new small::Conv2DLayer<BufferT>(input_shape,
                                            3U, 3U,
                                            stride, small::PADDING_F,
                                            output_channels,
                                            *filters[filter_num],
                                            filters_are_packed,
                                            RELU);
        this->m_layers.push_back(prev);
        max_buffer_size =
            std::max<size_t>(max_buffer_size, prev->output_size());

        // =================================================
        resid_idx = layer_idx;
        small::shape_type resid_shape(prev->output_shape());

        this->m_graph.add_vertex(layer_idx);
        //if (layer_idx > 0) this->m_graph.add_edge(layer_idx-1, layer_idx);
        ++layer_idx;
        // =================================================

        // First Stack
        ++filter_num;
        prev = new small::Conv2DLayer<BufferT>(prev->output_shape(),
                                               3U, 3U,
                                               stride, small::PADDING_F,
                                               output_channels,
                                               *filters[filter_num],
                                               filters_are_packed,
                                               RELU);
        this->m_layers.push_back(prev);
        max_buffer_size =
            std::max<size_t>(max_buffer_size, prev->output_size());

        // =================================================
        this->m_graph.add_vertex(layer_idx);
        this->m_graph.add_edge(layer_idx-1, layer_idx);
        ++layer_idx;
        // =================================================

        ++filter_num;
        prev = new small::PartialConv2DLayer<BufferT>(prev->output_shape(),
                                                      3U, 3U,
                                                      stride, small::PADDING_F,
                                                      output_channels,
                                                      *filters[filter_num],
                                                      filters_are_packed,
                                                      RELU);
        this->m_layers.push_back(prev);
        max_buffer_size =
            std::max<size_t>(max_buffer_size, prev->output_size());

        // =================================================
        this->m_graph.add_vertex(layer_idx);
        this->m_graph.add_edge(resid_idx,   layer_idx);
        this->m_graph.add_edge(layer_idx-1, layer_idx);
        ++layer_idx;
        // =================================================

        // Second and Third Stacks
        for (auto ix = 0; ix < 2; ++ix)
        {
            small::Layer<BufferT> *block_prev = prev;
            size_t block_input_idx = layer_idx - 1;
            resid_shape = block_prev->output_shape();

            output_channels = 2*output_channels;

            //==================
            prev = new small::Conv2DLayer<BufferT>(block_prev->output_shape(),
                                                   1U, 1U, 2U, // 1x1, s:2
                                                   small::PADDING_V,
                                                   output_channels,
                                                   *filters[filter_num + 3],
                                                   filters_are_packed);
            this->m_layers.push_back(prev);
            max_buffer_size =
                std::max<size_t>(max_buffer_size, prev->output_size());

            max_buffer_size = std::max<size_t>(max_buffer_size, max_buffer_size);

            // =================================================
            size_t conv1x1_idx = layer_idx;
            this->m_graph.add_vertex(layer_idx);
            this->m_graph.add_edge(block_input_idx, layer_idx);
            ++layer_idx;
            // =================================================

            //==================
            prev = new small::Conv2DLayer<BufferT>(block_prev->output_shape(),
                                                   3U, 3U,
                                                   2U, small::PADDING_F,
                                                   output_channels,
                                                   *filters[filter_num + 1],
                                                   filters_are_packed,
                                                   RELU);
            this->m_layers.push_back(prev);
            max_buffer_size =
                std::max<size_t>(max_buffer_size, prev->output_size());

            // =================================================
            size_t conv3x3_idx = layer_idx;
            this->m_graph.add_vertex(layer_idx);
            this->m_graph.add_edge(block_input_idx, layer_idx);
            ++layer_idx;
            // =================================================

            // Performs computation:  ReLU(input0 + Conv2D(input1))
            prev = new small::PartialConv2DLayer<BufferT>(
                prev->output_shape(),
                3U, 3U, 1U, // 3x3, s:1
                small::PADDING_F,
                output_channels,
                *filters[filter_num + 2],
                filters_are_packed,
                RELU);

            this->m_layers.push_back(prev);
            max_buffer_size =
                std::max<size_t>(max_buffer_size, prev->output_size());

            // =================================================
            this->m_graph.add_vertex(layer_idx);
            this->m_graph.add_edge(conv1x1_idx, layer_idx); // edge order important
            this->m_graph.add_edge(conv3x3_idx, layer_idx);
            ++layer_idx;
            // =================================================

            filter_num += 3;
        }

        uint32_t kernel_size =
            this->m_layers.back()->output_shape()[small::HEIGHT]; //image_size;

        /// @todo should be AveragePooling2D
        prev = new small::MaxPool2DLayer<BufferT>(prev->output_shape(),
                                                  kernel_size, kernel_size,
                                                  1U, small::PADDING_V);
        this->m_layers.push_back(prev);
        max_buffer_size = std::max<size_t>(max_buffer_size, prev->output_size());

        // =================================================
        this->m_graph.add_vertex(layer_idx);
        this->m_graph.add_edge(layer_idx-1, layer_idx);
        ++layer_idx;
        // =================================================

        ++filter_num;
        prev = new small::Conv2DLayer<BufferT>(prev->output_shape(),
                                               1U, 1U, 1U, // 1x1, s:1
                                               small::PADDING_V,
                                               num_classes,
                                               *filters[filter_num],
                                               filters_are_packed);
        this->m_layers.push_back(prev);
        max_buffer_size = std::max<size_t>(max_buffer_size, prev->output_size());

        // =================================================
        this->m_graph.add_vertex(layer_idx);
        this->m_graph.add_edge(layer_idx-1, layer_idx);
        ++layer_idx;
        // =================================================

        std::cerr << "Filters consumed: " << ++filter_num << ","
                  << filters.size() << std::endl;
        std::cerr << "Layers created:   "
                  << this->m_layers.size() << std::endl;

        return max_buffer_size;
    }
};

}
