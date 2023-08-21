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

#define PARALLEL 1

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <vector>

#include <small.h>
#include <small/utils/Timer.hpp>
#include "utils.h"

/// @todo Which of these defines are needed?
#ifndef RUNS
#define RUNS 10
#endif

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
/* This is the runtime recording:

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

#include<small/Layer.hpp>
#include<small/PartialConv2DLayer.hpp>
#include<small/Conv2DLayer.hpp>
#include<small/MaxPool2DLayer.hpp>
#include<small/ReLULayer.hpp>

template <class BufferT>
std::vector<small::Layer<BufferT>*> create_model(
    std::vector<BufferT*> const &filters)
{
    std::vector<small::Layer<BufferT>*> layers;

    // settings for first layers
    uint32_t kernel_size = 3;
    uint32_t stride = 1;
    uint32_t input_channels = 3;
    uint32_t output_channels = 16;
    uint32_t image_size = 32;
    uint32_t num_classes = 16;  /// @todo should be 10
    size_t   filter_num = 0;

    small::shape_type input_shape(
        {1UL, input_channels, image_size, image_size});

    small::Layer<BufferT> *prev =
        new small::Conv2DLayer<BufferT>(input_shape,
                                        kernel_size, kernel_size,
                                        stride, small::PADDING_F,
                                        output_channels,
                                        *filters[filter_num], true);
    layers.push_back(prev);

    prev = new small::ReLULayer<BufferT>(prev->output_shape());
    layers.push_back(prev);

    // First Stack
    ++filter_num;
    prev = new small::Conv2DLayer<BufferT>(prev->output_shape(),
                                           kernel_size, kernel_size,
                                           stride, small::PADDING_F,
                                           output_channels,
                                           *filters[filter_num], true);
    layers.push_back(prev);

    prev =  new small::ReLULayer<BufferT>(prev->output_shape());
    layers.push_back(prev);

    ++filter_num;
    prev = new small::PartialConv2DLayer<BufferT>(prev->output_shape(),
                                                  kernel_size, kernel_size,
                                                  stride, small::PADDING_F,
                                                  output_channels,
                                                  *filters[filter_num], true);
    layers.push_back(prev);

    prev =  new small::ReLULayer<BufferT>(prev->output_shape());
    layers.push_back(prev);

    // Second and Third Stacks
    for (auto ix = 0; ix < 2; ++ix)
    {
        small::Layer<BufferT> *block_prev = prev;

        //==================
        ++filter_num;

        output_channels = 2*output_channels;
        stride = 2;
        prev = new small::Conv2DLayer<BufferT>(block_prev->output_shape(),
                                               kernel_size, kernel_size,
                                               stride, small::PADDING_F,
                                               output_channels,
                                               *filters[filter_num], true);
        layers.push_back(prev);

        prev = new small::ReLULayer<BufferT>(prev->output_shape());
        layers.push_back(prev);

        //==================
        ++filter_num;  ///@note out of order filter access here/next

        prev = new small::Conv2DLayer<BufferT>(block_prev->output_shape(),
                                               1U, 1U,
                                               stride, small::PADDING_V,
                                               output_channels,
                                               *filters[filter_num+1], true);
        layers.push_back(prev);

        stride = 1;
        prev = new small::PartialConv2DLayer<BufferT>(prev->output_shape(),
                                                      kernel_size, kernel_size,
                                                      stride, small::PADDING_F,
                                                      output_channels,
                                                      *filters[filter_num], true);
        layers.push_back(prev);
        ++filter_num;  /// @note 1x1 filter order swapped

        prev = new small::ReLULayer<BufferT>(prev->output_shape());
        layers.push_back(prev);
    }

    kernel_size = layers.back()->output_shape()[small::HEIGHT]; //image_size;
    stride = 1;
    /// @todo should be AveragePooling2D
    prev = new small::MaxPool2DLayer<BufferT>(prev->output_shape(),
                                              kernel_size, kernel_size,
                                              stride, small::PADDING_V);
    layers.push_back(prev);
    image_size = 1;
    kernel_size = 1;

    ++filter_num;
    prev = new small::Conv2DLayer<BufferT>(prev->output_shape(),
                                           kernel_size, kernel_size,
                                           stride, small::PADDING_V,
                                           num_classes,
                                           *filters[filter_num], true);
    layers.push_back(prev);

    std::cerr << "Filters consumed: " << ++filter_num << ","
              << filters.size() << std::endl;
    std::cerr << "Layers created:   " << layers.size() << std::endl;
    return layers;
}

//****************************************************************************
template <class BufferT>
small::Tensor<BufferT> &model_inference(
    std::vector<small::Layer<BufferT>*> const &layers,
    small::Tensor<BufferT>                             const &input_dc,
    small::Tensor<BufferT>                                   &inter_0_dc,
    small::Tensor<BufferT>                                   &inter_1_dc,
    small::Tensor<BufferT>                                   &inter_2_dc)
{
    size_t layer_num = 0;
    layers[layer_num++]->compute_output({&input_dc}, &inter_0_dc);   //Conv2D
    layers[layer_num++]->compute_output({&inter_0_dc}, &inter_0_dc); //ReLU

    layers[layer_num++]->compute_output({&inter_0_dc}, &inter_1_dc); // Conv2D
    layers[layer_num++]->compute_output({&inter_1_dc}, &inter_1_dc); // ReLU
    layers[layer_num++]->compute_output({&inter_1_dc}, &inter_0_dc); // buf0+=Conv2D(buf1)
    layers[layer_num++]->compute_output({&inter_0_dc}, &inter_0_dc); // ReLU

    for (auto ix = 0U; ix < 2; ++ix)
    {
        layers[layer_num++]->compute_output({&inter_0_dc}, &inter_1_dc); // Conv2D
        layers[layer_num++]->compute_output({&inter_1_dc}, &inter_1_dc); // ReLU

        layers[layer_num++]->compute_output({&inter_0_dc}, &inter_2_dc); // Conv2D

        layers[layer_num++]->compute_output({&inter_1_dc}, &inter_2_dc); // buf2+=Conv2D(buf1)
        layers[layer_num++]->compute_output({&inter_2_dc}, &inter_2_dc); // ReLU

        inter_0_dc.swap(inter_2_dc);
    }

    layers[layer_num++]->compute_output({&inter_0_dc}, &inter_1_dc);
    layers[layer_num++]->compute_output({&inter_1_dc}, &inter_0_dc);

    return inter_0_dc;
}

//****************************************************************************

#define REDUCTION_C(layer_num) layer_params[layer_num][0]
#define GROUP_C(layer_num) layer_params[layer_num][1]
#define GROUPS(layer_num) layer_params[layer_num][2]
#define REDUCTION_HW(layer_num) layer_params[layer_num][3]
#define STRIDE(layer_num) layer_params[layer_num][4]
#define TYPE(layer_num) layer_params[layer_num][9]

#define SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad) layer_params[layer_num][5] = t_pad, layer_params[layer_num][6] = b_pad, layer_params[layer_num][7] = l_pad, layer_params[layer_num][8] = r_pad;
#define PADDING(layer_num) layer_params[layer_num][5], layer_params[layer_num][6], layer_params[layer_num][7], layer_params[layer_num][8]

#define PADDING_TORCH(layer_num) layer_params[layer_num][7], layer_params[layer_num][8], layer_params[layer_num][5], layer_params[layer_num][6]

#define I_WIDTH(layer_num) intermediate_dims[layer_num][0]
#define I_HEIGHT(layer_num) intermediate_dims[layer_num][1]

#define O_HEIGHT(layer_num) (((I_HEIGHT(layer_num - 1) + layer_params[layer_num - 1][5] + layer_params[layer_num - 1][6]) - REDUCTION_HW(layer_num - 1)) / STRIDE(layer_num - 1) + 1)
#define O_WIDTH(layer_num) (((I_WIDTH(layer_num - 1) + layer_params[layer_num - 1][7] + layer_params[layer_num - 1][8]) - REDUCTION_HW(layer_num - 1)) / STRIDE(layer_num - 1) + 1)

#define OUTPUT_DIMS(layer_num)                  \
    {                                           \
        O_HEIGHT(layer_num), O_WIDTH(layer_num) \
    }

#define INPUT_NUMEL(layer_num) \
    (O_HEIGHT(layer_num) * O_WIDTH(layer_num) * GROUP_C(layer_num - 1) * GROUPS(layer_num - 1))

//****************************************************************************
// The output of the block is stored in O
//
template <class BufferT>
inline void resnet_block(
    uint32_t in_dims[2], uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t output_channels,
    uint8_t t_pad_0,
    uint8_t b_pad_0,
    uint8_t l_pad_0,
    uint8_t r_pad_0,
    uint8_t t_pad_1,
    uint8_t b_pad_1,
    uint8_t l_pad_1,
    uint8_t r_pad_1,
    BufferT const &I,
    BufferT const &F_conv0,
    BufferT const &F_conv1,
    BufferT const &F_conv_1x1,
    BufferT       &O_intermediate,
    BufferT       &O)
{
    // printf("before: %d, %.2f %.2f %.2f %.2f\n", 0, I[0], I[1], I[2], I[3]);

    small::Conv2D(kernel_size, kernel_size, stride,
                  t_pad_0, b_pad_0, l_pad_0, r_pad_0,
                  output_channels, input_channels,
                  in_dims[0], in_dims[1],
                  I, F_conv0, O_intermediate);

    uint32_t o_h = small::output_dim(in_dims[0] + t_pad_0 + b_pad_0,
                                     stride, kernel_size);
    uint32_t o_w = small::output_dim(in_dims[1] + l_pad_0 + r_pad_0,
                                     stride, kernel_size);

    small::ReLUActivation(output_channels,
                          o_h, o_w,
                          O_intermediate, O_intermediate);

    small::Conv2D(1, 1, stride,
                  0, 0, 0, 0,
                  output_channels, input_channels,
                  in_dims[0], in_dims[1],
                  I, F_conv_1x1, O);

    small::PartialConv2D(kernel_size, kernel_size, 1,
                         t_pad_1, b_pad_1, l_pad_1, r_pad_1,
                         output_channels, output_channels,
                         o_h, o_w,
                         O_intermediate, F_conv1, O);
    small::ReLUActivation(output_channels, o_h, o_w, O, O);
}

//****************************************************************************
// The output of the block is stored in O
//
template <class BufferT>
inline void resnet_block(
    uint32_t in_dims[2], uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t output_channels,
    uint8_t t_pad_0,
    uint8_t b_pad_0,
    uint8_t l_pad_0,
    uint8_t r_pad_0,
    uint8_t t_pad_1,
    uint8_t b_pad_1,
    uint8_t l_pad_1,
    uint8_t r_pad_1,
    BufferT const &I,
    BufferT const &F_conv0,
    BufferT const &F_conv1,
    BufferT       &O_intermediate,
    BufferT       &O)
{
    small::Conv2D(kernel_size, kernel_size, stride,
                  t_pad_0, b_pad_0, l_pad_0, r_pad_0,
                  output_channels, input_channels,
                  in_dims[0], in_dims[1],
                  I, F_conv0, O_intermediate);

    uint32_t o_h = small::output_dim(in_dims[0] + t_pad_0 + b_pad_0,
                                     stride, kernel_size);
    uint32_t o_w = small::output_dim(in_dims[1] + l_pad_0 + r_pad_0,
                                     stride, kernel_size);

    small::ReLUActivation(output_channels,
                          o_h, o_w,
                          O_intermediate, O_intermediate);

    /// @todo Should this really be Partial Conv2D if no 1x1?
    small::PartialConv2D(kernel_size, kernel_size, 1,
                         t_pad_1, b_pad_1, l_pad_1, r_pad_1,
                         output_channels, output_channels,
                         o_h, o_w,
                         O_intermediate, F_conv1, O);
    small::ReLUActivation(output_channels, o_h, o_w, O, O);
}

//****************************************************************************
template <class BufferT>
BufferT &
model_inference(uint32_t layer_num_total,
                uint16_t layer_params[30][10],
                uint32_t intermediate_dims[30][2],
                std::vector<BufferT *> const &filter_buf_ptrs,
                BufferT const &input_dc,
                BufferT       &inter_0_dc,
                BufferT       &inter_1_dc,
                BufferT       &inter_2_dc)
{
    auto layer_num = 0;
    small::Conv2D(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                  STRIDE(layer_num), PADDING(layer_num),
                  GROUP_C(layer_num), REDUCTION_C(layer_num),
                  I_HEIGHT(layer_num), I_WIDTH(layer_num),
                  input_dc,
                  *filter_buf_ptrs[layer_num],
                  inter_0_dc);

    layer_num++;
    small::ReLUActivation(GROUP_C(0),
                          I_HEIGHT(layer_num), I_WIDTH(layer_num),
                          inter_0_dc,
                          inter_0_dc);

    resnet_block(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
                 REDUCTION_HW(layer_num),
                 STRIDE(layer_num),
                 GROUP_C(layer_num),
                 PADDING(layer_num),
                 PADDING(layer_num + 1),
                 inter_0_dc,
                 *filter_buf_ptrs[layer_num],
                 *filter_buf_ptrs[layer_num + 1],
                 inter_1_dc,
                 inter_0_dc);

    layer_num += 2;
    auto resnet_blocks = 3;
    auto num_filters = layer_num_total - 1;
    for (int ds_layer = 1; ds_layer < resnet_blocks; ds_layer++)
    {
        resnet_block(intermediate_dims[layer_num], REDUCTION_C(layer_num),
                     REDUCTION_HW(layer_num),
                     STRIDE(layer_num),
                     GROUP_C(layer_num),
                     PADDING(layer_num),
                     PADDING(layer_num + 1),
                     inter_0_dc,
                     *filter_buf_ptrs[layer_num],
                     *filter_buf_ptrs[layer_num + 1],
                     *filter_buf_ptrs[layer_num + 2],
                     inter_1_dc,
                     inter_2_dc);

        layer_num += 3;
        // Since channels were scaled, switch the pointers between inter_2 and inter_0
        inter_0_dc.swap(inter_2_dc);
    }

    small::MaxPool2D(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                     STRIDE(layer_num),
                     PADDING(layer_num),
                     GROUPS(layer_num),
                     I_HEIGHT(layer_num), I_WIDTH(layer_num),
                     inter_0_dc,
                     inter_1_dc);
    small::Conv2D(1, 1, 1,
                  0, 0, 0, 0,
                  GROUP_C(layer_num_total - 1), REDUCTION_C(layer_num_total - 1),
                  1, 1,
                  inter_1_dc,
                  *filter_buf_ptrs[num_filters - 1],
                  inter_0_dc);

    return inter_0_dc;
}

//****************************************************************************
//****************************************************************************
template <class BufferT>
void inference()
{
    uint32_t C_i = 3;
    uint32_t N = 32;
    uint32_t M = 32;
    uint32_t num_classes = 16;

    // Create input tensor
    uint32_t input_dimensions = C_i * N * M;
    BufferT input_dc(input_dimensions);
    init(input_dc, input_dimensions);

    // ================================================

    uint16_t layer_params[30][10] = {1};
    uint32_t intermediate_dims[30][2];

    uint8_t t_pad, b_pad, r_pad, l_pad;

    // Set up model parameters
    int layer_num = 0;
    uint32_t max_numel_inter_0 = 0, max_numel_inter_1 = 0;

    intermediate_dims[layer_num][0] = M;
    intermediate_dims[layer_num][1] = N;

    // conv
    REDUCTION_C(layer_num) = C_i;
    GROUP_C(layer_num) = 16;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 3;
    STRIDE(layer_num) = 1;
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                        STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                        STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

    layer_num++; // 1
    intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
    intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);
    auto inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_0 =
        (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

    // common set up for model architecture
    auto resnet_blocks = 3;
    int layer_strides[] = {1, 2, 2};
    // dwise 1
    for (int ds_layer = 0; ds_layer < resnet_blocks; ds_layer++)
    {
        int channel_multiplier = (ds_layer > 0) ? 2 : 1;

        uint32_t in_channels = GROUP_C(layer_num - 1); // output channels from the previous block

        REDUCTION_C(layer_num) = in_channels; // input channels
        GROUP_C(layer_num) = in_channels * channel_multiplier;
        GROUPS(layer_num) = 1;                       // output channels
        REDUCTION_HW(layer_num) = 3;                 // kernel size
        STRIDE(layer_num) = layer_strides[ds_layer]; // stride
        small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                            STRIDE(layer_num), t_pad, b_pad);
        small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                            STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

        layer_num++; // 2,4,7
        intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
        intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);

        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_1 =
            (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;

        REDUCTION_C(layer_num) = GROUP_C(layer_num - 1);
        GROUP_C(layer_num) = GROUP_C(layer_num - 1);
        GROUPS(layer_num) = 1;
        REDUCTION_HW(layer_num) = 3;
        STRIDE(layer_num) = 1;
        small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                            STRIDE(layer_num), t_pad, b_pad);
        small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                            STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

        layer_num++; // 3,5,8
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_0 =
            (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

        if (channel_multiplier != 1)
        {
            intermediate_dims[layer_num][0] = O_WIDTH(layer_num - 2);
            intermediate_dims[layer_num][1] = O_HEIGHT(layer_num - 2);
            REDUCTION_C(layer_num) = in_channels; // input channels
            GROUP_C(layer_num) = in_channels * channel_multiplier;
            GROUPS(layer_num) = 1;       // output channels
            REDUCTION_HW(layer_num) = 1; // kernel size
            STRIDE(layer_num) = 2;       // stride
            SET_PADDING(layer_num, 0, 0, 0, 0);
            layer_num++; // 6,9
            inter_dim = INPUT_NUMEL(layer_num);
            max_numel_inter_0 =
                (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;
        }
        intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
        intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);
    }

    // pooling dims
    REDUCTION_C(layer_num) = 1;
    GROUP_C(layer_num) = 1;
    GROUPS(layer_num) = GROUP_C(layer_num - 1);
    REDUCTION_HW(layer_num) = I_HEIGHT(layer_num);
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0);

    layer_num++;
    intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
    intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);
    REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
    GROUP_C(layer_num) = num_classes;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 1;
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0);
    layer_num++;

    size_t layer_num_total = layer_num;
    size_t num_filters = layer_num_total - 1;

#if SUMMARY == 1
    printf("Layer num total: %d", layer_num_total);
    for (auto i = 0; i < layer_num_total; i++)
    {
        printf("layer %d: ", i);
        printf(" input_dims: %d %d ", I_HEIGHT(i), I_WIDTH(i));
        for (auto j = 0; j < 10; j++)
        {
            printf("%d, ", layer_params[i][j]);
        }
        printf(", intermediate dims: %d %d\n",
               intermediate_dims[i][0], intermediate_dims[i][1]);
    }
#endif

    /// @todo use a vector of smart pointers if possible
    std::vector<BufferT *> filter_buf_ptrs;

    for (size_t l = 0; l < num_filters - 1; l++)
    {
        uint32_t filter_dimensions =
            REDUCTION_HW(l) * REDUCTION_HW(l) * REDUCTION_C(l) *
            GROUP_C(l) * GROUPS(l);

        BufferT *filter_buf_ptr =
            small::alloc_buffer<BufferT>(filter_dimensions);
        init(*filter_buf_ptr, filter_dimensions);
        filter_buf_ptrs.push_back(filter_buf_ptr);
    }

    uint32_t filter_dimensions =
        GROUP_C(layer_num_total - 1) * REDUCTION_C(layer_num_total - 1);
    BufferT *filter_fc_dc_ptr =
        small::alloc_buffer<BufferT>(filter_dimensions);
    init(*filter_fc_dc_ptr, filter_dimensions);
    filter_buf_ptrs.push_back(filter_fc_dc_ptr);
    /// @todo assert(filter_buf_ptrs.size() == num_filters)

    // allocate space for intermediate outputs
    // (use the max sizes calculated previously)
#if defined(QUANTIZED)
    BufferT inter_0_dc(max_numel_inter_0 + QUINT8_C_ob*16*16*32); // HACK
    BufferT inter_1_dc(max_numel_inter_1 + QUINT8_C_ob*16*16*32); // HACK
    BufferT inter_2_dc((max_numel_inter_0 / 2) + QUINT8_C_ob*16*16*3);
#else
    BufferT inter_0_dc(max_numel_inter_0);
    BufferT inter_1_dc(max_numel_inter_1);
    BufferT inter_2_dc(max_numel_inter_0);
#endif

    //======================================================
    small::Timer my_timer;

    std::cerr << "Warm up run (ORIG)" << std::endl;
    my_timer.start();
    auto &output_dc =
        model_inference(layer_num_total, layer_params, intermediate_dims,
                        filter_buf_ptrs,
                        input_dc,
                        inter_0_dc,
                        inter_1_dc,
                        inter_2_dc);
    my_timer.stop();
    printf("\nElapsed time: %lf ns.\n", my_timer.elapsed());

    //======================================================

    double min_small = std::numeric_limits<double>::max();
    std::vector<double> small_timing;

    for (int r = 0; r < RUNS; r++)
    {
        my_timer.start();

        //auto &output_dc =
            model_inference(layer_num_total, layer_params, intermediate_dims,
                            filter_buf_ptrs,
                            input_dc,
                            inter_0_dc,
                            inter_1_dc,
                            inter_2_dc);

        my_timer.stop();
        auto diff = my_timer.elapsed();
        min_small = std::min<double>(min_small, diff);
        small_timing.push_back(diff);
    }

    std::cout << "Minimum time: " << min_small << " ns.\n";

    int num_th = 1;
#if PARALLEL == 1
    char const *env_nt(std::getenv("OMP_NUM_THREADS"));
    if (nullptr != env_nt)
    {
        num_th = atoi(std::getenv("OMP_NUM_THREADS"));
    }
#endif
    std::cout << "Num Threads: " << num_th << std::endl;
    print_stats(small_timing, "\nSMaLL:resnet");

    //======================================================
    auto layers(create_model<BufferT>(filter_buf_ptrs));

    small::Tensor<BufferT> input_tensor({1, 3, 32, 32}, input_dc); // B, C_i, N, M
#if defined(QUANTIZED)
    small::Tensor<BufferT> inter_0_tensor(max_numel_inter_0 + QUINT8_C_ob*16*16*32);
    small::Tensor<BufferT> inter_1_tensor(max_numel_inter_1 + QUINT8_C_ob*16*16*32);
    small::Tensor<BufferT> inter_2_tensor((max_numel_inter_0 / 2) + QUINT8_C_ob*16*16*3);
#else
    small::Tensor<BufferT> inter_0_tensor(max_numel_inter_0);
    small::Tensor<BufferT> inter_1_tensor(max_numel_inter_1);
    small::Tensor<BufferT> inter_2_tensor(max_numel_inter_0);
#endif

    std::cerr << "Warm up run (LAYERS)" << std::endl;
    my_timer.start();
    auto &output_tensor =
        model_inference(layers, input_tensor,
                        inter_0_tensor, inter_1_tensor, inter_2_tensor);
    my_timer.stop();
    printf("\nElapsed time: %lf ns.\n", my_timer.elapsed());

    // Compare the results
    size_t num_outputs = layers.back()->output_size();
    std::cout << "\nCHECK RESULTS: Num output elements: " << num_outputs
              << std::endl;
    for (size_t ix = 0; ix < num_outputs; ++ix)
    {
        std::cout << "Current, new " << ix
                  << ": " << (float)output_dc[ix]
                  << ", " << (float)output_tensor.buffer()[ix]
                  << ((output_dc[ix] == output_tensor.buffer()[ix]) ? " (pass)" : " (fail)")
                  << std::endl;
    }

    // clean up model (move to model class destructor when built
    for (auto layer : layers) delete layer;
    //======================================================

    printf("deallocing %ld filters\n", filter_buf_ptrs.size());
    for (size_t l = 0; l < filter_buf_ptrs.size(); l++)
    {
        small::free_buffer(filter_buf_ptrs[l]);
    }

#if defined(NANO33BLE)
    small::detail::free_all();
#endif
}

//****************************************************************************
// For non-arduino platforms.  ... move to driver.cpp?
//****************************************************************************
#ifndef NANO33BLE
int main()
{
#if defined(QUANTIZED)
    inference<small::QUInt8Buffer>();
#else
    inference<small::FloatBuffer>();
#endif

    return 0;
}
#endif
