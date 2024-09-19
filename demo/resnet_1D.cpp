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

#include <small/Layer.hpp>
#include <small/PartialConv2DLayer.hpp>
#include <small/Conv2DLayer.hpp>
#include <small/Conv1DLayer.hpp> // Work in progress
#include <small/MaxPool2DLayer.hpp>
#include <small/ReLULayer.hpp>

/// @todo Which of these defines are needed?
#ifndef RUNS
#define RUNS 10
#endif

#ifndef TRIALS
#define TRIALS 100
#endif

#define TIME_LAYER 0

#define SUMMARY 1
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

// //****************************************************************************

#define REDUCTION_C(layer_num) layer_params[layer_num][0]
#define GROUP_C(layer_num) layer_params[layer_num][1]
#define GROUPS(layer_num) layer_params[layer_num][2]
#define STRIDE(layer_num) layer_params[layer_num][4]
#define REDUCTION_W(layer_num) layer_params[layer_num][9]

#define SET_PADDING(layer_num, l_pad, r_pad) layer_params[layer_num][7] = l_pad, layer_params[layer_num][8] = r_pad;
#define PADDING(layer_num) layer_params[layer_num][7], layer_params[layer_num][8]

#define I_WIDTH(layer_num) intermediate_dims[layer_num][0]
#define I_HEIGHT(layer_num) intermediate_dims[layer_num][1]

#define O_HEIGHT(layer_num) ((I_HEIGHT(layer_num - 1) - 1) / STRIDE(layer_num - 1) + 1)
#define O_WIDTH(layer_num) (((I_WIDTH(layer_num - 1) + layer_params[layer_num - 1][7] + layer_params[layer_num - 1][8]) - REDUCTION_W(layer_num - 1)) / STRIDE(layer_num - 1) + 1)

#define OUTPUT_DIMS(layer_num)                  \
    {                                           \
        O_HEIGHT(layer_num), O_WIDTH(layer_num) \
    }

#define INPUT_NUMEL(layer_num) \
    (O_HEIGHT(layer_num) * O_WIDTH(layer_num) * GROUP_C(layer_num - 1) * GROUPS(layer_num - 1))

// Layer by Layer Timers
double layer_timers[3][100];
double min_layer_timers[3][100];
double avg_layer_timers[3][100];

int layer_timer_count = 0;

//****************************************************************************
// The output of the block is stored in O
// Resnet block when stride == 2 (added 1x1 conv layer)
template <class BufferT>
inline void resnet_block(
    uint32_t in_dims[2], uint32_t input_channels, // Input dimensions
    uint32_t kernel_size_width,
    uint32_t stride,
    uint32_t output_channels,
    uint8_t l_pad_0,
    uint8_t r_pad_0,
    uint8_t l_pad_1,
    uint8_t r_pad_1,
    BufferT const &I,
    BufferT const &F_conv0,
    BufferT const &F_conv1,
    BufferT const &F_conv_1x1,
    BufferT &O_intermediate,
    BufferT &O)
{
    // printf("before: %d, %.2f %.2f %.2f %.2f\n", 0, I[0], I[1], I[2], I[3]);
#if TIME_LAYER
    small::Timer my_timer;
    my_timer.start();
#endif
    small::Conv1D(kernel_size_width, stride,
                  l_pad_0, r_pad_0,
                  output_channels, input_channels,
                  in_dims[0], in_dims[1],
                  I, F_conv0, O_intermediate);
#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif

    uint32_t o_h = in_dims[0];
    uint32_t o_w = small::output_dim(in_dims[1] + l_pad_0 + r_pad_0,
                                     stride, kernel_size_width);

#if TIME_LAYER
    my_timer.start();
#endif

    small::ReLUActivation1D(output_channels,
                            o_h, o_w,
                            O_intermediate, O_intermediate);

#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif

    // //split the O into 2 parts
    // BufferT O_1;
    // O_1.data() = O.data();
    // BufferT O_2;
    // O_2.data() = O.data() + (o_h*o_w*input_channels);

#if TIME_LAYER
    my_timer.start();
#endif

    small::AveragePool1D(1, stride,
                         0, 0,
                         input_channels,
                         in_dims[0], in_dims[1],
                         I,
                         O);

#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif


#if TIME_LAYER
    my_timer.start();
#endif
    small::PartialConv1D(kernel_size_width, 1,
                         l_pad_1, r_pad_1,
                         output_channels, output_channels,
                         o_h, o_w,
                         O_intermediate, F_conv1, O);
#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif

#if TIME_LAYER
    my_timer.start();
#endif

    small::ReLUActivation1D(output_channels, o_h, o_w, O, O);

#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif
}

//****************************************************************************
template <class BufferT>
inline void ewise_fused_resnet_block(
    uint32_t in_dims[2], uint32_t input_channels, // Input dimensions
    uint32_t kernel_size_width,
    uint32_t stride,
    uint32_t output_channels,
    uint8_t l_pad_0,
    uint8_t r_pad_0,
    uint8_t l_pad_1,
    uint8_t r_pad_1,
    BufferT const &I,
    BufferT const &F_conv0,
    BufferT const &F_conv1,
    // BufferT const &F_conv_1x1,
    BufferT &O_intermediate,
    BufferT &O)
{
    // printf("before: %d, %.2f %.2f %.2f %.2f\n", 0, I[0], I[1], I[2], I[3]);

#if TIME_LAYER
    small::Timer my_timer;
    my_timer.start();
#endif

    small::Conv1D_ReLU(kernel_size_width, stride,
                       l_pad_0, r_pad_0,
                       output_channels, input_channels,
                       in_dims[0], in_dims[1],
                       I, F_conv0, O_intermediate);

#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
    layer_timer_count++;
#endif

    //uint32_t o_h = small::output_dim(in_dims[0] + t_pad_0 + b_pad_0,
    //                                 stride, kernel_size_height);
    uint32_t o_h = in_dims[0];
    uint32_t o_w = small::output_dim(in_dims[1] + l_pad_0 + r_pad_0,
                                     stride, kernel_size_width);

    // small::ReLUActivation(output_channels,
    //                       o_h, o_w,
    //                       O_intermediate, O_intermediate);


    // #if TIME_LAYER
    // my_timer.start();
    // #endif

    // small::Conv2D(1, stride,
    //               0, 0,
    //               output_channels, input_channels,
    //               in_dims[0], in_dims[1],
    //               I, F_conv_1x1, O);
    // #if TIME_LAYER
    // my_timer.stop();
    // layer_timers[0][layer_timer_count] = my_timer.elapsed();
    // layer_timer_count++;
    // #endif

#if TIME_LAYER
    my_timer.start();
#endif
    small::PartialConv1D(kernel_size_width, 1,
                         l_pad_1, r_pad_1,
                         output_channels, output_channels,
                         o_h, o_w,
                         O_intermediate, F_conv1, O);
#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif

#if TIME_LAYER
    my_timer.start();
#endif
    small::ReLUActivation1D(output_channels, o_h, o_w, O, O);
#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif
}

//****************************************************************************
// The output of the block is stored in O
// block when stride = 1
template <class BufferT>
inline void resnet_block(
    uint32_t in_dims[2], uint32_t input_channels, // Input dimensions
    uint32_t kernel_size_width,
    uint32_t stride,
    uint32_t output_channels,
    uint8_t l_pad_0,
    uint8_t r_pad_0,
    uint8_t l_pad_1,
    uint8_t r_pad_1,
    BufferT const &I,
    BufferT const &F_conv0,
    BufferT const &F_conv1,
    BufferT &O_intermediate,
    BufferT &O)
{
#if TIME_LAYER
    small::Timer my_timer;
    my_timer.start();
#endif

    small::Conv1D(kernel_size_width, stride,
                  l_pad_0, r_pad_0,
                  output_channels, input_channels,
                  in_dims[0], in_dims[1],
                  I, F_conv0, O_intermediate);

#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif

    //uint32_t o_h = small::output_dim(in_dims[0] + t_pad_0 + b_pad_0,
    //                                 stride, kernel_size_height);
    uint32_t o_h = in_dims[0];
    uint32_t o_w = small::output_dim(in_dims[1] + l_pad_0 + r_pad_0,
                                     stride, kernel_size_width);
#if TIME_LAYER
    my_timer.start();
#endif

    small::ReLUActivation1D(output_channels,
                            o_h, o_w,
                            O_intermediate, O_intermediate);
#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif
    /// @todo Should this really be Partial Conv2D if no 1x1?

#if TIME_LAYER
    my_timer.start();
#endif
    small::PartialConv1D(kernel_size_width, 1,
                         l_pad_1, r_pad_1,
                         output_channels, output_channels,
                         o_h, o_w,
                         O_intermediate, F_conv1, O);
#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif

#if TIME_LAYER
    my_timer.start();
#endif
    small::ReLUActivation1D(output_channels, o_h, o_w, O, O);
#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif
}

//****************************************************************************
template <class BufferT>
BufferT &
model_inference(uint32_t layer_num_total,
                uint32_t n_blocks,
                uint32_t resnets_per_block,
                uint16_t layer_params[60][10],
                uint32_t intermediate_dims[60][2],
                std::vector<BufferT *> const &filter_buf_ptrs,
                BufferT const &input_dc,
                BufferT &inter_0_dc,
                BufferT &inter_1_dc,
                BufferT &inter_2_dc,
                uint32_t B = 1)
{
    auto layer_num = 0u;
#if TIME_LAYER
    small::Timer my_timer;
    layer_timer_count = 0;

    my_timer.start();
#endif

    small::Conv1D(REDUCTION_W(layer_num),
                  STRIDE(layer_num), PADDING(layer_num),
                  GROUP_C(layer_num), REDUCTION_C(layer_num),
                  I_HEIGHT(layer_num), I_WIDTH(layer_num),
                  input_dc,
                  *filter_buf_ptrs[layer_num],
                  inter_0_dc);

#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;

    layer_timer_count++;
#endif

#if TIME_LAYER
    my_timer.start();
#endif
    small::ReLUActivation1D(GROUP_C(0),
                          I_HEIGHT(layer_num), I_WIDTH(layer_num),
                          inter_0_dc,
                          inter_0_dc);
    // printf("Layer %d complete\n", layer_num);

#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif
    layer_num++;

// #if TIME_LAYER
//     my_timer.start();
// #endif
    resnet_block(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
                 REDUCTION_W(layer_num),
                 STRIDE(layer_num),
                 GROUP_C(layer_num),
                 PADDING(layer_num),
                 PADDING(layer_num + 1),
                 inter_0_dc,
                 *filter_buf_ptrs[layer_num],
                 *filter_buf_ptrs[layer_num + 1],
                 inter_1_dc,
                 inter_0_dc);
    // printf("Layer %d complete\n", layer_num);
// #if TIME_LAYER
//     my_timer.stop();
//     layer_timers[0][1] = my_timer.elapsed();
// #endif
    layer_num += 2;
    // 1 - n-1th blocks
    for (uint32_t n = 0; n < resnets_per_block - 1; n++)
    {
// #if TIME_LAYER
//         my_timer.start();
// #endif
        resnet_block(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
                     REDUCTION_W(layer_num),
                     STRIDE(layer_num),
                     GROUP_C(layer_num),
                     PADDING(layer_num),
                     PADDING(layer_num + 1),
                     inter_0_dc,
                     *filter_buf_ptrs[layer_num],
                     *filter_buf_ptrs[layer_num + 1],
                     inter_1_dc,
                     inter_0_dc);
// #if TIME_LAYER
//         my_timer.stop();
//         layer_timers[0][layer_num] = my_timer.elapsed();
// #endif
        layer_num += 2;
        // printf("Layer %d complete\n", layer_num);
    }

    auto resnet_blocks = n_blocks;
    auto num_filters = layer_num_total - 1;
    for (uint32_t ds_layer = 1u; ds_layer < resnet_blocks; ds_layer++)
    {
// #if TIME_LAYER
//         my_timer.start();
// #endif
        resnet_block(intermediate_dims[layer_num], REDUCTION_C(layer_num),
                     REDUCTION_W(layer_num),
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
// #if TIME_LAYER
//         my_timer.stop();
//         layer_timers[0][layer_num] = my_timer.elapsed();
// #endif
        // printf("Layer %d complete\n", layer_num);

        layer_num += 3;
        // Since channels were scaled, switch the pointers between inter_2 and inter_0

        inter_0_dc.swap(inter_2_dc);
        // printf("layer num %d 0  %d  2 %d\n", layer_num, inter_0_dc.data(), inter_2_dc.data());

        for (uint32_t n = 0; n < resnets_per_block - 1; n++)
        {
// #if TIME_LAYER
//             my_timer.start();
// #endif
            resnet_block(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
                         REDUCTION_W(layer_num),
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
            // printf("layer num %d 0  %d  2 %d\n", layer_num, inter_0_dc.data(), inter_2_dc.data());

// #if TIME_LAYER
//             my_timer.stop();
//             layer_timers[0][layer_num] = my_timer.elapsed();
// #endif

            // printf("Layer %d complete\n", layer_num);
        }
    }

#if TIME_LAYER
    my_timer.start();
#endif
    small::MaxPool1D(REDUCTION_W(layer_num),
                     STRIDE(layer_num),
                     PADDING(layer_num),
                     GROUPS(layer_num),
                     I_HEIGHT(layer_num), I_WIDTH(layer_num),
                     inter_0_dc,
                     inter_1_dc);
#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif

    layer_num++;
    // printf("Layer %d complete\n", layer_num);

#if TIME_LAYER
    my_timer.start();
#endif
    small::Conv1D(1, 1,
                  0, 0,
                  GROUP_C(layer_num_total - 1), REDUCTION_C(layer_num_total - 1),
                  1, 1,
                  inter_1_dc,
                  *filter_buf_ptrs[num_filters - 1],
                  inter_0_dc);
#if TIME_LAYER
    my_timer.stop();
    layer_timers[0][layer_timer_count] = my_timer.elapsed();
    layer_timer_count++;
#endif
    // printf("Layer %d complete\n", layer_num);

    return inter_0_dc;
}

//****************************************************************************
//****************************************************************************
template <class BufferT>
void inference(uint32_t const n_blocks = 2,
               uint32_t const resnets_per_block = 5, // # of repeated resnet blocks
               uint32_t const batch = 64,
               uint32_t const vector_length = 1024)
{
    uint32_t C_i = 2;  // 2 channels for I/Q data
    uint32_t N = batch;
    uint32_t M = vector_length;
    uint32_t num_classes = 16;

    int B = 1;
    char const *env_bt(std::getenv("OMP_BATCH_NUM_THREADS"));
    if (nullptr != env_bt)
    {
        B = atoi(std::getenv("OMP_BATCH_NUM_THREADS"));
    }

    auto threads_for_batch = B;
    auto thread_local_prob = N / threads_for_batch;
    auto thread_remainder = N % threads_for_batch;

    // split up batch into reused section within abstract layer(N), and loop section (batch_p_thread)
    auto batch_p_thread = thread_local_prob + (thread_remainder > 0);
    N = (batch_p_thread <= 8) ? batch_p_thread : 8;
    N = batch_p_thread;

    // Create input tensor
    srand(1729);
    uint32_t input_dimensions = C_i * N * M;
    BufferT input_dc(input_dimensions);
    init(input_dc, input_dimensions);

    // ================================================

    uint16_t layer_params[60][10] = {1};
    uint32_t intermediate_dims[60][2];

    uint8_t r_pad, l_pad;

    // Set up model parameters
    int layer_num = 0;
    uint32_t max_numel_inter_0 = 0, max_numel_inter_1 = 0;

    intermediate_dims[layer_num][0] = M;
    intermediate_dims[layer_num][1] = N;

    // conv
    REDUCTION_C(layer_num) = C_i;
    GROUP_C(layer_num) = 16;
    GROUPS(layer_num) = 1;
    REDUCTION_W(layer_num) = 3;
    STRIDE(layer_num) = 1;

    small::calc_padding(I_WIDTH(layer_num), REDUCTION_W(layer_num),
                        STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, l_pad, r_pad);

    layer_num++; // 1
    intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
    intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);
    auto inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_0 =
        (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

    // common set up for model architecture
    // auto resnet_blocks = 3;
    uint32_t layer_strides[n_blocks];
    layer_strides[0]= 1;
    for(uint32_t i = 1; i < n_blocks; i++)
    {
        layer_strides[i] = 2;
    }

    // dwise 1
    for (uint32_t ds_layer = 0; ds_layer < n_blocks; ds_layer++)
    {
        uint32_t channel_multiplier = (ds_layer > 0) ? 2 : 1;

        uint32_t in_channels = GROUP_C(layer_num - 1); // output channels from the previous block

        // First block in the resnet repeat
        REDUCTION_C(layer_num) = in_channels; // input channels
        GROUP_C(layer_num) = in_channels * channel_multiplier;
        GROUPS(layer_num) = 1; // output channels
        REDUCTION_W(layer_num) = 3;                  // kernel size
        STRIDE(layer_num) = layer_strides[ds_layer]; // stride

        small::calc_padding(I_WIDTH(layer_num), REDUCTION_W(layer_num),
                            STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, l_pad, r_pad);

        layer_num++; // 2,4,7
        intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
        intermediate_dims[layer_num][1] = N; // O_HEIGHT(layer_num);

        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_1 =
            (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;

        REDUCTION_C(layer_num) = GROUP_C(layer_num - 1);
        GROUP_C(layer_num) = GROUP_C(layer_num - 1);
        GROUPS(layer_num) = 1;
        REDUCTION_W(layer_num) = 3;
        STRIDE(layer_num) = 1;

        small::calc_padding(I_WIDTH(layer_num), REDUCTION_W(layer_num),
                            STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, l_pad, r_pad);

        layer_num++; // 3,5,8
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_0 =
            (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

        if (channel_multiplier != 1)
        {
            intermediate_dims[layer_num][0] = O_WIDTH(layer_num - 2);
            intermediate_dims[layer_num][1] = N;  // O_HEIGHT(layer_num - 2);
            REDUCTION_C(layer_num) = in_channels; // input channels
            GROUP_C(layer_num) = in_channels * channel_multiplier;
            GROUPS(layer_num) = 1;      // output channels
            REDUCTION_W(layer_num) = 1;
            STRIDE(layer_num) = 2; // stride
            SET_PADDING(layer_num, 0, 0);
            layer_num++; // 6,9
            inter_dim = INPUT_NUMEL(layer_num);
            max_numel_inter_0 =
                (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;
        }
        intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
        intermediate_dims[layer_num][1] = N; // O_HEIGHT(layer_num);

        // Remaining resnets_per_block - 1 blocks
        for (uint32_t b = 0; b < resnets_per_block - 1; b++)
        {
            REDUCTION_C(layer_num) = in_channels * channel_multiplier; // input channels
            GROUP_C(layer_num) = in_channels * channel_multiplier;
            GROUPS(layer_num) = 1; // output channels
            REDUCTION_W(layer_num) = 3; // kernel size
            STRIDE(layer_num) = 1;      // stride
            small::calc_padding(I_WIDTH(layer_num), REDUCTION_W(layer_num),
                                STRIDE(layer_num), l_pad, r_pad);
            SET_PADDING(layer_num, l_pad, r_pad);

            layer_num++; // 2,4,7
            intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
            intermediate_dims[layer_num][1] = N; // O_HEIGHT(layer_num);

            inter_dim = INPUT_NUMEL(layer_num);
            max_numel_inter_1 =
                (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;

            REDUCTION_C(layer_num) = GROUP_C(layer_num - 1);
            GROUP_C(layer_num) = GROUP_C(layer_num - 1);
            GROUPS(layer_num) = 1;
            REDUCTION_W(layer_num) = 3;
            STRIDE(layer_num) = 1;
            small::calc_padding(I_WIDTH(layer_num), REDUCTION_W(layer_num),
                                STRIDE(layer_num), l_pad, r_pad);
            SET_PADDING(layer_num, l_pad, r_pad);

            layer_num++; // 3,5,8
            inter_dim = INPUT_NUMEL(layer_num);
            max_numel_inter_0 =
                (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

            intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
            intermediate_dims[layer_num][1] = N; // O_HEIGHT(layer_num);
        }
    }

    // pooling dims
    REDUCTION_C(layer_num) = 1;
    GROUP_C(layer_num) = 1;
    GROUPS(layer_num) = GROUP_C(layer_num - 1);
    //REDUCTION_H(layer_num) = I_HEIGHT(layer_num);
    REDUCTION_W(layer_num) = I_WIDTH(layer_num);
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0);

    layer_num++;
    intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
    intermediate_dims[layer_num][1] = N; // O_HEIGHT(layer_num);
    REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
    GROUP_C(layer_num) = num_classes;
    GROUPS(layer_num) = 1;
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0);
    layer_num++;

    size_t layer_num_total = layer_num;
    size_t num_filters = layer_num_total - 1;

#if SUMMARY == 1
    printf("Layer num total: %ld", layer_num_total);
    for (size_t i = 0; i < layer_num_total; i++)
    {
        printf("layer %ld: ", i);
        printf(" input_dims: %d %d , \t", I_HEIGHT(i), I_WIDTH(i));
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
            REDUCTION_W(l) * REDUCTION_C(l) *
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
    BufferT inter_0_dc(max_numel_inter_0 + QUINT8_C_ob * 16 * 16 * 32); // HACK
    BufferT inter_1_dc(max_numel_inter_1 + QUINT8_C_ob * 16 * 16 * 32); // HACK
    BufferT inter_2_dc((max_numel_inter_0 / 2) + QUINT8_C_ob * 16 * 16 * 3);
#else
    BufferT inter_0_dc(max_numel_inter_0);
    BufferT inter_1_dc(max_numel_inter_1);
    BufferT inter_2_dc(max_numel_inter_0);
#endif

    //======================================================
    small::Timer my_timer;

    std::cerr << "Warm up run (ORIG)" << std::endl;
    my_timer.start();
    //auto &output_dc =
        model_inference(layer_num_total, n_blocks,resnets_per_block,
                        layer_params, intermediate_dims,
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
    int P = 1;
    char const *env_nt(std::getenv("OMP_INTRA_OP_NUM_THREADS"));
    if (nullptr != env_nt)
    {
        P = atoi(std::getenv("OMP_INTRA_OP_NUM_THREADS"));
    }

    BufferT output_dc_p[B];
    BufferT input_dc_p[B];

    BufferT inter_0_dc_p[B];
    BufferT inter_1_dc_p[B];
    BufferT inter_2_dc_p[B];

    for (int i = 0; i < B; i++)
    {
        input_dc_p[i] = BufferT(input_dimensions);
        init(input_dc_p[i], input_dimensions);

        inter_0_dc_p[i] = BufferT(max_numel_inter_0);
        inter_1_dc_p[i] = BufferT(max_numel_inter_1);
        inter_2_dc_p[i] = BufferT(max_numel_inter_0);

        // init(input_dc_p[i], input_dimensions);
    }

    for (int r = 0; r < RUNS; r++)
    {
        my_timer.start();

        for (int t = 0; t < TRIALS; t++)
        {
// auto &output_dc =
#pragma omp parallel num_threads(B)
            {
                uint32_t tid = omp_get_thread_num();  /// @todo bad cast?
                //auto t_start = tid * thread_local_prob + ((tid < thread_remainder) ? (tid % thread_remainder) : thread_remainder);
                //auto t_end = t_start + thread_local_prob + (1 * (tid < thread_remainder));
                // printf("tid: %d s%d e%d\n", tid, t_start, t_end);
                // auto batch_tid = t_end - t_start;
                // for(int i =  t_start; i < t_end; i++)
                // {
                // init(input_dc_p[tid], input_dimensions);

                output_dc_p[tid] = model_inference(layer_num_total, n_blocks, resnets_per_block,
                                                   layer_params, intermediate_dims,
                                                   filter_buf_ptrs,
                                                   input_dc_p[tid],
                                                   inter_0_dc_p[tid],
                                                   inter_1_dc_p[tid],
                                                   inter_2_dc_p[tid]);
                // }
            }
        }

        my_timer.stop();
        auto diff = my_timer.elapsed() / TRIALS;
        min_small = std::min<double>(min_small, diff);

        // small_timing.push_back(diff);
    }

#if TIME_LAYER
    int impl = 0;

    for (int r = 0; r < RUNS; r++)
    {
        for (int i = 0; i < TRIALS; i++)
        {
            // for (int b = 0; b < 64; b++)
            // {
                small::init(input_dc, input_dimensions);

                auto out_dc = model_inference(layer_num_total, n_blocks, resnets_per_block,
                                              layer_params, intermediate_dims,
                                              filter_buf_ptrs,
                                              input_dc,
                                              inter_0_dc,
                                              inter_1_dc,
                                              inter_2_dc);
            // }
            for (int timer = 0; timer < 100; timer++)
            {
                if (0 < i)
                {
                    avg_layer_timers[impl][timer] += layer_timers[impl][timer];
                }
                else
                {
                    avg_layer_timers[impl][timer] = layer_timers[impl][timer];
                }
            }
        }

        for (int timer = 0; timer < 100; timer++)
        {
            avg_layer_timers[impl][timer] /= TRIALS;
            if (0 < r)
            {
                min_layer_timers[impl][timer] =
                    std::min<double>(min_layer_timers[impl][timer],
                                     avg_layer_timers[impl][timer]);
            }
            else
            {
                min_layer_timers[impl][timer] = avg_layer_timers[impl][timer];
            }
        }
    }
#endif
    std::cout << resnets_per_block << " Minimum time: " << min_small << " ns.\n";

#if TIME_LAYER
    double sum_unfused = 0, sum_ewise = 0, sum_fused = 0;
    for (int layer = 0; layer < 100; layer++)
    {
        printf("%d , ", layer);
        for (int impl = 0; impl < 1; impl++)
        {
            printf("%f ,", min_layer_timers[impl][layer]);
        }
        printf("\n");
        sum_unfused += min_layer_timers[0][layer];
        sum_ewise += min_layer_timers[1][layer];
        sum_fused += min_layer_timers[2][layer];
    }

    printf("sum , %f, %f, %f ,%d, %d \n",
           sum_unfused, sum_ewise, sum_fused, resnets_per_block, 64);
#endif

    int num_th = 1;
#if PARALLEL == 1
    char const *env_num_th(std::getenv("OMP_NUM_THREADS"));
    if (nullptr != env_num_th)
    {
        num_th = atoi(std::getenv("OMP_NUM_THREADS"));
    }
#endif
    std::cout << "Num Threads: " << num_th << " " << min_small << std::endl;
    printf(" Threads for batch: %d, %d, Thread local prob: %d, Thread remainder: %d\n",
           B, P, thread_local_prob, thread_remainder);

    // print_stats(small_timing, "\nSMaLL:resnet_1D");

    //======================================================
    //     auto layers(create_model<BufferT>(filter_buf_ptrs));

    //     small::Tensor<BufferT> input_tensor({1, 3, 32, 32}, input_dc); // B, C_i, N, M
    // #if defined(QUANTIZED)
    //     small::Tensor<BufferT> inter_0_tensor(max_numel_inter_0 + QUINT8_C_ob*16*16*32);
    //     small::Tensor<BufferT> inter_1_tensor(max_numel_inter_1 + QUINT8_C_ob*16*16*32);
    //     small::Tensor<BufferT> inter_2_tensor((max_numel_inter_0 / 2) + QUINT8_C_ob*16*16*3);
    // #else
    //     small::Tensor<BufferT> inter_0_tensor(max_numel_inter_0);
    //     small::Tensor<BufferT> inter_1_tensor(max_numel_inter_1);
    //     small::Tensor<BufferT> inter_2_tensor(max_numel_inter_0);
    // #endif

    //     std::cerr << "Warm up run (LAYERS)" << std::endl;
    //     my_timer.start();
    //     auto &output_tensor =
    //         model_inference(layers, input_tensor,
    //                         inter_0_tensor, inter_1_tensor, inter_2_tensor);
    //     my_timer.stop();
    //     printf("\nElapsed time: %lf ns.\n", my_timer.elapsed());

    //     // Compare the results
    //     size_t num_outputs = layers.back()->output_size();
    //     std::cout << "\nCHECK RESULTS: Num output elements: " << num_outputs
    //               << std::endl;
    // for (size_t ix = 0; ix < num_classes; ++ix)
    // {
    //     std::cout << "Current, new " << ix
    //               << ": " << (float)output_dc[ix]
    //               << std::endl;
    // }

#if defined(NANO33BLE)
    small::detail::free_all();
#endif
}

//****************************************************************************
// For non-arduino platforms.  ... move to driver.cpp?
//****************************************************************************
#ifndef NANO33BLE
int main(int argc, char **argv)
{
    uint32_t resnets_per_block = 5;
    uint32_t batch = 64;
    uint32_t vector_length = 1024;
    uint32_t n_blocks = 3;
    printf("argc: %d\n", argc);
    switch(argc-1)
    {
        case 4:
            n_blocks = atoi(argv[4]);
        case 3:
            vector_length = atoi(argv[3]);
        case 2:
            resnets_per_block = atoi(argv[2]);
        case 1:
            batch = atoi(argv[1]);
    }


#if defined(QUANTIZED)
    inference<small::QUInt8Buffer>();
#else
    inference<small::FloatBuffer>(n_blocks, resnets_per_block, batch, vector_length);
#endif

    return 0;
}
#endif
