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
#define RUNS 100
#endif

//****************************************************************************

/* From https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/keras_model.py#L26

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
def get_model(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder
    (128*128*128*128*8*128*128*128*128)
    """
    inputLayer = Input(shape=(inputDim,)) // batches (unspecified) of 'inputDim' sized vectors

    h = Dense(128)(inputLayer)            // input=inputDim, output=128 so weights = 128xinputDim?
    h = BatchNormalization()(h)           // pointwise normalization using set mean, std values
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(8)(h)               <---------- 8
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(inputDim)(h)

    return Model(inputs=inputLayer, outputs=h)
 */

//****************************************************************************
/*
Conv2D(k:1, s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
ReLUActivation(chans:128,img:1x1,I,O)
Conv2D(k:1, s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
ReLUActivation(chans:128,img:1x1,I,O)
Conv2D(k:1, s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
ReLUActivation(chans:128,img:1x1,I,O)
Conv2D(k:1, s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
ReLUActivation(chans:128,img:1x1,I,O)
Conv2D(k:1, s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
ReLUActivation(chans:128,img:1x1,I,O)
Conv2D(k:1, s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
ReLUActivation(chans:128,img:1x1,I,O)
Conv2D(k:1, s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
ReLUActivation(chans:128,img:1x1,I,O)
Conv2D(k:1, s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
ReLUActivation(chans:128,img:1x1,I,O)
Conv2D(k:1, s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
ReLUActivation(chans:128,img:1x1,I,O)
*/

#include<small/Layer.hpp>
#include<small/Conv2DLayer.hpp>
#include<small/ReLULayer.hpp>

template <class BufferT>
std::vector<small::Layer<BufferT>*> create_model(
    std::vector<BufferT*> const &filters)
{
    std::vector<small::Layer<BufferT>*> layers;

    uint32_t kernel_size = 1;
    uint32_t stride = 1;
    uint32_t input_height = 1, input_width = 1;
    uint32_t input_channels = 128, output_channels = 128;

    small::shape_type input_shape(
        {1UL, input_channels, input_height, input_width});

    for (auto ix = 0U; ix < filters.size(); ++ix)
    {
        output_channels = 128;
        if (ix + 1 == filters.size()) output_channels = 16;  /// @todo ix == 4

        small::Layer<BufferT> *prev =
            new small::Conv2DLayer<BufferT>(input_shape,
                                            kernel_size, kernel_size,
                                            stride, small::PADDING_V,
                                            output_channels,
                                            *filters[ix], true);
        layers.push_back(prev);

        input_shape = prev->output_shape();

        prev = new small::ReLULayer<BufferT>(input_shape);
        layers.push_back(prev);
    }

    return layers;
}

//****************************************************************************
template <class BufferT>
small::Tensor<BufferT> &model_inference(
    std::vector<small::Layer<BufferT>*> const &layers,
    small::Tensor<BufferT>                             const &input_dc,
    small::Tensor<BufferT>                                   &inter_0_dc,
    small::Tensor<BufferT>                                   &inter_1_dc)
{
    size_t layer_num = 0;
    layers[layer_num++]->compute_output({&input_dc}, &inter_0_dc);   // Conv2D
    layers[layer_num++]->compute_output({&inter_0_dc}, &inter_0_dc); // ReLU

    while (layer_num < layers.size() - 1)
    {
        layers[layer_num]->compute_output({&inter_0_dc}, &inter_1_dc);
        layers[layer_num + 1]->compute_output({&inter_1_dc}, &inter_1_dc);
        layer_num += 2;

        inter_0_dc.swap(inter_1_dc);
    }

    return inter_0_dc;
}

//****************************************************************************

#define REDUCTION_C(layer_num) layer_params[layer_num][0]
#define GROUP_C(layer_num) layer_params[layer_num][1]
#define GROUPS(layer_num) layer_params[layer_num][2]
#define REDUCTION_HW(layer_num) layer_params[layer_num][3]
#define STRIDE(layer_num) layer_params[layer_num][4]

// In the case of non-square kernels
#define REDUCTION_H(layer_num) layer_params[layer_num][3]
#define REDUCTION_W(layer_num) layer_params[layer_num][9]

#define SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad) layer_params[layer_num][5] = t_pad, layer_params[layer_num][6] = b_pad, layer_params[layer_num][7] = l_pad, layer_params[layer_num][8] = r_pad;
#define PADDING(layer_num) layer_params[layer_num][5], layer_params[layer_num][6], layer_params[layer_num][7], layer_params[layer_num][8]

#define BOTTOM_PADDING(layer_num) layer_params[layer_num][6]

#define RIGHT_PADDING(layer_num) layer_params[layer_num][8]

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
template <class BufferT>
BufferT &model_inference(
    uint32_t layer_num_total,
    uint16_t layer_params[30][10],
    std::vector<BufferT *> const &filter_buf_ptrs,
    BufferT  const &input_dc,
    BufferT        &inter_0_dc,
    BufferT        &inter_1_dc)
{
    auto layer_num = 0;
    small::Conv2D(1, 1, 1,
                  0, 0, 0, 0,
                  GROUP_C(layer_num), REDUCTION_C(layer_num),
                  1, 1,
                  input_dc,
                  *filter_buf_ptrs[layer_num],
                  inter_0_dc);

    small::ReLUActivation(GROUP_C(layer_num),
                          1, 1,
                          inter_0_dc, inter_0_dc);

    for (uint32_t cur_layer = 1; cur_layer < layer_num_total; cur_layer++)
    {
        layer_num++;
        small::Conv2D(1, 1, 1,
                      0, 0, 0, 0,
                      GROUP_C(layer_num), REDUCTION_C(layer_num),
                      1, 1,
                      inter_0_dc,
                      *filter_buf_ptrs[layer_num],
                      inter_1_dc);

        small::ReLUActivation(GROUP_C(layer_num),
                              1, 1,
                              inter_1_dc,
                              inter_1_dc);

        inter_0_dc.swap(inter_1_dc);
    }
    return inter_0_dc;
}

//****************************************************************************
//****************************************************************************
template <class BufferT>
void inference()
{
    uint32_t C_i = 128;
    uint32_t N = 1;
    uint32_t M = 1;
    uint32_t num_classes = 16;

    // Create input tensor
    uint32_t input_dimensions = C_i * N * M;
    BufferT  input_dc(input_dimensions);
    small::init(input_dc, input_dimensions);

    // ================================================

    uint16_t layer_params[30][10] = {1};
    uint32_t intermediate_dims[30][2];

    // Set up model parameters
    auto layer_num_total = 9U;
    uint32_t max_numel_inter_0 = 128, max_numel_inter_1 = 128;

    uint32_t layer_num = 0;
    intermediate_dims[layer_num][0] = 1;
    intermediate_dims[layer_num][1] = 1;

    // conv
    REDUCTION_C(layer_num) = C_i; // input channels
    GROUP_C(layer_num) = 128;     // output channels
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 1;  // kernel size
    STRIDE(layer_num) = 1;        // stride
    SET_PADDING(layer_num, 0, 0, 0, 0);

    layer_num++;
    intermediate_dims[layer_num][0] = 1;
    intermediate_dims[layer_num][1] = 1;

    // common set up for model architecture
    for (uint32_t cur_layer = 1; cur_layer+1 < layer_num_total; cur_layer++)
    {
        REDUCTION_C(layer_num) = GROUP_C(layer_num - 1); // input channels
        GROUP_C(layer_num) = GROUP_C(layer_num - 1);     // output channels
        GROUPS(layer_num) = 1;
        REDUCTION_HW(layer_num) = 1;                 // kernel size
        STRIDE(layer_num) = 1; // stride
        SET_PADDING(layer_num, 0, 0, 0, 0);

        layer_num++; // 2
        intermediate_dims[layer_num][0] = 1;
        intermediate_dims[layer_num][1] = 1;
    }

    REDUCTION_C(layer_num) = GROUP_C(layer_num-1);
    GROUP_C(layer_num) = num_classes;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) =   1;
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0);

    layer_num++;
    intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
    intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);

#if SUMMARY == 1
    printf("Layer num total: %d\n", layer_num_total);
    printf("Layer: Red_C(in_chan), Grp_C(out_chan), Grps, Red_HW(k), Stride(s)\n");
    for (uint32_t i = 0; i < layer_num_total; i++)
    {
        printf("%d: ", i);
        for (auto j = 0; j < 10; j++)
        {
            printf("%d, ", layer_params[i][j]);
        }
        printf("inter_dims %d,%d\n", intermediate_dims[i][0], intermediate_dims[i][1]);
    }
#endif

    std::vector<BufferT *> filter_buf_ptrs;

    // Direct Convolution Setup
    for (uint32_t l = 0; l < layer_num_total; l++)
    {
        uint32_t filter_dimensions =
            REDUCTION_HW(l) * REDUCTION_HW(l) * REDUCTION_C(l) *
            GROUP_C(l) * GROUPS(l);
        BufferT *filter_buf_ptr =
            small::alloc_buffer<BufferT>(filter_dimensions);
        init(*filter_buf_ptr, filter_dimensions);
        filter_buf_ptrs.push_back(filter_buf_ptr);
    }

#if defined QUANTIZED
    // std::cerr << "Intermediate buffer sizes: "
    //           << max_numel_inter_0 << ", " << max_numel_inter_1
    //           << std::endl;
    BufferT inter_0_dc(max_numel_inter_0*4);
    BufferT inter_1_dc(max_numel_inter_1*4);
#else
    BufferT inter_0_dc(max_numel_inter_0);
    BufferT inter_1_dc(max_numel_inter_1);
#endif

    //======================================================
    small::Timer my_timer;

    std::cerr << "Warm up run (ORIG)" << std::endl;
    my_timer.start();
    auto &output_dc =
        model_inference(layer_num_total, layer_params,
                        filter_buf_ptrs,
                        input_dc,
                        inter_0_dc,
                        inter_1_dc);
    my_timer.stop();
    printf("\nElapsed time: %lf ns.\n", my_timer.elapsed());

    //======================================================

    double min_small = std::numeric_limits<double>::max();
    std::vector<double> small_timing;

    for (int r = 0; r < RUNS; r++)
    {
        my_timer.start();

        //auto &output_dc =
            model_inference(layer_num_total, layer_params,
                            filter_buf_ptrs,
                            input_dc,
                            inter_0_dc,
                            inter_1_dc);

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
    print_stats(small_timing, "\nSMaLL:autoencoder");

    //======================================================
    auto layers(create_model<BufferT>(filter_buf_ptrs));

    small::Tensor<BufferT> input_tensor({1, 128, 1, 1}, input_dc); // B, C_i, N, M
#if defined(QUANTIZED)
    small::Tensor<BufferT> inter_0_tensor(max_numel_inter_0*4);
    small::Tensor<BufferT> inter_1_tensor(max_numel_inter_1*4);
#else
    small::Tensor<BufferT> inter_0_tensor(max_numel_inter_0);
    small::Tensor<BufferT> inter_1_tensor(max_numel_inter_1);
    std::cout << "buffer sizes: " << max_numel_inter_0 << ", "
              << max_numel_inter_1 << std::endl;
#endif

    std::cerr << "Warm up run (LAYERS)" << std::endl;
    my_timer.start();
    auto &output_tensor =
        model_inference(layers, input_tensor,
                        inter_0_tensor, inter_1_tensor);
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
