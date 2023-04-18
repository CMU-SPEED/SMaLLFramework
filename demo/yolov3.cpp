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

#include <math.h>
#include <assert.h>
#include <omp.h>
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
#ifndef PARALLEL
#define PARALLEL 0
#endif

//****************************************************************************
/* This is the runtime recording

   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:32,ichans:3,img:416x416,I,F,O)
   ReLUActivation(chans:32,img:416x416,I,O)

   Conv2D(k:3,s:2,pad:[0,1,0,1],ochans:64,ichans:32,img:416x416,I,F,O)
   ReLUActivation(chans:64,img:208x208,I,O)
   ================
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:32,ichans:64,img:208x208,I,F,O)
   ReLUActivation(chans:32,img:208x208,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:64,ichans:32,img:208x208,I,F,O)
   ReLUActivation(chans:64,img:208x208,I,O)

   ================
   Conv2D(k:3,s:2,pad:[0,1,0,1],ochans:128,ichans:64,img:208x208,I,F,O)
   ReLUActivation(chans:128,img:104x104,I,O)
   ================ x2
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:64,ichans:128,img:104x104,I,F,O)
   ReLUActivation(chans:64,img:104x104,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:128,ichans:64,img:104x104,I,F,O)
   ReLUActivation(chans:128,img:104x104,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:64,ichans:128,img:104x104,I,F,O)
   ReLUActivation(chans:64,img:104x104,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:128,ichans:64,img:104x104,I,F,O)
   ReLUActivation(chans:128,img:104x104,I,O)

   ================
   Conv2D(k:3,s:2,pad:[0,1,0,1],ochans:256,ichans:128,img:104x104,I,F,O)
   ReLUActivation(chans:256,img:52x52,I,O)
   ================ x8
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:256,img:52x52,I,F,O)
   ReLUActivation(chans:128,img:52x52,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:256,ichans:128,img:52x52,I,F,O)
   ReLUActivation(chans:256,img:52x52,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:256,img:52x52,I,F,O)
   ReLUActivation(chans:128,img:52x52,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:256,ichans:128,img:52x52,I,F,O)
   ReLUActivation(chans:256,img:52x52,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:256,img:52x52,I,F,O)
   ReLUActivation(chans:128,img:52x52,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:256,ichans:128,img:52x52,I,F,O)
   ReLUActivation(chans:256,img:52x52,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:256,img:52x52,I,F,O)
   ReLUActivation(chans:128,img:52x52,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:256,ichans:128,img:52x52,I,F,O)
   ReLUActivation(chans:256,img:52x52,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:256,img:52x52,I,F,O)
   ReLUActivation(chans:128,img:52x52,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:256,ichans:128,img:52x52,I,F,O)
   ReLUActivation(chans:256,img:52x52,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:256,img:52x52,I,F,O)
   ReLUActivation(chans:128,img:52x52,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:256,ichans:128,img:52x52,I,F,O)
   ReLUActivation(chans:256,img:52x52,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:256,img:52x52,I,F,O)
   ReLUActivation(chans:128,img:52x52,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:256,ichans:128,img:52x52,I,F,O)
   ReLUActivation(chans:256,img:52x52,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:256,img:52x52,I,F,O)
   ReLUActivation(chans:128,img:52x52,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:256,ichans:128,img:52x52,I,F,O)
   ReLUActivation(chans:256,img:52x52,I,O)

   ================
   Conv2D(k:3,s:2,pad:[0,1,0,1],ochans:512,ichans:256,img:52x52,I,F,O)
   ReLUActivation(chans:512,img:26x26,I,O)
   ================ x8
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:256,ichans:512,img:26x26,I,F,O)
   ReLUActivation(chans:256,img:26x26,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:512,ichans:256,img:26x26,I,F,O)
   ReLUActivation(chans:512,img:26x26,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:256,ichans:512,img:26x26,I,F,O)
   ReLUActivation(chans:256,img:26x26,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:512,ichans:256,img:26x26,I,F,O)
   ReLUActivation(chans:512,img:26x26,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:256,ichans:512,img:26x26,I,F,O)
   ReLUActivation(chans:256,img:26x26,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:512,ichans:256,img:26x26,I,F,O)
   ReLUActivation(chans:512,img:26x26,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:256,ichans:512,img:26x26,I,F,O)
   ReLUActivation(chans:256,img:26x26,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:512,ichans:256,img:26x26,I,F,O)
   ReLUActivation(chans:512,img:26x26,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:256,ichans:512,img:26x26,I,F,O)
   ReLUActivation(chans:256,img:26x26,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:512,ichans:256,img:26x26,I,F,O)
   ReLUActivation(chans:512,img:26x26,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:256,ichans:512,img:26x26,I,F,O)
   ReLUActivation(chans:256,img:26x26,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:512,ichans:256,img:26x26,I,F,O)
   ReLUActivation(chans:512,img:26x26,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:256,ichans:512,img:26x26,I,F,O)
   ReLUActivation(chans:256,img:26x26,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:512,ichans:256,img:26x26,I,F,O)
   ReLUActivation(chans:512,img:26x26,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:256,ichans:512,img:26x26,I,F,O)
   ReLUActivation(chans:256,img:26x26,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:512,ichans:256,img:26x26,I,F,O)
   ReLUActivation(chans:512,img:26x26,I,O)

   ================
   Conv2D(k:3,s:2,pad:[0,1,0,1],ochans:1024,ichans:512,img:26x26,I,F,O)
   ReLUActivation(chans:1024,img:13x13,I,O)
   ================ x4
   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:1024,img:13x13,I,F,O)
   ReLUActivation(chans:512,img:13x13,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:1024,ichans:512,img:13x13,I,F,O)
   ReLUActivation(chans:1024,img:13x13,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:1024,img:13x13,I,F,O)
   ReLUActivation(chans:512,img:13x13,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:1024,ichans:512,img:13x13,I,F,O)
   ReLUActivation(chans:1024,img:13x13,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:1024,img:13x13,I,F,O)
   ReLUActivation(chans:512,img:13x13,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:1024,ichans:512,img:13x13,I,F,O)
   ReLUActivation(chans:1024,img:13x13,I,O)

   Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:512,ichans:1024,img:13x13,I,F,O)
   ReLUActivation(chans:512,img:13x13,I,O)
   Conv2D(k:3,s:1,pad:[1,1,1,1],ochans:1024,ichans:512,img:13x13,I,F,O)
   ReLUActivation(chans:1024,img:13x13,I,O)

 */
//****************************************************************************

/* This is the config file:
[net]
# Testing
#batch=1
#subdivisions=1
# Training
batch=16
subdivisions=1
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

[conv2d], batch_norm=1, filters=32, size=3, stride=1, pad=1, activation=leaky   Output: 416x416

# Downsample
[conv2d], batch_norm=1, filters=64, size=3, stride=2, pad=1, activation=leaky   Output: 208x208
 |     |
 |     |
 |   [conv2d], batch_norm=1, filters=32, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=64, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear


# Downsample
[conv2d], batch_norm=1, filters=128, size=3, stride=2, pad=1, activation=leaky  Output: 104x104
 |     |
 |     |
 |   [conv2d], batch_norm=1, filters=64, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=128, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |
 |   [conv2d], batch_norm=1, filters=64, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=128, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear


# Downsample
[conv2d], batch_norm=1, filters=256, size=3, stride=2, pad=1, activation=leaky  Output: 52x52
 |     |
 |     |
 |   [conv2d], batch_norm=1, filters=128, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=256, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=128, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=256, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=128, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=256, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=128, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=256, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=128, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=256, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=128, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=256, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=128, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=256, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=128, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=256, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear


# Downsample
[conv2d], batch_norm=1, filters=512, size=3, stride=2, pad=1, activation=leaky  Output 26x26
 |     |
 |     |
 |   [conv2d], batch_norm=1, filters=256, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=512, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=256, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=512, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=256, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=512, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=256, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=512, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=256, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=512, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=256, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=512, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=256, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=512, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=256, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=512, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear


# Downsample
[conv2d], batch_norm=1, filters=1024, size=3, stride=2, pad=1, activation=leaky     Output 13x13
 |     |
 |     |
 |   [conv2d], batch_norm=1, filters=512, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=1024, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=512, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=1024, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=512, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=1024, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear
 |     |
 |   [conv2d], batch_norm=1, filters=512, size=1, stride=1, pad=1, activation=leaky
 |   [conv2d], batch_norm=1, filters=1024, size=3, stride=1, pad=1, activation=leaky
 |     |
[shortcut], from=-3, activation=linear,

######################
######################

[conv2d], batch_norm=1, filters=512,  size=1, stride=1, pad=1, activation=leaky	  Output 13x13?
[conv2d], batch_norm=1, size=3, stride=1, pad=1, filters=1024, activation=leaky

[conv2d], batch_norm=1, filters=512,  size=1, stride=1, pad=1, activation=leaky
[conv2d], batch_norm=1, size=3, stride=1, pad=1, filters=1024, activation=leaky

[conv2d], batch_norm=1, filters=512,  size=1, stride=1, pad=1, activation=leaky
[conv2d], batch_norm=1, size=3, stride=1, pad=1, filters=1024, activation=leaky

[conv2d], size=1, stride=1, pad=1, filters=18, activation=linear		Output 13x13


[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=1
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1


[route], layers = -4

[conv2d], batch_norm=1, filters=256, size=1, stride=1, pad=1, activation=leaky

[upsample], stride=2

[route], layers = -1, 61


[conv2d], batch_norm=1, filters=256, size=1, stride=1, pad=1, activation=leaky
[conv2d], batch_norm=1, size=3, stride=1, pad=1, filters=512, activation=leaky

[conv2d], batch_norm=1, filters=256, size=1, stride=1, pad=1, activation=leaky
[conv2d], batch_norm=1, size=3, stride=1, pad=1, filters=512, activation=leaky

[conv2d], batch_norm=1, filters=256, size=1, stride=1, pad=1, activation=leaky
[conv2d], batch_norm=1, size=3, stride=1, pad=1, filters=512, activation=leaky


[conv2d], size=1, stride=1, pad=1, filters=18, activation=linear


[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=1
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1


[route], layers = -4

[conv2d], batch_norm=1, filters=128, size=1, stride=1, pad=1, activation=leaky

[upsample], stride=2

[route], layers = -1, 36


[conv2d], batch_norm=1, filters=128, size=1, stride=1, pad=1, activation=leaky
[conv2d], batch_norm=1, size=3, stride=1, pad=1, filters=256, activation=leaky

[conv2d], batch_norm=1, filters=128, size=1, stride=1, pad=1, activation=leaky
[conv2d], batch_norm=1, size=3, stride=1, pad=1, filters=256, activation=leaky

[conv2d], batch_norm=1, filters=128, size=1, stride=1, pad=1, activation=leaky
[conv2d], batch_norm=1, size=3, stride=1, pad=1, filters=256, activation=leaky


[conv2d], size=1, stride=1, pad=1, filters=18. activation=linear


[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=1
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

 */

//****************************************************************************

#include<small/Layer.hpp>
#include<small/Conv2DLayer.hpp>
#include<small/MaxPool2DLayer.hpp>
#include<small/ReLULayer.hpp>

//****************************************************************************
template <class BufferT>
void create_conv_block(uint32_t kernel_size,
                       uint32_t stride,
                       uint32_t input_channels,
                       uint32_t output_channels,
                       uint32_t &image_height,
                       uint32_t &image_width,
                       //std::vector<float> batch_norm_params,
                       //BufferT const &filter,
                       //bool pack_filters,
                       std::vector<BufferT*> &filters,
                       std::vector<small::Layer<BufferT>*> &layers)
{
    std::cerr << "conv_block(k:" << kernel_size << ",s" << stride
              << "), in(chans:" << input_channels
              << ",img:" << image_height << "x" << image_width << ")->";

    // ---------- TEMPORARY CODE --------------
    // create an unpacked filter
    size_t filter_numel(kernel_size*kernel_size*input_channels*output_channels);
    auto *filter = small::alloc_buffer(filter_numel);
    init(*filter, filter_numel);
    filters.push_back(filter);
    // ---------- TEMPORARY CODE --------------

    layers.push_back(
        new small::Conv2DLayer<BufferT>(kernel_size, kernel_size,
                                        stride, small::PADDING_F,
                                        input_channels, output_channels,
                                        image_height, image_width,
                                        // batch_norm_params, pack_filters,
                                        *filter));

    image_height = small::compute_output_dim(image_height, kernel_size,
                                             stride, small::PADDING_F);
    image_width  = small::compute_output_dim(image_width, kernel_size,
                                             stride, small::PADDING_F);

    layers.push_back(
        //new small::LeakyReLULayer(output_channels, image_height, image_width)
        new small::ReLULayer<BufferT>(
            output_channels, image_height, image_width));
    std::cerr << ", out(chans:" << output_channels
              << ",img:" << image_height << "x" << image_width << ")\n";
}

//****************************************************************************

template <class BufferT>
std::vector<small::Layer<BufferT>*> create_model(
    uint32_t image_height,
    uint32_t image_width,
    uint32_t model_input_channels,
    uint32_t model_output_channels,
    std::vector<BufferT*> &filters)
{
    std::vector<small::Layer<BufferT>*> layers;

    // settings for first layer
    bool pack_filters = true;
    uint32_t kernel_size = 3U;
    uint32_t stride = 1U;
    uint32_t input_channels = model_input_channels;
    uint32_t output_channels = 32U;

    // first conv block
    //size_t filter_num = 0U;
    create_conv_block(kernel_size, stride,
                      input_channels, output_channels,
                      image_height, image_width,
                      //batch_norm_params[filter_num],
                      //filters[filter_num++], pack_filters,
                      filters,
                      layers);

    uint32_t residual_blocks[] = {1,2,8,8,4};

    for (size_t stage_num = 0; stage_num < 5; ++stage_num)
    {
        std::cerr << "== Begin Stage " << stage_num << " ==\n";
        kernel_size = 3U;
        stride = 2U;
        input_channels = output_channels;
        output_channels = 2*input_channels;

        // halve the image size and double the number of channels
        create_conv_block(kernel_size, stride,
                          input_channels, output_channels,
                          image_height, image_width,
                          //batch_norm_params[filter_num],
                          //filters[filter_num++], pack_filters,
                          filters,
                          layers);

        for (size_t block_num = 0;
             block_num < residual_blocks[stage_num];
             ++block_num)
        {
            std::cerr << "===== Begin Residual Block " << stage_num
                      << "/" << block_num << " =====\n";
            // ================= Begin Residual Block =================
            kernel_size = 1U;
            stride = 1U;
            input_channels = output_channels;
            output_channels = input_channels/2;
            create_conv_block(kernel_size, stride,
                              input_channels, output_channels,
                              image_height, image_width,
                              //batch_norm_params[filter_num],
                              //filters[filter_num++], pack_filters,
                              filters,
                              layers);

            input_channels = output_channels;
            output_channels = 2*input_channels;
            kernel_size = 3U;
            create_conv_block(kernel_size, stride,
                              input_channels, output_channels,
                              image_height, image_width,
                              //batch_norm_params[filter_num],
                              //filters[filter_num++], pack_filters,
                              filters,
                              layers);

            /// @todo for adding two tensors
            //layers.push_back(
            //  new small::LinearLayer(output_channels,image_height,image_width));
            // ================== End Residual Block ==================
            std::cerr << "====== End Residual Block ======\n";
        }
    }


    // Begin detection layers
    //=====================================

    return layers;
}

//****************************************************************************
template <class BufferT>
void compute_buffer_sizes(
    std::vector<small::Layer<BufferT>*> const &layers,
    size_t                                    &max_numel_0,
    size_t                                    &max_numel_1,
    size_t                                    &max_numel_2)
{
    size_t layer_num = 0;

    //layers[layer_num++]->compute_output(input_dc, inter_1_dc);   // conv 3x3/1
    //layers[layer_num++]->compute_output(inter_1_dc, inter_1_dc); // ReLU
    max_numel_1 = std::max<size_t>(max_numel_1,
                                   layers[layer_num]->output_buffer_size());
    layer_num += 2;

    uint32_t residual_blocks[] = {1,2,8,8,4};
    for (size_t stage_num = 0; stage_num < 5; ++stage_num)
    {
        //layers[layer_num++]->compute_output(inter_1_dc, inter_0_dc); //conv 3x3/2
        //layers[layer_num++]->compute_output(inter_0_dc, inter_0_dc); //relu
        max_numel_0 = std::max<size_t>(max_numel_0,
                                       layers[layer_num]->output_buffer_size());
        layer_num += 2;

        for (size_t block_num = 0; block_num < residual_blocks[stage_num];
             ++block_num)
        {
            //layers[layer_num++]->compute_output(inter_0_dc, inter_1_dc); //conv 1x1/1
            //layers[layer_num++]->compute_output(inter_1_dc, inter_1_dc); //relu
            max_numel_1 = std::max<size_t>(max_numel_1,
                                           layers[layer_num]->output_buffer_size());
            layer_num += 2;

            //layers[layer_num++]->compute_output(inter_1_dc, inter_2_dc); //conv 3x3/1
            //layers[layer_num++]->compute_output(inter_2_dc, inter_2_dc); //relu
            max_numel_2 = std::max<size_t>(max_numel_2,
                                           layers[layer_num]->output_buffer_size());
            layer_num += 2;

            /// @todo add inter_1 to inter_0 to complete residual layer
            /// layers[layer_num++]->compute_output(inter_2_dc, inter_0_dc); // linear
            //inter_0_dc.swap(inter_2_dc); // placeholder for (buf0 += buf2)
            std::cerr << "should be the same: " << max_numel_0 << "?="
                      << max_numel_2 << std::endl;
            //max_numel_0 = std::max<size_t>(max_numel_0, max_numel_2); // NOT OPTIMAL
        }
        //inter_1_dc.swap(inter_0_dc)
        max_numel_1 = std::max<size_t>(max_numel_1, max_numel_0); // NOT OPTIMAL
    }

    std::cerr << "Max num elements: " << max_numel_0 << ", " << max_numel_1
              << ", " << max_numel_2 << std::endl;
}

//****************************************************************************
template <class BufferT>
BufferT &model_inference(
    std::vector<small::Layer<BufferT>*> const &layers,
    BufferT                             const &input_dc,
    BufferT                                   &inter_0_dc,
    BufferT                                   &inter_1_dc,
    BufferT                                   &inter_2_dc)
{
    size_t layer_num = 0;

    // yolo_block = 0
    layers[layer_num++]->compute_output(input_dc, inter_1_dc);   // conv 3x3/1
    layers[layer_num++]->compute_output(inter_1_dc, inter_1_dc); // ReLU

    uint32_t residual_blocks[] = {1,2,8,8,4};
    for (size_t stage_num = 0; stage_num < 5; ++stage_num)
    {
        layers[layer_num++]->compute_output(inter_1_dc, inter_0_dc); //conv 3x3/2
        layers[layer_num++]->compute_output(inter_0_dc, inter_0_dc); //relu

        for (size_t block_num = 0; block_num < residual_blocks[stage_num];
             ++block_num)
        {
            layers[layer_num++]->compute_output(inter_0_dc, inter_1_dc); //conv 1x1/1
            layers[layer_num++]->compute_output(inter_1_dc, inter_1_dc); //relu

            layers[layer_num++]->compute_output(inter_1_dc, inter_2_dc); //conv 3x3/1
            layers[layer_num++]->compute_output(inter_2_dc, inter_2_dc); //relu

            /// @todo add inter_1 to inter_0 to complete residual layer
            // layers[layer_num++]->compute_output(inter_2_dc, inter_0_dc); // linear
            inter_0_dc.swap(inter_2_dc); // placeholder for (buf0 += buf2)
        }
        inter_1_dc.swap(inter_0_dc);
    }

    return inter_0_dc;
}

//****************************************************************************


//****************************************************************************
//****************************************************************************
template <class BufferT>
void inference(uint32_t C_i,
               uint32_t N,   // I_h
               uint32_t M,   // I_w
               uint32_t num_classes)
{
    // Create and Initialize Input tensors
    uint32_t input_dimensions = C_i * N * M;
    BufferT input_dc(input_dimensions);
    init(input_dc, input_dimensions);

    std::cerr << "\ncreate_model (LAYERS)\n";
    std::vector<BufferT *> filter_buf_ptrs;
    size_t max_numel_0(0UL), max_numel_1(0UL), max_numel_2(0UL);

    auto layers(create_model<BufferT>(N, M,
                                      C_i, num_classes,
                                      filter_buf_ptrs));
    compute_buffer_sizes<BufferT>(layers, max_numel_0, max_numel_1, max_numel_2);

    // allocate space for intermediate outputs (use the max sizes calculated previously)

#if defined(QUANTIZED)
    BufferT inter_0_dc(max_numel_0*2);  /// @todo HACK need to determine correct size
    BufferT inter_1_dc(max_numel_1*2);  /// @todo HACK need to determine correct size
    BufferT inter_2_dc(max_numel_2*2);  /// @todo HACK need to determine correct size
#else
    BufferT inter_0_dc(max_numel_0);
    BufferT inter_1_dc(max_numel_1);
    BufferT inter_2_dc(max_numel_2);
#endif

    //========================================================================

    std::cerr << "\nWarm up run (LAYERS)\n";
    auto &output_a_dc =
        model_inference(layers, input_dc,
                        inter_0_dc, inter_1_dc, inter_2_dc);

    // Compare the results
    size_t num_outputs = layers.back()->output_buffer_size();
    std::cout << "\nNum output elements: " << num_outputs << std::endl;
    // for (size_t ix = 0; ix < num_outputs; ++ix)
    // {
    //     std::cout << "Current, new " << ix << ": "
    //               << (float)output_dc[ix] << ", " << (float)output_a_dc[ix]
    //               << std::endl;
    // }

    // clean up model (move to model class destructor when built
    std::cout << "Num layers to delete: " << layers.size() << std::endl;
    for (auto layer : layers) delete layer;

    //========================================================================

    // Free allocated weight buffers

    std::cout << "Num filters to delete: " << filter_buf_ptrs.size() << std::endl;
    for (size_t l = 0; l < filter_buf_ptrs.size(); l++)
    {
        std::cerr << l << ": filter_size = " << filter_buf_ptrs[l]->size()
                  << ", data = " << (void*)(filter_buf_ptrs[l]->data())
                  << std::endl;
        small::free_buffer(filter_buf_ptrs[l]);
    }

    //===============================End SMaLL================================

    //___________________________Correctness check____________________________
    // bool check = 1;
    // std::vector<uint32_t>
    //     inter_0_dims, inter_1_dims;
    // for (int tens_dim_i = 0; tens_dim_i < inter_1.dim(); tens_dim_i++)
    // {
    //     inter_1_dims.push_back(inter_1.size(tens_dim_i));
    // }
    // check = check_eqivalence<C_ob, C_ib>(inter_1, 'o', inter_1_dims, inter_1_dc, LIMIT);
    // std::cout << inter_1_dims << std::endl;

    // assert(check == 1);

    // inter_0_dims.clear();
    // for (int tens_dim_i = 0; tens_dim_i < inter_0.dim(); tens_dim_i++)
    // {
    //     inter_0_dims.push_back(inter_0.size(tens_dim_i));
    // }
    // check = check_eqivalence<C_ob, C_ib>(inter_0, 'o', inter_0_dims, inter_0_dc, LIMIT);
    // std::cout << inter_0_dims << std::endl;

    // assert(check == 1);

    // Free input and output buffers
    //free(input_dc);
    //free(inter_0_dc);
    //free(inter_1_dc);
}

//****************************************************************************
// For non-arduino platforms.  ... move to driver.cpp?
//****************************************************************************
#ifndef NANO33BLE
int main(int argc, char **argv)
{
    uint32_t C_i = 3U;
    uint32_t I_h = 416U;
    uint32_t I_w = 416U;
    uint32_t num_classes = 16U;

    if (argc == 5)
    {
        C_i = atoi(argv[1]);
        I_h = atol(argv[2]);  //N
        I_w = atol(argv[3]);  //M
        num_classes = atol(argv[4]);
    }
    else if (argc != 1)
    {
        printf("\nUsage ERROR: %s "
               "[<Input Channels> <Input H> <Input W> <Output Classes>]\n",
               argv[0]);
        printf("Default: %s 3 416 416 16\n", argv[0]);
        return 0;
    }


    if (num_classes % 16 != 0)
    {
        printf("Number of output classes must be a multiple of 16\n");
        exit(-1);
    }

    small::Timer my_timer;
    my_timer.start();
#if defined(QUANTIZED)
    inference<small::QUInt8Buffer>(C_i, I_h, I_w, num_classes);
#else
    inference<small::FloatBuffer>(C_i, I_h, I_w, num_classes);
#endif
    my_timer.stop();
    print_cycles(my_timer.elapsed());

    return 0;
}
#endif
