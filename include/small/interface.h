/*
 * SMaLL framework
 *
 * Copyright 2022 Carnegie Mellon University and Authors.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DMxx-xxxx
 */

/// @todo OBE: Use an enum for padding  enum Padding { PAD_FULL, PAD_VALID };
/// @todo Put interface in small namespace, details in small::detail namespace
/// @todo Consider changing to unsigned integer types for dimensions
/// @todo How should errors be reported (throw exceptions, return codes?)
/// @todo add interface documentation for possible errors

/// #include <detail/kernel.h

#pragma once

/**
 * Perform the computation for a 2D convolution layer.
 *
 * @param[in]  layer_num       Unused
 * @param[in]  kernel_size     Height and width dimensions of convolution window
 * @param[in]  stride          Number of pixels to skip in height and width
 *                             dimensions of the input between convolutions
 * @param[in]  padding         'v' is for valid padding, no additional padding;
 *                             'f' is for full padding, works only for square
 *                             images and square kernels and adds enough boundary
 *                             pixels to have the output image dimensions to be
 *                             the same as the input image dimension.
 * @param[in]  output_channels Number of channels produced by layer
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_ptr       Pointer to input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[in]  filter_ptr      Pointer to convolution filter weights
 *                             size = Ci x kernel x kernel x Co
 * @param[out] output_ptr      Pointer to output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
void Conv2D(int layer_num,
            int kernel_size, int stride, char padding,
            int output_channels, int input_channels,
            int input_height, int input_width,
            float *input_ptr, float *filter_ptr, float *output_ptr);

/**
 * Perform the computation for a partial Conv2D layer??
 *
 * @param[in]  layer_num       Unused
 * @param[in]  kernel_size     Height and width dimensions of convolution window
 * @param[in]  stride          Number of pixels to skip in height and width
 *                             dimensions of the input between convolutions
 * @param[in]  padding         'v' is for valid padding, no additional padding;
 *                             'f' is for full padding, works only for square
 *                             images and square kernels and adds enough boundary
 *                             pixels to have the output image dimensions to be
 *                             the same as the input image dimension.
 * @param[in]  output_channels Number of channels produced by layer
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_ptr       Pointer to input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[in]  filter_ptr      Pointer to convolution filter weights
 *                             size = Ci x kernel x kernel x Co
 * @param[out] output_ptr      Pointer to output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
void PartialConv2D(int layer_num,
                   int kernel_size, int stride, char padding,
                   int output_channels, int input_channels,
                   int input_height, int input_width,
                   float *input_ptr, float *filter_ptr, float *output_ptr);

/**
 * Perform the computation for a group of Conv2D layers??
 *
 * @param[in]  layer_num       Unused
 * @param[in]  kernel_size     Height and width dimensions of convolution window
 * @param[in]  stride          Number of pixels to skip in height and width
 *                             dimensions of the input between convolutions
 * @param[in]  padding         'v' is for valid padding, no additional padding;
 *                             'f' is for full padding, works only for square
 *                             images and square kernels and adds enough boundary
 *                             pixels to have the output image dimensions to be
 *                             the same as the input image dimension.
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_ptr       Pointer to input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[in]  filter_ptr      Pointer to convolution filter weights
 *                             size = Ci x kernel x kernel x Co
 * @param[out] output_ptr      Pointer to output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
void GroupConv2D(int layer_num,
                 int kernel_size, int stride, char padding,
                 int input_channels,
                 int input_height, int input_width,
                 float *input_ptr, float *filter_ptr, float *output_ptr);

/**
 * Perform the computation for a depth-wise Conv2D layer.
 *
 * @param[in]  layer_num       Unused
 * @param[in]  kernel_size     Height and width dimensions of convolution window
 * @param[in]  stride          Number of pixels to skip in height and width
 *                             dimensions of the input between convolutions
 * @param[in]  padding         'v' is for valid padding, no additional padding;
 *                             'f' is for full padding, works only for square
 *                             images and square kernels and adds enough boundary
 *                             pixels to have the output image dimensions to be
 *                             the same as the input image dimension.
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_ptr       Pointer to input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[in]  filter_ptr      Pointer to convolution filter weights
 *                             size = Ci x kernel x kernel x Co
 * @param[out] output_ptr      Pointer to output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
void DepthwiseConv2D(int layer_num,
                     int kernel_size, int stride, char padding,
                     int input_channels,
                     int input_height, int input_width,
                     float *input_ptr, float *filter_ptr, float *output_ptr);

/**
 * Perform the computation for a 2D maxpool layer.
 *
 * @param[in]  layer_num       Unused
 * @param[in]  kernel_size     Height and width dimensions of convolution window
 * @param[in]  stride          Number of pixels to skip in height and width
 *                             dimensions of the input between convolutions
 * @param[in]  padding         'v' is for valid padding, no additional padding;
 *                             'f' is for full padding, works only for square
 *                             images and square kernels and adds enough boundary
 *                             pixels to have the output image dimensions to be
 *                             the same as the input image dimension.
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_ptr       Pointer to input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[out] output_ptr      Pointer to output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
void Maxpool2D(int layer_num,
               int kernel_size, int stride, char padding,
               int input_channels,
               int input_height, int input_width,
               float *input_ptr, float *output_ptr);

/**
 * Perform the computation for a rectified linear unit (ReLU) layer.
 *
 * @param[in]  layer_num       Unused
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_ptr       Pointer to input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[out] output_ptr      Pointer to output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
void ReLUActivation(int layer_num,
                    int input_channels,
                    int input_height, int input_width,
                    float *input_ptr, float *output_ptr);

/**
 * Perform the computation for a fully-connected layer?
 *
 * @param[in]  layer_num       Unused
 * @param[in]  output_elements ???
 * @param[in]  input_elements  ???
 * @param[in]  input_ptr       Pointer to input data (how big?)
 * @param[in]  filter_ptr      Pointer to convolution filter weights (how big?)
 * @param[out] output_ptr      Pointer to output data (how big?)
 */
void Dense(int layer_num,
           int output_elements, int input_elements,
           float *input_ptr, float *filter_ptr, float *output_ptr);
