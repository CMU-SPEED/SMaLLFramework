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

/// @todo OBE: Use an enum for padding  enum Padding { PAD_FULL, PAD_VALID };
/// @todo Consider changing to unsigned integer types for dimensions
/// @todo How should errors be reported (throw exceptions, return codes?)
/// @todo add interface documentation for possible errors

#pragma once

#include <stdint.h> // for uint8_t

namespace small
{

//****************************************************************************
/**
 * Perform the computation for a 2D convolution layer.
 *
 * @param[in]  kernel_size     Height and width dimensions of convolution window
 * @param[in]  stride          Number of pixels to skip in height and width
 *                             dimensions of the input between convolutions
 * @param[in]  t_pad           number of pixels of top padding
 * @param[in]  b_pad           number of pixels of bottom padding
 * @param[in]  l_pad           number of pixels of left padding
 * @param[in]  r_pad           number of pixels of right padding
 * @param[in]  output_channels Number of channels produced by layer
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_buf       Buffer of input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[in]  filter_buf      Buffer of convolution filter weights
 *                             size = Ci x kernel x kernel x Co
 * @param[out] output_buf      Buffer of output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
template <class BufferT>                    /// @todo Create a buffer concept?
void Conv2D(int kernel_size, int stride,    /// @todo use dim_t?
            uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
            int output_channels, int input_channels,
            int input_height, int input_width,
            BufferT const &input_buf,
            BufferT const &filter_buf,
            BufferT       &output_buf);

//****************************************************************************
/**
 * Perform the computation for a 2D convolution layer.
 *
 * @param[in]  kernel_size_h   Height dimension of convolution window
 * @param[in]  kernel_size_w   Width dimension of convolution window
 * @param[in]  stride          Number of pixels to skip in height and width
 *                             dimensions of the input between convolutions
 * @param[in]  t_pad           number of pixels of top padding
 * @param[in]  b_pad           number of pixels of bottom padding
 * @param[in]  l_pad           number of pixels of left padding
 * @param[in]  r_pad           number of pixels of right padding
 * @param[in]  output_channels Number of channels produced by layer
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_buf       Buffer of input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[in]  filter_buf      Buffer of convolution filter weights
 *                             size = Ci x kernel x kernel x Co
 * @param[out] output_buf      Buffer of output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
template <class BufferT>
void Conv2D_rect(int kernel_size_h, int kernel_size_w, int stride,
                 uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                 int output_channels, int input_channels,
                 int input_height, int input_width,
                 BufferT const &input_buf,
                 BufferT const &filter_buf,
                 BufferT       &output_buf);

//****************************************************************************
/**
 * Perform the computation for a partial Conv2D layer??
 *
 * @param[in]  kernel_size     Height and width dimensions of convolution window
 * @param[in]  stride          Number of pixels to skip in height and width
 *                             dimensions of the input between convolutions
 * @param[in]  t_pad           number of pixels of top padding
 * @param[in]  b_pad           number of pixels of bottom padding
 * @param[in]  l_pad           number of pixels of left padding
 * @param[in]  r_pad           number of pixels of right padding
 * @param[in]  output_channels Number of channels produced by layer
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_buf       Buffer of input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[in]  filter_buf      Buffer of convolution filter weights
 *                             size = Ci x kernel x kernel x Co
 * @param[out] output_buf      Buffer of output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
template <class BufferT>
void PartialConv2D(int kernel_size, int stride,
                   uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                   int output_channels, int input_channels,
                   int input_height, int input_width,
                   BufferT const &input_buf,
                   BufferT const &filter_buf,
                   BufferT       &output_buf);

/**
 * Perform the computation for a depth-wise Conv2D layer.
 *
 * @param[in]  kernel_size     Height and width dimensions of convolution window
 * @param[in]  stride          Number of pixels to skip in height and width
 *                             dimensions of the input between convolutions
 * @param[in]  t_pad           number of pixels of top padding
 * @param[in]  b_pad           number of pixels of bottom padding
 * @param[in]  l_pad           number of pixels of left padding
 * @param[in]  r_pad           number of pixels of right padding
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_buf       Buffer of input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[in]  filter_buf      Buffer of convolution filter weights
 *                             size = Ci x kernel x kernel x Co
 * @param[out] output_buf      Buffer of output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
template <class BufferT>
void DepthwiseConv2D(int kernel_size, int stride,
                     uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                     int input_channels,
                     int input_height, int input_width,
                     BufferT const &input_buf,
                     BufferT const &filter_buf,
                     BufferT       &output_buf);

/**
 * Perform the computation for a 2D maxpool layer.
 *
 * @param[in]  kernel_size     Height and width dimensions of convolution window
 * @param[in]  stride          Number of pixels to skip in height and width
 *                             dimensions of the input between convolutions
 * @param[in]  t_pad           number of pixels of top padding
 * @param[in]  b_pad           number of pixels of bottom padding
 * @param[in]  l_pad           number of pixels of left padding
 * @param[in]  r_pad           number of pixels of right padding
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_buf       Buffer of input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[out] output_buf      Buffer of output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
template <class BufferT>
void MaxPool2D(int kernel_size, int stride,
               uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
               int input_channels,
               int input_height, int input_width,
               BufferT const &input_buf,
               BufferT       &output_buf);

//****************************************************************************
/**
 * Perform the computation for a 2D maxpool layer with a rectangular window
 *
 * @param[in]  kernel_size_h   Height dimension of convolution window
 * @param[in]  kernel_size_w   Width dimension of convolution window
 * @param[in]  stride          Number of pixels to skip in height and width
 *                             dimensions of the input between convolutions
 * @param[in]  t_pad           number of pixels of top padding
 * @param[in]  b_pad           number of pixels of bottom padding
 * @param[in]  l_pad           number of pixels of left padding
 * @param[in]  r_pad           number of pixels of right padding
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_buf       Buffer of input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[out] output_buf      Buffer of output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
template <class BufferT>
void MaxPool2D_rect(int kernel_size_h, int kernel_size_w, int stride,
                    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                    int input_channels,
                    int input_height, int input_width,
                    BufferT const &input_buf,
                    BufferT       &output_buf);

//****************************************************************************
/**
 * Perform the computation for a rectified linear unit (ReLU) layer.
 *
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_buf       Buffer of input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[out] output_buf      Buffer of output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
template <class BufferT>
void ReLUActivation(int input_channels,
                    int input_height, int input_width,
                    BufferT const &input_buf,
                    BufferT       &output_buf);

//****************************************************************************
/**
 * Perform the computation for a "leaky" rectified linear unit (ReLU) layer.
 *
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  negative_slope  The attenuation factor for negative values (1e-2)
 * @param[in]  input_buf       Buffer of input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[out] output_buf      Buffer of output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
template <class BufferT>
void LeakyReLUActivation(int input_channels,
                         int input_height, int input_width,
                         float negative_slope,  /// @todo should this be valueT
                         BufferT const &input_buf,
                         BufferT       &output_buf);

//****************************************************************************
/**
 * Element-wise accumulation of input buffer into output buffer.
 *
 * @param[in]  input_channels  Number of channels associated with input image
 * @param[in]  input_height    Height of input image in pixels
 * @param[in]  input_width     Width of input image in pixels
 * @param[in]  input_buf       Buffer of input (image x channels) data
 *                             size = Ci x iH x iW
 * @param[out] output_buf      Buffer of output data computed for layer
 *                             size = oH x oW x Co where the output image size
 *                             depends on input image size, kernel, padding
 *                             and stride parameters.
 */
template <class BufferT>
void Accum(int input_channels,
           int input_height, int input_width,
           BufferT const &input_buf,
           BufferT       &output_buf);

//****************************************************************************
/**
 * Perform the computation for a fully-connected layer?
 *
 * @param[in]  output_elements ???
 * @param[in]  input_elements  ???
 * @param[in]  input_buf       Buffer of input data (how big?)
 * @param[in]  filter_buf      Buffer of convolution filter weights (how big?)
 * @param[out] output_buf      Buffer of output data (how big?)
 */
template <class BufferT>
void Dense(int output_elements, int input_elements,
           BufferT const &input_buf,
           BufferT const &filter_buf,
           BufferT       &output_buf);
} // ns small
