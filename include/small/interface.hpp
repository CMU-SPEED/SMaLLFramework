/*
 * SMaLL Framework
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
template <class ScalarT>
void Conv2D(int kernel_size, int stride,  /// @todo dim_t?
            uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
            int output_channels, int input_channels,
            int input_height, int input_width,
            Buffer<ScalarT> const &input_buf,
            Buffer<ScalarT> const &filter_buf,
            Buffer<ScalarT>       &output_buf);

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
template <class ScalarT>
void Conv2D_rect(int kernel_size_h, int kernel_size_w, int stride,
                 uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                 int output_channels, int input_channels,
                 int input_height, int input_width,
                 Buffer<ScalarT> const &input_buf,
                 Buffer<ScalarT> const &filter_buf,
                 Buffer<ScalarT>       &output_buf);

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
template <class ScalarT>
void PartialConv2D(int kernel_size, int stride,
                   uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                   int output_channels, int input_channels,
                   int input_height, int input_width,
                   Buffer<ScalarT> const &input_buf,
                   Buffer<ScalarT> const &filter_buf,
                   Buffer<ScalarT>       &output_buf);

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
template <class ScalarT>
void DepthwiseConv2D(int kernel_size, int stride,
                     uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                     int input_channels,
                     int input_height, int input_width,
                     Buffer<ScalarT> const &input_buf,
                     Buffer<ScalarT> const &filter_buf,
                     Buffer<ScalarT>       &output_buf);

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
template <class ScalarT>
void Maxpool2D(int kernel_size, int stride,
               uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
               int input_channels,
               int input_height, int input_width,
               Buffer<ScalarT> const &input_buf,
               Buffer<ScalarT>       &output_buf);

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
template <class ScalarT>
void MaxPool2D_rect(int kernel_size_h, int kernel_size_w, int stride,
                    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                    int input_channels,
                    int input_height, int input_width,
                    Buffer<ScalarT> const &input_buf,
                    Buffer<ScalarT>       &output_buf);

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
template <class ScalarT>
void ReLUActivation(int input_channels,
                    int input_height, int input_width,
                    Buffer<ScalarT> const &input_buf,
                    Buffer<ScalarT>       &output_buf);

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
template <class ScalarT>
void Dense(int output_elements, int input_elements,
           Buffer<ScalarT> const &input_buf,
           Buffer<ScalarT> const &filter_buf,
           Buffer<ScalarT>       &output_buf);
} // ns small
