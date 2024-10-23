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

/// @todo Decide if we want to be generic wrt Buffer::value_type.
///       Currently definitions in check_interface_abstract.cpp are
///       fixed to float.

template <class BufferT>
void check_Conv2D(int kernel_height, int kernel_width, int stride,
                  uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                  int output_channels, int input_channels,
                  int input_height, int input_width,
                  BufferT const &input_buf,
                  BufferT const &filter_buf,
                  BufferT       &output_buf);

template <class BufferT>
void check_PartialConv2D(int kernel_size, int stride,
                         uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                         int output_channels, int input_channels,
                         int input_height, int input_width,
                         BufferT const &input_buf,
                         BufferT const &filter_buf,
                         BufferT       &output_buf);

template <class BufferT>
void check_DepthwiseConv2D(int kernel_height, int kernel_width, int stride,
                           uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                           int input_channels,
                           int input_height, int input_width,
                           BufferT const &input_buf,
                           BufferT const &filter_buf,
                           BufferT &output_buf);

template <class BufferT>
void check_MaxPool2D(int kernel_height, int kernel_width, int stride,
                     uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                     int input_channels,
                     int input_height, int input_width,
                     BufferT const &input_buf,
                     BufferT &output_buf);

template <class BufferT>
void check_ReLUActivation(int input_channels,
                          int input_height, int input_width,
                          BufferT const &input_buf,
                          BufferT       &output_buf);

template <class BufferT>
void check_Dense(int output_elements, int input_elements,
                 BufferT const &input_buf,
                 BufferT const &filter_buf,
                 BufferT       &output_buf);
