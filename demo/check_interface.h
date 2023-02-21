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

#pragma once

/// @todo Decide if we want to be generic wrt Buffer::value_type.
///       Currently definitions in check_interface_abstract.cpp are
///       fixed to float.

template <class ScalarT>
void check_Conv2D(int kernel_size, int stride,
                  uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                  int output_channels, int input_channels,
                  int input_height, int input_width,
                  small::Buffer<ScalarT> const &input_buf,
                  small::Buffer<ScalarT> const &filter_buf,
                  small::Buffer<ScalarT>       &output_buf);

template <class ScalarT>
void check_PartialConv2D(int kernel_size, int stride,
                         uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                         int output_channels, int input_channels,
                         int input_height, int input_width,
                         small::Buffer<ScalarT> const &input_buf,
                         small::Buffer<ScalarT> const &filter_buf,
                         small::Buffer<ScalarT>       &output_buf);

template <class ScalarT>
void check_DepthwiseConv2D(int kernel_size, int stride,
                           uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                           int input_channels,
                           int input_height, int input_width,
                           small::Buffer<ScalarT> const &input_buf,
                           small::Buffer<ScalarT> const &filter_buf,
                           small::Buffer<ScalarT>       &output_buf);

template <class ScalarT>
void check_Maxpool2D(int kernel_size, int stride,
                     uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
                     int input_channels,
                     int input_height, int input_width,
                     small::Buffer<ScalarT> const &input_buf,
                     small::Buffer<ScalarT>       &output_buf);

template <class ScalarT>
void check_ReLUActivation(int input_channels,
                          int input_height, int input_width,
                          small::Buffer<ScalarT> const &input_buf,
                          small::Buffer<ScalarT>       &output_buf);

template <class ScalarT>
void check_Dense(int output_elements, int input_elements,
                 small::Buffer<ScalarT> const &input_buf,
                 small::Buffer<ScalarT> const &filter_buf,
                 small::Buffer<ScalarT>       &output_buf);
