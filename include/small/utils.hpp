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
#include <stdint.h> // for uint8_t

namespace small
{
/// @ todo only need one type.
typedef uint32_t index_t;
typedef uint32_t dim_t;
//typedef float operand_t;


//****************************************************************************
// Useful utility functions
//****************************************************************************

//****************************************************************************
// @todo make constexpr(?) function
inline dim_t output_dim(dim_t input_dim, dim_t stride, dim_t kernel_dim)
{
    int out_elems = (int(input_dim) - int(kernel_dim))/ stride + 1;
    return ((out_elems > 0) ? dim_t(out_elems) : 0U);
}

inline dim_t output_dim_new(dim_t input_dim, dim_t stride, dim_t kernel_dim)
{
    return ((kernel_dim > input_dim)
            ? 0U
            : ((input_dim - kernel_dim)/stride + 1));
}


//****************************************************************************
/**
 * When padding mode is 'f', compute front and back padding (for either
 * horizontal or vertical dimension) based on corresponding image dimension
 * and kernel dimension.  Call this twice to compute l,r and t,b pairs.
 *
 */
inline void calc_padding(uint32_t  I_dim,
                         uint32_t  K_dim,
                         uint16_t  stride,
                         uint8_t  &padding_front,
                         uint8_t  &padding_back)
{
    uint32_t padding;
    if (I_dim % stride == 0)
    {
        padding = (K_dim > stride) ?
                   K_dim - stride :
                   0;
    }
    else
    {
        padding = (K_dim > (I_dim % stride)) ?
                  (K_dim - (I_dim % stride)) :
                  0;
    }
    padding_front = padding / 2;
    padding_back  = padding - padding_front;
}

//****************************************************************************
/**
 * @todo Find a better way to set const members of the Layer classes
 * @todo consider moving to Layer class.
 */
inline uint8_t calc_front_padding(char      padding_type,
                                  uint32_t  I_dim,
                                  uint32_t  K_dim,
                                  uint16_t  stride)
{
    if (padding_type == 'v') return 0U;

    uint8_t front_padding, back_padding;
    calc_padding(I_dim, K_dim, stride, front_padding, back_padding);
    return front_padding;
}

//****************************************************************************
/**
 * @todo Find a better way to set const members of the Layer classes
 * @todo consider moving to Layer class.
 */
inline uint8_t calc_back_padding(char      padding_type,
                                 uint32_t  I_dim,
                                 uint32_t  K_dim,
                                 uint16_t  stride)
{
    if (padding_type == 'v') return 0U;

    uint8_t front_padding, back_padding;
    calc_padding(I_dim, K_dim, stride, front_padding, back_padding);
    return back_padding;
}


} // small
