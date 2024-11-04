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

#include <stdint.h>
#include <stdio.h>
#if PARALLEL == 1
#include <omp.h>
#endif

#include <small/op_type.hpp> /// @todo Add support: ADD, AVG_POOL, etc
#include <small/utils.hpp>

#define DEBUG 0

namespace small
{
/// @note the following namespace is tied to the buffer type because c_tile_t
///       and c_tile_out_t are tied to typedefs within the buffer class but
///       the buffer templat has been stripped from the signatures
namespace quint8_detail
{

//****************************************************************************
//abstract layer where the filter height is set to 1
template <typename BufferT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          OpType op_type,
          int8_t op_class,     //  2  (conv),  1  (dense,pool), or '0' (activation, upsample)
          bool rewrite_output> // 0 (partial conv, accum), 1 (otherwise)
void abstract_layer_1D(
    dim_t G,   // Output Channel Grouping
    dim_t K,   // Output Channels per group
    dim_t F_c, // Channel Reduction Dimension
    dim_t I_h, // Input Height
    dim_t I_w, // Input Width

    //dim_t F_h, // Filter height
    dim_t F_w, // Filter width

    //dim_t pad_top, // Padding values
    dim_t pad_left,
    dim_t pad_right,
    //dim_t pad_bottom,

    BufferT const * /*__restrict__*/ I, // Data
    BufferT const *__restrict__ F,
    BufferT * /*__restrict__*/ O)
{
    //small::unimplemented_function();
}

} // ns detail
} // ns small
