//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2024 by The SMaLL Contributors, All Rights Reserved.
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

#include <small/op_type.hpp>
#include <small/q_compute_with_padding.hpp>

namespace small
{
namespace quint8_detail
{

//****************************************************************************
template <typename ScalarT,
          typename AccumT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          OpType op_type,
          int8_t op_class,
          bool   quantize = false,
          ScalarT max_val = 255, // std::numeric_limits<ScalarT>::max()
          ScalarT min_val = 0>   // std::numeric_limits<ScalarT>::lowest()
void inline kernel_left(
    bool first,
    dim_t F_h,
    dim_t F_w,
    dim_t input_col_stride,
    dim_t l_pad_el,
    dim_t l_pad,
    ScalarT const *I,
    ScalarT const *F,
    AccumT        *O,  // ScalarT -> AccumT
    dim_t H_lb = 0,
    dim_t H_ub = 0,
    int k_zero = 0,
    AccumT I_offset = 0,
    AccumT F_offset = 0,
    ScalarT *O_out = NULL,
    AccumT lshift = 0,
    AccumT rshift = 0,
    AccumT q_mul = 1,
    AccumT zero = 0)
{
    constexpr dim_t _C_ob = _G_b * _K_b;
    // constexpr dim_t _C_ib = _G_b * _F_cb;
    // constexpr dim_t step = _stride * _C_ib;

    const dim_t H_UPPER = ((!H_ub) * (F_h)) + (H_ub);
    QUINT8_DEF_END_C(_O_wb, _C_ob);

    // left padding elements
    AccumT        *O_ptr = O;  // ScalarT -> AccumT
    ScalarT const *I_ptr = I;

    int W_i_valid = l_pad;

    if (first)
    {
        QUINT8_ZERO_END_C(l_pad_el, _C_ob, k_zero);
    }
    else
    {
        QUINT8_LOAD_END_C(O_ptr, l_pad_el, _C_ob);
    }

    c_tile_t *c_cur = c_tile;
    // dim_t c_cur = 0;
    for (uint32_t k_p = 0; k_p < l_pad_el; k_p++)
    {
        compute_with_padding<ScalarT, AccumT,
                             _G_b, _K_b, _F_cb, _O_wb, _stride,
                             _UNROLL, op_type, op_class>(
                                 H_lb, H_UPPER,
                                 W_i_valid, F_w,
                                 F_w,
                                 1,
                                 input_col_stride,
                                 F,
                                 I_ptr,
                                 c_cur,
                                 I_offset,
                                 F_offset);

        c_cur += (_K_b * _G_b) / (QUINT8_SIMD_EPILOGUE);
        // c_cur += 1;
        W_i_valid -= _stride;
        // I_ptr += ()*(_stride * _F_cb * _G_b);
    }

    if constexpr (quantize)
    {
        ScalarT *O_out_ptr = O_out;
        QUINT8_QUANTIZE_END_C(l_pad_el, _C_ob, lshift, rshift, q_mul, zero, max_val, min_val);
        QUINT8_STORE_Q_END_C(O_out_ptr, l_pad_el, _C_ob);
    }
    else
    {
        QUINT8_STORE_END_C(O_ptr, l_pad_el, _C_ob);
    }

    O_ptr += _G_b * _K_b;
}

} // ns quint8_detail
} // ns small
