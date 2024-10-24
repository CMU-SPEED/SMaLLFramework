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
          bool quantize = false,
          ScalarT max_val = 255, // std::numeric_limits<ScalarT>::max()
          ScalarT min_val = 0>   // std::numeric_limits<ScalarT>::lowest()
void inline kernel_right(
    bool first,
    dim_t F_h,
    dim_t F_w,
    dim_t input_col_stride,
    dim_t O_w_left,
    dim_t r_pad_el,
    dim_t r_pad,
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
    constexpr dim_t _C_ib = _G_b * _F_cb;
    constexpr dim_t step = _stride * _C_ib;
    const dim_t H_UPPER = ((!H_ub) * (F_h)) + (H_ub);
    QUINT8_DEF_END_C(_O_wb, _C_ob);

    if (O_w_left)
    {
        if (first)
        {
            QUINT8_ZERO_END_C(O_w_left, _C_ob, k_zero);

            if (op_type == OP_MAX_POOL && H_lb == 0 && H_ub == 0)
            {
                QUINT8_LOAD_END_C_strided(I, step, O_w_left, _C_ob);
            }
            else if (op_type == OP_UPSAMPLE)
            {
                //QUINT8_LOAD_END_C_upsample(I, _stride, _C_ib, O_w_left, _C_ob);
                throw std::invalid_argument("*kernel_right ERROR: "
                                            "no support for op_type OP_UPSAMPLE.");
            }
        }
        else
        {
            QUINT8_LOAD_END_C(O, O_w_left, _C_ob);
        }

        compute_with_padding<ScalarT, AccumT,
                             _G_b, _K_b, _F_cb, _O_wb, _stride,
                             _UNROLL, op_type, op_class>(
                                 H_lb, H_UPPER,
                                 0, F_w,
                                 F_w,
                                 O_w_left,
                                 input_col_stride,
                                 F,
                                 I,
                                 c_tile,
                                 I_offset,
                                 F_offset);

        if constexpr(quantize)
        {
            ScalarT * O_out_ptr = O_out;
            QUINT8_QUANTIZE_END_C(O_w_left, _C_ob, lshift, rshift, q_mul, zero, max_val, min_val);
            QUINT8_STORE_Q_END_C(O_out_ptr, O_w_left, _C_ob);
        }
        else
        {
            QUINT8_STORE_END_C(O, O_w_left, _C_ob);
        }
    }

    // right padding elements
    AccumT        *O_ptr = O + O_w_left * _C_ob;  // ScalarT --> AccumT
    ScalarT const *I_ptr = I + O_w_left * step;
    int W_i_valid = F_w - 1;

    if (first)
    {
        QUINT8_ZERO_END_C(r_pad_el, _C_ob, k_zero);

        // Initialize with 0 for the padding elements

        // if (op_type==OP_MAX_POOL)
        // {
        //     LOAD_END_C_strided(I_ptr, step, r_pad_el, _C_ob);
        // }
    }
    else
    {
        QUINT8_LOAD_END_C(O_ptr, r_pad_el, _C_ob);
    }

    c_tile_t *c_cur = c_tile;
    // dim_t c_cur = 0;
    for (uint32_t k_p = 0; k_p < r_pad_el; k_p++)
    {
        compute_with_padding<ScalarT, AccumT,
                             _G_b, _K_b, _F_cb, _O_wb, _stride,
                             _UNROLL, op_type, op_class>(
                                 H_lb, H_UPPER,
                                 0, W_i_valid,
                                 F_w,
                                 1,
                                 input_col_stride,
                                 F,
                                 I_ptr,
                                 c_cur,
                                 I_offset,
                                 F_offset);

        c_cur += (_K_b * _G_b) / (QUINT8_SIMD_EPILOGUE);
        W_i_valid -= _stride;
        I_ptr += _stride * _F_cb * _G_b;
    }

    if constexpr (quantize)
    {
        ScalarT *O_out_ptr = O_out + O_w_left * _C_ob;
        QUINT8_QUANTIZE_END_C(r_pad_el, _C_ob, lshift, rshift, q_mul, zero, max_val, min_val);
        QUINT8_STORE_Q_END_C(O_out_ptr, r_pad_el, _C_ob);
    }
    else
    {
        QUINT8_STORE_END_C(O_ptr, r_pad_el, _C_ob);
    }
}

} // ns quint8_detail
} // ns small
