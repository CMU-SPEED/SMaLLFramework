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
#include <small/utils.hpp>

#include <small/q_abstract_op.hpp>

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
void inline kernel(
    bool first,
    dim_t F_h,
    dim_t F_w,
    dim_t input_col_stride,
    ScalarT const *I,
    ScalarT const *F,
    AccumT        *O,  // ScalarT -> AccumT
    dim_t H_lb = 0,
    dim_t H_ub = 0,
    dim_t W_lb = 0,
    dim_t W_ub = 0,
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
    // const dim_t W_UPPER = ((!W_ub) * (F_w)) + (W_ub);

    QUINT8_DEF_TILE_C(_O_wb, _C_ob);
    if (first)
    {
        QUINT8_ZERO_TILE_C(_O_wb, _C_ob, k_zero);
        if (op_type == OP_MAX_POOL)
        {
            /// @note using platform C_ob
            QUINT8_LOAD_TILE_C_strided(I, step, _O_wb, QUINT8_C_ob);
        }
        else if (op_type == OP_UPSAMPLE)
        {
            //QUINT8_LOAD_TILE_C_upsample(I, _stride, _C_ib, _O_wb, _C_ob);
            throw std::invalid_argument("*kernel ERROR: "
                                        "no support for op_type OP_UPSAMPLE.");

        }
    }
    else
    {
        QUINT8_LOAD_TILE_C(O, _O_wb, _C_ob);
    }

    for (uint32_t n = H_lb; n < H_UPPER; n++)
    {
        int filter_offset_h = n * F_w * _F_cb * _G_b * _K_b;
        int input_stencil_h = (n - H_lb) * input_col_stride; /*+ input_col_offset + input_row_offset*/

        for (uint32_t m = 0; m < F_w; m++)
        {
            int filter_offset_w = m * _F_cb * _G_b * _K_b + filter_offset_h;
            // This is C_ob because the microkernel stretches across groups
            int input_stencil_w = (m - W_lb) * _C_ib + input_stencil_h;

            ScalarT const *b = F + filter_offset_w;
            ScalarT const *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _F_cb / _UNROLL; ii++)
            {
                /// @note using platform C_ob
                ScalarT const *b_cur = b + ii * _UNROLL * QUINT8_C_ob;
                ScalarT const *a_cur = a + ii * _UNROLL;
                QUINT8_ABSTRACT_OP(op_type, op_class, a_cur, b_cur,
                                   I_offset, F_offset); /// @todo pass _C_ob
            }
        }
    }

    if constexpr(quantize)
    {
        ScalarT *O_out_ptr = O_out;
        QUINT8_QUANTIZE_TILE_C(_O_wb, _C_ob, lshift, rshift, q_mul, zero, max_val, min_val);
        QUINT8_STORE_Q_TILE_C(O_out_ptr, _O_wb, _C_ob);
    }
    else
    {
        QUINT8_STORE_TILE_C(O, _O_wb, _C_ob);
    }
}

} // ns quint8_detail
} // ns small
