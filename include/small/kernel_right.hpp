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
#include <stdio.h>
#if PARALLEL == 1
#include <omp.h>
#endif

#include <small/op_type.hpp>
#include <small/utils.hpp>

#include <small/abstract_op.hpp>
#include <small/compute_with_padding.hpp>

namespace small
{

namespace detail
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
          OpType fused_single_element_before = OP_NONE,
          OpType fused_single_element_after = OP_NONE,
          dim_t _stride_before = 1,
          dim_t _stride_after = 1>
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
    AccumT *O, // ScalarT -> AccumT
    dim_t H_lb = 0,
    dim_t H_ub = 0,
    ScalarT const *F_b = NULL,
    ScalarT const *F_a = NULL)
{
    constexpr dim_t _C_ob = _G_b * _K_b;
    constexpr dim_t _C_ib = _G_b * _F_cb;
    constexpr dim_t step = _stride * _C_ib;
    const dim_t H_UPPER = ((!H_ub) * (F_h)) + (H_ub);
    FLOAT_DEF_END_C(_O_wb, _C_ob);
#if DEBUG
    printf("O_W_left %d r_pad_el %d\n", O_w_left, r_pad_el);
#endif
    if (O_w_left)
    {
        if (first)
        {
            FLOAT_ZERO_END_C(O_w_left, _C_ob);

            if ( (op_type == OP_MUL)|| (op_type == OP_MAX_POOL && H_lb == 0 && H_ub == 0))
            {
                FLOAT_LOAD_END_C_strided(I, step, O_w_left, _C_ob);
            }
            else if (op_type == OP_UPSAMPLE)
            {
                FLOAT_LOAD_END_C_upsample(I, _stride, _C_ib, O_w_left, _C_ob);
            }
        }
        else
        {
            //Global Reduction
            if constexpr(op_type == OP_ADD && op_class == 3)
            {
                FLOAT_ZERO_END_C(O_w_left, _C_ob);
            }
            FLOAT_LOAD_END_C(O, O_w_left, _C_ob);
            if constexpr (op_type == OP_UPSAMPLE)
            {
                FLOAT_ACCUM_END_C_upsample(I, _stride, _C_ib, O_w_left, _C_ob);
            }
        }

        if constexpr (fused_single_element_before == OP_UPSAMPLE)
        {
            FLOAT_ACCUM_END_C_upsample(F_b, _stride_before, _C_ib, O_w_left, _C_ob);
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
                                 c_tile);

        if constexpr(op_type == OP_AVERAGE_POOL)
        {
            float norm = 1.0 / (1.0 * F_h * F_w);
            FLOAT_DIV_END_C(c_tile, norm, O_w_left, _C_ob);
        }
        if constexpr(op_type == OP_ADD && op_class == 3 && _C_ob == 1)
        {
            /* If the operation reduces the channel dimension,
               reduce across channel dimension of simd tile*/
            FLOAT_REDUCE_CHANNEL_END_C(O_w_left, _C_ob);
        }

        dim_t step_after = _stride_after * _C_ib;
        FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_END(step_after, fused_single_element_after,
                                             0, F_a, c_tile, O_w_left, _C_ob);

        // in-place store
        FLOAT_STORE_END_C(O, O_w_left, _C_ob);
    }

    // right padding elements
    AccumT *O_ptr = O + O_w_left * _C_ob; // ScalarT --> AccumT
    ScalarT const *I_ptr = I + O_w_left * step;
    int W_i_valid = F_w - 1;

    if (first)
    {
        FLOAT_ZERO_END_C(r_pad_el, _C_ob);

        // Initialize with 0 for the padding elements

        //@note padding should always be 'v' for pointwise operations,
        //      so this code path should not be used
        if (op_type == OP_MUL)
        {
            FLOAT_LOAD_END_C_strided(I_ptr, step, r_pad_el, _C_ob);
        }
    }
    else
    {
        FLOAT_LOAD_END_C(O_ptr, r_pad_el, _C_ob);
    }

    if constexpr (fused_single_element_before == OP_UPSAMPLE)
    {
        FLOAT_ACCUM_END_C_upsample(F_b, _stride_before, _C_ib, r_pad_el, _C_ob);
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
                                 c_cur);

        c_cur += (_K_b * _G_b) / (FLOAT_SIMD_EPILOGUE);
        W_i_valid -= _stride;
        I_ptr += _stride * _F_cb * _G_b;
    }

    if (op_type == OP_AVERAGE_POOL)
    {
        float norm = 1.0 / (1.0 * F_h * F_w);
        FLOAT_DIV_END_C(c_tile, norm, r_pad_el, _C_ob);
    }

    dim_t step_after = _stride_after * _C_ib;
    FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_END(step_after, fused_single_element_after,
                                         0, F_a, c_tile, r_pad_el, _C_ob);

    FLOAT_STORE_END_C(O_ptr, r_pad_el, _C_ob);
}

}
}
