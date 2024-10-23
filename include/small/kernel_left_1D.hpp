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
#include <small/compute_with_padding_1D.hpp>

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
void inline kernel_left_1D(
    bool first,
    //dim_t F_h,
    dim_t F_w,
    dim_t input_col_stride,
    dim_t l_pad_el,
    dim_t l_pad,
    ScalarT const *I,
    ScalarT const *F,
    AccumT *O, // ScalarT -> AccumT
    //dim_t H_lb = 0,
    //dim_t H_ub = 0,
    ScalarT const *F_b = NULL,
    ScalarT const *F_a = NULL)
{
    constexpr dim_t _C_ob = _G_b * _K_b;
    constexpr dim_t _C_ib = _G_b * _F_cb;
    // constexpr dim_t step = _stride * _C_ib;

    //const dim_t H_UPPER = ((!H_ub) * (F_h)) + (H_ub);
    FLOAT_DEF_END_C(_O_wb, _C_ob);

    // left padding elements
    AccumT *O_ptr = O; // ScalarT -> AccumT
    ScalarT const *I_ptr = I;

    int W_i_valid = l_pad;

    if (first)
    {
        FLOAT_ZERO_END_C(l_pad_el, _C_ob);
    }
    else
    {
        FLOAT_LOAD_END_C(O_ptr, l_pad_el, _C_ob);
    }

    if constexpr (fused_single_element_before == OP_UPSAMPLE)
    {
        FLOAT_ACCUM_END_C_upsample(F_b, _stride_before, _C_ib, l_pad_el, _C_ob);
    }

    c_tile_t *c_cur = c_tile;
    // dim_t c_cur = 0;
    for (uint32_t k_p = 0; k_p < l_pad_el; k_p++)
    {
        compute_with_padding_1D<ScalarT, AccumT,
                                _G_b, _K_b, _F_cb, _O_wb, _stride,
                                _UNROLL, op_type, op_class>(
                                    //H_lb, H_UPPER,
                                    W_i_valid, F_w,
                                    F_w,
                                    1,
                                    input_col_stride,
                                    F,
                                    I_ptr,
                                    c_cur);

        c_cur += (_K_b * _G_b) / (FLOAT_SIMD_EPILOGUE);
        // c_cur += 1;
        W_i_valid -= _stride;
        // I_ptr += ()*(_stride * _F_cb * _G_b);
    }
    // Fusion Slot # 1
    //  Include division for Average Pooling e.g. fused single element multiplication
    if (op_type == OP_AVERAGE_POOL)
    {
        float norm = 1.0 / (1.0 * F_w);
        FLOAT_DIV_END_C(c_tile, norm, l_pad_el, _C_ob);
    }

    dim_t step_after = _stride_after * _C_ib;
    FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_END(step_after, fused_single_element_after,
                                         0, F_a, c_tile, l_pad_el, _C_ob);

    FLOAT_STORE_END_C(O_ptr, l_pad_el, _C_ob);
    O_ptr += _G_b * _K_b;
}

}
}
