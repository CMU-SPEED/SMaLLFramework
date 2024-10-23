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

#include <small/kernel_left.hpp>
#include <small/kernel_pad.hpp>
#include <small/kernel_right.hpp>

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
void inline kernel_bottom(
    bool first,
    dim_t F_h,
    dim_t F_w,
    dim_t input_col_stride,
    dim_t b_pad_el,
    dim_t b_pad,
    dim_t W_full_index,
    dim_t l_pad_el,
    dim_t l_pad,
    dim_t O_w_w_pad,
    dim_t O_w_full,
    dim_t O_w_left,
    dim_t r_pad_el,
    dim_t r_pad,
    ScalarT const *I,
    ScalarT const *F,
    AccumT *O,
    ScalarT const * F_b = NULL,
    ScalarT const * F_a = NULL) // ScalarT -> AccumT
{
    ScalarT const *I_ptr = I;
    AccumT *O_ptr = O; // ScalarT -> AccumT

    int H_i_valid = F_h - 1;

    for (uint32_t j_p = 0; j_p < b_pad_el; j_p++)
    {
        // Prologue with left padding
        kernel_left<ScalarT, AccumT,
                    _G_b, _K_b, _F_cb, _O_wb, _stride,
                    _UNROLL, op_type, op_class, fused_single_element_before,
                    fused_single_element_after, _stride_before, _stride_after>(
                        first,
                        F_h,
                        F_w,
                        input_col_stride,
                        l_pad_el,
                        l_pad,
                        I_ptr,
                        F,
                        O_ptr,
                        0,
                        H_i_valid,
                        F_b,
                        F_a);

        ScalarT const *I_row_full = I + W_full_index * (_F_cb * _G_b);
        AccumT *O_row_full = O + l_pad_el * (_G_b * _K_b); // ScalarT -> AccumT
        // Steady State with microkernel
        for (index_t l = 0; l < O_w_full; l += _O_wb)
        {
            ScalarT const *I_col = I_row_full + (l * _stride) * (_F_cb * _G_b);
            ScalarT const *F_col = F + 0;
            AccumT *O_col = O_row_full + l * (_G_b * _K_b); // ScalarT -> AccumT

            kernel_pad<ScalarT, AccumT,
                       _G_b, _K_b, _F_cb, _O_wb, _stride,
                       _UNROLL, op_type, op_class, fused_single_element_before,
                       fused_single_element_after, _stride_before, _stride_after>(
                           first,
                           F_h,
                           F_w,
                           input_col_stride,
                           I_col,
                           F_col,
                           O_col,
                           0,
                           H_i_valid,
                           0,  /// @todo This was added, W_lb. Is it right?
                           0,
                           F_b,
                           F_a); /// @todo This was added, W_ub. Is it right?);
        }

        // Epilogue for microkernel + right padding elements
        ScalarT const *I_col_left =
            I_row_full + (O_w_full * _stride) * (_F_cb * _G_b);
        ScalarT const *F_col_left = F + 0;
        AccumT *O_col_left = O_row_full + O_w_full * (_G_b * _K_b); // ScalarT -> AccumT

        kernel_right<ScalarT, AccumT,
                     _G_b, _K_b, _F_cb, _O_wb, _stride,
                     _UNROLL, op_type, op_class, fused_single_element_before,
                     fused_single_element_after, _stride_before, _stride_after>(
                         first,
                         F_h,
                         F_w,
                         input_col_stride,
                         O_w_left,
                         r_pad_el,
                         r_pad,
                         I_col_left,
                         F_col_left,
                         O_col_left,
                         0,          /// @todo confirm this, H_lb
                         H_i_valid,
                         F_b,
                         F_a); /// @todo confirm this, H_ub

        O_ptr += O_w_w_pad * _K_b * _G_b;

        H_i_valid -= _stride;
        I_ptr += _stride * _F_cb * _G_b;
    }
}

}
}
