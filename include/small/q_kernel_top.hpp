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
#include <small/q_kernel_left.hpp>
#include <small/q_kernel_pad.hpp>
#include <small/q_kernel_right.hpp>

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
void inline kernel_top(
    bool first,
    dim_t F_h,
    dim_t F_w,
    dim_t input_col_stride,
    dim_t t_pad_el,
    dim_t t_pad,
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
    AccumT        *O,  // ScalarT --> AccumT
    int k_zero = 0,
    AccumT I_offset = 0,
    AccumT F_offset = 0,
    ScalarT *O_out = NULL,
    AccumT lshift = 0,
    AccumT rshift = 0,
    AccumT q_mul = 1,
    AccumT zero = 0)
{
    ScalarT const *I_ptr = I;
    AccumT        *O_ptr = O;  // ScalarT --> AccumT
    ScalarT       *O_ptr_out = O_out;
    int H_i_valid = t_pad;

    for (uint32_t j_p = 0; j_p < t_pad_el; j_p++)
    {
        // Prologue with left padding
        kernel_left<ScalarT, AccumT,
                    _G_b, _K_b, _F_cb, _O_wb, _stride,
                    _UNROLL, op_type, op_class, quantize, max_val, min_val>(
                        first,
                        F_h,
                        F_w,
                        input_col_stride,
                        l_pad_el,
                        l_pad,
                        I_ptr,
                        F,
                        O_ptr,
                        H_i_valid,
                        F_h,
                        k_zero,
                        I_offset,
                        F_offset,
                        O_ptr_out,
                        lshift,
                        rshift,
                        q_mul,
                        zero);

        ScalarT const *I_row_full = I + W_full_index * (_F_cb * _G_b);
        AccumT        *O_row_full = O + l_pad_el * (_G_b * _K_b);  // ScalarT --> AccumT
        ScalarT       *O_row_full_out = O_out + l_pad_el * (_G_b * _K_b);

        // Steady State with microkernel
        for (index_t l = 0; l < O_w_full; l += _O_wb)
        {
            ScalarT const *I_col =
                I_row_full + (l * _stride) * (_F_cb * _G_b);
            ScalarT const *F_col = F + 0;
            AccumT        *O_col = O_row_full + l * (_G_b * _K_b);  // ScalarT --> AccumT
            ScalarT       *O_col_out = O_row_full_out + l * (_G_b * _K_b);

            kernel_pad<ScalarT, AccumT,
                       _G_b, _K_b, _F_cb, _O_wb, _stride,
                       _UNROLL, op_type, op_class, quantize, max_val, min_val>(
                           first,
                           F_h,
                           F_w,
                           input_col_stride,
                           I_col,
                           F_col,
                           O_col,
                           H_i_valid,  // H_lb
                           F_h,        // H_ub
                           0,          // W_lb
                           F_w,        // W_ub
                           k_zero,
                           I_offset,
                           F_offset,
                           O_col_out,
                           lshift,
                           rshift,
                           q_mul,
                           zero);
        }

        // Epilogue for microkernel + right padding elements
        ScalarT const *I_col_left =
            I_row_full + (O_w_full * _stride) * (_F_cb * _G_b);
        ScalarT const *F_col_left = F + 0;
        AccumT        *O_col_left =
            O_row_full + O_w_full * (_G_b * _K_b);  // ScalarT --> AccumT
        ScalarT       *O_col_left_out =
            O_row_full_out + O_w_full * (_G_b * _K_b);

        kernel_right<ScalarT, AccumT,
                     _G_b, _K_b, _F_cb, _O_wb, _stride,
                     _UNROLL, op_type, op_class, quantize, max_val, min_val>(
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
                         H_i_valid,
                         F_h,
                         k_zero,
                         I_offset,
                         F_offset,
                         O_col_left_out,
                         lshift,
                         rshift,
                         q_mul,
                         zero);

        O_ptr += O_w_w_pad * _K_b * _G_b;
        O_ptr_out += O_w_w_pad * _K_b * _G_b;
        H_i_valid += _stride;
        // I_ptr += _stride * _F_cb * _G_b;
    }
}

} // ns quint8_detail
} // ns small
