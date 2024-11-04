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

#include <small/float_detail/kernel_left_remainder.hpp>
#include <small/float_detail/kernel_pad_remainder.hpp>
#include <small/float_detail/kernel_right_remainder.hpp>

namespace small
{
namespace float_detail
{

//****************************************************************************
//****************************************************************************
// Kernel top for remainder channels
//****************************************************************************
//****************************************************************************
// Edge case to handle remainder output channels
template <typename ScalarT,
          typename AccumT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          OpType op_type,
          int8_t op_class>
void inline rem_kernel_top(
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
    AccumT *O,
    const dim_t G_left,
    const dim_t K_left) // ScalarT --> AccumT
{
    const dim_t _C_ob = G_left * K_left;
    const dim_t _C_ib = G_left * _F_cb;
    //const dim_t step = _stride * _C_ib;

    ScalarT const *I_ptr = I;
    AccumT *O_ptr = O; // ScalarT --> AccumT

    int H_i_valid = t_pad;

    for (uint32_t j_p = 0; j_p < t_pad_el; j_p++)
    {
        // Prologue with left padding
        rem_kernel_left<ScalarT, AccumT,
                        _G_b, _K_b, _F_cb, _O_wb, _stride,
                        _UNROLL, op_type, op_class>(
            first,
            F_h,
            F_w,
            input_col_stride,
            l_pad_el,
            l_pad,
            I_ptr,
            F,
            O_ptr,
            G_left,
            K_left,
            H_i_valid,
            F_h);

        ScalarT const *I_row_full = I + W_full_index * (_C_ib);
        AccumT *O_row_full = O + l_pad_el * (_C_ob); // ScalarT --> AccumT

        // Steady State with microkernel
        for (index_t l = 0; l < O_w_full; l += _O_wb)
        {
            ScalarT const *I_col =
                I_row_full + (l * _stride) * (_C_ib);
            ScalarT const *F_col = F + 0;
            AccumT *O_col = O_row_full + l * (_C_ob); // ScalarT --> AccumT

            rem_kernel_pad<ScalarT, AccumT,
                           _G_b, _K_b, _F_cb, _O_wb, _stride,
                           _UNROLL, op_type, op_class>(
                first,
                F_h,
                F_w,
                input_col_stride,
                I_col,
                F_col,
                O_col,
                G_left,
                K_left,
                H_i_valid, // H_lb
                F_h,       // H_ub
                0,         // W_lb
                0);        /// @todo Confirm this, W_ub. Is it right? q_abstract_layer has F_w
        }

        // Epilogue for microkernel + right padding elements
        ScalarT const *I_col_left =
            I_row_full + (O_w_full * _stride) * (_C_ib);
        ScalarT const *F_col_left = F + 0;
        AccumT *O_col_left =
            O_row_full + O_w_full * (_C_ob); // ScalarT --> AccumT

        rem_kernel_right<ScalarT, AccumT,
                         _G_b, _K_b, _F_cb, _O_wb, _stride,
                         _UNROLL, op_type, op_class>(
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
            G_left,
            K_left,
            H_i_valid,
            F_h);

        O_ptr += O_w_w_pad * _C_ob;
        H_i_valid += _stride;
    }
}

//****************************************************************************
// Edge case to handle remainder input channels
template <typename ScalarT,
          typename AccumT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          OpType op_type,
          int8_t op_class>
void inline kernel_top_rem(
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
    AccumT *O,
    const dim_t F_c_left) // ScalarT --> AccumT
{
    ScalarT const *I_ptr = I;
    AccumT *O_ptr = O; // ScalarT --> AccumT

    int H_i_valid = t_pad;

    for (uint32_t j_p = 0; j_p < t_pad_el; j_p++)
    {
        // Prologue with left padding
        kernel_left_rem<ScalarT, AccumT,
                        _G_b, _K_b, _F_cb, _O_wb, _stride,
                        _UNROLL, op_type, op_class>(
            first,
            F_h,
            F_w,
            input_col_stride,
            l_pad_el,
            l_pad,
            I_ptr,
            F,
            O_ptr,
            F_c_left,
            H_i_valid,
            F_h);

        ScalarT const *I_row_full = I + W_full_index * (F_c_left * _G_b);
        AccumT *O_row_full = O + l_pad_el * (_G_b * _K_b); // ScalarT --> AccumT

        // Steady State with microkernel
        for (index_t l = 0; l < O_w_full; l += _O_wb)
        {
            ScalarT const *I_col =
                I_row_full + (l * _stride) * (F_c_left * _G_b);
            ScalarT const *F_col = F + 0;
            AccumT *O_col = O_row_full + l * (_G_b * _K_b); // ScalarT --> AccumT

            kernel_pad_rem<ScalarT, AccumT,
                           _G_b, _K_b, _F_cb, _O_wb, _stride,
                           _UNROLL, op_type, op_class>(
                first,
                F_h,
                F_w,
                input_col_stride,
                I_col,
                F_col,
                O_col,
                F_c_left,
                H_i_valid, // H_lb
                F_h,       // H_ub
                0,         // W_lb
                0);        /// @todo Confirm this, W_ub. Is it right? q_abstract_layer has F_w
        }

        // Epilogue for microkernel + right padding elements
        ScalarT const *I_col_left =
            I_row_full + (O_w_full * _stride) * (F_c_left * _G_b);
        ScalarT const *F_col_left = F + 0;
        AccumT *O_col_left =
            O_row_full + O_w_full * (_G_b * _K_b); // ScalarT --> AccumT

        kernel_right_rem<ScalarT, AccumT,
                         _G_b, _K_b, _F_cb, _O_wb, _stride,
                         _UNROLL, op_type, op_class>(
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
            F_c_left,
            H_i_valid,
            F_h);

        O_ptr += O_w_w_pad * _K_b * _G_b;
        H_i_valid += _stride;
        // I_ptr += _stride * _F_cb * _G_b;
    }
}

//****************************************************************************
// Edge case to handle remainder input and output channels
template <typename ScalarT,
          typename AccumT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          OpType op_type,
          int8_t op_class>
void inline rem_kernel_top_rem(
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
    AccumT *O,
    const dim_t F_c_left,
    const dim_t G_left,
    const dim_t K_left) // ScalarT --> AccumT
{
    const dim_t _C_ob = G_left * K_left;
    const dim_t _C_ib = G_left * F_c_left;
    //const dim_t step = _stride * _C_ib;

    ScalarT const *I_ptr = I;
    AccumT *O_ptr = O; // ScalarT --> AccumT

    int H_i_valid = t_pad;

    for (uint32_t j_p = 0; j_p < t_pad_el; j_p++)
    {
        // Prologue with left padding
        rem_kernel_left_rem<ScalarT, AccumT,
                            _G_b, _K_b, _F_cb, _O_wb, _stride,
                            _UNROLL, op_type, op_class>(
            first,
            F_h,
            F_w,
            input_col_stride,
            l_pad_el,
            l_pad,
            I_ptr,
            F,
            O_ptr,
            F_c_left,
            G_left,
            K_left,
            H_i_valid,
            F_h);

        ScalarT const *I_row_full = I + W_full_index * (_C_ib);
        AccumT *O_row_full = O + l_pad_el * (_C_ob); // ScalarT --> AccumT

        // Steady State with microkernel
        for (index_t l = 0; l < O_w_full; l += _O_wb)
        {
            ScalarT const *I_col =
                I_row_full + (l * _stride) * (_C_ib);
            ScalarT const *F_col = F + 0;
            AccumT *O_col = O_row_full + l * (_C_ob); // ScalarT --> AccumT

            rem_kernel_pad_rem<ScalarT, AccumT,
                               _G_b, _K_b, _F_cb, _O_wb, _stride,
                               _UNROLL, op_type, op_class>(
                first,
                F_h,
                F_w,
                input_col_stride,
                I_col,
                F_col,
                O_col,
                F_c_left,
                G_left,
                K_left,
                H_i_valid, // H_lb
                F_h,       // H_ub
                0,         // W_lb
                0);        /// @todo Confirm this, W_ub. Is it right? q_abstract_layer has F_w
        }

        // Epilogue for microkernel + right padding elements
        ScalarT const *I_col_left =
            I_row_full + (O_w_full * _stride) * (_C_ib);
        ScalarT const *F_col_left = F + 0;
        AccumT *O_col_left =
            O_row_full + O_w_full * (_C_ob); // ScalarT --> AccumT

        rem_kernel_right_rem<ScalarT, AccumT,
                             _G_b, _K_b, _F_cb, _O_wb, _stride,
                             _UNROLL, op_type, op_class>(
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
            F_c_left,
            G_left,
            K_left,
            H_i_valid,
            F_h);

        O_ptr += O_w_w_pad * _C_ob;
        H_i_valid += _stride;
        // I_ptr += _stride * _F_cb * _G_b;
    }
}

} // ns float_detail
} // ns small
