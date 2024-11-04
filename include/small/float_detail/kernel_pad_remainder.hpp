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

namespace small
{
namespace float_detail
{

//****************************************************************************
//****************************************************************************
// Kernel pad for remainder channels
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
void inline rem_kernel_pad(
    bool first,
    dim_t F_h,
    dim_t F_w,
    dim_t input_col_stride,
    ScalarT const *I,
    ScalarT const *F,
    AccumT *O, // ScalarT -> AccumT
    const dim_t G_left,
    const dim_t K_left,
    dim_t H_lb = 0,
    dim_t H_ub = 0,
    dim_t W_lb = 0,
    dim_t W_ub = 0)
{
    const dim_t _C_ob = G_left * K_left;
    const dim_t _C_ib = G_left * _F_cb;
    const dim_t step = _stride * _C_ib;

    const dim_t H_UPPER = ((!H_ub) * (F_h)) + (H_ub);
    // const dim_t W_UPPER = ((!W_ub) * (F_w)) + (W_ub);

    FLOAT_DEF_END_C(_O_wb, _C_ob);
    if (first)
    {
        FLOAT_ZERO_END_C(_O_wb, _C_ob);

        //@note padding  should always be 'v' for pointwise operations, so this code path should not be used
        if (op_type == OP_MUL)
        {
            FLOAT_LOAD_END_C_strided(I, step, _O_wb, _C_ob);
        }
    }
    else
    {
        FLOAT_LOAD_END_C(O, _O_wb, _C_ob);
    }

    for (uint32_t n = H_lb; n < H_UPPER; n++)
    {
        int filter_offset_h = n * F_w * _F_cb * _C_ob;
        int input_stencil_h = /*input_col_offset + input_row_offset +*/
            (n - H_lb) * input_col_stride;

        for (uint32_t m = 0; m < F_w; m++)
        {
            int filter_offset_w = m * _F_cb * _C_ob + filter_offset_h;
            // This is C_ob because the microkernel stretches across groups
            int input_stencil_w = (m - W_lb) * _C_ib + input_stencil_h;

            ScalarT const *b = F + filter_offset_w;
            ScalarT const *a = I + input_stencil_w;

            for (uint32_t ii = 0; ii < _F_cb / _UNROLL; ii++)
            {
                /// @note using platform C_ob
                ScalarT const *b_cur = b + ii * _UNROLL * _C_ob;
                ScalarT const *a_cur = a + ii * _UNROLL;

                FLOAT_ABSTRACT_OP_END(step, op_type, op_class, a_cur, b_cur, c_tile, _O_wb, _C_ob);
            }
        }
    }
    if (op_type == OP_AVERAGE_POOL)
    {
        float norm = 1.0 / (1.0 * F_h * F_w);
        FLOAT_DIV_END_C(c_tile, norm, _O_wb, _C_ob);
    }
    FLOAT_STORE_END_C(O, _O_wb, _C_ob);
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
void inline kernel_pad_rem(
    bool first,
    dim_t F_h,
    dim_t F_w,
    dim_t input_col_stride,
    ScalarT const *I,
    ScalarT const *F,
    AccumT *O, // ScalarT -> AccumT
    const dim_t F_c_left,
    dim_t H_lb = 0,
    dim_t H_ub = 0,
    dim_t W_lb = 0,
    dim_t W_ub = 0)
{
    constexpr dim_t _C_ob = _G_b * _K_b;
    const dim_t _C_ib = _G_b * F_c_left;
    const dim_t step = _stride * _C_ib;

    const dim_t H_UPPER = ((!H_ub) * (F_h)) + (H_ub);
    // const dim_t W_UPPER = ((!W_ub) * (F_w)) + (W_ub);

    FLOAT_DEF_TILE_C(_O_wb, _C_ob);
    if (first)
    {
        FLOAT_ZERO_TILE_C(_O_wb, _C_ob);

        //@note padding  should always be 'v' for pointwise operations, so this code path should not be used
        if (op_type == OP_MUL)
        {
            FLOAT_LOAD_TILE_C_strided(I, step, _O_wb, _C_ob);
        }
    }
    else
    {
        FLOAT_LOAD_TILE_C(O, _O_wb, _C_ob);
    }

    for (uint32_t n = H_lb; n < H_UPPER; n++)
    {
        int filter_offset_h = n * F_w * F_c_left * _G_b * _K_b;
        int input_stencil_h = /*input_col_offset + input_row_offset +*/
            (n - H_lb) * input_col_stride;

        for (uint32_t m = 0; m < F_w; m++)
        {
            int filter_offset_w = m * F_c_left * _G_b * _K_b + filter_offset_h;
            // This is C_ob because the microkernel stretches across groups
            int input_stencil_w = (m - W_lb) * _C_ib + input_stencil_h;

            ScalarT const *b = F + filter_offset_w;
            ScalarT const *a = I + input_stencil_w;

            for (uint32_t ii = 0; ii < F_c_left / _UNROLL; ii++)
            {
                /// @note using platform C_ob
                ScalarT const *b_cur = b + ii * _UNROLL * _C_ob;
                ScalarT const *a_cur = a + ii * _UNROLL;

                FLOAT_ABSTRACT_OP(step, op_type, op_class, a_cur, b_cur, _O_wb, _C_ob);
            }
        }
    }
    if (op_type == OP_AVERAGE_POOL)
    {
        float norm = 1.0 / (1.0 * F_h * F_w);
        FLOAT_DIV_TILE_C(norm, _O_wb, _C_ob);
    }
    FLOAT_STORE_TILE_C(O, _O_wb, _C_ob);
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
void inline rem_kernel_pad_rem(
    bool first,
    dim_t F_h,
    dim_t F_w,
    dim_t input_col_stride,
    ScalarT const *I,
    ScalarT const *F,
    AccumT *O, // ScalarT -> AccumT
    const dim_t F_c_left,
    const dim_t G_left,
    const dim_t K_left,
    dim_t H_lb = 0,
    dim_t H_ub = 0,
    dim_t W_lb = 0,
    dim_t W_ub = 0)
{
    const dim_t _C_ob = G_left * K_left;
    const dim_t _C_ib = G_left * F_c_left;
    const dim_t step = _stride * _C_ib;

    const dim_t H_UPPER = ((!H_ub) * (F_h)) + (H_ub);
    // const dim_t W_UPPER = ((!W_ub) * (F_w)) + (W_ub);

    FLOAT_DEF_END_C(_O_wb, _C_ob);
    if (first)
    {
        FLOAT_ZERO_END_C(_O_wb, _C_ob);

        //@note padding  should always be 'v' for pointwise operations, so this code path should not be used
        if (op_type == OP_MUL)
        {
            FLOAT_LOAD_END_C_strided(I, step, _O_wb, _C_ob);
        }
    }
    else
    {
        FLOAT_LOAD_END_C(O, _O_wb, _C_ob);
    }

    for (uint32_t n = H_lb; n < H_UPPER; n++)
    {
        int filter_offset_h = n * F_w * F_c_left * G_left * K_left;
        int input_stencil_h = /*input_col_offset + input_row_offset +*/
            (n - H_lb) * input_col_stride;

        for (uint32_t m = 0; m < F_w; m++)
        {
            int filter_offset_w = m * F_c_left * G_left * K_left + filter_offset_h;
            // This is C_ob because the microkernel stretches across groups
            int input_stencil_w = (m - W_lb) * _C_ib + input_stencil_h;

            ScalarT const *b = F + filter_offset_w;
            ScalarT const *a = I + input_stencil_w;

            for (uint32_t ii = 0; ii < F_c_left / _UNROLL; ii++)
            {
                /// @note using platform C_ob
                ScalarT const *b_cur = b + ii * _UNROLL * _C_ob;
                ScalarT const *a_cur = a + ii * _UNROLL;

                FLOAT_ABSTRACT_OP_END(step, op_type, op_class, a_cur, b_cur, c_tile, _O_wb, _C_ob);
            }
        }
    }
    if (op_type == OP_AVERAGE_POOL)
    {
        float norm = 1.0 / (1.0 * F_h * F_w);
        FLOAT_DIV_END_C(c_tile, norm, _O_wb, _C_ob);
    }
    FLOAT_STORE_END_C(O, _O_wb, _C_ob);
    // prinf the stored _O_wb x _C_ob output tile
}

} // ns float_detail
} // ns small
