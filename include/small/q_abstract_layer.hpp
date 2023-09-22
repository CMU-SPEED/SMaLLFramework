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

#include <small/op_type.hpp>
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
/// @todo add parameters: step, _O_wb, _C_ob
/// @todo use constexpr if on op_type and op_class in calling code
#define QUINT8_ABSTRACT_OP(op_type, op_class, a_cur, b_cur, a_offset, b_offset) \
    if constexpr (op_type == 'c')                                       \
    {                                                                   \
        if constexpr (op_class == 1)                                    \
        {                                                               \
            QUINT8_DW_TILE_C(step, a_cur, b_cur, _O_wb, _C_ob, a_offset, b_offset); \
        }                                                               \
        else if constexpr (op_class == 2)                               \
        {                                                               \
            QUINT8_CONV_TILE_C(step, a_cur, b_cur, _O_wb, _C_ob, a_offset, b_offset); \
        }                                                               \
    }                                                                   \
    else if constexpr (op_type == 'a' || op_type == 'p')                \
    {                                                                   \
        QUINT8_MAX_TILE_C(step, a_cur, _O_wb, _C_ob, a_offset);         \
    }                                                                   \
    else if constexpr (op_type == 'l')                                  \
    {                                                                   \
        /*QUINT8_COND_SCALE_TILE_C(step, a_cur, b_cur, _O_wb, _C_ob);*/ \
        throw std::invalid_argument("*_ABSTRACT_OP ERROR: no support for op_type 'l'"); \
    }                                                                   \
    else if constexpr (op_type == 'd')                                  \
    {                                                                   \
        /*QUINT8_ACCUM_TILE_C(step, a_cur, _O_wb, _C_ob);*/             \
        throw std::invalid_argument("*_ABSTRACT_OP ERROR: no support for op_type 'd'"); \
    }

//****************************************************************************
/// @todo add parameters: step, W_elements, _C_ob
#define QUINT8_ABSTRACT_OP_END(op_type, op_class, a_cur, b_cur, c_cur, a_offset, b_offset) \
    if constexpr (op_type == 'c')                                       \
    {                                                                   \
        if constexpr (op_class == 1)                                    \
        {                                                               \
            QUINT8_DW_END_C(step, a_cur, b_cur, c_cur, W_elements, _C_ob, a_offset, b_offset); \
        }                                                               \
        else if constexpr (op_class == 2)                               \
        {                                                               \
            QUINT8_CONV_END_C(step, a_cur, b_cur, c_cur, W_elements, _C_ob, a_offset, b_offset); \
        }                                                               \
    }                                                                   \
    else if constexpr (op_type == 'a' || op_type == 'p')                \
    {                                                                   \
        QUINT8_MAX_END_C(step, a_cur, c_cur, W_elements, _C_ob, a_offset); \
    }                                                                   \
    else if constexpr (op_type == 'l')                                  \
    {                                                                   \
        /*QUINT8_COND_SCALE_END_C(step, a_cur, b_cur, c_cur, W_elements, _C_ob, a_offset, b_offset);*/ \
        throw std::invalid_argument("*_ABSTRACT_OP_END ERROR: no support for op_type 'l'"); \
    }                                                                   \
    else if constexpr (op_type == 'd')                                  \
    {                                                                   \
        /*QUINT8_ACCUM_END_C(step, a_cur, c_cur, W_elements, _C_ob, a_offset);*/ \
        throw std::invalid_argument("*_ABSTRACT_OP_END ERROR: no support for op_type 'd'"); \
    }

//****************************************************************************
template <typename ScalarT,
          typename AccumT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          // TODO: add a bool to switch between microkernel and default imp
          // Leaf to describe abstract operation
          char   op_type,
          int8_t op_class>
void inline compute_with_padding(dim_t H_lb, dim_t H_ub,
                                 dim_t W_lb, dim_t W_ub,
                                 dim_t F_w,
                                 dim_t W_elements,
                                 dim_t input_col_stride,
                                 ScalarT const *F,
                                 ScalarT const *I,
                                 c_tile_t *c_cur, /// @todo need to deal with type
                                 AccumT I_offset = 0,
                                 AccumT F_offset = 0)
{
    constexpr dim_t _C_ob = _G_b * _K_b;
    constexpr dim_t _C_ib = _G_b * _F_cb;
    constexpr dim_t step = _stride * _C_ib;
    for (uint32_t n = H_lb; n < H_ub; n++)
    {
        int filter_offset_h = n * F_w * _F_cb * _G_b * _K_b;
        int input_stencil_h = (n - H_lb) * input_col_stride; /*+ input_col_offset + input_row_offset*/

        for (uint32_t m = W_lb; m < W_ub; m++)
        {
            int filter_offset_w = m * _F_cb * _G_b * _K_b + filter_offset_h;
            /* This is C_ib because the microkernel stretches across groups*/
            int input_stencil_w = (m - W_lb) * _C_ib + input_stencil_h;

            ScalarT const *b = F + filter_offset_w;
            ScalarT const *a = I + input_stencil_w;

            // TODO: reintroduce convolution
            for (uint32_t ii = 0; ii < _F_cb / _UNROLL; ii++)
            {
                /// @note using platform C_ob
                ScalarT const *b_cur = b + ii * _UNROLL * QUINT8_C_ob;
                ScalarT const *a_cur = a + ii * _UNROLL;
                QUINT8_ABSTRACT_OP_END(op_type, op_class, a_cur, b_cur, c_cur,
                                       I_offset, F_offset);
            }
        }
    }
}

//****************************************************************************
template <typename ScalarT,
          typename AccumT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          char   op_type,
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

//****************************************************************************
template <typename ScalarT,
          typename AccumT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          char   op_type,
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
        if (op_type == 'p')
        {
            /// @note using platform C_ob
            QUINT8_LOAD_TILE_C_strided(I, step, _O_wb, QUINT8_C_ob);
        }
        else if (op_type == 'u')
        {
            //QUINT8_LOAD_TILE_C_upsample(I, _stride, _C_ib, _O_wb, _C_ob);
            throw std::invalid_argument("*kernel ERROR: "
                                        "no support for op_type 'u'.");

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

//****************************************************************************
// TODO: Explain the difference between kernel and kernel_pad
template <typename ScalarT,
          typename AccumT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          char   op_type,
          int8_t op_class,
          bool   quantize = false,
          ScalarT max_val = 255, // std::numeric_limits<ScalarT>::max()
          ScalarT min_val = 0>   // std::numeric_limits<ScalarT>::lowest()
void inline kernel_pad(
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
    int k_zero = 0,  /// @todo why both zero and k_zero?  Should this be AccumT?
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
    }
    else
    {
        QUINT8_LOAD_TILE_C(O, _O_wb, _C_ob);
    }

    // int updates = 0;
    // uint32_t step = _C_ob;//stride*_C_ob;
    // int count = 0;
    for (uint32_t n = H_lb; n < H_UPPER; n++)
    {
        int filter_offset_h = n * F_w * _F_cb * _G_b * _K_b;
        int input_stencil_h = /*input_col_offset + input_row_offset +*/
            (n - H_lb) * input_col_stride;

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
                                   I_offset, F_offset);
            }
        }
    }

    if constexpr (quantize)
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

//****************************************************************************
template <typename ScalarT,
          typename AccumT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          char   op_type,
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

            if (op_type == 'p' && H_lb == 0 && H_ub == 0)
            {
                QUINT8_LOAD_END_C_strided(I, step, O_w_left, _C_ob);
            }
            else if (op_type == 'u')
            {
                //QUINT8_LOAD_END_C_upsample(I, _stride, _C_ib, O_w_left, _C_ob);
                throw std::invalid_argument("*kernel_right ERROR: "
                                            "no support for op_type 'u'.");
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

        // if (op_type=='p')
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

//****************************************************************************
template <typename ScalarT,
          typename AccumT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          char   op_type,
          int8_t op_class,
          bool   quantize = false,
          ScalarT max_val = 255, // std::numeric_limits<ScalarT>::max()
          ScalarT min_val = 0>   // std::numeric_limits<ScalarT>::lowest()
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
    AccumT        *O,  // ScalarT -> AccumT
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
    AccumT        *O_ptr = O;  // ScalarT -> AccumT
    ScalarT       *O_ptr_out = O_out;

    int H_i_valid = F_h - 1;

    for (uint32_t j_p = 0; j_p < b_pad_el; j_p++)
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
                        0,
                        H_i_valid,
                        k_zero,
                        I_offset,
                        F_offset,
                        O_ptr_out,
                        lshift,
                        rshift,
                        q_mul,
                        zero);

        ScalarT const *I_row_full = I + W_full_index * (_F_cb * _G_b);
        AccumT        *O_row_full = O + l_pad_el * (_G_b * _K_b);  // ScalarT -> AccumT
        ScalarT       *O_row_full_out = O_out + l_pad_el * (_G_b * _K_b);
        // Steady State with microkernel
        for (index_t l = 0; l < O_w_full; l += _O_wb)
        {
            ScalarT const *I_col = I_row_full + (l * _stride) * (_F_cb * _G_b);
            ScalarT const *F_col = F + 0;
            AccumT        *O_col = O_row_full + l * (_G_b * _K_b);  // ScalarT -> AccumT
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
                           0,
                           H_i_valid,
                           0,
                           F_w,
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
        AccumT        *O_col_left = O_row_full + O_w_full * (_G_b * _K_b);  // ScalarT -> AccumT
        ScalarT       *O_col_left_out = O_row_full_out + O_w_full * (_G_b * _K_b);

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
                         0,
                         H_i_valid,
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

        H_i_valid -= _stride;
        I_ptr += _stride * _F_cb * _G_b;
    }
}

//****************************************************************************
template <typename ScalarT,
          typename AccumT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          char   op_type,
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

//****************************************************************************
//****************************************************************************

//****************************************************************************
template <typename BufferT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          char   op_type,  // 'c' (conv,dense), 'p' (pool), or 'a' (activation)
          int8_t op_class, //  2  (conv),  1  (dense,pool), or '0' (activation)
          bool   rewrite_output>  // 0 (partial conv), 1 (otherwise)
void abstract_layer(
    dim_t G,   // Output Channel Grouping
    dim_t K,   // Output Channels per group
    dim_t F_c, // Channel Reduction Dimension
    dim_t I_h, // Input Height
    dim_t I_w, // Input Width

    dim_t F_h, // Filter height
    dim_t F_w, // Filter width

    dim_t pad_top,    // Padding values
    dim_t pad_left,
    dim_t pad_right,
    dim_t pad_bottom,

    BufferT const *__restrict__ I,    // Data
    BufferT const *__restrict__ F,
    BufferT       *__restrict__ O,

    /// @todo should this type be 'typename BufferT::accum_type'?
    /// @todo should this be moved inside function and access BufferT::zero?
    int k_zero = 0)
{
    using ScalarT = typename BufferT::value_type;
    using AccumT  = typename BufferT::accum_type;

    // Pointers to buffers inside Buffer class
    ScalarT const *I_buf = I->data();  //__restrict__ ?
    AccumT         I_offset = I->zero();

    ScalarT const *F_buf = nullptr;
    AccumT         F_offset(0);
    if constexpr (op_type == 'c' ||  op_type == 'l')  // if (F != nullptr)
    {
        F_buf = F->data();
        F_offset = F->zero();
    }

    ScalarT *O_buf = O->data();  //__restrict__ ?

    //============QUANTIZED==============
    AccumT        lshift = O->lshift;
    AccumT        rshift = O->rshift;
    AccumT         q_mul = O->multiplier;
    AccumT        offset = O->zero();
    //============QUANTIZED==============

#if DEBUG == 1
    if (op_type == 'c')
    {
        printf("conv class: %d \n", op_class);
    }
    else if (op_type == 'p')
    {
        printf("pool class: %d \n", op_class);
    }
    else if (op_type == 'a')
    {
        printf("activation class: %d \n", op_class);
    }
#endif

    // calculate output dimensions based on input params.
    constexpr dim_t _C_ib = _F_cb * _G_b;

    /*
     * Data layout (slowest to fastest changing dimensions):
     *    blocks of groups
     *       blocks of channels within groups
     *          blocks of weights in the same group
     *             spatial dimensions
     *                weights in the same group
     *                   weights across groups in a block
     *                      channels in a block
     *
     * I: [G/G_b,        F_c/F_cb, I_h, I_w, F_cb, G_b     ]
     * F: [G/G_b, K/K_b, F_c/F_cb, F_h, F_w, F_cb, G_b, K_b]
     * O: [G/G_b, K/K_b,           O_h, O_w,       G_b, K_b]
     */

    //************************************************************************
    // Deriving padding parameters

    //  To calculate offsets to next output row, next output block
    // @todo fix this in small::output_dim
    dim_t H_o_w_pad, W_o_w_pad;
    if constexpr (op_type == 'u')
    {
        /// @todo VERIFY Copied from float version
        if constexpr(_stride == std::numeric_limits<dim_t>::max())
        {
            H_o_w_pad = I_h;
            W_o_w_pad = I_w;
        }
        else
        {
            H_o_w_pad = I_h * _stride;
            W_o_w_pad = I_w * _stride;
        }
    }
    else
    {
        H_o_w_pad = small::output_dim((I_h + pad_top + pad_bottom),
                                      _stride, F_h);
        W_o_w_pad = small::output_dim((I_w + pad_left + pad_right),
                                      _stride, F_w);
    }
    const dim_t O_h_w_pad = H_o_w_pad;
    const dim_t O_w_w_pad = W_o_w_pad;

    dim_t t_pad_el = pad_top / _stride + (pad_top % _stride != 0);
    dim_t l_pad_el = pad_left / _stride + (pad_left % _stride != 0);

    dim_t H_full_index = t_pad_el * _stride - pad_top;
    dim_t W_full_index = l_pad_el * _stride - pad_left;

    // Full kernel output elements
    dim_t H_o, W_o_full;
    if constexpr (op_type == 'u')
    {
        /// @todo VERIFY Copied from float version
        H_o = H_o_w_pad;
        W_o_full = W_o_w_pad;
    }
    else
    {
        H_o      = small::output_dim((I_h - H_full_index), _stride, F_h);
        W_o_full = small::output_dim((I_w - W_full_index), _stride, F_w);
    }

    // back padding elements
    dim_t H_back_index = H_full_index + _stride * (H_o);
    dim_t W_back_index = W_full_index + _stride * (W_o_full);
    dim_t b_pad_el, r_pad_el;
    if constexpr (op_type == 'u')
    {
        /// @todo VERIFY Copied from float version
        b_pad_el = 0;
        r_pad_el = 0;
    }
    else
    {
        b_pad_el = small::output_dim((I_h + pad_bottom - H_back_index),
                                     _stride, F_h);
        r_pad_el = small::output_dim((I_w + pad_right - W_back_index),
                                     _stride, F_w);
    }


    const dim_t O_h = H_o;
    const dim_t O_w = W_o_full;
    //************************************************************************

    // setting up microkernel specific parameters
    const dim_t O_w_full = (O_w / _O_wb) * _O_wb;
    const dim_t O_w_left = O_w - O_w_full;
    const dim_t O_hxO_w = O_h_w_pad * O_w_w_pad;

#if DEBUG == 1
    printf("\t\t I_h %d I_w %d F_C %d G %d \n", I_h, I_w, F_c, G);
    printf("\t\t O_h_pad: %d O_w_w_pad %d \n", O_h_w_pad, O_w_w_pad);
    printf("O_h %d O_w %d O_w_left %d \n", O_h, O_w_full, O_w_left);

    printf("bottom padding index into input: %d \t bottom padding elements: %d \n",
           H_back_index, b_pad_el);
    printf("no padding index into input: %d \t top padding elements: %d \n",
           H_full_index, t_pad_el);
    printf("right padding index into input: %d \t right padding elements: %d \n",
           W_back_index, r_pad_el);
    printf("no padding index into input: %d \t left padding elements: %d \n",
           W_full_index, l_pad_el);
    printf("O_w_full: %d O_w_left: %d \n", O_w_full, O_w_left);
    printf("params: F_Cb %d G_b %d K_b %d\n", _F_cb, _G_b, _K_b);
    printf("rewrite output?: %d, op type/class:  %c/%d\n",
           rewrite_output, op_type, op_class);
#endif

    // Set up parallelism for the channel loops

    //  Get total available threads
    int N = 1;
#if PARALLEL == 1
    char const *env_nt(std::getenv("OMP_NUM_THREADS"));
    if (nullptr != env_nt)
    {
        N = atoi(std::getenv("OMP_NUM_THREADS"));
    }
#endif

    int T_channel = N, T_group = 1, T_height = 1;

    // If dwise, parallelize on groups
    if (K == 1)
    {
        T_channel = 1;
        T_group = N;
    }

    // create parallel region with all threads
#if PARALLEL == 1
#pragma omp parallel num_threads(N)
#endif
    {
#if PARALLEL == 1
        auto t_id = omp_get_thread_num();
#else
        auto t_id = 0;
#endif
        auto height_tid = t_id % T_height;
        auto channel_tid = ((t_id) / (T_height)) % T_channel;
        auto group_tid = ((t_id / (T_channel * T_height))) % T_group;

        // loops over output channels
        for (index_t g = group_tid; g < G / _G_b; g += T_group)
        {
            ScalarT const *I_group;
            if constexpr (op_type == 'u' && _stride == std::numeric_limits<dim_t>::max())
            {
                /// @todo VERIFY Copied from float version
                I_group = I_buf + g * (F_c * 1 * 1* _G_b);
            }
            else
            {
                I_group = I_buf + g * (F_c * I_h * I_w * _G_b);
            }
            ScalarT       *O_group = O_buf + g * (K * O_hxO_w * _G_b);
            //if leaky relu, the weight pointer does not change with the group id

            ScalarT const *F_group ;
            if constexpr (op_type == 'l')
            {
                /// @todo VERIFY Copied from float version
                F_group = F_buf;
            }
            else
            {
                F_group = F_buf + g * (K * F_c * F_h * F_w * _G_b);
            }

            // resuse O_group as a uint32_t array
            for (index_t k = channel_tid; k < K / _K_b; k += T_channel)
            {
                ScalarT const *I_channel_block_output =
                    I_group + 0;
                ScalarT const *F_channel_block_output =
                    F_group + k * (F_c * F_h * F_w * _G_b * _K_b);
                ScalarT       *O_channel_block_output =
                    O_group + k * (O_hxO_w * _G_b * _K_b);

                //============QUANTIZED==============
                AccumT *O_channel_block_accumulator = (AccumT *)O_channel_block_output;
                //============QUANTIZED==============

                //************************************************************
                // Loop over input channel reduction
                for (index_t i = 0; i < (F_c / _F_cb) - 1; i++)
                {
                    bool first = rewrite_output && (i == 0);

                    ScalarT const *I_channel_block_input =
                        I_channel_block_output + i * (I_h * I_w * _F_cb * _G_b);
                    ScalarT const *F_channel_block_input =
                        F_channel_block_output + i * (F_h * F_w * _F_cb * _G_b * _K_b);
                    AccumT        *O_channel_block_input =
                        O_channel_block_accumulator + 0;

                    // Loops over spatial dimensions of output

                    // Prologue with top padding
                    ScalarT const *I_row_top = I_channel_block_input;
                    ScalarT const *F_row_top = F_channel_block_input + 0;
                    AccumT        *O_row_top = O_channel_block_input;  // ScalarT --> AccumT

                    kernel_top<ScalarT, AccumT,
                               _G_b, _K_b, _F_cb, _O_wb, _stride,
                               _UNROLL, op_type, op_class>(
                                   first,
                                   F_h,
                                   F_w,
                                   I_w * _C_ib,
                                   t_pad_el,
                                   pad_top,
                                   W_full_index,
                                   l_pad_el,
                                   pad_left,
                                   O_w_w_pad,
                                   O_w_full,
                                   O_w_left,
                                   r_pad_el,
                                   pad_right,
                                   I_row_top,
                                   F_row_top,
                                   O_row_top,
                                   k_zero,
                                   I_offset,
                                   F_offset);

                    ScalarT const *I_row_full =
                        I_row_top + H_full_index * I_w * (_F_cb * _G_b);
                    AccumT        *O_row_full =
                        O_row_top + t_pad_el * O_w_w_pad * (_G_b * _K_b); // ScalarT --> AccumT

                    // Steady State over rows
                    for (index_t j = height_tid; j < O_h; j += T_height)
                    {
                        ScalarT const *I_row;
                        // @todo cast index calculation as int and make stride a float value.
                        //I_x = I_x + (int)(j * _stride) * (<remaining dimensions>)
                        if constexpr (op_type == 'u')
                        {
                            /// @todo VERIFY Copied from float version
                            I_row = I_row_full + (j / _stride) * (I_w * _F_cb * _G_b);
                        }
                        else
                        {
                            I_row = I_row_full + (j * _stride) * (I_w * _F_cb * _G_b);
                        }
                        ScalarT const *F_row = F_channel_block_input + 0;
                        AccumT        *O_row =
                            O_row_full + j * (O_w_w_pad * _G_b * _K_b); // ScalarT --> AccumT
                        // Prologue with left padding
                        kernel_left<ScalarT, AccumT,
                                    _G_b, _K_b, _F_cb, _O_wb, _stride,
                                    _UNROLL, op_type, op_class>(
                                        first,
                                        F_h,
                                        F_w,
                                        I_w * _C_ib,
                                        l_pad_el,
                                        pad_left,
                                        I_row,
                                        F_row,
                                        O_row,
                                        0,
                                        0,
                                        k_zero,
                                        I_offset,
                                        F_offset);

                        ScalarT const *I_col_full =
                            I_row + W_full_index * (_F_cb * _G_b);
                        AccumT        *O_col_full = O_row + l_pad_el * (_G_b * _K_b); // ScalarT --> AccumT
                        // Steady State with microkernel
                        for (index_t l = 0; l < O_w_full; l += _O_wb)
                        {
                            ScalarT const *I_col;
                            // @todo cast index calculation as int and make stride a float value.
                            //I_x = I_x + (int)(j * _stride) * (<remaining dimensions>)
                            if constexpr (op_type == 'u')
                            {
                                /// @todo VERIFY Copied from float version
                                I_col = I_col_full + (l / _stride) * (_F_cb * _G_b);
                            }
                            else
                            {
                                I_col = I_col_full + (l * _stride) * (_F_cb * _G_b);
                            }
                            ScalarT const *F_col = F_row + 0;
                            AccumT        *O_col = O_col_full + l * (_G_b * _K_b); // ScalarT --> AccumT

                            kernel<ScalarT, AccumT,
                                   _G_b, _K_b, _F_cb, _O_wb, _stride,
                                   _UNROLL, op_type, op_class>(
                                       first,
                                       F_h,
                                       F_w,
                                       I_w * _C_ib,
                                       I_col,
                                       F_col,
                                       O_col,
                                       0,
                                       0,
                                       0,
                                       0,
                                       k_zero,
                                       I_offset,
                                       F_offset);
                        }

                        // Epilogue for microkernel + right padding elements
                        ScalarT const *I_col_left;
                        if constexpr (op_type == 'u')
                        {
                            /// @todo VERIFY Copied from float version
                            I_col_left =
                                I_col_full + (O_w_full / _stride) * (_F_cb * _G_b);
                        }
                        else
                        {
                            I_col_left =
                                I_col_full + (O_w_full * _stride) * (_F_cb * _G_b);
                        }

                        ScalarT const *F_col_left = F_row + 0;
                        AccumT        *O_col_left = O_col_full + O_w_full * (_G_b * _K_b); // ScalarT --> AccumT

                        kernel_right<ScalarT, AccumT,
                                     _G_b, _K_b, _F_cb, _O_wb, _stride,
                                     _UNROLL, op_type, op_class>(
                                         first,
                                         F_h,
                                         F_w,
                                         I_w * _C_ib,
                                         O_w_left,
                                         r_pad_el,
                                         pad_right,
                                         I_col_left,
                                         F_col_left,
                                         O_col_left,
                                         0,
                                         0,
                                         k_zero,
                                         I_offset,
                                         F_offset);
                    }
                    // Epilogue with bottom padding
                    ScalarT const *I_row_bot =
                        I_row_full + (O_h * _stride) * (I_w * _F_cb * _G_b);
                    ScalarT const *F_row_bot = F_channel_block_input + 0;
                    AccumT        *O_row_bot = O_row_full + O_h * (O_w_w_pad * _G_b * _K_b); // ScalarT --> AccumT

                    kernel_bottom<ScalarT, AccumT,
                                  _G_b, _K_b, _F_cb, _O_wb, _stride,
                                  _UNROLL, op_type, op_class>(
                                      first,
                                      F_h,
                                      F_w,
                                      I_w * _C_ib,
                                      b_pad_el,
                                      pad_bottom,
                                      W_full_index,
                                      l_pad_el,
                                      pad_left,
                                      O_w_w_pad,
                                      O_w_full,
                                      O_w_left,
                                      r_pad_el,
                                      pad_right,
                                      I_row_bot,
                                      F_row_bot,
                                      O_row_bot,
                                      k_zero,
                                      I_offset,
                                      F_offset);
                }

                //************************************************************
                // Loop over the last of the input channel reduction
                // The last iterations produce ScalarT's instead of AccumT's.
                for (index_t i = (F_c / _F_cb) - 1; i < (F_c / _F_cb); i++)
                {
                    bool first = rewrite_output && (i == 0);

                    ScalarT const *I_channel_block_input =
                        I_channel_block_output + i * (I_h * I_w * _F_cb * _G_b);
                    ScalarT const *F_channel_block_input =
                        F_channel_block_output + i * (F_h * F_w * _F_cb * _G_b * _K_b);
                    AccumT        *O_channel_block_input = // ScalarT --> AccumT
                        O_channel_block_accumulator + 0;
                    ScalarT * O_channel_block_input_out = O_channel_block_output;

                    // Loops over spatial dimensions of output

                    // Prologue with top padding
                    ScalarT const *I_row_top = I_channel_block_input;
                    ScalarT const *F_row_top = F_channel_block_input + 0;
                    AccumT        *O_row_top = O_channel_block_input; // ScalarT --> AccumT
                    ScalarT * O_row_top_out = O_channel_block_input_out;

                    kernel_top<ScalarT, AccumT,
                               _G_b, _K_b, _F_cb, _O_wb, _stride,
                               _UNROLL, op_type, op_class, true>(
                                   first,
                                   F_h,
                                   F_w,
                                   I_w * _C_ib,
                                   t_pad_el,
                                   pad_top,
                                   W_full_index,
                                   l_pad_el,
                                   pad_left,
                                   O_w_w_pad,
                                   O_w_full,
                                   O_w_left,
                                   r_pad_el,
                                   pad_right,
                                   I_row_top,
                                   F_row_top,
                                   O_row_top,
                                   k_zero,
                                   I_offset,
                                   F_offset,
                                   O_row_top_out,
                                   lshift,
                                   rshift,
                                   q_mul,
                                   offset);

                    ScalarT const *I_row_full =
                        I_row_top + H_full_index * I_w * (_F_cb * _G_b);
                    AccumT        *O_row_full =
                        O_row_top + t_pad_el * O_w_w_pad * (_G_b * _K_b); // ScalarT --> AccumT

                    ScalarT *O_row_full_out =
                        O_row_top_out + t_pad_el * O_w_w_pad * (_G_b * _K_b);

                    // Steady State over rows
                    for (index_t j = height_tid; j < O_h; j += T_height)
                    {
                        ScalarT const *I_row;
                        if constexpr (op_type == 'u')
                        {
                            /// @todo VERIFY Copied from float version
                            I_row = I_row_full + (j / _stride) * (I_w * _F_cb * _G_b);
                        }
                        else
                        {
                            I_row = I_row_full + (j * _stride) * (I_w * _F_cb * _G_b);
                        }
                        ScalarT const *F_row = F_channel_block_input + 0;
                        AccumT        *O_row =
                            O_row_full + j * (O_w_w_pad * _G_b * _K_b); // ScalarT --> AccumT

                        ScalarT *O_row_out =
                            O_row_full_out + j * (O_w_w_pad * _G_b * _K_b);

                        // Prologue with left padding
                        kernel_left<ScalarT, AccumT,
                                    _G_b, _K_b, _F_cb, _O_wb, _stride,
                                    _UNROLL, op_type, op_class, true>(
                                        first,
                                        F_h,
                                        F_w,
                                        I_w * _C_ib,
                                        l_pad_el,
                                        pad_left,
                                        I_row,
                                        F_row,
                                        O_row,
                                        0,
                                        0,
                                        k_zero,
                                        I_offset,
                                        F_offset,
                                        O_row_out,
                                        lshift,
                                        rshift,
                                        q_mul,
                                        offset);

                        ScalarT const *I_col_full =
                            I_row + W_full_index * (_F_cb * _G_b);
                        AccumT        *O_col_full = O_row + l_pad_el * (_G_b * _K_b); // ScalarT --> AccumT
                        ScalarT       *O_col_full_out = O_row_out + l_pad_el * (_G_b * _K_b);

                        // Steady State with microkernel
                        for (index_t l = 0; l < O_w_full; l += _O_wb)
                        {
                            ScalarT const *I_col;
                            // @todo cast index calculation as int and make stride a float value.
                            //I_x = I_x + (int)(j * _stride) * (<remaining dimensions>)
                            if constexpr (op_type == 'u')
                            {
                                /// @todo VERIFY Copied from float version
                                I_col = I_col_full + (l / _stride) * (_F_cb * _G_b);
                            }
                            else
                            {
                                I_col = I_col_full + (l * _stride) * (_F_cb * _G_b);
                            }
                            ScalarT const *F_col = F_row + 0;
                            AccumT        *O_col = O_col_full + l * (_G_b * _K_b); // ScalarT --> AccumT
                            ScalarT       *O_col_out = O_col_full_out + l * (_G_b * _K_b);

                            kernel<ScalarT, AccumT,
                                   _G_b, _K_b, _F_cb, _O_wb, _stride,
                                   _UNROLL, op_type, op_class, true>(
                                       first,
                                       F_h,
                                       F_w,
                                       I_w * _C_ib,
                                       I_col,
                                       F_col,
                                       O_col,
                                       0,
                                       0,
                                       0,
                                       0,
                                       k_zero,
                                       I_offset,
                                       F_offset,
                                       O_col_out,
                                       lshift,
                                       rshift,
                                       q_mul,
                                       offset);
                        }

                        // Epilogue for microkernel + right padding elements
                        ScalarT const *I_col_left;
                        if constexpr (op_type == 'u')
                        {
                            /// @todo VERIFY Copied from float version
                            I_col_left = I_col_full + (O_w_full / _stride) * (_F_cb * _G_b);
                        }
                        else
                        {
                            I_col_left = I_col_full + (O_w_full * _stride) * (_F_cb * _G_b);
                        }

                        ScalarT const *F_col_left = F_row + 0;
                        AccumT        *O_col_left = O_col_full + O_w_full * (_G_b * _K_b);  // ScalarT --> AccumT
                        ScalarT       *O_col_left_out = O_col_full_out + O_w_full * (_G_b * _K_b);

                        kernel_right<ScalarT, AccumT,
                                     _G_b, _K_b, _F_cb, _O_wb, _stride,
                                     _UNROLL, op_type, op_class, true>(
                                         first,
                                         F_h,
                                         F_w,
                                         I_w * _C_ib,
                                         O_w_left,
                                         r_pad_el,
                                         pad_right,
                                         I_col_left,
                                         F_col_left,
                                         O_col_left,
                                         0,
                                         0,
                                         k_zero,
                                         I_offset,
                                         F_offset,
                                         O_col_left_out,
                                         lshift,
                                         rshift,
                                         q_mul,
                                         offset);
                    }
                    // Epilogue with bottom padding
                    ScalarT const *I_row_bot =
                        I_row_full + (O_h * _stride) * (I_w * _F_cb * _G_b);
                    ScalarT const *F_row_bot = F_channel_block_input + 0;
                    AccumT        *O_row_bot = O_row_full + O_h * (O_w_w_pad * _G_b * _K_b); // ScalarT --> AccumT
                    ScalarT       *O_row_bot_out = O_row_full_out + O_h * (O_w_w_pad * _G_b * _K_b);

                    kernel_bottom<ScalarT, AccumT,
                                  _G_b, _K_b, _F_cb, _O_wb, _stride,
                                  _UNROLL, op_type, op_class, true>(
                                      first,
                                      F_h,
                                      F_w,
                                      I_w * _C_ib,
                                      b_pad_el,
                                      pad_bottom,
                                      W_full_index,
                                      l_pad_el,
                                      pad_left,
                                      O_w_w_pad,
                                      O_w_full,
                                      O_w_left,
                                      r_pad_el,
                                      pad_right,
                                      I_row_bot,
                                      F_row_bot,
                                      O_row_bot,
                                      k_zero,
                                      I_offset,
                                      F_offset,
                                      O_row_bot_out,
                                      lshift,
                                      rshift,
                                      q_mul,
                                      offset);
                }
            }
        }
    }
}

} // ns detail
} // ns small
