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
// Kernel right for remainder channels
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
void inline rem_kernel_right(
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
    const dim_t G_left,
    const dim_t K_left,
    dim_t H_lb = 0,
    dim_t H_ub = 0)
{
    const dim_t _C_ob = G_left * K_left;
    const dim_t _C_ib = G_left * _F_cb;
    const dim_t step = _stride * _C_ib;
    const dim_t H_UPPER = ((!H_ub) * (F_h)) + (H_ub);
    FLOAT_DEF_END_C(_O_wb, _C_ob);

#if DEBUG == 1
    printf("G_left %d K_left %d\n", G_left, K_left);
    printf("kernel_right_rem\n");
    printf("First 5 input values: %f %f %f %f %f\n", I[0], I[1], I[2], I[3], I[4]);
    if (op_type == OP_CONV)
        printf("First 5 Filter values for output channel 0 : %f %f %f %f %f \n", F[0], F[1], F[2], F[3], F[4]);
    printf("O_W_left %d r_pad_el %d\n", O_w_left, r_pad_el);
    printf("input col stride %d\n", input_col_stride);
#endif
    if (O_w_left)
    {
        if (first)
        {
            FLOAT_ZERO_END_C(O_w_left, _C_ob);

            if ((op_type == OP_MUL) || (op_type == OP_MAX_POOL && H_lb == 0 && H_ub == 0))
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
            if constexpr (op_type == OP_ADD && op_class == 3)
            {
                FLOAT_ZERO_END_C(O_w_left, _C_ob);
            }
            FLOAT_LOAD_END_C(O, O_w_left, _C_ob);
            if constexpr (op_type == OP_UPSAMPLE)
            {
                FLOAT_ACCUM_END_C_upsample(I, _stride, _C_ib, O_w_left, _C_ob);
            }
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
            _F_cb,
            G_left,
            K_left);

        if (op_type == OP_AVERAGE_POOL)
        {
            float norm = 1.0 / (1.0 * F_h * F_w);
            FLOAT_DIV_END_C(c_tile, norm, O_w_left, _C_ob);
        }
        if (op_type == OP_ADD && op_class == 3 && _C_ob == 1)
        {
            /* If the operation reduces the channel dimension, reduce across channel dimension of simd tile*/
            FLOAT_REDUCE_REM_CHANNEL_END_C(O_w_left, _C_ob)
        }

        FLOAT_STORE_END_C(O, O_w_left, _C_ob);
#if DEBUG == 1
        printf("First output value: %f %f %f %f \n", O[0], O[1], O[2], O[3]);
#endif
    }

    // right padding elements
    AccumT *O_ptr = O + O_w_left * _C_ob; // ScalarT --> AccumT
    ScalarT const *I_ptr = I + O_w_left * step;
    int W_i_valid = F_w - 1;

    if (first)
    {
        FLOAT_ZERO_END_C(r_pad_el, _C_ob);

        // Initialize with 0 for the padding elements

        //@note padding  should always be 'v' for pointwise operations, so this code path should not be used
        if (op_type == OP_MUL)
        {
            FLOAT_LOAD_END_C_strided(I_ptr, step, r_pad_el, _C_ob);
        }
    }
    else
    {
        FLOAT_LOAD_END_C(O_ptr, r_pad_el, _C_ob);
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
            _F_cb,
            G_left,
            K_left);

        c_cur += (K_left * G_left) / (FLOAT_SIMD_EPILOGUE);
        W_i_valid -= _stride;
        I_ptr += _stride * _C_ib;
    }

    if (op_type == OP_AVERAGE_POOL)
    {
        float norm = 1.0 / (1.0 * F_h * F_w);
        FLOAT_DIV_END_C(c_tile, norm, r_pad_el, _C_ob);
    }

    FLOAT_STORE_END_C(O_ptr, r_pad_el, _C_ob);
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
void inline kernel_right_rem(
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
    const dim_t F_c_left,
    dim_t H_lb = 0,
    dim_t H_ub = 0)
{
    constexpr dim_t _C_ob = _G_b * _K_b;
    const dim_t _C_ib = _G_b * F_c_left;
    const dim_t step = _stride * _C_ib;
    const dim_t H_UPPER = ((!H_ub) * (F_h)) + (H_ub);
    FLOAT_DEF_END_C(_O_wb, _C_ob);

#if DEBUG
    printf("kernel_right_rem\n");
    printf("First 5 input values: %f %f %f %f %f\n", I[0], I[1], I[2], I[3], I[4]);
    printf("First 5 Filter values for output channel 0 : %f %f %f %f %f \n", F[0 * _C_ob], F[1 * _C_ob], F[2 * _C_ob], F[3 * _C_ob], F[4 * _C_ob]);
    printf("O_W_left %d r_pad_el %d\n", O_w_left, r_pad_el);
    printf("input col stride %d\n", input_col_stride);
#endif
    if (O_w_left)
    {
        if (first)
        {
            FLOAT_ZERO_END_C(O_w_left, _C_ob);

            if ((op_type == OP_MUL) || (op_type == OP_MAX_POOL && H_lb == 0 && H_ub == 0))
            {
                FLOAT_LOAD_END_C_strided(I, step, O_w_left, _C_ob);
            }
            else if (op_type == OP_UPSAMPLE)
            {
                FLOAT_LOAD_END_C_upsample(I, _stride, _C_ib, O_w_left, _C_ob);
            }
        }
        else
        {//Global Reduction
            if constexpr (op_type == OP_ADD && op_class == 3)
            {
                FLOAT_ZERO_END_C(O_w_left, _C_ob);
            }
            FLOAT_LOAD_END_C(O, O_w_left, _C_ob);
            if constexpr (op_type == OP_UPSAMPLE)
            {
                FLOAT_ACCUM_END_C_upsample(I, _stride, _C_ib, O_w_left, _C_ob);
            }
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
            F_c_left);

        if (op_type == OP_AVERAGE_POOL)
        {
            float norm = 1.0 / (1.0 * F_h * F_w);
            FLOAT_DIV_END_C(c_tile, norm, O_w_left, _C_ob);
        }
        if constexpr (op_type == OP_ADD && op_class == 3 && _C_ob == 1)
        {
            /* If the operation reduces the channel dimension, reduce across channel dimension of simd tile*/
            FLOAT_REDUCE_CHANNEL_END_C(O_w_left, _C_ob)
        }

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

        //@note padding  should always be 'v' for pointwise operations, so this code path should not be used
        if (op_type == OP_MUL)
        {
            FLOAT_LOAD_END_C_strided(I_ptr, step, r_pad_el, _C_ob);
        }
    }
    else
    {
        FLOAT_LOAD_END_C(O_ptr, r_pad_el, _C_ob);
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
            F_c_left);

        c_cur += (_K_b * _G_b) / (FLOAT_SIMD_EPILOGUE);
        W_i_valid -= _stride;
        I_ptr += step;
    }

    if (op_type == OP_AVERAGE_POOL)
    {
        float norm = 1.0 / (1.0 * F_h * F_w);
        FLOAT_DIV_END_C(c_tile, norm, r_pad_el, _C_ob);
    }

    FLOAT_STORE_END_C(O_ptr, r_pad_el, _C_ob);

#if DEBUG
    printf("First output value: %f %f %f %f \n", O[0], O[1], O[2], O[3]);
#endif
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
void inline rem_kernel_right_rem(
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
    const dim_t F_c_left,
    const dim_t G_left,
    const dim_t K_left,
    dim_t H_lb = 0,
    dim_t H_ub = 0)
{
    const dim_t _C_ob = G_left * K_left;
    const dim_t _C_ib = G_left * F_c_left;
    const dim_t step = _stride * _C_ib;
    const dim_t H_UPPER = ((!H_ub) * (F_h)) + (H_ub);
    FLOAT_DEF_END_C(_O_wb, _C_ob);

#if DEBUG
    printf("kernel_right_rem\n");
    printf("First 5 input values: %f %f %f %f %f\n", I[0], I[1], I[2], I[3], I[4]);
    printf("First 5 Filter values for output channel 0 : %f %f %f %f %f \n", F[0 * _C_ob], F[1 * _C_ob], F[2 * _C_ob], F[3 * _C_ob], F[4 * _C_ob]);
    printf("O_W_left %d r_pad_el %d\n", O_w_left, r_pad_el);
    printf("input col stride %d\n", input_col_stride);
#endif
    if (O_w_left)
    {
        if (first)
        {
            FLOAT_ZERO_END_C(O_w_left, _C_ob);

            if ((op_type == OP_MUL) || (op_type == OP_MAX_POOL && H_lb == 0 && H_ub == 0))
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
            if constexpr (op_type == OP_ADD && op_class == 3)
            {
                FLOAT_ZERO_END_C(O_w_left, _C_ob);
            }
            FLOAT_LOAD_END_C(O, O_w_left, _C_ob);
            if constexpr (op_type == OP_UPSAMPLE)
            {
                FLOAT_ACCUM_END_C_upsample(I, _stride, _C_ib, O_w_left, _C_ob);
            }
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
            F_c_left,
            G_left,
            K_left);

        if (op_type == OP_AVERAGE_POOL)
        {
            float norm = 1.0 / (1.0 * F_h * F_w);
            FLOAT_DIV_END_C(c_tile, norm, O_w_left, _C_ob);
        }
        if (op_type == OP_ADD && op_class == 3 && _C_ob == 1)
        {
            /* If the operation reduces the channel dimension, reduce across channel dimension of simd tile*/
            FLOAT_REDUCE_REM_CHANNEL_END_C(O_w_left, _C_ob)
        }

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

        //@note padding  should always be 'v' for pointwise operations, so this code path should not be used
        if (op_type == OP_MUL)
        {
            FLOAT_LOAD_END_C_strided(I_ptr, step, r_pad_el, _C_ob);
        }
    }
    else
    {
        FLOAT_LOAD_END_C(O_ptr, r_pad_el, _C_ob);
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
            F_c_left,
            G_left,
            K_left);

        c_cur += (G_left * K_left) / (FLOAT_SIMD_EPILOGUE);
        W_i_valid -= _stride;
        I_ptr += step;
    }

    if (op_type == OP_AVERAGE_POOL)
    {
        float norm = 1.0 / (1.0 * F_h * F_w);
        FLOAT_DIV_END_C(c_tile, norm, r_pad_el, _C_ob);
    }

    FLOAT_STORE_END_C(O_ptr, r_pad_el, _C_ob);

#if DEBUG
    printf("First output value: %f %f %f %f \n", O[0], O[1], O[2], O[3]);
#endif
}

} // ns float_detail
} // ns small
