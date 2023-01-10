/*
 * SMaLL Framework
 *
 * Copyright 2022 Carnegie Mellon University and Authors.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DMxx-xxxx
 */

#pragma once

#include <stdint.h>
#include <string.h>
#include <omp.h>
// #define G_b    16
// #define K_b    1
// #define F_cb   1
// #define O_wb   6
// #define stride 1

#define DEBUG 0

typedef uint32_t index_t;
typedef uint32_t dim_t;
typedef float operand_t;  /// @todo template on operand type?

//****************************************************************************
// @todo unify op_dim and output_dim
// @todo replace macro with inline/constexpr(?) function
#define op_dim(IN_dim, stride, K_dim, OUT_dim)   \
    {                                            \
        int out_elems = (int(IN_dim) - int(K_dim)) / stride + 1;        \
        OUT_dim = (out_elems > 0 ) ? out_elems : 0;\
    }

//****************************************************************************
inline dim_t output_dim(dim_t idim, dim_t fdim, dim_t stride) {
    dim_t odim = ((idim - fdim) / stride) + 1;
    return odim < 0 ? 0 : odim; /// @todo FIXME: odim is unsigned, cant be negative
}


//TODO: Make this work
#define ABSTRACT_OP_END(op_type, op_class, a_cur, b_cur, c_cur)       \
    if (op_type == 'c')                                               \
    {                                                                 \
        if (op_class == 1)                                            \
        {                                                             \
            DW_END_C(step, a_cur, b_cur, c_cur, W_elements, _C_ob);   \
        }                                                             \
        else if (op_class == 2)                                       \
        {                                                             \
            CONV_END_C(step, a_cur, b_cur, c_cur, W_elements, _C_ob); \
        }                                                             \
    }                                                                 \
    else if (op_type == 'a' || op_type == 'p')                        \
    {                                                                 \
        MAX_END_C(step, a_cur, b_cur, c_cur, W_elements, _C_ob);      \
    }

#define ABSTRACT_OP(op_type, op_class, a_cur, b_cur)       \
    if (op_type == 'c')                                    \
    {                                                      \
        if (op_class == 1)                                 \
        {                                                  \
            DW_TILE_C(step, a_cur, b_cur, _O_wb, _C_ob);   \
        }                                                  \
        else if (op_class == 2)                            \
        {                                                  \
            CONV_TILE_C(step, a_cur, b_cur, _O_wb, _C_ob); \
        }                                                  \
    }                                                      \
    else if (op_type == 'a' || op_type == 'p')             \
    {                                                      \
        MAX_TILE_C(step, a_cur, _O_wb, _C_ob);             \
    }

#define ABSTRACT_EWISE_OP_END(op_type, c_cur, W_elements)     \
    if (op_type == 'a')                     \
    {                                       \
        RELU_REGISTER_END_C(c_cur, W_elements, _C_ob); \
    }

#define ABSTRACT_EWISE_OP(op_type)                    \
    if (op_type == 'a')                               \
    {                                                 \
        RELU_REGISTER_TILE_C(_O_wb, _C_ob);           \
    }                                      \
    // #define DEBUG 0

template <
    // standard template params
    dim_t _G_b,
    dim_t _K_b,
    dim_t _F_cb,
    dim_t _O_wb,
    dim_t _stride,
    dim_t _UNROLL,
    // TODO: add a bool to switch between microkernel and default imp
    // Leaf to describe abstract operation
    char op_type,
    int8_t op_class>
void inline compute_with_padding(dim_t H_lb, dim_t H_ub,
                                 dim_t W_lb, dim_t W_ub,
                                 dim_t F_w,
                                 dim_t W_elements,
                                 dim_t input_col_stride,
                                 operand_t *F,
                                 operand_t *I,
                                 c_tile_t *c_cur
                                 //dim_t c_cur
                                )
{
    constexpr dim_t _C_ob = _G_b * _K_b;
    constexpr dim_t _C_ib = _G_b * _F_cb;
    constexpr dim_t step = _stride * _C_ib;
    for (uint32_t n = H_lb; n < H_ub; n++)
    {

        int filter_offset_h = n * F_w * _F_cb * _G_b * _K_b;
        int input_stencil_h = /*input_col_offset +*/ (n - H_lb) * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = W_lb; m < W_ub; m++)
        {

            int filter_offset_w = m * _F_cb * _G_b * _K_b + filter_offset_h;
            /* This is C_ib because the microkernel stretches across groups*/
            int input_stencil_w = (m - W_lb) * _C_ib + input_stencil_h;

            operand_t *b = F + filter_offset_w;
            operand_t *a = I + input_stencil_w;
	    //TODO: reintroduce convolution
            for (uint32_t ii = 0; ii < _F_cb/_UNROLL ; ii++)
            {

                operand_t *b_cur = b + ii * _UNROLL * C_ob;
                operand_t *a_cur = a + ii * _UNROLL;
                // printf("unrol k %d %f\n", ii*_UNROLL, a[ii * _UNROLL]);
                // if (op_type == 'c')
                // {
                //     if (op_class == 1)
                //     {
                //         DW_END_C(step, a_cur, b_cur, c_cur, W_elements, _C_ob);
                //     }
                //     else if (op_class == 2)
                //     {
                //         CONV_END_C(step, a_cur, b_cur, c_cur, W_elements, _C_ob);
                //     }
                // }
                // else if (op_type == 'a' || op_type == 'p')
                // {
                //     MAX_END_C(step, a_cur, b_cur, c_cur, W_elements, _C_ob);
                // }
                ABSTRACT_OP_END(op_type, op_class, a_cur, b_cur, c_cur);
            }
        }
    }
}

#define CALC_FULL_DIMS(I_h, I_w, _stride, H_f, W_f, H, W) \
    op_dim((I_h), _stride, H_f, H);                      \
    op_dim((I_w), _stride, W_f, W);

// dim_t H_o_w_pad, W_o_w_pad;
// op_dim((I_h + t_pad + b_pad), _stride, H_f, H_o_w_pad);
// op_dim((I_w + l_pad + r_pad), _stride, W_f, W_o_w_pad);
// O_h_w_pad = H_o_w_pad;
// O_w_w_pad = W_o_w_pad;

#define SET_PADDING_PARAMS(I_h, I_w, t_pad, b_pad, l_pad, r_pad, _stride, H_f, W_f,\
                           t_pad_el, b_pad_el, l_pad_el, r_pad_el,\
                           H_full_index, W_full_index, H_back_index, W_back_index,\
                           H_o, W_o_full)                                     \
    /*    //  To calculate offsets to next output row, next output block*/                                              \
                                                                                                                        \
    t_pad_el = t_pad / _stride + (t_pad % _stride != 0);                                                                \
    l_pad_el = l_pad / _stride + (l_pad % _stride != 0);                                                                \
                                                                                                                        \
    H_full_index = t_pad_el * _stride - t_pad;                                                                          \
    W_full_index = l_pad_el * _stride - l_pad;                                                                          \
                                                                                                                        \
    /*// Full kernel output elements*/                                                                                  \
    CALC_FULL_DIMS((I_h - H_full_index), (I_w - W_full_index), _stride, H_f, W_f, H_o, W_o_full);                       \
                                                                                                                        \
    /*// back padding elements*/                                                                                        \
                                                                                                                        \
    H_back_index = H_full_index + _stride * (H_o);                                                                      \
    W_back_index = W_full_index + _stride * (W_o_full);                                                                 \
                                                                                                                        \
    CALC_FULL_DIMS((I_h + b_pad - H_back_index), (I_w + r_pad - W_back_index), _stride, H_f, W_f, b_pad_el, r_pad_el); \
                                                                                    

//****************************************************************************
template <
    dim_t _G_b,
    dim_t _K_b,
    dim_t _F_cb,
    dim_t _O_wb,
    dim_t _stride,
    dim_t _UNROLL,
    char op_type,
    int8_t op_class,
    int8_t op_class_fused_ewise = -1,
    char op_type_fused_ewise = 'a'>
void inline kernel_left(
    bool first,
    dim_t F_h,
    dim_t F_w,
    dim_t input_col_stride,
    dim_t r_pad_el,
    dim_t r_pad,
    operand_t *I,
    operand_t *F,
    operand_t *O,
    dim_t H_lb = 0,
    dim_t H_ub = 0)
{

    constexpr dim_t _C_ob = _G_b * _K_b;
    //constexpr dim_t _C_ib = _G_b * _F_cb;
    //constexpr dim_t step = _stride * _C_ib;

    const dim_t H_UPPER = ((!H_ub) * (F_h)) + (H_ub);
    // printf("%d \n", H_UPPER);
    DEF_END_C(_O_wb, _C_ob);


    // left padding elements
    operand_t *O_ptr = O;
    operand_t *I_ptr = I;

    int W_i_valid = r_pad;

    if (first)
    {
        ZERO_END_C(r_pad_el, _C_ob);

        // Initialize with 0 for the padding elements

        // if (op_type == 'p')
        // {
        //     LOAD_END_C_strided(I, step, r_pad_el, _C_ob);
        // }
    }
    else
    {
        LOAD_END_C(O_ptr, r_pad_el, _C_ob);
    }

    c_tile_t *c_cur = c_tile;
    // dim_t c_cur = 0;
    for (uint32_t k_p = 0; k_p < r_pad_el; k_p++)
    {
        // printf("calling left padding using weights from %d to %d\n", W_i_valid, F_w);

        // printf("c offset = %d", c_cur - c_tile);
        compute_with_padding<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL,

                             op_type, op_class>(H_lb, H_UPPER,
                                                W_i_valid, F_w,
                                                F_w,
                                                1,
                                                input_col_stride,
                                                F,
                                                I_ptr,
                                                c_cur);
        // printf("%.2f \t padded op el%d \n", c_cur[0], k_p);
        // printf("end16 f1rom %d to %d\n", W_i_valid, F_w);

        c_cur += (_K_b * _G_b)/(SIMD_EPILOGUE);
        // c_cur += 1;
        W_i_valid -= _stride;
        // I_ptr += ()*(_stride * _F_cb * _G_b);
    }

    if(op_class_fused_ewise == 0)
    {
        ABSTRACT_EWISE_OP_END(op_type_fused_ewise, c_tile,r_pad_el);
    }    
    STORE_END_C(O_ptr, r_pad_el, _C_ob);
    O_ptr += _G_b * _K_b;
}

//****************************************************************************
template <
    dim_t _G_b,
    dim_t _K_b,
    dim_t _F_cb,
    dim_t _O_wb,
    dim_t _stride,
    dim_t _UNROLL,
    char op_type,
    int8_t op_class,
    int8_t op_class_fused_ewise = -1,
    char op_type_fused_ewise = 'a'>
void inline kernel(
    bool first,
    dim_t F_h,
    dim_t F_w,
    dim_t input_col_stride,
    operand_t *I,
    operand_t *F,
    operand_t *O,
    dim_t H_lb = 0,
    dim_t H_ub = 0,
    dim_t W_lb = 0,
    dim_t W_ub = 0)
{

    constexpr dim_t _C_ob = _G_b * _K_b;
    constexpr dim_t _C_ib = _G_b * _F_cb;
    constexpr dim_t step  = _stride * _C_ib;

    const dim_t H_UPPER = ((!H_ub)*(F_h)) + (H_ub);
    // const dim_t W_UPPER = ((!W_ub) * (F_w)) + (W_ub);
    // printf("op_type: %c op_class: %d\n", op_type, op_class);
    DEF_TILE_C(_O_wb, _C_ob);
    if (first)
    {
        ZERO_TILE_C(_O_wb, _C_ob);
        if (op_type == 'p')
        {
            // printf("pooling params: %d %d \n", _C_ob, _O_wb);
            LOAD_TILE_C_strided(I, step, _O_wb, _C_ob);
            // for(int i = 0; i < _C_ob; i++)
            // {
            //     printf("%d i %f\n", i, I[i]);
            // }
        }
    }
    else
    {
        LOAD_TILE_C(O, _O_wb, _C_ob);
    }

    // int updates = 0;
    // uint32_t step = _C_ob;//stride*_C_ob;
    // int count = 0;
    // printf(" \n\n\nop_class %d op_type %c \n", op_class, op_type);

    for (uint32_t n = H_lb; n < H_UPPER; n++)
    {

        int filter_offset_h = n * F_w * _F_cb * _G_b * _K_b;
        int input_stencil_h = /*input_col_offset +*/ (n - H_lb) * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0 ; m < F_w; m++)
        {

            int filter_offset_w = m * _F_cb * _G_b * _K_b + filter_offset_h;
            //This is C_ob because the microkernel stretches across groups
            int input_stencil_w = (m - W_lb) * _C_ib + input_stencil_h;

            operand_t *b = F + filter_offset_w;
            operand_t *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _F_cb/_UNROLL; ii++)
            {
                operand_t *b_cur = b + ii * _UNROLL * C_ob;
                operand_t *a_cur = a + ii * _UNROLL;
                // if (op_type == 'p')
                // {
                //     printf("%d %d %d input_offset %d col_stride: %d  %f\n", n, m, ii, input_stencil_w, input_col_stride, a[0]);

                //     // for (int i = 0; i < _C_ob; i++)
                //     // {
                //     //     printf("%d i %f\n", i, a[i]);
                //     // }
                // }
                ABSTRACT_OP(op_type, op_class, a_cur, b_cur);
            }
        }
    }

    // printf("op_class %d op_type %c \n\n\n", op_class, op_type);
    if (op_class_fused_ewise == 0)
    {
        ABSTRACT_EWISE_OP(op_type_fused_ewise);
    }
    STORE_TILE_C(O, _O_wb, _C_ob);
    // for(int i = 0; i < _C_ob; i++)
    // {
    //     printf("%d %f\n", i, O[i] );
    // }
}


//****************************************************************************
//TODO: Explain the difference between kernel and kernel_pad
template <
    dim_t _G_b,
    dim_t _K_b,
    dim_t _F_cb,
    dim_t _O_wb,
    dim_t _stride,
    dim_t _UNROLL,
    char op_type,
    int8_t op_class,
    int8_t op_class_fused_ewise = -1,
    char op_type_fused_ewise = 'a'>
void inline kernel_pad(
    bool first,
    dim_t F_h,
    dim_t F_w,
    dim_t input_col_stride,
    operand_t *I,
    operand_t *F,
    operand_t *O,
    dim_t H_lb = 0,
    dim_t H_ub = 0,
    dim_t W_lb = 0,
    dim_t W_ub = 0)
{

    constexpr dim_t _C_ob = _G_b * _K_b;
    constexpr dim_t _C_ib = _G_b * _F_cb;
    constexpr dim_t step = _stride * _C_ib;

    const dim_t H_UPPER = ((!H_ub) * (F_h)) + (H_ub);
    // const dim_t W_UPPER = ((!W_ub) * (F_w)) + (W_ub);

    DEF_TILE_C(_O_wb, _C_ob);
    if (first)
    {
        ZERO_TILE_C(_O_wb, _C_ob);
    }
    else
    {
        LOAD_TILE_C(O, _O_wb, _C_ob);
    }

    // int updates = 0;
    // uint32_t step = _C_ob;//stride*_C_ob;
    // int count = 0;
    for (uint32_t n = H_lb; n < H_UPPER; n++)
    {

        int filter_offset_h = n * F_w * _F_cb * _G_b * _K_b;
        int input_stencil_h = /*input_col_offset +*/ (n - H_lb) * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < F_w; m++)
        {

            int filter_offset_w = m * _F_cb * _G_b * _K_b + filter_offset_h;
            // This is C_ob because the microkernel stretches across groups
            int input_stencil_w = (m - W_lb) * _C_ib + input_stencil_h;

            operand_t *b = F + filter_offset_w;
            operand_t *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _F_cb / _UNROLL; ii++)
            {

                operand_t *b_cur = b + ii * _UNROLL * C_ob;
                operand_t *a_cur = a + ii * _UNROLL;

                // if (op_type == 'c')
                // {
                //     if (op_class == 1)
                //     {
                //         DW_TILE_C(step, a_cur, b_cur, _O_wb, _C_ob);
                //     }
                //     else if (op_class == 2)
                //     {
                //         CONV_TILE_C(step, a_cur, b_cur, _O_wb, _C_ob);
                //     }
                // }
                // else if (op_type == 'a' || op_type == 'p')
                // {
                //     MAX_TILE_C(step, a_cur, _O_wb, _C_ob);
                // }
                ABSTRACT_OP(op_type, op_class, a_cur, b_cur);
            }
        }
    }
    if (op_class_fused_ewise == 0)
    {
        ABSTRACT_EWISE_OP(op_type_fused_ewise);
    }
    STORE_TILE_C(O, _O_wb, _C_ob);
}

//****************************************************************************
template <
    dim_t _G_b,
    dim_t _K_b,
    dim_t _F_cb,
    dim_t _O_wb,
    dim_t _stride,
    dim_t _UNROLL,
    char op_type,
    int8_t op_class,
    int8_t op_class_fused_ewise = -1,
    char op_type_fused_ewise = 'a'>
void inline kernel_right(
    bool first,
    dim_t F_h,
    dim_t F_w,
    dim_t input_col_stride,
    dim_t O_w_left,
    dim_t r_pad_el,
    dim_t r_pad,
    operand_t *I,
    operand_t *F,
    operand_t *O,
    dim_t H_lb = 0,
    dim_t H_ub = 0)
{
    constexpr dim_t _C_ob = _G_b * _K_b;
    constexpr dim_t _C_ib = _G_b * _F_cb;
    constexpr dim_t step = _stride * _C_ib;

    const dim_t H_UPPER = ((!H_ub) * (F_h)) + (H_ub);
    DEF_END_C(_O_wb, _C_ob);



    if(O_w_left)
    {



        if (first)
        {
            ZERO_END_C(O_w_left, _C_ob);

            if(op_type == 'p' && H_lb == 0 && H_ub == 0)
            {
                LOAD_END_C_strided(I, step, O_w_left, _C_ob);

            }

        }
        else
        {
            LOAD_END_C(O, O_w_left, _C_ob);
        }

        compute_with_padding<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL,

                             op_type, op_class>(H_lb, H_UPPER,
                                                0, F_w,
                                                F_w,
                                                O_w_left,
                                                input_col_stride,
                                                F,
                                                I,
                                                c_tile);
        // for (size_t i = 0; i < O_w_left; i++)
        // {
        //     for (size_t j = 0; j < _C_ob; j++)
        //     {
        //         printf("%.2f \t", c_tile[i*_C_ob + j]);
        //     }
        //     printf("\n");
        // }

        if (op_class_fused_ewise == 0)
        {
            ABSTRACT_EWISE_OP_END(op_type_fused_ewise, c_tile, O_w_left);
        }

        // for (size_t i = 0; i < O_w_left; i++)
        // {
        //     for (size_t j = 0; j < _C_ob; j++)
        //     {
        //         printf("%.2f \t", c_tile[i * _C_ob + j]);
        //     }
        //     printf("\n");
        // }

        STORE_END_C(O, O_w_left, _C_ob);
    }


    //right padding elements
    operand_t *O_ptr = O + O_w_left * _C_ob;
    operand_t *I_ptr = I + O_w_left * step;
    // printf("padding %d \n", I_ptr - I);
    int W_i_valid = F_w - 1;


    if (first)
    {
        ZERO_END_C(r_pad_el, _C_ob);

        // Initialize with 0 for the padding elements

        // if(op_type=='p')
        // {
        //     LOAD_END_C_strided(I_ptr, step, r_pad_el, _C_ob);
        // }
    }
    else
    {
        LOAD_END_C(O_ptr, r_pad_el, _C_ob);
    }


    c_tile_t * c_cur = c_tile;
    // dim_t c_cur = 0;
    for (uint32_t k_p = 0; k_p < r_pad_el; k_p ++)
    {
        // printf("calling right padding use weights from 1 to %d\n", W_i_valid);

        // printf("c offset = %d", c_cur - c_tile);
        // COMPUTE_W_PADDING(0, F_h, 0, W_i_valid, F_w, _C_ib, 1, op_type, op_class, F, I_ptr, c_cur);
        compute_with_padding<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL,

                             op_type, op_class>(H_lb, H_UPPER,
                                                0, W_i_valid,
                                                F_w,
                                                1,
                                                input_col_stride,
                                                F,
                                                I_ptr,
                                                c_cur);
        // printf("%.2f \t padded op el%d \n", c_cur[0], k_p);

        c_cur += (_K_b * _G_b)/(SIMD_EPILOGUE);
	//printf("c_cur index %d %d \n", c_cur - c_tile, SIMD_EPILOGUE);
        // c_cur += 1;
        W_i_valid -= _stride;
        I_ptr += _stride * _F_cb * _G_b;
    }
    if (op_class_fused_ewise == 0)
    {
        ABSTRACT_EWISE_OP_END(op_type_fused_ewise, c_tile, r_pad_el);
    }
    STORE_END_C(O_ptr, r_pad_el, _C_ob);
    // O_ptr += r_pad_el * _G_b * _K_b;
}

//****************************************************************************
template <
    dim_t _G_b,
    dim_t _K_b,
    dim_t _F_cb,
    dim_t _O_wb,
    dim_t _stride,
    dim_t _UNROLL,
    char op_type,
    int8_t op_class,
    int8_t op_class_fused_ewise = -1,
    char op_type_fused_ewise = 'a'>
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
    operand_t *I,
    operand_t *F,
    operand_t *O)
{
    operand_t *I_ptr = I;
    operand_t *O_ptr = O;

    int H_i_valid = F_h - 1;

    for (uint32_t j_p = 0; j_p < b_pad_el; j_p++)
    {
        // Prologue with left padding
        kernel_left<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL, op_type, op_class, op_class_fused_ewise, op_type_fused_ewise>(
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
            H_i_valid);

        float *I_row_full = I + W_full_index * (_F_cb * _G_b);
        float *O_row_full = O + l_pad_el * (_G_b * _K_b);
        // Steady State with microkernel
        for (index_t l = 0; l < O_w_full; l += _O_wb)
        {
            operand_t *I_col = I_row_full + (l * _stride) * (_F_cb * _G_b);
            operand_t *F_col = F + 0;
            operand_t *O_col = O_row_full + l * (_G_b * _K_b);

            kernel_pad<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL, op_type, op_class, op_class_fused_ewise, op_type_fused_ewise>(
                first,
                F_h,
                F_w,
                input_col_stride,
                I_col,
                F_col,
                O_col,
                0,
                H_i_valid);
        }

        // Epilogue for microkernel + right padding elements
        operand_t *I_col_left = I_row_full + (O_w_full * _stride) * (_F_cb * _G_b);
        operand_t *F_col_left = F + 0;
        operand_t *O_col_left = O_row_full + O_w_full * (_G_b * _K_b);
        kernel_right<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL, op_type, op_class, op_class_fused_ewise, op_type_fused_ewise>(
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
            H_i_valid);

        O_ptr += O_w_w_pad * _K_b * _G_b;
        H_i_valid -= _stride;
        I_ptr += _stride * _F_cb * _G_b;
    }
}

//****************************************************************************
template <
    dim_t _G_b,
    dim_t _K_b,
    dim_t _F_cb,
    dim_t _O_wb,
    dim_t _stride,
    dim_t _UNROLL,
    char op_type,
    int8_t op_class,
    int8_t op_class_fused_ewise = -1,
    char op_type_fused_ewise = 'a'>
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
    operand_t *I,
    operand_t *F,
    operand_t *O)
{
    operand_t *I_ptr = I;
    operand_t *O_ptr = O;

    int H_i_valid = t_pad;

    for (uint32_t j_p = 0; j_p < t_pad_el; j_p++)
    {
        // Prologue with left padding
        kernel_left<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL, op_type, op_class, op_class_fused_ewise, op_type_fused_ewise>(
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
            F_h);

        float *I_row_full = I + W_full_index * (_F_cb * _G_b);
        float *O_row_full = O + l_pad_el * (_G_b * _K_b);
        // Steady State with microkernel
        for (index_t l = 0; l < O_w_full; l += _O_wb)
        {
            operand_t *I_col = I_row_full + (l * _stride) * (_F_cb * _G_b);
            operand_t *F_col = F + 0;
            operand_t *O_col = O_row_full + l * (_G_b * _K_b);

            kernel_pad<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL, op_type, op_class, op_class_fused_ewise, op_type_fused_ewise>(
                first,
                F_h,
                F_w,
                input_col_stride,
                I_col,
                F_col,
                O_col,
                H_i_valid,
                F_h);
        }

        // Epilogue for microkernel + right padding elements
        operand_t *I_col_left = I_row_full + (O_w_full * _stride) * (_F_cb * _G_b);
        operand_t *F_col_left = F + 0;
        operand_t *O_col_left = O_row_full + O_w_full * (_G_b * _K_b);
        kernel_right<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL, op_type, op_class, op_class_fused_ewise, op_type_fused_ewise>(
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
            F_h);

        O_ptr += O_w_w_pad * _K_b * _G_b;
        H_i_valid += _stride;
        // I_ptr += _stride * _F_cb * _G_b;
    }
}

//****************************************************************************
template <
    dim_t _G_b,
    dim_t _K_b,
    dim_t _F_cb,
    dim_t _O_wb,
    dim_t _stride,
    dim_t _UNROLL,
    char op_type,
    int8_t op_class,
    bool rewrite_output>
void abstract_layer(
    dim_t G,   // Output Channel Grouping
    dim_t K,   // Output Channels per group
    dim_t F_c, // Channel Reduction Dimension
    dim_t I_h, // Input Height
    dim_t I_w, // Input Width

    dim_t F_h, // Filter height
    dim_t F_w, // Filter width

    // Padding values
    dim_t pad_top,
    dim_t pad_left,
    dim_t pad_right,
    dim_t pad_bottom,

    // Data
    operand_t * __restrict__ I,
    operand_t * __restrict__ F,
    operand_t * __restrict__ O)
{

    // uint32_t H_padding = 0, W_padding = 0;
    // uint32_t H_f_padding = 0, W_f_padding = 0;
    // uint32_t H_b_padding = 0, W_b_padding = 0;
    // printf(" %d unroll\n", _UNROLL);
    dim_t t_pad_el = 0, l_pad_el = 0;
    dim_t b_pad_el = 0, r_pad_el = 0;

    uint32_t H_full_index = 0, W_full_index = 0;
    uint32_t H_back_index = 0, W_back_index = 0;

    // Output Elements with padding
    // Output Elements using the full filter

    uint32_t H_o = 0, W_o_full = 0;
#if DEBUG == 1
    if (op_type == 'c')
    {
        printf("conv class: %d \n", op_class);
    }
    else if (op_type == 'a')
    {
        printf("activation class: %d \n", op_class);
    }
#endif
    // calculate output dimensions based on input params.
    constexpr dim_t _C_ib = _F_cb * _G_b;
    // constexpr dim_t _C_ob = _K_b * _G_b;

    // constexpr dim_t _UNROLL = 1;

    // const dim_t O_h = output_dim(I_h, F_h, _stride);
    // const dim_t O_w = output_dim(I_w, F_w, _stride);

    /*
        Data layout (slowest to fastest changing dimensions):
            blocks of groups
                   blocks of channels within groups
                          blocks of weights in the same group
                                    spatial dimensions
                                              weights in the same group
                                                    weights across groups in a block
                                                         channels in a block
        I: [G/G_b,        F_c/F_cb, I_h, I_w, F_cb, G_b     ]
        F: [G/G_b, K/K_b, F_c/F_cb, F_h, F_w, F_cb, G_b, K_b]
        O: [G/G_b, K/K_b,           O_h, O_w,       G_b, K_b]
    */

    // Deriving padding parameters
    dim_t O_h_w_pad;
    dim_t O_w_w_pad;

    // hardware specific kernel iterations
    dim_t O_h;
    dim_t O_w;
    CALC_FULL_DIMS((I_h + pad_top + pad_bottom), (I_w + pad_left + pad_right), _stride, F_h, F_w, O_h_w_pad, O_w_w_pad);
    SET_PADDING_PARAMS(I_h, I_w, pad_top, pad_bottom, pad_left, pad_right, _stride, F_h, F_w,\
                       t_pad_el, b_pad_el, l_pad_el, r_pad_el, H_full_index, W_full_index, H_back_index, W_back_index, O_h, O_w);

    // setting up microkernel specific parameters
    /*// setting up microkernel specific parameters*/

    const dim_t O_w_full = (O_w / _O_wb) * _O_wb;
    const dim_t O_w_left = O_w - O_w_full;

    const dim_t O_hxO_w = O_h_w_pad * O_w_w_pad;
#if DEBUG == 1

    printf("\t\t I_h %d I_w %d F_C %d G %d \n", I_h, I_w, F_c, G);
    printf("\t\t O_h_pad: %d O_w_w_pad %d \n", O_h_w_pad, O_w_w_pad);
    printf("O_h %d O_w %d O_w_left %d \n", O_h, O_w_full, O_w_left);

    printf("bottom padding index into input: %d \t bottom padding elements: %d \n", H_back_index, b_pad_el);
    printf("no padding index into input: %d \t top padding elements: %d \n", H_full_index, t_pad_el);

    printf("right padding index into input: %d \t right padding elements: %d \n", W_back_index, r_pad_el);
    printf("no padding index into input: %d \t left padding elements: %d \n", W_full_index, l_pad_el);

    printf("O_w_full: %d O_w_left: %d \n", O_w_full, O_w_left);
    printf("params: F_Cb %d G_b %d K_b %d\n", _F_cb, _G_b, _K_b);

#endif
    // printf("\t rewrite output?: %d  %c %d\n", rewrite_output, op_type, op_class);

    // Set up parallelism for the channel loops

    //  Get total available threads
    // TODO: add error checking in case env variable isn't defined.
    int N = 1;
    char const *env_nt(std::getenv("OMP_NUM_THREADS"));
    if (nullptr != env_nt)
    {
        N = atoi(std::getenv("OMP_NUM_THREADS"));
    }

    //const int Channel_blocks = K / _K_b;
    //const int Group_blocks = G / _G_b;

    int T_channel = N, T_group = 1, T_height = 1;

    // If dwise, parallelize on groups
    if (K == 1)
    {
        T_channel = 1;
        T_group = N;
    }

    // printf("T_Group: %d T_channel: %d\n",T_group, T_channel);
    // TODO: parallelize

    //create parallel region with all threads
    #if PARALLEL == 1
    #pragma omp parallel num_threads(N)
    #endif
    {
        auto t_id = omp_get_thread_num();
        auto height_tid = t_id % T_height;
        auto channel_tid = ((t_id) / (T_height)) % T_channel;
        auto group_tid = ((t_id / (T_channel * T_height))) % T_group;
        // loops over output channels
        for (index_t g = group_tid; g < G / _G_b; g+= T_group)
        {
            operand_t *I_group = I + g * (F_c * I_h * I_w * _G_b);
            operand_t *F_group = F + g * (K * F_c * F_h * F_w * _G_b);
            operand_t *O_group = O + g * (K * O_hxO_w * _G_b);
            // printf("I group offset %d \n", (I_group - I)/(I_h * I_w));
            // printf("F group offset %d \n", (F_group - F) / (F_h * F_w));

            for (index_t k = channel_tid; k < K / _K_b; k+= T_channel)
            {
                operand_t *I_channel_block_output = I_group + 0;
                operand_t *F_channel_block_output = F_group + k * (F_c * F_h * F_w * _G_b * _K_b);
                operand_t *O_channel_block_output = O_group + k * (O_hxO_w * _G_b * _K_b);

                // Loop over input channel reduction
                for (index_t i = 0; i < F_c / _F_cb; i++)
                {
                    bool first = rewrite_output && (i == 0);

                    // printf("\t out_channel:%d, %d before: in_channel %d, %.2f %.2f %.2f %.2f\n", g, k, i,O_channel_block_output[0], O_channel_block_output[1], O_channel_block_output[2], O_channel_block_output[3]);

                    operand_t *I_channel_block_input = I_channel_block_output + i * (I_h * I_w * _F_cb * _G_b);
                    // printf("I block offset %d %d \n", (I_channel_block_input - I), i * (I_h * I_w * _F_cb * _G_b));
                    operand_t *F_channel_block_input = F_channel_block_output + i * (F_h * F_w * _F_cb * _G_b * _K_b);
                    operand_t *O_channel_block_input = O_channel_block_output + 0;

                    // Loops over spatial dimensions of output

                    // Prologue with top padding
                    operand_t *I_row_top = I_channel_block_input;
                    operand_t *F_row_top = F_channel_block_input + 0;
                    operand_t *O_row_top = O_channel_block_input;

                    kernel_top<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL, op_type, op_class>(
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
                        O_row_top);

                    float *I_row_full = I_row_top + H_full_index * I_w * (_F_cb * _G_b);
                    float *O_row_full = O_row_top + t_pad_el * O_w_w_pad * (_G_b * _K_b);
                    // Steady State over rows
                    for (index_t j = height_tid; j < O_h; j+= T_height)
                    {
                        operand_t *I_row = I_row_full + (j * _stride) * (I_w * _F_cb * _G_b);
                        operand_t *F_row = F_channel_block_input + 0;
                        operand_t *O_row = O_row_full + j * (O_w_w_pad * _G_b * _K_b);
                        // Prologue with left padding
                        kernel_left<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL, op_type, op_class>(
                            first,
                            F_h,
                            F_w,
                            I_w * _C_ib,
                            l_pad_el,
                            pad_left,
                            I_row,
                            F_row,
                            O_row);

                        float *I_col_full = I_row + W_full_index * (_F_cb * _G_b);
                        float *O_col_full = O_row + l_pad_el * (_G_b * _K_b);
                        // Steady State with microkernel
                        for (index_t l = 0; l < O_w_full; l += _O_wb)
                        {
                            operand_t *I_col = I_col_full + (l * _stride) * (_F_cb * _G_b);
                            operand_t *F_col = F_row + 0;
                            operand_t *O_col = O_col_full + l * (_G_b * _K_b);
                            // printf("O_col_full %p O_col %p l %d _C_ob %d _O_wb %d\n", O_col, O_col_full, l, (_G_b * _K_b), _O_wb);
                                kernel<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL, op_type, op_class>(
                                    first,
                                    F_h,
                                    F_w,
                                    I_w * _C_ib,
                                    I_col,
                                    F_col,
                                    O_col);
                            // for (uint32_t o_idx = 0; o_idx < _K_b; o_idx++)
                            // {
                            //     printf("idx: %d offset %ld value %f\n", o_idx, O_col - O, O_col[o_idx]);
                            // }
                        }


                        // Epilogue for microkernel + right padding elements
                        operand_t *I_col_left = I_col_full + (O_w_full * _stride) * (_F_cb * _G_b);
                        operand_t *F_col_left = F_row + 0;
                        operand_t *O_col_left = O_col_full + O_w_full * (_G_b * _K_b);
                        kernel_right<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL, op_type, op_class>(
                            first,
                            F_h,
                            F_w,
                            I_w * _C_ib,
                            O_w_left,
                            r_pad_el,
                            pad_right,
                            I_col_left,
                            F_col_left,
                            O_col_left);

                        
                    }
                    // Epilogue with bottom padding
                    operand_t *I_row_bot = I_row_full + (O_h * _stride) * (I_w * _F_cb * _G_b);
                    operand_t *F_row_bot = F_channel_block_input + 0;
                    operand_t *O_row_bot = O_row_full + O_h * (O_w_w_pad * _G_b * _K_b);

                    kernel_bottom<_G_b, _K_b, _F_cb, _O_wb, _stride, _UNROLL, op_type, op_class>(
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
                        O_row_bot);

                        // printf("\t out_channel:%d %d after: in_channel %d %.2f %.2f %.2f %.2f\n", g, k, i, O_channel_block_output[0], O_channel_block_output[1], O_channel_block_output[2], O_channel_block_output[3]);
                }
            }
        }
    }

}

// // Fused abstract layer
// //  Fused call params
// template <
//     dim_t _G_b_0,
//     dim_t _K_b_0,
//     dim_t _F_cb_0,
//     dim_t _O_wb,
//     dim_t _stride_0,
//     dim_t _UNROLL,
//     char op_type_0,
//     int8_t op_class_0,
//     bool rewrite_output_0,
//     dim_t _G_b_1,
//     dim_t _K_b_1,
//     dim_t _F_cb_1,
//     dim_t _stride_1,
//     char op_type_1,
//     int8_t op_class_1,
//     bool rewrite_output_1>
// void fused_abstract_layer(
//     // Params for first layer
//     dim_t G_0,   // Output Channel Grouping
//     dim_t K_0,   // Output Channels per group
//     dim_t F_c_0, // Channel Reduction Dimension
//     dim_t I_h_0, // Input Height
//     dim_t I_w_0, // Input Width

//     dim_t F_h_0, // Filter height
//     dim_t F_w_0, // Filter width

//     // Padding values
//     dim_t pad_top_0,
//     dim_t pad_left_0,
//     dim_t pad_right_0,
//     dim_t pad_bottom_0,

//     // Params for second layer
//     dim_t G_1,   // Output Channel Grouping
//     dim_t K_1,   // Output Channels per group
//     dim_t F_c_1, // Channel Reduction Dimension
//     dim_t I_h_1, // Input Height ------------------> These could be inferred
//     dim_t I_w_1, // Input Width

//     dim_t F_h_1, // Filter height
//     dim_t F_w_1, // Filter width

//     // Padding values
//     dim_t pad_top_1,
//     dim_t pad_left_1,
//     dim_t pad_right_1,
//     dim_t pad_bottom_1,
//     // Data
//     operand_t *__restrict__ I,
//     operand_t *__restrict__ F_0,
//     operand_t *__restrict__ O_inter,
//     operand_t *__restrict__ F_1,
//     operand_t *__restrict__ O)
// {

//     abstract_layer<
//         _G_b_0,
//         _K_b_0,
//         _F_cb_0,
//         _O_wb,
//         _stride_0,
//         _UNROLL,
//         op_type_0,
//         op_class_0,
//         rewrite_output_0>(
//         // Params for first layer
//         G_0,   // Output Channel Grouping
//         K_0,   // Output Channels per group
//         F_c_0, // Channel Reduction Dimension
//         I_h_0, // Input Height
//         I_w_0, // Input Width

//         F_h_0, // Filter height
//         F_w_0, // Filter width

//         // Padding values
//         pad_top_0,
//         pad_left_0,
//         pad_right_0,
//         pad_bottom_0,

//         I,
//         F_0,
//         O_inter);

//     abstract_layer<
//         _G_b_1,
//         _K_b_1,
//         _F_cb_1,
//         _O_wb,
//         _stride_1,
//         _UNROLL,
//         op_type_1,
//         op_class_1,
//         rewrite_output_1>(
//         // Params for first layer
//         G_1,   // Output Channel Grouping
//         K_1,   // Output Channels per group
//         F_c_1, // Channel Reduction Dimension
//         I_h_1, // Input Height
//         I_w_1, // Input Width

//         F_h_1, // Filter height
//         F_w_1, // Filter width

//         // Padding values
//         pad_top_1,
//         pad_left_1,
//         pad_right_1,
//         pad_bottom_1,

//         O_inter,
//         F_1,
//         O);
// }

// Fused abstract layer
//  Fused call param
template <
    dim_t _G_b_0,
    dim_t _K_b_0,
    dim_t _F_cb_0,
    dim_t _O_wb,
    dim_t _stride_0,
    dim_t _UNROLL_0,
    char op_type_0,
    int8_t op_class_0,
    bool rewrite_output_0,
    dim_t _G_b_1,
    dim_t _K_b_1,
    dim_t _F_cb_1,
    dim_t _stride_1,
    char fused_ewise_op_type_0,
    int8_t fused_ewise_op_class_0,
    bool rewrite_output_1>
void fused_abstract_layer(
    // Params for first layer
    dim_t G_0,   // Output Channel Grouping
    dim_t K_0,   // Output Channels per group
    dim_t F_c_0, // Channel Reduction Dimension
    dim_t I_h_0, // Input Height
    dim_t I_w_0, // Input Width

    dim_t F_h_0, // Filter height
    dim_t F_w_0, // Filter width

    // Padding values
    dim_t pad_top_0,
    dim_t pad_left_0,
    dim_t pad_right_0,
    dim_t pad_bottom_0,

    // Params for second layer
    dim_t G_1,   // Output Channel Grouping
    dim_t K_1,   // Output Channels per group
    dim_t F_c_1, // Channel Reduction Dimension
    dim_t I_h_1, // Input Height ------------------> These could be inferred
    dim_t I_w_1, // Input Width

    dim_t F_h_1, // Filter height
    dim_t F_w_1, // Filter width

    // Padding values
    dim_t pad_top_1,
    dim_t pad_left_1,
    dim_t pad_right_1,
    dim_t pad_bottom_1,
    // Data
    operand_t *__restrict__ I,
    operand_t *__restrict__ F_0,
    operand_t *__restrict__ O_inter,
    operand_t *__restrict__ F_1,
    operand_t *__restrict__ _interO)
{

    // Parameters for the first operation
    dim_t t_pad_el_0 = 0, l_pad_el_0 = 0;
    dim_t b_pad_el_0 = 0, r_pad_el_0 = 0;

    // Only need this for debugging
    uint32_t H_back_index = 0, W_back_index = 0;

    uint32_t H_full_index_0 = 0, W_full_index_0 = 0;

    constexpr dim_t _C_ib = _F_cb_0 * _G_b_0;

    // Derive parameters for first operation
    // Deriving padding parameters
    dim_t O_h_w_pad_0 = I_h_1;
    dim_t O_w_w_pad_0 = I_w_1;

    // hardware specific kernel iterations
    dim_t O_h_0;
    dim_t O_w_0;
    // CALC_FULL_DIMS((I_h_0 + pad_top_0 + pad_bottom_0), (I_w_0 + pad_left_0 + pad_right_0), _stride_0, F_h_0, F_w_0, O_h_w_pad_0, O_w_w_pad_0);
    SET_PADDING_PARAMS(I_h_0, I_w_0, pad_top_0, pad_bottom_0, pad_left_0, pad_right_0, _stride_0, F_h_0, F_w_0,
                       t_pad_el_0, b_pad_el_0, l_pad_el_0, r_pad_el_0, H_full_index_0, W_full_index_0, H_back_index, W_back_index, O_h_0, O_w_0);

    // setting up microkernel specific parameters
    /*// setting up microkernel specific parameters*/

    const dim_t O_w_full_0 = (O_w_0 / _O_wb) * _O_wb;
    const dim_t O_w_left_0 = O_w_0 - O_w_full_0;

    const dim_t O_hxO_w_0 = O_h_w_pad_0 * O_w_w_pad_0;

    //
#if DEBUG == 1

    printf("\t\t I_h_0 %d I_w_0 %d F_C %d G_0 %d \n", I_h_0, I_w_0, F_c_0, G_0);
    printf("\t\t O_h_pad: %d O_w_w_pad_0 %d \n", O_h_w_pad_0, O_w_w_pad_0);
    printf("O_h_0 %d O_w_0 %d O_w_left_0 %d \n", O_h_0, O_w_full_0, O_w_left_0);

    printf("bottom padding index into input: %d \t bottom padding elements: %d \n", H_back_index, b_pad_el_0);
    printf("no padding index into input: %d \t top padding elements: %d \n", H_full_index_0, t_pad_el_0);

    printf("right padding index into input: %d \t right padding elements: %d \n", W_back_index, r_pad_el_0);
    printf("no padding index into input: %d \t left padding elements: %d \n", W_full_index_0, l_pad_el_0);

    printf("O_w_full_0: %d O_w_left_0: %d \n", O_w_full_0, O_w_left_0);
    printf("new params: _O_wb %d F_Cb %d G_b %d K_b %d\n", _O_wb, _F_cb_0, _G_b_0, _K_b_0);

#endif
    // printf("\t rewrite output?: %d  %c %d\n", rewrite_output, op_type_0, op_class_0);

    // Set up parallelism for the channel loops

    //  Get total available threads
    // TODO: add error checking in case env variable isn't defined.
    int N = 1;
    char const *env_nt(std::getenv("OMP_NUM_THREADS"));
    if (nullptr != env_nt)
    {
        N = atoi(std::getenv("OMP_NUM_THREADS"));
    }

    // const int Channel_blocks = K_0 / _K_b_0;
    // const int Group_blocks = G_0 / _G_b_0;

    int T_channel = N, T_group = 1, T_height = 1;

    // If dwise, parallelize on groups
    if (K_0 == 1)
    {
        T_channel = 1;
        T_group = N;
    }

//_____________________________________________________________________________________________
    // Parameters for the second operation

    dim_t t_pad_el_1 = 0, l_pad_el_1 = 0;
    dim_t b_pad_el_1 = 0, r_pad_el_1 = 0;

    // uint32_t H_back_index = 0, W_back_index = 0;

    uint32_t H_full_index_1 = 0, W_full_index_1 = 0;


    // Derive parameters for second operation
    // Deriving padding parameters
    dim_t O_h_w_pad_1;
    dim_t O_w_w_pad_1;

    // hardware specific kernel iterations
    dim_t O_h_1;
    dim_t O_w_1;
    CALC_FULL_DIMS((I_h_1 + pad_top_1 + pad_bottom_1), (I_w_1 + pad_left_1 + pad_right_1), _stride_1, F_h_1, F_w_1, O_h_w_pad_1, O_w_w_pad_1);
    SET_PADDING_PARAMS(I_h_1, I_w_1, pad_top_1, pad_bottom_1, pad_left_1, pad_right_1, _stride_1, F_h_1, F_w_1,
                       t_pad_el_1, b_pad_el_1, l_pad_el_1, r_pad_el_1, H_full_index_1, W_full_index_1, H_back_index, W_back_index, O_h_1, O_w_1);

    // setting up microkernel specific parameters
    /*// setting up microkernel specific parameters*/

    const dim_t O_w_full_1 = (O_w_1 / _O_wb) * _O_wb;
    const dim_t O_w_left_1 = O_w_1 - O_w_full_1;

    const dim_t O_hxO_w_1 = O_h_w_pad_1 * O_w_w_pad_1;

#if DEBUG == 1

    printf("\t\t I_h_1 %d I_w_1 %d F_C %d G_1 %d \n", I_h_1, I_w_1, F_c_1, G_1);
    printf("\t\t O_h_pad: %d O_w_w_pad_1 %d \n", O_h_w_pad_1, O_w_w_pad_1);
    printf("O_h_1 %d O_w_1 %d O_w_left_1 %d \n", O_h_1, O_w_full_1, O_w_left_1);
make 
    printf("bottom padding index into input: %d \t bottom padding elements: %d \n", H_back_index, b_pad_el_1);
    printf("no padding index into input: %d \t top padding elements: %d \n", H_full_index_1, t_pad_el_1);

    printf("right padding index into input: %d \t right padding elements: %d \n", W_back_index, r_pad_el_1);
    printf("no padding index into input: %d \t left padding elements: %d \n", W_full_index_1, l_pad_el_1);

    printf("O_w_full_1: %d O_w_left_1: %d \n", O_w_full_1, O_w_left_1);
    printf("new params: _O_wb %d F_Cb %d G_b %d K_b %d\n", _O_wb, _F_cb_1, _G_b_1, _K_b_1);

#endif

// printf("T_Group: %d T_channel: %d\n",T_group, T_channel);
// TODO: parallelize

// create parallel region with all threads
#if PARALLEL == 1
#pragma omp parallel num_threads(N)
#endif
    {
        auto t_id = omp_get_thread_num();
        auto height_tid = t_id % T_height;
        auto channel_tid = ((t_id) / (T_height)) % T_channel;
        auto group_tid = ((t_id / (T_channel * T_height))) % T_group;
        // loops over output channels
        for (index_t g = group_tid; g < G_0 / _G_b_0; g += T_group)
        {
            operand_t *I_group = I + g * (F_c_0 * I_h_0 * I_w_0 * _G_b_0);
            operand_t *F_group = F_0 + g * (K_0 * F_c_0 * F_h_0 * F_w_0 * _G_b_0);
            operand_t *O_group = O_inter + g * (K_0 * O_hxO_w_0 * _G_b_0);
            // printf("I group offset %d \n", (I_group - I)/(I_h_0 * I_w_0));
            // printf("F group offset %d \n", (F_group - F) / (F_h_0 * F_w_0));

            for (index_t k = channel_tid; k < K_0 / _K_b_0; k += T_channel)
            {
                operand_t *I_channel_block_output = I_group + 0;
                operand_t *F_channel_block_output = F_group + k * (F_c_0 * F_h_0 * F_w_0 * _G_b_0 * _K_b_0);
                operand_t *O_channel_block_output = O_group + k * (O_hxO_w_0 * _G_b_0 * _K_b_0);

                // Loop over input channel reduction

                // Peel the last iteration to fuse
                for (index_t i = 0; i < (F_c_0 / _F_cb_0) - 1; i++)
                {
                    bool first = rewrite_output_0 && (i == 0);

                    // printf("\t out_channel:%d, %d before: in_channel %d, %.2f %.2f %.2f %.2f\n", g, k, i,O_channel_block_output[0], O_channel_block_output[1], O_channel_block_output[2], O_channel_block_output[3]);

                    operand_t *I_channel_block_input = I_channel_block_output + i * (I_h_0 * I_w_0 * _F_cb_0 * _G_b_0);
                    // printf("I block offset %d %d \n", (I_channel_block_input - I), i * (I_h_0 * I_w_0 * _F_cb_0 * _G_b_0));
                    operand_t *F_channel_block_input = F_channel_block_output + i * (F_h_0 * F_w_0 * _F_cb_0 * _G_b_0 * _K_b_0);
                    operand_t *O_channel_block_input = O_channel_block_output + 0;

                    // Loops over spatial dimensions of output

                    // Prologue with top padding
                    operand_t *I_row_top = I_channel_block_input;
                    operand_t *F_row_top = F_channel_block_input + 0;
                    operand_t *O_row_top = O_channel_block_input;

                    kernel_top<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0>(
                        first,
                        F_h_0,
                        F_w_0,
                        I_w_0 * _C_ib,
                        t_pad_el_0,
                        pad_top_0,
                        W_full_index_0,
                        l_pad_el_0,
                        pad_left_0,
                        O_w_w_pad_0,
                        O_w_full_0,
                        O_w_left_0,
                        r_pad_el_0,
                        pad_right_0,
                        I_row_top,
                        F_row_top,
                        O_row_top);

                    float *I_row_full = I_row_top + H_full_index_0 * I_w_0 * (_F_cb_0 * _G_b_0);
                    float *O_row_full = O_row_top + t_pad_el_0 * O_w_w_pad_0 * (_G_b_0 * _K_b_0);
                    // Steady State over rows
                    for (index_t j = height_tid; j < O_h_0; j += T_height)
                    {
                        operand_t *I_row = I_row_full + (j * _stride_0) * (I_w_0 * _F_cb_0 * _G_b_0);
                        operand_t *F_row = F_channel_block_input + 0;
                        operand_t *O_row = O_row_full + j * (O_w_w_pad_0 * _G_b_0 * _K_b_0);
                        // Prologue with left padding
                        kernel_left<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0>(
                            first,
                            F_h_0,
                            F_w_0,
                            I_w_0 * _C_ib,
                            l_pad_el_0,
                            pad_left_0,
                            I_row,
                            F_row,
                            O_row);

                        float *I_col_full = I_row + W_full_index_0 * (_F_cb_0 * _G_b_0);
                        float *O_col_full = O_row + l_pad_el_0 * (_G_b_0 * _K_b_0);
                        // Steady State with microkernel
                        for (index_t l = 0; l < O_w_full_0; l += _O_wb)
                        {
                                operand_t *I_col = I_col_full + (l * _stride_0) * (_F_cb_0 * _G_b_0);
                                operand_t *F_col = F_row + 0;
                                operand_t *O_col = O_col_full + l * (_G_b_0 * _K_b_0);
                                // printf("O_col_full %p O_col %p l %d _C_ob %d _O_wb %d\n", O_col, O_col_full, l, (_G_b_0 * _K_b_0), _O_wb);
                                kernel<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0>(
                                    first,
                                    F_h_0,
                                    F_w_0,
                                    I_w_0 * _C_ib,
                                    I_col,
                                    F_col,
                                    O_col);
                                // for (uint32_t o_idx = 0; o_idx < _K_b_0; o_idx++)
                                // {
                                //     printf("idx: %d offset %ld value %f\n", o_idx, O_col - O, O_col[o_idx]);
                                // }
                        }

                        // Epilogue for microkernel + right padding elements
                        operand_t *I_col_left = I_col_full + (O_w_full_0 * _stride_0) * (_F_cb_0 * _G_b_0);
                        operand_t *F_col_left = F_row + 0;
                        operand_t *O_col_left = O_col_full + O_w_full_0 * (_G_b_0 * _K_b_0);
                        kernel_right<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0>(
                            first,
                            F_h_0,
                            F_w_0,
                            I_w_0 * _C_ib,
                            O_w_left_0,
                            r_pad_el_0,
                            pad_right_0,
                            I_col_left,
                            F_col_left,
                            O_col_left);
                    }
                    // Epilogue with bottom padding
                    operand_t *I_row_bot = I_row_full + (O_h_0 * _stride_0) * (I_w_0 * _F_cb_0 * _G_b_0);
                    operand_t *F_row_bot = F_channel_block_input + 0;
                    operand_t *O_row_bot = O_row_full + O_h_0 * (O_w_w_pad_0 * _G_b_0 * _K_b_0);

                    kernel_bottom<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0>(
                        first,
                        F_h_0,
                        F_w_0,
                        I_w_0 * _C_ib,
                        b_pad_el_0,
                        pad_bottom_0,
                        W_full_index_0,
                        l_pad_el_0,
                        pad_left_0,
                        O_w_w_pad_0,
                        O_w_full_0,
                        O_w_left_0,
                        r_pad_el_0,
                        pad_right_0,
                        I_row_bot,
                        F_row_bot,
                        O_row_bot);

                    // printf("\t out_channel:%d %d after: in_channel %d %.2f %.2f %.2f %.2f\n", g, k, i, O_channel_block_output[0], O_channel_block_output[1], O_channel_block_output[2], O_channel_block_output[3]);
                }

                // Last iteration of updates for the first layer

                {
                    index_t i = (F_c_0 / _F_cb_0) - 1;
                    bool first = rewrite_output_0 && (i == 0);

                    // printf("\t out_channel:%d, %d before: in_channel %d, %.2f %.2f %.2f %.2f\n", g, k, i,O_channel_block_output[0], O_channel_block_output[1], O_channel_block_output[2], O_channel_block_output[3]);

                    operand_t *I_channel_block_input = I_channel_block_output + i * (I_h_0 * I_w_0 * _F_cb_0 * _G_b_0);
                    // printf("I block offset %d %d \n", (I_channel_block_input - I), i * (I_h_0 * I_w_0 * _F_cb_0 * _G_b_0));
                    operand_t *F_channel_block_input = F_channel_block_output + i * (F_h_0 * F_w_0 * _F_cb_0 * _G_b_0 * _K_b_0);
                    operand_t *O_channel_block_input = O_channel_block_output + 0;

                    // Loops over spatial dimensions of output

                    // Prologue with top padding
                    operand_t *I_row_top = I_channel_block_input;
                    operand_t *F_row_top = F_channel_block_input + 0;
                    operand_t *O_row_top = O_channel_block_input;

                    kernel_top<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0, fused_ewise_op_class_0, fused_ewise_op_type_0>(
                        first,
                        F_h_0,
                        F_w_0,
                        I_w_0 * _C_ib,
                        t_pad_el_0,
                        pad_top_0,
                        W_full_index_0,
                        l_pad_el_0,
                        pad_left_0,
                        O_w_w_pad_0,
                        O_w_full_0,
                        O_w_left_0,
                        r_pad_el_0,
                        pad_right_0,
                        I_row_top,
                        F_row_top,
                        O_row_top);

                    float *I_row_full = I_row_top + H_full_index_0 * I_w_0 * (_F_cb_0 * _G_b_0);
                    float *O_row_full = O_row_top + t_pad_el_0 * O_w_w_pad_0 * (_G_b_0 * _K_b_0);
                    
                    // Steady State over rows
                    for (index_t j = height_tid; j < O_h_0; j += T_height)
                    {
                        operand_t *I_row = I_row_full + (j * _stride_0) * (I_w_0 * _F_cb_0 * _G_b_0);
                        operand_t *F_row = F_channel_block_input + 0;
                        operand_t *O_row = O_row_full + j * (O_w_w_pad_0 * _G_b_0 * _K_b_0);
                        // Prologue with left padding
                        kernel_left<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0, fused_ewise_op_class_0, fused_ewise_op_type_0>(
                            first,
                            F_h_0,
                            F_w_0,
                            I_w_0 * _C_ib,
                            l_pad_el_0,
                            pad_left_0,
                            I_row,
                            F_row,
                            O_row);

                        float *I_col_full = I_row + W_full_index_0 * (_F_cb_0 * _G_b_0);
                        float *O_col_full = O_row + l_pad_el_0 * (_G_b_0 * _K_b_0);
                        // Steady State with microkernel
                        for (index_t l = 0; l < O_w_full_0; l += _O_wb)
                        {
                                operand_t *I_col = I_col_full + (l * _stride_0) * (_F_cb_0 * _G_b_0);
                                operand_t *F_col = F_row + 0;
                                operand_t *O_col = O_col_full + l * (_G_b_0 * _K_b_0);
                                // printf("O_col_full %p O_col %p l %d _C_ob %d _O_wb %d\n", O_col, O_col_full, l, (_G_b_0 * _K_b_0), _O_wb);
                                kernel<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0, fused_ewise_op_class_0, fused_ewise_op_type_0>(
                                    first,
                                    F_h_0,
                                    F_w_0,
                                    I_w_0 * _C_ib,
                                    I_col,
                                    F_col,
                                    O_col);
                                // for (uint32_t o_idx = 0; o_idx < _K_b_0; o_idx++)
                                // {
                                //     printf("idx: %d offset %ld value %f\n", o_idx, O_col - O, O_col[o_idx]);
                                // }
                        }

                        // Epilogue for microkernel + right padding elements
                        operand_t *I_col_left = I_col_full + (O_w_full_0 * _stride_0) * (_F_cb_0 * _G_b_0);
                        operand_t *F_col_left = F_row + 0;
                        operand_t *O_col_left = O_col_full + O_w_full_0 * (_G_b_0 * _K_b_0);
                        kernel_right<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0, fused_ewise_op_class_0, fused_ewise_op_type_0>(
                            first,
                            F_h_0,
                            F_w_0,
                            I_w_0 * _C_ib,
                            O_w_left_0,
                            r_pad_el_0,
                            pad_right_0,
                            I_col_left,
                            F_col_left,
                            O_col_left);
                    }
                    // Epilogue with bottom padding
                    operand_t *I_row_bot = I_row_full + (O_h_0 * _stride_0) * (I_w_0 * _F_cb_0 * _G_b_0);
                    operand_t *F_row_bot = F_channel_block_input + 0;
                    operand_t *O_row_bot = O_row_full + O_h_0 * (O_w_w_pad_0 * _G_b_0 * _K_b_0);

                    kernel_bottom<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0, fused_ewise_op_class_0, fused_ewise_op_type_0>(
                        first,
                        F_h_0,
                        F_w_0,
                        I_w_0 * _C_ib,
                        b_pad_el_0,
                        pad_bottom_0,
                        W_full_index_0,
                        l_pad_el_0,
                        pad_left_0,
                        O_w_w_pad_0,
                        O_w_full_0,
                        O_w_left_0,
                        r_pad_el_0,
                        pad_right_0,
                        I_row_bot,
                        F_row_bot,
                        O_row_bot);

                    // printf("\t out_channel:%d %d after: in_channel %d %.2f %.2f %.2f %.2f\n", g, k, i, O_channel_block_output[0], O_channel_block_output[1], O_channel_block_output[2], O_channel_block_output[3]);
                }
            }
        }
    }
}

// Fused abstract layer
//  Fused call param
template <
    dim_t _G_b_0,
    dim_t _K_b_0,
    dim_t _F_cb_0,
    dim_t _O_wb,
    dim_t _stride_0,
    dim_t _UNROLL_0,
    char op_type_0,
    int8_t op_class_0,
    bool rewrite_output_0,
    dim_t _G_b_1,
    dim_t _K_b_1,
    dim_t _F_cb_1,
    dim_t _stride_1,
    char fused_ewise_op_type_0,
    int8_t fused_ewise_op_class_0,
    bool rewrite_output_1>
void fused_newise_abstract_layer(
    // Params for first layer
    dim_t G_0,   // Output Channel Grouping
    dim_t K_0,   // Output Channels per group
    dim_t F_c_0, // Channel Reduction Dimension
    dim_t I_h_0, // Input Height
    dim_t I_w_0, // Input Width

    dim_t F_h_0, // Filter height
    dim_t F_w_0, // Filter width

    // Padding values
    dim_t pad_top_0,
    dim_t pad_left_0,
    dim_t pad_right_0,
    dim_t pad_bottom_0,

    // Params for second layer
    dim_t G_1,   // Output Channel Grouping
    dim_t K_1,   // Output Channels per group
    dim_t F_c_1, // Channel Reduction Dimension
    dim_t I_h_1, // Input Height ------------------> These could be inferred
    dim_t I_w_1, // Input Width

    dim_t F_h_1, // Filter height
    dim_t F_w_1, // Filter width

    // Padding values
    dim_t pad_top_1,
    dim_t pad_left_1,
    dim_t pad_right_1,
    dim_t pad_bottom_1,
    // Data
    operand_t *__restrict__ I,
    operand_t *__restrict__ F_0,
    operand_t *__restrict__ O_inter,
    operand_t *__restrict__ F_1,
    operand_t *__restrict__ O)
{

    // Parameters for the first operation
    dim_t t_pad_el_0 = 0, l_pad_el_0 = 0;
    dim_t b_pad_el_0 = 0, r_pad_el_0 = 0;

    // Only need this for debugging
    uint32_t H_back_index = 0, W_back_index = 0;

    uint32_t H_full_index_0 = 0, W_full_index_0 = 0;

    constexpr dim_t _C_ib = _F_cb_0 * _G_b_0;
    constexpr dim_t _C_ib_1 = _F_cb_1 * _G_b_1;
    // Derive parameters for first operation
    // Deriving padding parameters
    dim_t O_h_w_pad_0 = I_h_1;
    dim_t O_w_w_pad_0 = I_w_1;

    // hardware specific kernel iterations
    dim_t O_h_0;
    dim_t O_w_0;
    // CALC_FULL_DIMS((I_h_0 + pad_top_0 + pad_bottom_0), (I_w_0 + pad_left_0 + pad_right_0), _stride_0, F_h_0, F_w_0, O_h_w_pad_0, O_w_w_pad_0);
    SET_PADDING_PARAMS(I_h_0, I_w_0, pad_top_0, pad_bottom_0, pad_left_0, pad_right_0, _stride_0, F_h_0, F_w_0,
                       t_pad_el_0, b_pad_el_0, l_pad_el_0, r_pad_el_0, H_full_index_0, W_full_index_0, H_back_index, W_back_index, O_h_0, O_w_0);

    // setting up microkernel specific parameters
    /*// setting up microkernel specific parameters*/

    const dim_t O_w_full_0 = (O_w_0 / _O_wb) * _O_wb;
    const dim_t O_w_left_0 = O_w_0 - O_w_full_0;

    const dim_t O_hxO_w_0 = O_h_w_pad_0 * O_w_w_pad_0;

    //
#if DEBUG == 1

    printf("\t\t I_h_0 %d I_w_0 %d F_C %d G_0 %d \n", I_h_0, I_w_0, F_c_0, G_0);
    printf("\t\t O_h_pad: %d O_w_w_pad_0 %d \n", O_h_w_pad_0, O_w_w_pad_0);
    printf("O_h_0 %d O_w_0 %d O_w_left_0 %d \n", O_h_0, O_w_full_0, O_w_left_0);

    printf("bottom padding index into input: %d \t bottom padding elements: %d \n", H_back_index, b_pad_el_0);
    printf("no padding index into input: %d \t top padding elements: %d \n", H_full_index_0, t_pad_el_0);

    printf("right padding index into input: %d \t right padding elements: %d \n", W_back_index, r_pad_el_0);
    printf("no padding index into input: %d \t left padding elements: %d \n", W_full_index_0, l_pad_el_0);

    printf("O_w_full_0: %d O_w_left_0: %d \n", O_w_full_0, O_w_left_0);
    printf("new params: _O_wb %d F_Cb %d G_b %d K_b %d\n", _O_wb, _F_cb_0, _G_b_0, _K_b_0);

#endif
    // printf("\t rewrite output?: %d  %c %d\n", rewrite_output, op_type_0, op_class_0);

    // Set up parallelism for the channel loops

    //  Get total available threads
    // TODO: add error checking in case env variable isn't defined.
    int N = 1;
    char const *env_nt(std::getenv("OMP_NUM_THREADS"));
    if (nullptr != env_nt)
    {
        N = atoi(std::getenv("OMP_NUM_THREADS"));
    }

    // const int Channel_blocks = K_0 / _K_b_0;
    // const int Group_blocks = G_0 / _G_b_0;

    int T_channel = N, T_group = 1, T_height_0 = 1, T_height_1;

    // If dwise, parallelize on groups
    if (K_0 == 1)
    {
        T_channel = 1;
        T_group = N;
    }

    //_____________________________________________________________________________________________
    // Parameters for the second operation

    dim_t t_pad_el_1 = 0, l_pad_el_1 = 0;
    dim_t b_pad_el_1 = 0, r_pad_el_1 = 0;

    // uint32_t H_back_index = 0, W_back_index = 0;

    uint32_t H_full_index_1 = 0, W_full_index_1 = 0;

    // Derive parameters for second operation
    // Deriving padding parameters
    dim_t O_h_w_pad_1;
    dim_t O_w_w_pad_1;

    // hardware specific kernel iterations
    dim_t O_h_1;
    dim_t O_w_1;
    CALC_FULL_DIMS((I_h_1 + pad_top_1 + pad_bottom_1), (I_w_1 + pad_left_1 + pad_right_1), _stride_1, F_h_1, F_w_1, O_h_w_pad_1, O_w_w_pad_1);
    SET_PADDING_PARAMS(I_h_1, I_w_1, pad_top_1, pad_bottom_1, pad_left_1, pad_right_1, _stride_1, F_h_1, F_w_1,
                       t_pad_el_1, b_pad_el_1, l_pad_el_1, r_pad_el_1, H_full_index_1, W_full_index_1, H_back_index, W_back_index, O_h_1, O_w_1);

    // setting up microkernel specific parameters
    /*// setting up microkernel specific parameters*/

    const dim_t O_w_full_1 = (O_w_1 / _O_wb) * _O_wb;
    const dim_t O_w_left_1 = O_w_1 - O_w_full_1;

    const dim_t O_hxO_w_1 = O_h_w_pad_1 * O_w_w_pad_1;

#if DEBUG == 1

    printf("\t\t I_h_1 %d I_w_1 %d F_C %d G_1 %d \n", I_h_1, I_w_1, F_c_1, G_1);
    printf("\t\t O_h_pad: %d O_w_w_pad_1 %d \n", O_h_w_pad_1, O_w_w_pad_1);
    printf("O_h_1 %d O_w_1 %d O_w_left_1 %d \n", O_h_1, O_w_full_1, O_w_left_1);

    printf("bottom padding index into input: %d \t bottom padding elements: %d \n", H_back_index, b_pad_el_1);
    printf("no padding index into input: %d \t top padding elements: %d \n", H_full_index_1, t_pad_el_1);

    printf("right padding index into input: %d \t right padding elements: %d \n", W_back_index, r_pad_el_1);
    printf("no padding index into input: %d \t left padding elements: %d \n", W_full_index_1, l_pad_el_1);

    printf("O_w_full_1: %d O_w_left_1: %d \n", O_w_full_1, O_w_left_1);
    printf("new params: _O_wb %d F_Cb %d G_b %d K_b %d\n", _O_wb, _F_cb_1, _G_b_1, _K_b_1);

#endif

// printf("T_Group: %d T_channel: %d\n",T_group, T_channel);
// TODO: parallelize

// create parallel region with all threads
#if PARALLEL == 1
#pragma omp parallel num_threads(N)
#endif
    {
        auto t_id = omp_get_thread_num();
        auto height_tid = t_id % T_height_0;
        auto channel_tid = ((t_id) / (T_height_0)) % T_channel;
        auto group_tid = ((t_id / (T_channel * T_height_0))) % T_group;
        // loops over output channels
        for (index_t g = group_tid; g < G_0 / _G_b_0; g += T_group)
        {
            operand_t *I_group = I + g * (F_c_0 * I_h_0 * I_w_0 * _G_b_0);
            operand_t *F_group = F_0 + g * (K_0 * F_c_0 * F_h_0 * F_w_0 * _G_b_0);
            operand_t *O_group = O_inter + g * (K_0 * O_hxO_w_0 * _G_b_0);
            // printf("I group offset %d \n", (I_group - I)/(I_h_0 * I_w_0));
            // printf("F group offset %d \n", (F_group - F) / (F_h_0 * F_w_0));

            for (index_t k = channel_tid; k < K_0 / _K_b_0; k += T_channel)
            {
                operand_t *I_channel_block_output = I_group + 0;
                operand_t *F_channel_block_output = F_group + k * (F_c_0 * F_h_0 * F_w_0 * _G_b_0 * _K_b_0);
                operand_t *O_channel_block_output = O_group + k * (O_hxO_w_0 * _G_b_0 * _K_b_0);

                // Loop over input channel reduction

                // Peel the last iteration to fuse
                for (index_t i = 0; i < (F_c_0 / _F_cb_0) - 1; i++)
                {
                    bool first = rewrite_output_0 && (i == 0);

                    // printf("\t out_channel:%d, %d before: in_channel %d, %.2f %.2f %.2f %.2f\n", g, k, i,O_channel_block_output[0], O_channel_block_output[1], O_channel_block_output[2], O_channel_block_output[3]);

                    operand_t *I_channel_block_input = I_channel_block_output + i * (I_h_0 * I_w_0 * _F_cb_0 * _G_b_0);
                    // printf("I block offset %d %d \n", (I_channel_block_input - I), i * (I_h_0 * I_w_0 * _F_cb_0 * _G_b_0));
                    operand_t *F_channel_block_input = F_channel_block_output + i * (F_h_0 * F_w_0 * _F_cb_0 * _G_b_0 * _K_b_0);
                    operand_t *O_channel_block_input = O_channel_block_output + 0;

                    // Loops over spatial dimensions of output

                    // Prologue with top padding
                    operand_t *I_row_top = I_channel_block_input;
                    operand_t *F_row_top = F_channel_block_input + 0;
                    operand_t *O_row_top = O_channel_block_input;

                    kernel_top<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0>(
                        first,
                        F_h_0,
                        F_w_0,
                        I_w_0 * _C_ib,
                        t_pad_el_0,
                        pad_top_0,
                        W_full_index_0,
                        l_pad_el_0,
                        pad_left_0,
                        O_w_w_pad_0,
                        O_w_full_0,
                        O_w_left_0,
                        r_pad_el_0,
                        pad_right_0,
                        I_row_top,
                        F_row_top,
                        O_row_top);

                    float *I_row_full = I_row_top + H_full_index_0 * I_w_0 * (_F_cb_0 * _G_b_0);
                    float *O_row_full = O_row_top + t_pad_el_0 * O_w_w_pad_0 * (_G_b_0 * _K_b_0);
                    // Steady State over rows
                    for (index_t j = height_tid; j < O_h_0; j += T_height_0)
                    {
                        operand_t *I_row = I_row_full + (j * _stride_0) * (I_w_0 * _F_cb_0 * _G_b_0);
                        operand_t *F_row = F_channel_block_input + 0;
                        operand_t *O_row = O_row_full + j * (O_w_w_pad_0 * _G_b_0 * _K_b_0);
                        // Prologue with left padding
                        kernel_left<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0>(
                            first,
                            F_h_0,
                            F_w_0,
                            I_w_0 * _C_ib,
                            l_pad_el_0,
                            pad_left_0,
                            I_row,
                            F_row,
                            O_row);

                        float *I_col_full = I_row + W_full_index_0 * (_F_cb_0 * _G_b_0);
                        float *O_col_full = O_row + l_pad_el_0 * (_G_b_0 * _K_b_0);
                        // Steady State with microkernel
                        for (index_t l = 0; l < O_w_full_0; l += _O_wb)
                        {
                                operand_t *I_col = I_col_full + (l * _stride_0) * (_F_cb_0 * _G_b_0);
                                operand_t *F_col = F_row + 0;
                                operand_t *O_col = O_col_full + l * (_G_b_0 * _K_b_0);
                                // printf("O_col_full %p O_col %p l %d _C_ob %d _O_wb %d\n", O_col, O_col_full, l, (_G_b_0 * _K_b_0), _O_wb);
                                kernel<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0>(
                                    first,
                                    F_h_0,
                                    F_w_0,
                                    I_w_0 * _C_ib,
                                    I_col,
                                    F_col,
                                    O_col);
                                // for (uint32_t o_idx = 0; o_idx < _K_b_0; o_idx++)
                                // {
                                //     printf("idx: %d offset %ld value %f\n", o_idx, O_col - O, O_col[o_idx]);
                                // }
                        }

                        // Epilogue for microkernel + right padding elements
                        operand_t *I_col_left = I_col_full + (O_w_full_0 * _stride_0) * (_F_cb_0 * _G_b_0);
                        operand_t *F_col_left = F_row + 0;
                        operand_t *O_col_left = O_col_full + O_w_full_0 * (_G_b_0 * _K_b_0);
                        kernel_right<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0>(
                            first,
                            F_h_0,
                            F_w_0,
                            I_w_0 * _C_ib,
                            O_w_left_0,
                            r_pad_el_0,
                            pad_right_0,
                            I_col_left,
                            F_col_left,
                            O_col_left);
                    }
                    // Epilogue with bottom padding
                    operand_t *I_row_bot = I_row_full + (O_h_0 * _stride_0) * (I_w_0 * _F_cb_0 * _G_b_0);
                    operand_t *F_row_bot = F_channel_block_input + 0;
                    operand_t *O_row_bot = O_row_full + O_h_0 * (O_w_w_pad_0 * _G_b_0 * _K_b_0);

                    kernel_bottom<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0>(
                        first,
                        F_h_0,
                        F_w_0,
                        I_w_0 * _C_ib,
                        b_pad_el_0,
                        pad_bottom_0,
                        W_full_index_0,
                        l_pad_el_0,
                        pad_left_0,
                        O_w_w_pad_0,
                        O_w_full_0,
                        O_w_left_0,
                        r_pad_el_0,
                        pad_right_0,
                        I_row_bot,
                        F_row_bot,
                        O_row_bot);

                    // printf("\t out_channel:%d %d after: in_channel %d %.2f %.2f %.2f %.2f\n", g, k, i, O_channel_block_output[0], O_channel_block_output[1], O_channel_block_output[2], O_channel_block_output[3]);
                }

                // Last iteration of updates for the first layer

                {
                    index_t i = (F_c_0 / _F_cb_0) - 1;
                    bool first = rewrite_output_0 && (i == 0);

                    // printf("\t out_channel:%d, %d before: in_channel %d, %.2f %.2f %.2f %.2f\n", g, k, i,O_channel_block_output[0], O_channel_block_output[1], O_channel_block_output[2], O_channel_block_output[3]);

                    operand_t *I_channel_block_input = I_channel_block_output + i * (I_h_0 * I_w_0 * _F_cb_0 * _G_b_0);
                    // printf("I block offset %d %d \n", (I_channel_block_input - I), i * (I_h_0 * I_w_0 * _F_cb_0 * _G_b_0));
                    operand_t *F_channel_block_input = F_channel_block_output + i * (F_h_0 * F_w_0 * _F_cb_0 * _G_b_0 * _K_b_0);
                    operand_t *O_channel_block_input = O_channel_block_output + 0;

                    // Pointers for 2nd operation
                    // total channel offset for the 1st operation
                    dim_t output_0_channel_offset = g * (K_0) + k;
                    operand_t *F_1_channel_block = F_1 + output_0_channel_offset * (K_1 * F_c_1 * F_h_1 * F_w_1 * _G_b_1);
                    operand_t *O_1_channel_offset = O + output_0_channel_offset * (O_hxO_w_1 * _G_b_1);

                    // Loops over spatial dimensions of output
                    // Peel First F_h_1 - S_h_1 elements of the loop over output rows of the 1st operation
                    // TODO: work out the case where t_pad_el_0 is greater than the number of rows required to be peeled.
                    int req_peeled_output_rows = F_h_1 - _stride_1 - t_pad_el_0;
                    dim_t O_h_0_peeled = (req_peeled_output_rows >= 0) ? req_peeled_output_rows : 0;
                    // printf("F_h_1 : %d Stride: %d req_peel: %d O_h_0_peel: %d\n", F_h_1, _stride_1, req_peeled_output_rows, O_h_0_peeled);
                    // Prologue with top padding
                    operand_t *I_row_top = I_channel_block_input;
                    operand_t *F_row_top = F_channel_block_input + 0;
                    operand_t *O_row_top = O_channel_block_input;

                    kernel_top<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0>(
                        first,
                        F_h_0,
                        F_w_0,
                        I_w_0 * _C_ib,
                        t_pad_el_0,
                        pad_top_0,
                        W_full_index_0,
                        l_pad_el_0,
                        pad_left_0,
                        O_w_w_pad_0,
                        O_w_full_0,
                        O_w_left_0,
                        r_pad_el_0,
                        pad_right_0,
                        I_row_top,
                        F_row_top,
                        O_row_top);

                    float *I_row_full = I_row_top + H_full_index_0 * I_w_0 * (_F_cb_0 * _G_b_0);
                    float *O_row_full = O_row_top + t_pad_el_0 * O_w_w_pad_0 * (_G_b_0 * _K_b_0);

                    for (index_t j_peel = 0; j_peel < O_h_0_peeled; j_peel++)
                    {
                        operand_t *I_row = I_row_full + (j_peel * _stride_0) * (I_w_0 * _F_cb_0 * _G_b_0);
                        operand_t *F_row = F_channel_block_input + 0;
                        operand_t *O_row = O_row_full + j_peel * (O_w_w_pad_0 * _G_b_0 * _K_b_0);
                        // Prologue with left padding
                        kernel_left<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0, fused_ewise_op_class_0, fused_ewise_op_type_0>(
                            first,
                            F_h_0,
                            F_w_0,
                            I_w_0 * _C_ib,
                            l_pad_el_0,
                            pad_left_0,
                            I_row,
                            F_row,
                            O_row);

                        float *I_col_full = I_row + W_full_index_0 * (_F_cb_0 * _G_b_0);
                        float *O_col_full = O_row + l_pad_el_0 * (_G_b_0 * _K_b_0);
                        // Steady State with microkernel
                        for (index_t l = 0; l < O_w_full_0; l += _O_wb)
                        {
                                operand_t *I_col = I_col_full + (l * _stride_0) * (_F_cb_0 * _G_b_0);
                                operand_t *F_col = F_row + 0;
                                operand_t *O_col = O_col_full + l * (_G_b_0 * _K_b_0);
                                // printf("O_col_full %p O_col %p l %d _C_ob %d _O_wb %d\n", O_col, O_col_full, l, (_G_b_0 * _K_b_0), _O_wb);
                                kernel<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0, fused_ewise_op_class_0, fused_ewise_op_type_0>(
                                    first,
                                    F_h_0,
                                    F_w_0,
                                    I_w_0 * _C_ib,
                                    I_col,
                                    F_col,
                                    O_col);
                                // for (uint32_t o_idx = 0; o_idx < _K_b_0; o_idx++)
                                // {
                                //     printf("idx: %d offset %ld value %f\n", o_idx, O_col - O, O_col[o_idx]);
                                // }
                        }

                        // Epilogue for microkernel + right padding elements
                        operand_t *I_col_left = I_col_full + (O_w_full_0 * _stride_0) * (_F_cb_0 * _G_b_0);
                        operand_t *F_col_left = F_row + 0;
                        operand_t *O_col_left = O_col_full + O_w_full_0 * (_G_b_0 * _K_b_0);
                        kernel_right<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0, fused_ewise_op_class_0, fused_ewise_op_type_0>(
                            first,
                            F_h_0,
                            F_w_0,
                            I_w_0 * _C_ib,
                            O_w_left_0,
                            r_pad_el_0,
                            pad_right_0,
                            I_col_left,
                            F_col_left,
                            O_col_left);
                    }


                    // TODO: top padding row for second operation
                    operand_t *O_1_row_top = O_1_channel_offset;
                    // float *I_row_full = I_row_top + H_full_index_0 * I_w_0 * (_F_cb_0 * _G_b_0);
                    float *O_1_row_full = O_1_row_top + t_pad_el_1 * O_w_w_pad_1 * (_G_b_1 * _K_b_1);
                    // Steady State over output rows o the second operation
                    for (index_t j = 0; j < O_h_1; j ++)
                    {
                        index_t j_0_offset = j * _stride_1;
                        
                        index_t j_end = (j + 1) * _stride_1;
                        // printf("Calc 2nd output row %d starting with input row  %d - %d \n", j, j_0_offset, j_end);
                        for (index_t j_peel = j_0_offset; j_peel < j_end; j_peel++)
                        {
                                operand_t *I_row = I_row_full + (j_peel * _stride_0) * (I_w_0 * _F_cb_0 * _G_b_0);
                                operand_t *F_row = F_channel_block_input + 0;
                                operand_t *O_row = O_row_full + j_peel * (O_w_w_pad_0 * _G_b_0 * _K_b_0);
                                // Prologue with left padding
                                kernel_left<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0>(
                                    first,
                                    F_h_0,
                                    F_w_0,
                                    I_w_0 * _C_ib,
                                    l_pad_el_0,
                                    pad_left_0,
                                    I_row,
                                    F_row,
                                    O_row);

                                float *I_col_full = I_row + W_full_index_0 * (_F_cb_0 * _G_b_0);
                                float *O_col_full = O_row + l_pad_el_0 * (_G_b_0 * _K_b_0);
                                // Steady State with microkernel
                                for (index_t l = 0; l < O_w_full_0; l += _O_wb)
                                {
                                    operand_t *I_col = I_col_full + (l * _stride_0) * (_F_cb_0 * _G_b_0);
                                    operand_t *F_col = F_row + 0;
                                    operand_t *O_col = O_col_full + l * (_G_b_0 * _K_b_0);
                                    // printf("O_col_full %p O_col %p l %d _C_ob %d _O_wb %d\n", O_col, O_col_full, l, (_G_b_0 * _K_b_0), _O_wb);
                                    kernel<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0, fused_ewise_op_class_0, fused_ewise_op_type_0>(
                                        first,
                                        F_h_0,
                                        F_w_0,
                                        I_w_0 * _C_ib,
                                        I_col,
                                        F_col,
                                        O_col);
                                    // for (uint32_t o_idx = 0; o_idx < _K_b_0; o_idx++)
                                    // {
                                    //     printf("idx: %d offset %ld value %f\n", o_idx, O_col - O, O_col[o_idx]);
                                    // }
                                }

                                // Epilogue for microkernel + right padding elements
                                operand_t *I_col_left = I_col_full + (O_w_full_0 * _stride_0) * (_F_cb_0 * _G_b_0);
                                operand_t *F_col_left = F_row + 0;
                                operand_t *O_col_left = O_col_full + O_w_full_0 * (_G_b_0 * _K_b_0);
                                kernel_right<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0, fused_ewise_op_class_0, fused_ewise_op_type_0>(
                                    first,
                                    F_h_0,
                                    F_w_0,
                                    I_w_0 * _C_ib,
                                    O_w_left_0,
                                    r_pad_el_0,
                                    pad_right_0,
                                    I_col_left,
                                    F_col_left,
                                    O_col_left);
                        }

                        // printf("I[416]: %f\n", O_inter[432]);
                        operand_t *I_row_1 = O_channel_block_input + (j * _stride_1) * (I_w_1 * _F_cb_1 * _G_b_1);
                        operand_t *F_1_row = F_1_channel_block + 0;
                        operand_t *O_1_row = O_1_row_full + j * (O_w_w_pad_0 * _G_b_0 * _K_b_0);
                        // Prologue with left padding
                        kernel_left<_G_b_1, _K_b_1, _F_cb_1, _O_wb, _stride_1, _UNROLL_0, fused_ewise_op_type_0,  fused_ewise_op_class_0>(
                            1,
                            F_h_1,
                            F_w_1,
                            I_w_1 * _C_ib_1,
                            l_pad_el_1,
                            pad_left_1,
                            I_row_1,
                            F_1_row,
                            O_1_row);

                        float *I_col_full_1 = I_row_1 + W_full_index_1 * (_F_cb_1 * _G_b_1);
                        float *O_1_col_full = O_1_row + l_pad_el_1 * (_G_b_1 * _K_b_1);
                        // Steady State with microkernel
                        for (index_t l = 0; l < O_w_full_1; l += _O_wb)
                        {
                                operand_t *I_col = I_col_full_1 + (l * _stride_1) * (_F_cb_1 * _G_b_1);
                                operand_t *F_col = F_1_row + 0;
                                operand_t *O_col = O_1_col_full + l * (_G_b_1 * _K_b_1);
                                // printf("O_col_full %p O_col %p l %d _C_ob %d _O_wb %d\n", O_col, O_col_full, l, (_G_b_1 * _K_b_1), _O_wb);
                                kernel<_G_b_1, _K_b_1, _F_cb_1, _O_wb, _stride_1, _UNROLL_0, fused_ewise_op_type_0, fused_ewise_op_class_0>(
                                    1,
                                    F_h_1,
                                    F_w_1,
                                    I_w_1 * _C_ib_1,
                                    I_col,
                                    F_col,
                                    O_col);
                                // for (uint32_t o_idx = 0; o_idx < _G_b_1; o_idx++)
                                // {
                                //     printf("idx: %d offset %ld value %f on %f %d %d %d %c\n", o_idx, O_col - O, O_col[o_idx], I_col[o_idx], _F_cb_1, _UNROLL_0, fused_ewise_op_class_0, fused_ewise_op_type_0);
                                // }
                        }

                        // Epilogue for microkernel + right padding elements
                        operand_t *I_col_left = I_col_full_1 + (O_w_full_1 * _stride_1) * (_F_cb_1 * _G_b_1);
                        operand_t *F_col_left = F_1_row + 0;
                        operand_t *O_col_left = O_1_col_full + O_w_full_1 * (_G_b_1 * _K_b_1);
                        kernel_right<_G_b_1, _K_b_1, _F_cb_1, _O_wb, _stride_1, _UNROLL_0, fused_ewise_op_type_0, fused_ewise_op_class_0>(
                            1,
                            F_h_1,
                            F_w_1,
                            I_w_1 * _C_ib_1,
                            O_w_left_1,
                            r_pad_el_1,
                            pad_right_1,
                            I_col_left,
                            F_col_left,
                            O_col_left);
                    }

                    // First Operation
                    //TODO : calculate any full rows that were missed above

                    //  Epilogue with bottom padding
                    operand_t *I_row_bot = I_row_full + (O_h_0 * _stride_0) * (I_w_0 * _F_cb_0 * _G_b_0);
                    operand_t *F_row_bot = F_channel_block_input + 0;
                    operand_t *O_row_bot = O_row_full + O_h_0 * (O_w_w_pad_0 * _G_b_0 * _K_b_0);

                    kernel_bottom<_G_b_0, _K_b_0, _F_cb_0, _O_wb, _stride_0, _UNROLL_0, op_type_0, op_class_0>(
                        first,
                        F_h_0,
                        F_w_0,
                        I_w_0 * _C_ib,
                        b_pad_el_0,
                        pad_bottom_0,
                        W_full_index_0,
                        l_pad_el_0,
                        pad_left_0,
                        O_w_w_pad_0,
                        O_w_full_0,
                        O_w_left_0,
                        r_pad_el_0,
                        pad_right_0,
                        I_row_bot,
                        F_row_bot,
                        O_row_bot);

                    // printf("\t out_channel:%d %d after: in_channel %d %.2f %.2f %.2f %.2f\n", g, k, i, O_channel_block_output[0], O_channel_block_output[1], O_channel_block_output[2], O_channel_block_output[3]);
                }
            }
        }
    }
}
