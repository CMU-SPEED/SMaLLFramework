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

#include <small/op_type.hpp>

namespace small
{
namespace detail
{

//****************************************************************************
//****************************************************************************
/// @todo use constexpr if on op_type and op_class in calling code
#define FLOAT_ABSTRACT_OP(step, op_type, op_class, a_cur, b_cur, O_wb, C_ob) \
    if constexpr (op_type == OP_CONV)                                    \
    {                                                                    \
        if constexpr (op_class == 1)                                     \
        {                                                                \
            FLOAT_DW_TILE_C(step, a_cur, b_cur, O_wb, C_ob);             \
        }                                                                \
        else if constexpr (op_class == 2)                                \
        {                                                                \
            FLOAT_CONV_TILE_C(step, a_cur, b_cur, O_wb, C_ob);           \
        }                                                                \
    }                                                                    \
    else if constexpr (op_type == OP_RELU || op_type == OP_MAX_POOL)     \
    {                                                                    \
        FLOAT_MAX_TILE_C(step, a_cur, O_wb, C_ob);                       \
    }                                                                    \
    else if constexpr (op_type == OP_LEAKY_RELU)                         \
    {                                                                    \
        FLOAT_COND_SCALE_TILE_C(step, a_cur, b_cur, O_wb, C_ob);         \
    }                                                                    \
    else if constexpr (op_type == OP_ADD || op_type == OP_AVERAGE_POOL)  \
    {                                                                    \
        FLOAT_ACCUM_TILE_C(step, a_cur, O_wb, C_ob);                     \
    }                                                                    \
    else if constexpr (op_type == OP_MUL)                                \
    {                                                                    \
        float drop_out_rate = b_cur[0];                                  \
        FLOAT_DIV_TILE_C(drop_out_rate, O_wb, C_ob)                      \
    }                                                                    \
    else if constexpr (op_type == OP_EXP)                                \
    {                                                                    \
        FLOAT_EXP_TILE_C(step, a_cur, O_wb, C_ob)                        \
    }

//****************************************************************************
#define FLOAT_ABSTRACT_OP_END(step, op_type, op_class, a_cur, b_cur, c_cur, W_elements, C_ob) \
    if constexpr (op_type == OP_CONV)                                         \
    {                                                                         \
        if constexpr (op_class == 1)                                          \
        {                                                                     \
            FLOAT_DW_END_C(step, a_cur, b_cur, c_cur, W_elements, C_ob);      \
        }                                                                     \
        else if constexpr (op_class == 2)                                     \
        {                                                                     \
            FLOAT_CONV_END_C(step, a_cur, b_cur, c_cur, W_elements, C_ob);    \
        }                                                                     \
    }                                                                         \
    else if constexpr (op_type == OP_RELU || op_type == OP_MAX_POOL)          \
    {                                                                         \
        FLOAT_MAX_END_C(step, a_cur, c_cur, W_elements, C_ob);                \
    }                                                                         \
    else if constexpr (op_type == OP_LEAKY_RELU)                              \
    {                                                                         \
        FLOAT_COND_SCALE_END_C(step, a_cur, b_cur, c_cur, W_elements, C_ob);  \
    }                                                                         \
    else if constexpr (op_type == OP_ADD || op_type == OP_AVERAGE_POOL)       \
    {                                                                         \
        FLOAT_ACCUM_END_C(step, a_cur, c_cur, W_elements, C_ob);              \
    }                                                                         \
    else if constexpr (op_type == OP_MUL)                                     \
    {                                                                         \
        float drop_out_rate = b_cur[0];                                       \
        FLOAT_DIV_END_C(c_cur, drop_out_rate, W_elements, C_ob);              \
    }                                                                         \
    else if constexpr (op_type == OP_EXP)                                     \
    {                                                                         \
        FLOAT_EXP_END_C(step, a_cur, c_cur,  W_elements, C_ob);               \
    }

//****************************************************************************
#define FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, op_type, op_class, b_cur, O_wb, C_ob) \
    if constexpr (op_type == OP_RELU)                                   \
    {                                                                   \
        FLOAT_FUSED_RELU_TILE_C(O_wb, C_ob);                            \
    }                                                                   \
    else if constexpr (op_type == OP_LEAKY_RELU)                        \
    {                                                                   \
        FLOAT_FUSED_COND_SCALE_TILE_C(b_cur, O_wb, C_ob);               \
    }                                                                   \
    else if constexpr (op_type == OP_ADD)                               \
    {                                                                   \
        FLOAT_ACCUM_TILE_C(step, b_cur, O_wb, C_ob);                    \
    }                                                                   \
    else if constexpr (op_type == OP_MUL)                               \
    {                                                                   \
        float drop_out_rate = b_cur[0];                                 \
        FLOAT_DIV_TILE_C(drop_out_rate, O_wb, C_ob);                    \
    }                                                                   \
    else if constexpr (op_type == OP_EXP)                               \
    {                                                                   \
        FLOAT_FUSED_EXP_TILE_C(O_wb, C_ob) ;                            \
    }

//****************************************************************************
#define FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_END(step, op_type, op_class, b_cur, c_cur, W_elements, C_ob) \
    if constexpr (op_type == OP_RELU)                                   \
    {                                                                   \
        FLOAT_FUSED_RELU_END_C(c_cur, W_elements, C_ob);                \
    }                                                                   \
    else if constexpr (op_type == OP_LEAKY_RELU)                        \
    {                                                                   \
        FLOAT_FUSED_COND_SCALE_END_C(b_cur, c_cur, W_elements, C_ob);   \
    }                                                                   \
    else if constexpr (op_type == OP_ADD)                               \
    {                                                                   \
        FLOAT_ACCUM_END_C(step, b_cur, c_cur, W_elements, C_ob);        \
    }                                                                   \
    else if constexpr (op_type == OP_MUL)                               \
    {                                                                   \
        float drop_out_rate = b_cur[0];                                 \
        FLOAT_DIV_END_C(c_cur, drop_out_rate, W_elements, C_ob);        \
    }                                                                   \
    else if constexpr (op_type == OP_EXP)                               \
    {                                                                   \
        FLOAT_FUSED_EXP_END_C(c_cur, W_elements, C_ob) ;                \
    }

//****************************************************************************
#define FLOAT_ABSTRACT_SINGLE_ELEMENT_UPSAMPLE_OP_END(out_step, op_type, op_class) \
    else if constexpr (op_type == OP_UPSAMPLE)                              \
    {                                                                       \
        for (index_t j_s = 0; j_s < out_step / _C_ib; j_s++)                \
        { /*@todo: we need the output column width as a parameter*/         \
            for (index_t kk_s = 0; kk_s < out_step / _C_ib; kk_s++)         \
            { /*@todo: add a strided store, move this to the kernel stage*/ \
                FLOAT_STORE_END_C_strided();                                \
            }                                                               \
        }                                                                   \
    }


} // ns detail
} // ns small
