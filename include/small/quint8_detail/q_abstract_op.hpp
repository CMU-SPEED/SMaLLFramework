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

#include <stdexcept>
#include <small/op_type.hpp>

namespace small
{
namespace quint8_detail
{

//****************************************************************************
//****************************************************************************
/// @todo add parameters: step, _O_wb, _C_ob
/// @todo use constexpr if on op_type and op_class in calling code
#define QUINT8_ABSTRACT_OP(op_type, op_class, a_cur, b_cur, a_offset, b_offset) \
    if constexpr (op_type == OP_CONV)                                   \
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
    else if constexpr (op_type == OP_RELU || op_type == OP_MAX_POOL)    \
    {                                                                   \
        QUINT8_MAX_TILE_C(step, a_cur, _O_wb, _C_ob, a_offset);         \
    }                                                                   \
    else if constexpr (op_type == OP_LEAKY_RELU)                        \
    {                                                                   \
        /*QUINT8_COND_SCALE_TILE_C(step, a_cur, b_cur, _O_wb, _C_ob);*/ \
        throw std::invalid_argument("*_ABSTRACT_OP ERROR: no support for op_type OP_LEAKY_RELU"); \
    }                                                                   \
    else if constexpr (op_type == OP_ADD || op_type == OP_AVERAGE_POOL) \
    {                                                                   \
        /*QUINT8_ACCUM_TILE_C(step, a_cur, _O_wb, _C_ob);*/             \
        throw std::invalid_argument("*_ABSTRACT_OP ERROR: no support for op_type OP_ADD/AVERAGE_POOL"); \
    }                                                                    \
    else if constexpr (op_type == OP_MUL)                                \
    {                                                                    \
        /*float drop_out_rate = b_cur[0]; */                             \
        /*QUINT8_DIV_TILE_C(drop_out_rate, O_wb, C_ob) */                \
        throw std::invalid_argument("*_ABSTRACT_OP ERROR: no support for op_type OP_MUL"); \
    }                                                                       \
    else if constexpr (op_type == OP_EXP)                                \
    {                                                                    \
        /* FLOAT_EXP_TILE_C(step, a_cur, O_wb, C_ob) */                  \
        throw std::invalid_argument("*_ABSTRACT_OP ERROR: no support for op_type OP_EXP"); \
    }

//****************************************************************************
/// @todo add parameters: step, W_elements, _C_ob
#define QUINT8_ABSTRACT_OP_END(op_type, op_class, a_cur, b_cur, c_cur, a_offset, b_offset) \
    if constexpr (op_type == OP_CONV)                                   \
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
    else if constexpr (op_type == OP_RELU || op_type == OP_MAX_POOL)    \
    {                                                                   \
        QUINT8_MAX_END_C(step, a_cur, c_cur, W_elements, _C_ob, a_offset); \
    }                                                                   \
    else if constexpr (op_type == OP_LEAKY_RELU)                        \
    {                                                                   \
        /*QUINT8_COND_SCALE_END_C(step, a_cur, b_cur, c_cur, W_elements, _C_ob, a_offset, b_offset);*/ \
        throw std::invalid_argument("*_ABSTRACT_OP_END ERROR: no support for op_type OP_LEAKY_RELU"); \
    }                                                                   \
    else if constexpr (op_type == OP_ADD || op_type == OP_AVERAGE_POOL) \
    {                                                                   \
        /*QUINT8_ACCUM_END_C(step, a_cur, c_cur, W_elements, _C_ob, a_offset);*/ \
        throw std::invalid_argument("*_ABSTRACT_OP_END ERROR: no support for op_type OP_ADD/AVERAGE_POOL"); \
    }                                                    \
    else if constexpr (op_type == OP_MUL)                                \
    {                                                                    \
        /*float drop_out_rate = b_cur[0]; */                             \
        /*QUINT8_DIV_TILE_C(drop_out_rate, O_wb, C_ob) */                \
        throw std::invalid_argument("*_ABSTRACT_OP ERROR: no support for op_type OP_MUL"); \
    }                                                                       \
    else if constexpr (op_type == OP_EXP)                                \
    {                                                                    \
        /* FLOAT_EXP_TILE_C(step, a_cur, O_wb, C_ob) */                  \
        throw std::invalid_argument("*_ABSTRACT_OP ERROR: no support for op_type OP_EXP"); \
    }

} // ns quint8_detail
} // ns small
