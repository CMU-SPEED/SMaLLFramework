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
#include <small/abstract_op.hpp>

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
          // TODO: add a bool to switch between microkernel and default imp
          // Leaf to describe abstract operation
          OpType op_type,
          int8_t op_class>
void inline compute_with_padding_1D(dim_t W_lb, dim_t W_ub,
                                    dim_t F_w,
                                    dim_t W_elements,
                                    dim_t input_col_stride,
                                    ScalarT const *F,
                                    ScalarT const *I,
                                    c_tile_t *c_cur) /// @todo need to deal with type
{
    constexpr dim_t _C_ob = _G_b * _K_b;
    constexpr dim_t _C_ib = _G_b * _F_cb;
    constexpr dim_t step = _stride * _C_ib;

    for (uint32_t m = W_lb; m < W_ub; m++)
    {
        int filter_offset_w = m * _F_cb * _G_b * _K_b; // + filter_offset_h;
        /* This is C_ib because the microkernel stretches across groups*/
        int input_stencil_w = (m - W_lb) * _C_ib;// + input_stencil_h;

        ScalarT const *b = F + filter_offset_w;
        ScalarT const *a = I + input_stencil_w;

        // TODO: reintroduce convolution
        for (uint32_t ii = 0; ii < _F_cb / _UNROLL; ii++)
        {
            /// @note using platform C_ob
            ScalarT const *b_cur = b + ii * _UNROLL * FLOAT_C_ob;
            ScalarT const *a_cur = a + ii * _UNROLL;
            FLOAT_ABSTRACT_OP_END(step, op_type, op_class, a_cur, b_cur, c_cur, W_elements, _C_ob);
        }
    }
}

} // ns detail
} // ns small
