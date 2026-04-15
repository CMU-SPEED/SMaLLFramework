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

namespace small
{

/// op_types used to control how abstract_layer() performs computation.
enum OpType
{
    OP_CONV,          // 'c'
    OP_RELU,          // 'a'
    OP_MAX_POOL,      // 'p'
    OP_LEAKY_RELU,    // 'l'
    OP_ADD,           // 'd'
    OP_AVERAGE_POOL,  // 's'
    OP_MUL,
    OP_UPSAMPLE,      // 'u'
    OP_EXP,
    OP_INPLACE_ADD_SCALAR,
    OP_SOFTSIGN,
    OP_ABS,
    OP_DIV, // c_i = a_i / c_i
    OP_SLOPE_RELU,
    OP_CELU,
    OP_FUSED_SLOPE_RELU,
    OP_FUSED_CELU,
    OP_SOFTMAX,
    OP_FUSED_SOFTMAX,
    OP_NONE = -1
};

}
