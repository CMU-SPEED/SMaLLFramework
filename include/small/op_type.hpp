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
    OP_CONV = 0,          // 'c'
    OP_RELU = 1,          // 'a'
    OP_MAX_POOL = 2,      // 'p'
    OP_LEAKY_RELU = 3,    // 'l'
    OP_ADD = 4,           // 'd'
    OP_AVERAGE_POOL = 5,  // 's'
    OP_MUL = 6,
    OP_UPSAMPLE = 7,      // 'u'
    OP_EXP = 8
};

}
