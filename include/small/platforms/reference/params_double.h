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

#define DOUBLE_W_ob   1
#define DOUBLE_C_ob   1
#define DOUBLE_SIMD   1
#define DOUBLE_UNROLL 1
#define DOUBLE_C_ib   DOUBLE_C_ob

//Potential blocking parameters for packing
#define DOUBLE_NUM_FMA 1
#define DOUBLE_NUM_MAX 1
#define DOUBLE_NUM_LOAD 1
#define DOUBLE_NUM_STORE 1
