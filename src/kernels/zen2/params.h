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

typedef float dtype;

#define W_ob 6
#define C_ob 16
#define SIMD 8
#define UNROLL 1
#define C_ib C_ob


#define NUM_FMA 2
#define NUM_MAX 1
#define NUM_LOAD 2
#define NUM_STORE 1
