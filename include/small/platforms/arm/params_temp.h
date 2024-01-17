
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

#define SMALL_HAS_FLOAT_SUPPORT  1

#define FLOAT_W_ob 6
#define FLOAT_C_ob 16
#define FLOAT_SIMD 8
#define FLOAT_UNROLL 1
#define FLOAT_C_ib FLOAT_C_ob

// not used for kernels, but used in throughput calculation.
#define NUM_FMA 2
#define NUM_MAX 1
#define NUM_LOAD 2
#define NUM_STORE 1
        