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

#define QUINT8_W_ob   2
#define QUINT8_C_ob   1
#define QUINT8_SIMD   1
#define QUINT8_UNROLL 1
#define QUINT8_C_ib   QUINT8_C_ob

//Potential blocking parameters for packing
#define QUINT8_NUM_FMA 1
#define QUINT8_NUM_MAX 1
#define QUINT8_NUM_LOAD 1
#define QUINT8_NUM_STORE 1
