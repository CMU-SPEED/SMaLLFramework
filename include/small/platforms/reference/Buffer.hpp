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

#include <type_traits>
#include <exception>

#if defined(SMALL_HAS_FLOAT_SUPPORT)
#include <FloatBuffer.hpp>
#endif

#if defined(SMALL_HAS_QUINT8_SUPPORT)
#include <QUInt8Buffer.hpp>
#endif