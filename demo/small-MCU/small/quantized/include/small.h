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

#define SMALL_MAJOR_VERSION  1
#define SMALL_MINOR_VERSION  0
#define SMALL_SUB_VERSON     0

#define PARALLEL 0

// ============ Implementation details ==============
// Platform specific includes.
// Use -I compile option to point to correct platform

#include <kernels/params.h>      //"kernels/quantized_reference/params.h"
#include <kernels/Buffer.hpp>    //"kernels/quantized_reference/intrinsics.h"
#include <kernels/intrinsics.h>  //"kernels/quantized_reference/utils.h"
/// This must come after platform-specific includes.
// #include <small/abstract_layer.hpp>

// ================== Public API ====================
#include <small/utils.hpp>
#include <small/buffers.hpp>
#include <small/interface.hpp>           // template declarations only

#if defined(QUANTIZED)
#include <small/q_abstract_layer.hpp>
#include <small/q_interface_abstract.hpp>
#else
#include <small/abstract_layer.hpp>
#include <small/interface_abstract.hpp>
#endif
//USE THESE  IF Quantized
