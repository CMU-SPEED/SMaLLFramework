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
// The setup script must copy the selected platform headers to the correct location

#include <small/platforms/params.h>    //"kernels/quantized_reference/params.h"
#include <small/platforms/Buffer.hpp>  //"kernels/quantized_reference/intrinsics.h"
#include <small/platforms/intrinsics.h>//"kernels/quantized_reference/utils.h"

// ================== Public API ====================
#include <small/utils.hpp>
#include <small/buffers.hpp>
#include <small/interface.hpp>

#include <small/interface_abstract.hpp>
