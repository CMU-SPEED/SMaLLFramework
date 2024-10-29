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

#define PARALLEL 1

#include <acutest.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>

#include <small.h>
#include <small/utils/Timer.hpp>
#include <small/Conv1DLayer.hpp>

#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");

//****************************************************************************
void test_conv1d_layer(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,H,W,k,s,p,C_o
    LayerParams params {16, 1, 13, 3, 1, small::PADDING_F, 64};

    // Odd packed buffers should fail
    try
    {
        small::shape_type input_shape{1U, params.C_i, params.H, params.W};
        BufferT filters(params.C_i*params.k*params.C_o);
        small::Conv1DLayer conv1d(input_shape,
                                  params.k,
                                  params.s, params.p,
                                  params.C_o,
                                  filters,
                                  true);

        TEST_ASSERT(params.C_o % BufferT::C_ob == 0);
    }
    catch (std::invalid_argument &e_obj)
    {
        TEST_CHECK(params.C_o % BufferT::C_ob != 0);
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"conv1d_layer",   test_conv1d_layer},
    /// @todo need more tests
    {NULL, NULL}
};
