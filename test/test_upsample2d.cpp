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
#include <small/UpSample2DLayer.hpp>

#include "test_utils.hpp"

//****************************************************************************
template <class BufferT>
BufferT create_upsample2d_data(size_t num_elements)
{
    BufferT input_dc(num_elements);

    for (size_t ix = 0; ix < num_elements; ++ix)
    {
        input_dc[ix] = (typename BufferT::value_type)(ix);
    }

    return input_dc;
}

//****************************************************************************
void test_upsample2d_single_element(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1;
    uint32_t const scale_factor = 2;

    size_t const num_input_elts = C_i * H * W;
    size_t const num_output_elts = C_i * H * scale_factor * W * scale_factor;

#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc =
        create_upsample2d_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_output_elts);
#else
    small::FloatBuffer input_dc =
        create_upsample2d_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer output_dc(num_output_elts);
#endif

    small::UpSample2D(scale_factor, C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        // TEST_CHECK((input_dc[ix] >= 0) ?
        //            (output_dc[ix] == input_dc[ix]) :
        //            (output_dc[ix] == 0));
        std::cout << ix << ": Upsample in(" << (int)input_dc[ix] << ")\n";
    }
    for (size_t ix = 0; ix < num_output_elts; ++ix)
    {
        // TEST_CHECK((input_dc[ix] >= 0) ?
        //            (output_dc[ix] == input_dc[ix]) :
        //            (output_dc[ix] == 0));
        std::cout << ix << ": Upsample out(" << (int)output_dc[ix] << ")\n";
    }

    TEST_CHECK(false);
}

//****************************************************************************
void test_upsample2d_single_tile(void)
{
    TEST_CHECK(false);
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"upsample2d_single_element", test_upsample2d_single_element},
    {"upsample2d_single_tile", test_upsample2d_single_tile},
    {NULL, NULL}};