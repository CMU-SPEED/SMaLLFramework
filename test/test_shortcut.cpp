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
#include <small/AddLayer.hpp>

#include "test_utils.hpp"

//****************************************************************************
void test_shortcut(size_t C_i, size_t H, size_t W)
{
    size_t const num_input_elts = C_i*H*W;

    small::FloatBuffer input_buf(num_input_elts);
    small::init(input_buf, num_input_elts);
    small::Tensor<small::FloatBuffer> input({1, C_i, H, W}, input_buf);

    small::FloatBuffer output_ref_buf(num_input_elts);
    small::FloatBuffer output_res_buf(num_input_elts);

    small::init(output_res_buf, num_input_elts);
    std::copy(&output_res_buf[0], &output_res_buf[num_input_elts], &output_ref_buf[0]);

    small::Tensor<small::FloatBuffer> output({1, C_i, H, W}, output_res_buf);

    small::AddLayer<small::FloatBuffer> add(input.shape(), output.shape());

    add.compute_output({&input}, &output);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        if (output.buffer()[ix] != input_buf[ix] + output_ref_buf[ix])
        {
            std::cout << "Error at index " << ix << std::endl;
            std::cout << "Expected: " << input_buf[ix] + output_ref_buf[ix] << std::endl;
            std::cout << "Actual: " << output.buffer()[ix] << std::endl;
            TEST_CHECK(false);
        }
    }
    TEST_CHECK(true);

}

//****************************************************************************
void test_shortcut_single_element(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1;

    test_shortcut(C_i, H, W);

}

//****************************************************************************
void test_shortcut_single_tile(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 6;

    test_shortcut(C_i, H, W);
}

//****************************************************************************
void test_shortcut_large_tile(void)
{
    size_t const C_i = 16;
    size_t const H = 30;
    size_t const W = 30;

    test_shortcut(C_i, H, W);
}

//****************************************************************************


//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"shortcut_single_element",  test_shortcut_single_element},
    {"shortcut_single_tile",  test_shortcut_single_tile},
    {"shortcut_large_tile",  test_shortcut_large_tile},
    {NULL, NULL}
};
