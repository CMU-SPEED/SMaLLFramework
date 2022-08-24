//-----------------------------------------------------------------------------
// test/test_relu.cpp:  test cases for xxx
//-----------------------------------------------------------------------------

// LAGraph, (c) 2022 by the Authors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// See additional acknowledgments in the LICENSE file,

//-----------------------------------------------------------------------------

//#include <LAGraph.h>

#include <acutest.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <random>

#include <small/config.h>
#include <small/interface.h>
#include <params.h>
#include <intrinsics.h>

#include "Timer.hpp"
#include "test_utils.hpp"

//****************************************************************************
void test_maxpool_single_element(void)
{
    using RealT = float;

    size_t const C_i = 16;
    size_t const H = 3;
    size_t const W = 3;
    size_t const kernel_size = 3;
    size_t const stride = 2;
    char   const padding = 'v';
    //size_t const C_o = 16;

    std::string in_fname(
        "../test/regression_data/in__pool_Ci16_H3_W3_k3_s2_v_144.bin");
    RealT *input_dc = nullptr;
    uint32_t num_input_elts = read_float_inputs(in_fname, &input_dc);
    TEST_CHECK(num_input_elts == C_i*H*W);

    std::string out_fname(
        "../test/regression_data/out__pool_Ci16_H3_W3_k3_s2_v_16.bin");
    RealT *output_dc_answers = nullptr;
    uint32_t num_output_elts = read_float_inputs(out_fname, &output_dc_answers);

    size_t Ho(compute_output_dim(H, kernel_size, stride, padding));
    TEST_CHECK(1 == Ho);
    size_t Wo(compute_output_dim(W, kernel_size, stride, padding));
    TEST_CHECK(1 == Wo);
    TEST_CHECK(num_output_elts == C_i*Ho*Wo);

    RealT *output_dc = small_alloc<RealT>(num_output_elts);
    TEST_CHECK(nullptr != output_dc);

    Maxpool2D(0, kernel_size, stride, padding,
              C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_output_elts; ++ix)
    {
        TEST_CHECK(output_dc[ix] == output_dc_answers[ix]);
        //std::cout << ": Maxpool_out(" << ix << ")-->"
        //          << output_dc[ix] << " ?= " << output_dc_answers[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(output_dc);
    free(output_dc_answers);
}

//****************************************************************************
void test_maxpool_single_tile(void)
{
    using RealT = float;

    size_t const C_i = 16;
    size_t const H = 3;
    size_t const W = 13;
    size_t const kernel_size = 3;
    size_t const stride = 2;
    char   const padding = 'v';
    //size_t const C_o = 16;

    std::string in_fname(
        "../test/regression_data/in__pool_Ci16_H3_W13_k3_s2_v_624.bin");
    RealT *input_dc = nullptr;
    uint32_t num_input_elts = read_float_inputs(in_fname, &input_dc);
    TEST_CHECK(num_input_elts == C_i*H*W);

    std::string out_fname(
        "../test/regression_data/out__pool_Ci16_H3_W13_k3_s2_v_96.bin");
    RealT *output_dc_answers = nullptr;
    uint32_t num_output_elts = read_float_inputs(out_fname, &output_dc_answers);

    size_t Ho(compute_output_dim(H, kernel_size, stride, padding));
    size_t Wo(compute_output_dim(W, kernel_size, stride, padding));
    TEST_CHECK(1 == Ho);
    TEST_CHECK(6 == Wo);
    TEST_CHECK(num_output_elts == C_i*Ho*Wo);

    RealT *output_dc = small_alloc<RealT>(num_output_elts);
    TEST_CHECK(nullptr != output_dc);

    Maxpool2D(0, kernel_size, stride, padding,
              C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_output_elts; ++ix)
    {
        TEST_CHECK(output_dc[ix] == output_dc_answers[ix]);
        //std::cout << ": Maxpool_out(" << ix << ")-->"
        //          << output_dc[ix] << " ?= " << output_dc_answers[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(output_dc);
    free(output_dc_answers);
}

//****************************************************************************
void test_maxpool_large_tile(void)
{
    using RealT = float;

    size_t const C_i = 16;
    size_t const H = 30;
    size_t const W = 30;
    size_t const kernel_size = 3;
    size_t const stride = 2;
    char   const padding = 'v';
    //size_t const C_o = 16;

    std::string in_fname(
        "../test/regression_data/in__pool_Ci16_H30_W30_k3_s2_v_14400.bin");
    RealT *input_dc = nullptr;
    uint32_t num_input_elts = read_float_inputs(in_fname, &input_dc);
    TEST_CHECK(num_input_elts == C_i*H*W);

    std::string out_fname(
        "../test/regression_data/out__pool_Ci16_H30_W30_k3_s2_v_3136.bin");
    RealT *output_dc_answers = nullptr;
    uint32_t num_output_elts = read_float_inputs(out_fname, &output_dc_answers);

    size_t Ho(compute_output_dim(H, kernel_size, stride, padding));
    size_t Wo(compute_output_dim(W, kernel_size, stride, padding));
    TEST_CHECK(14 == Ho);
    TEST_CHECK(14 == Wo);
    TEST_CHECK(num_output_elts == C_i*Ho*Wo);

    RealT *output_dc = small_alloc<RealT>(num_output_elts);
    TEST_CHECK(nullptr != output_dc);

    Maxpool2D(0, kernel_size, stride, padding,
              C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_output_elts; ++ix)
    {
        TEST_CHECK(output_dc[ix] == output_dc_answers[ix]);
        //std::cout << ": Maxpool_out(" << ix << ")-->"
        //          << output_dc[ix] << " ?= " << output_dc_answers[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(output_dc);
    free(output_dc_answers);
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"maxpool_single_element",  test_maxpool_single_element},
    {"maxpool_single_tile",  test_maxpool_single_tile},
    {"maxpool_large_tile",  test_maxpool_large_tile},
    {NULL, NULL}
};
