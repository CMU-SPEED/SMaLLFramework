//-----------------------------------------------------------------------------
// test/test_conv2d.cpp:  test cases for xxx
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
void test_conv2d_single_element(void)
{
    using RealT = float;

    size_t const C_i = 16;
    size_t const H = 3;
    size_t const W = 3;
    size_t const kernel_size = 3;
    size_t const stride = 1;
    char   const padding = 'v';
    size_t const C_o = 16;

    std::string in_fname(
        "../test/regression_data/in__conv2d_Ci16_H3_W3_k3_s1_v_Co16_144.bin");
    RealT *input_dc = nullptr;
    uint32_t num_input_elts = read_float_inputs(in_fname, &input_dc);
    TEST_CHECK(num_input_elts == C_i*H*W);

    std::string filter_fname(
        "../test/regression_data/filter__conv2d_Ci16_H3_W3_k3_s1_v_Co16_2304.bin");
    RealT *filter_dc = nullptr;
    uint32_t num_filter_elts = read_float_inputs(filter_fname, &filter_dc);
    TEST_CHECK(num_filter_elts == C_i*kernel_size*kernel_size*C_o);

    std::string out_fname(
        "../test/regression_data/out__conv2d_Ci16_H3_W3_k3_s1_v_Co16_16.bin");
    RealT *output_dc_answers = nullptr;
    uint32_t num_output_elts = read_float_inputs(out_fname, &output_dc_answers);

    size_t Ho(compute_output_dim(H, kernel_size, stride, padding));
    TEST_CHECK(1 == Ho);
    size_t Wo(compute_output_dim(W, kernel_size, stride, padding));
    TEST_CHECK(1 == Wo);
    TEST_CHECK(num_output_elts == C_o*Ho*Wo);

    RealT *output_dc = small_alloc<RealT>(num_output_elts);
    TEST_CHECK(nullptr != output_dc);

    Conv2D(0, kernel_size, stride, padding,
           C_o, C_i, H, W, input_dc, filter_dc, output_dc);

    for (size_t ix = 0; ix < num_output_elts; ++ix)
    {
        TEST_CHECK(output_dc[ix] == output_dc_answers[ix]);
        //std::cout << ": Conv2d_out(" << ix << ")-->"
        //          << output_dc[ix] << " ?= " << output_dc_answers[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(output_dc);
    free(output_dc_answers);
}

//****************************************************************************
void test_conv2d_single_tile(void)
{
    using RealT = float;

    size_t const C_i = 16;
    size_t const H = 3;
    size_t const W = 8;
    size_t const kernel_size = 3;
    size_t const stride = 1;
    char   const padding = 'v';
    size_t const C_o = 16;

    std::string in_fname(
        "../test/regression_data/in__conv2d_Ci16_H3_W8_k3_s1_v_Co16_384.bin");
    RealT *input_dc = nullptr;
    uint32_t num_input_elts = read_float_inputs(in_fname, &input_dc);
    TEST_CHECK(num_input_elts == C_i*H*W);

    std::string filter_fname(
        "../test/regression_data/filter__conv2d_Ci16_H3_W8_k3_s1_v_Co16_2304.bin");
    RealT *filter_dc = nullptr;
    uint32_t num_filter_elts = read_float_inputs(filter_fname, &filter_dc);
    TEST_CHECK(num_filter_elts == C_i*kernel_size*kernel_size*C_o);

    std::string out_fname(
        "../test/regression_data/out__conv2d_Ci16_H3_W8_k3_s1_v_Co16_96.bin");
    RealT *output_dc_answers = nullptr;
    uint32_t num_output_elts = read_float_inputs(out_fname, &output_dc_answers);

    size_t Ho(compute_output_dim(H, kernel_size, stride, padding));
    size_t Wo(compute_output_dim(W, kernel_size, stride, padding));
    //std::cout << "n = 16*ho*wo: " << num_output_elts << " = 16*"
    //          << Ho << "*" << Wo << std::endl;
    TEST_CHECK(1 == Ho);
    TEST_CHECK(6 == Wo);
    TEST_CHECK(num_output_elts == C_o*Ho*Wo);

    RealT *output_dc = small_alloc<RealT>(num_output_elts);
    TEST_CHECK(nullptr != output_dc);

    Conv2D(0, kernel_size, stride, padding,
           C_o, C_i, H, W, input_dc, filter_dc, output_dc);

    for (size_t ix = 0; ix < num_output_elts; ++ix)
    {
        TEST_CHECK(output_dc[ix] == output_dc_answers[ix]);
        //std::cout << ": Conv2d_out(" << ix << ")-->"
        //          << output_dc[ix] << " ?= " << output_dc_answers[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(filter_dc);
    free(output_dc);
    free(output_dc_answers);
}

//****************************************************************************
void test_conv2d_large_tile(void)
{
    using RealT = float;

    size_t const C_i = 16;
    size_t const H = 30;
    size_t const W = 30;
    size_t const kernel_size = 3;
    size_t const stride = 1;
    char   const padding = 'v';
    size_t const C_o = 16;

    std::string in_fname(
        "../test/regression_data/in__conv2d_Ci16_H30_W30_k3_s1_v_Co16_14400.bin");
    RealT *input_dc = nullptr;
    uint32_t num_input_elts = read_float_inputs(in_fname, &input_dc);
    TEST_CHECK(num_input_elts == C_i*H*W);

    std::string filter_fname(
        "../test/regression_data/filter__conv2d_Ci16_H30_W30_k3_s1_v_Co16_2304.bin");
    RealT *filter_dc = nullptr;
    uint32_t num_filter_elts = read_float_inputs(filter_fname, &filter_dc);
    TEST_CHECK(num_filter_elts == C_i*kernel_size*kernel_size*C_o);

    std::string out_fname(
        "../test/regression_data/out__conv2d_Ci16_H30_W30_k3_s1_v_Co16_12544.bin");
    RealT *output_dc_answers = nullptr;
    uint32_t num_output_elts = read_float_inputs(out_fname, &output_dc_answers);

    size_t Ho(compute_output_dim(H, kernel_size, stride, padding));
    size_t Wo(compute_output_dim(W, kernel_size, stride, padding));
    //std::cout << "n = 16*ho*wo: " << num_output_elts << " = 16*"
    //          << Ho << "*" << Wo << std::endl;
    TEST_CHECK(28 == Ho);
    TEST_CHECK(28 == Wo);
    TEST_CHECK(num_output_elts == C_o*Ho*Wo);

    RealT *output_dc = small_alloc<RealT>(num_output_elts);
    TEST_CHECK(nullptr != output_dc);

    Conv2D(0, kernel_size, stride, padding,
           C_o, C_i, H, W, input_dc, filter_dc, output_dc);

    for (size_t ix = 0; ix < num_output_elts; ++ix)
    {
        TEST_CHECK(output_dc[ix] == output_dc_answers[ix]);
        //std::cout << ": Conv2d_out(" << ix << ")-->"
        //          << output_dc[ix] << " ?= " << output_dc_answers[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(filter_dc);
    free(output_dc);
    free(output_dc_answers);
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"conv2d_single_element", test_conv2d_single_element},
    {"conv2d_single_tile",    test_conv2d_single_tile},
    {"conv2d_large_tile",     test_conv2d_large_tile},
    {NULL, NULL}
};
