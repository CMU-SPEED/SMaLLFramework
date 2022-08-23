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
#include <random>
#include <chrono>

#include <small/config.h>
#include <small/interface.h>
#include <params.h>
#include <intrinsics.h>

#include "Timer.hpp"
#include "test_utils.hpp"

size_t const C_i_max = 16;
size_t const M_max = 30;
size_t const N_max = 30;
size_t const C_o_max = 16;

std::string const input_data_fname("relu_input_data_float.bin");
size_t const max_inputs(C_i_max * M_max * N_max);
//size_t const max_weights(C_i * M_max * N_max);

//****************************************************************************
void test_relu_setup_input(void)
{
    using RealT = float;

    std::default_random_engine generator;
    std::normal_distribution<RealT> distribution{0.f, 2.f};  // what distribution is Torch::Tensor::randn?

    std::vector<RealT> nums(max_inputs, 0.f);
    for (size_t ix = 0; ix < max_inputs; ++ix)
    {
        nums[ix] = distribution(generator);
    }

    {
        std::ofstream ofs(input_data_fname);
        size_t n = max_inputs;
        // TODO: endian details
        ofs.write(reinterpret_cast<char*>(&n), sizeof(size_t));
        ofs.write(reinterpret_cast<char*>(&nums[0]), n*sizeof(RealT));
    }

    //-------------------------
    std::vector<RealT> in_nums(max_inputs, 0.f);
    int ret = read_float_inputs(input_data_fname, &in_nums[0], max_inputs);
    TEST_CHECK(ret == 0);

    for (size_t ix = 0; ix < max_inputs; ++ix)
    {
        TEST_CHECK(nums[ix] == in_nums[ix]);
        //std::cout << ix << ": " << in_nums[ix] << std::endl;
    }
}

//****************************************************************************
void test_relu_single_element(void)
{
    using RealT = float;

    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1;
    //size_t const kernel_size = 1;
    //size_t const stride = 1;
    //char const type = 'v';
    //size_t const C_o = 16;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;

    RealT *input_dc = small_alloc<RealT>(num_input_elts);
    TEST_CHECK(nullptr != input_dc);

    int ret = read_float_inputs(input_data_fname, input_dc, num_input_elts);
    TEST_CHECK(0 == ret);

    //size_t num_output_elts = C_i*H*W;
    RealT *output_dc = small_alloc<RealT>(num_input_elts);
    TEST_CHECK(nullptr != output_dc);

    ReLUActivation(0, C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == 0.f));
        //std::cout << ix << ": ReLU(" << input_dc[ix] << ")-->" << output_dc[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(output_dc);
}

//****************************************************************************
void test_relu_single_tile(void)
{
    using RealT = float;

    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 6;
    //size_t const kernel_size = 1;
    //size_t const stride = 1;
    //char const type = 'v';
    //size_t const C_o = 16;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;

    RealT *input_dc = small_alloc<RealT>(num_input_elts);
    TEST_CHECK(nullptr != input_dc);

    int ret = read_float_inputs(input_data_fname, input_dc, num_input_elts);
    TEST_CHECK(0 == ret);

    //size_t num_output_elts = C_i*H*W;
    RealT *output_dc = small_alloc<RealT>(num_input_elts);
    TEST_CHECK(nullptr != output_dc);

    ReLUActivation(0, C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == 0.f));
        //std::cout << ix << ": ReLU(" << input_dc[ix] << ")-->" << output_dc[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(output_dc);
}

//****************************************************************************
void test_relu_large_tile(void)
{
    using RealT = float;

    size_t const C_i = 16;
    size_t const H = 30;
    size_t const W = 30;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;

    RealT *input_dc = small_alloc<RealT>(num_input_elts);
    TEST_CHECK(nullptr != input_dc);

    int ret = read_float_inputs(input_data_fname, input_dc, num_input_elts);
    TEST_CHECK(0 == ret);

    //size_t num_output_elts = C_i*H*W;
    RealT *output_dc = small_alloc<RealT>(num_input_elts);
    TEST_CHECK(nullptr != output_dc);

    ReLUActivation(0, C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == 0.f));
        //std::cout << ix << ": ReLU(" << input_dc[ix] << ")-->" << output_dc[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(output_dc);
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"relu_setup", test_relu_setup_input},
    {"relu_single_element",  test_relu_single_element},
    {"relu_single_tile",  test_relu_single_tile},
    {"relu_large_tile",  test_relu_large_tile},
    {NULL, NULL}
};
