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

size_t const C_i_max = 16;
size_t const M_max = 30;
size_t const N_max = 30;
size_t const C_o_max = 16;

std::string const input_data_fname("relu_input_data_float.bin");
size_t const max_inputs(C_i_max * M_max * N_max);
//size_t const max_weights(C_i * M_max * N_max);

//****************************************************************************


//****************************************************************************
/// @pre in_buf points to allocation at least as big as sizeof(float)*num_elts
int read_inputs(float *in_buf, size_t num_elts)
{
    using RealT = float;
    size_t in_n;
    std::ifstream ifs(input_data_fname);
    // TODO: endian details
    ifs.read(reinterpret_cast<char*>(&in_n), sizeof(size_t));

    if (num_elts > in_n)
        return 1;

    // TODO: endian details
    ifs.read(reinterpret_cast<char*>(in_buf), num_elts*sizeof(RealT));
    return 0;
}

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
    int ret = read_inputs(&in_nums[0], max_inputs);
    TEST_CHECK(ret == 0);

    for (size_t ix = 0; ix < max_inputs; ++ix)
    {
        TEST_CHECK(nums[ix] == in_nums[ix]);
        //std::cout << ix << ": " << in_nums[ix] << std::endl;
    }
}

//****************************************************************************
void test_relu_single_output_element(void)
{
    using RealT = float;

    size_t const C_i = 16;
    size_t const N = 1;
    size_t const M = 1;
    size_t const kernel_size = 1;
    size_t const stride = 1;
    char const type = 'v';
    size_t const C_o = 16;

    TEST_CHECK(C_i == C_o);
    size_t const num_elts = C_i*N*M;

    RealT *input_dc;
    RealT *output_dc;
    int ret = posix_memalign((void **)&input_dc, 4096, sizeof(RealT)*num_elts);
    TEST_CHECK(0 == ret);

    ret = read_inputs(input_dc, num_elts);
    TEST_CHECK(0 == ret);

    ret = posix_memalign((void**)&output_dc, 4096, sizeof(RealT)*num_elts);
    TEST_CHECK(0 == ret);

    ReLUActivation(0, C_i, N, M, input_dc, output_dc);
    for (size_t ix = 0; ix < num_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == 0.f));
        std::cout << ix << ": ReLU(" << input_dc[ix] << ")-->" << output_dc[ix]
                  << std::endl;
    }
}

//****************************************************************************
void test_relu_single_output_tile(void)
{
}

//****************************************************************************
void test_relu_many_output_channel_blocks_1group(void)
{
}

//****************************************************************************
void test_relu_many_input_channel_blocks(void)
{
}

//****************************************************************************
void test_relu_large_size(void)
{
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"endian", test_relu_setup_input},
    {"relu1",  test_relu_single_output_element},
    {"relu2",  test_relu_single_output_tile},
    {"relu3",  test_relu_many_output_channel_blocks_1group},
    {"relu4",  test_relu_many_input_channel_blocks},
    {"relu5",  test_relu_large_size},
    {NULL, NULL}
};
