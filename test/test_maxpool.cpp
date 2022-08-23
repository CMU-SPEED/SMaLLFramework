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

size_t const C_i_max = 16;
size_t const M_max = 30;
size_t const N_max = 30;

std::string const input_data_fname("maxpool_input_data_float.bin");
size_t const max_inputs(C_i_max * M_max * N_max);
//size_t const max_weights(C_i * M_max * N_max);

//****************************************************************************
// from https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
size_t compute_output_dim_old(size_t input_dim,
                              size_t kernel_dim,
                              size_t stride,
                              char   padding)
{
    if ((padding == 'v') && (input_dim >= kernel_dim))
    {
        return std::floor((input_dim - kernel_dim)/((float)stride)) + 1;
    }
    else if (padding == 's')
    {
        return std::floor((input_dim - 1)/((float)stride)) + 1;
    }
    else
    {
        throw std::invalid_argument(std::string("Bad combination"));
    }

    return 0;
}

//****************************************************************************
size_t compute_output_dim(size_t input_dim,
                          size_t kernel_dim,
                          size_t stride,
                          char   padding)
{
    if ((padding == 'v') && (input_dim >= kernel_dim))
    {
        return std::floor((input_dim - kernel_dim)/((float)stride)) + 1;
    }
    else if (padding == 's')
    {
        int padding_elements;
        if (input_dim % stride == 0)
        {
            padding_elements = ((kernel_dim - stride > 0) ?
                                kernel_dim - stride :
                                0);
        }
        else
        {
            padding_elements = ((kernel_dim - (input_dim % stride) > 0) ?
                                kernel_dim - (input_dim % stride) :
                                0);
        }
        size_t padded_input_dim = input_dim + padding_elements;
        size_t output_dim = ((padded_input_dim - kernel_dim)/stride) - 1;
        return std::max(output_dim, 0UL);
    }
    else
    {
        throw std::invalid_argument("Bad combination");
    }

    return 0;
}


//****************************************************************************
void test_maxpool_setup_input(void)
{
    using RealT = float;

    std::default_random_engine generator;
    std::normal_distribution<RealT> distribution{0.f, 1.f};  // what distribution is Torch::Tensor::randn?

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
void test_maxpool_single_element_old(void)
{
    using RealT = float;

    size_t const C_i = 16;
    size_t const H = 3;
    size_t const W = 3;
    size_t const kernel_size = 3;
    size_t const stride = 2;
    char   const padding = 'v';
    //size_t const C_o = 16;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;

    RealT *input_dc = small_alloc<RealT>(num_input_elts);
    TEST_CHECK(nullptr != input_dc);

    int ret = read_float_inputs(input_data_fname, input_dc, num_input_elts);
    TEST_CHECK(0 == ret);

    size_t Ho(compute_output_dim(H, kernel_size, stride, padding));
    TEST_CHECK(1 == Ho);
    size_t Wo(compute_output_dim(W, kernel_size, stride, padding));
    TEST_CHECK(1 == Wo);
    size_t num_output_elts = C_i*Ho*Wo;
    RealT *output_dc = small_alloc<RealT>(num_output_elts);
    TEST_CHECK(nullptr != output_dc);

    Maxpool2D(0, kernel_size, stride, padding,
              C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {

        //TEST_CHECK((output_dc[ix] == ));
        std::cout << ": Maxpool_in(" << ix << ")-->"
                  << input_dc[ix] << std::endl;
    }

    for (size_t ix = 0; ix < num_output_elts; ++ix)
    {

        //TEST_CHECK((output_dc[ix] == ));
        std::cout << ": Maxpool_out(" << ix << ")-->"
                  << output_dc[ix] << std::endl;
    }

    free(input_dc);
    free(output_dc);
}

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

    std::string in_fname("../test/regression_data/in__pool_Ci16_H3_W3_k3_s2_v_144.bin");
    RealT *input_dc = nullptr;
    uint32_t num_input_elts = read_float_inputs(in_fname, &input_dc);
    TEST_CHECK(num_input_elts == C_i*H*W);

    std::string out_fname("../test/regression_data/out__pool_Ci16_H3_W3_k3_s2_v_16.bin");
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

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {

        //TEST_CHECK((output_dc[ix] == ));
        std::cout << ": Maxpool_in(" << ix << ")-->"
                  << input_dc[ix] << std::endl;
    }

    for (size_t ix = 0; ix < num_output_elts; ++ix)
    {
        TEST_CHECK(output_dc[ix] == output_dc_answers[ix]);
        std::cout << ": Maxpool_out(" << ix << ")-->"
                  << output_dc[ix] << " ?= " << output_dc_answers[ix]
                  << std::endl;
    }

    free(input_dc);
    free(output_dc);
    free(output_dc_answers);
}

#if 0
//****************************************************************************
void test_maxpool_single_tile(void)
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

    Maxpool2D(0, C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == 0.f));
        //std::cout << ix << ": Maxpool(" << input_dc[ix] << ")-->" << output_dc[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(output_dc);
}

//****************************************************************************
void test_maxpool_large_tile(void)
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

    Maxpool2D(0, C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == 0.f));
        //std::cout << ix << ": Maxpool(" << input_dc[ix] << ")-->" << output_dc[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(output_dc);
}
#endif

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"maxpool_setup", test_maxpool_setup_input},
    {"maxpool_single_element_old",  test_maxpool_single_element_old},
    {"maxpool_single_element",  test_maxpool_single_element},
    //{"maxpool_single_tile",  test_maxpool_single_tile},
    //{"maxpool_large_tile",  test_maxpool_large_tile},
    {NULL, NULL}
};
