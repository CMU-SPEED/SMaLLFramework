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

#include <small.h>

#include "Timer.hpp"
#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");

//****************************************************************************
template <typename RealT>
bool run_conv2d_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nConv2D: input file = " << in_fname << std::endl;

    RealT *input_dc = nullptr;
    uint32_t num_input_elts = read_float_inputs(in_fname, &input_dc);
    TEST_ASSERT(num_input_elts == params.C_i*params.H*params.W);
    TEST_ASSERT(nullptr != input_dc);

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "Conv2D: filter file= " << filter_fname << std::endl;
    RealT *filter_dc = nullptr;
    uint32_t num_filter_elts = read_float_inputs(filter_fname, &filter_dc);
    TEST_ASSERT(num_filter_elts == params.C_i*params.k*params.k*params.C_o);
    TEST_ASSERT(nullptr != filter_dc);

    // Read output regression data
    size_t Ho(compute_output_dim(params.H, params.k, params.s, params.p));
    size_t Wo(compute_output_dim(params.W, params.k, params.s, params.p));
    std::cerr << "Output image dims: " << Ho << ", " << Wo << std::endl;
    std::string out_fname =
        get_pathname(data_dir, "out", "conv2d",
                     params,
                     params.C_o*Ho*Wo);
    std::cout << "Conv2D: output file= " << out_fname << std::endl;
    RealT *output_dc_answers = nullptr;
    uint32_t num_output_elts = read_float_inputs(out_fname, &output_dc_answers);
    TEST_ASSERT(num_output_elts == params.C_o*Ho*Wo);
    TEST_ASSERT(nullptr != output_dc_answers);

    // Allocate output buffer
    RealT *output_dc = small_alloc<RealT>(num_output_elts);
    TEST_ASSERT(nullptr != output_dc);

    uint8_t t_pad=0, b_pad=0, l_pad=0, r_pad=0;
    if (params.p == 'f')
    {
        CALC_PADDING(params.H, params.k, params.s, t_pad, b_pad);
        CALC_PADDING(params.W, params.k, params.s, l_pad, r_pad);
    }

    // Compute layer
    Conv2D(0,
           params.k, params.s,
           t_pad, b_pad, l_pad, r_pad,
           params.C_o, params.C_i, params.H, params.W,
           input_dc, filter_dc, output_dc);

    // Check answer
    bool passing = true;
    for (size_t ix = 0; ix < num_output_elts; ++ix)
    {
        if (output_dc[ix] != output_dc_answers[ix])
        {
            passing = false;
        }
        std::cout << ": Conv2D_out(" << ix << ")-->"
                  << output_dc[ix] << " ?= " << output_dc_answers[ix]
                  << std::endl;
    }

    free(input_dc);
    free(filter_dc);
    free(output_dc);
    free(output_dc_answers);

    return passing;
}

//****************************************************************************
void test_conv2d_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {16,  3,  3, 3, 1, 'v', 16},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3,  8, 3, 1, 'v', 16},
        {16, 30, 30, 3, 1, 'v', 16},

        {16,  3,  8, 3, 1, 'v', 96},
        {96,  3,  8, 3, 1, 'v', 16},
        {96, 30, 30, 3, 1, 'v', 96}
#if 0
      , {16,  3,  3, 3, 1, 'f', 16},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3,  8, 3, 1, 'f', 16},
        {96, 30, 30, 3, 1, 'f', 96},
        {96,  3,  8, 3, 1, 'f', 16}
#endif
    };
    for (LayerParams const &p: params)
    {
        TEST_CHECK(true == run_conv2d_config<float>(p));
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"conv2d_regression_data",     test_conv2d_regression_data},
    {NULL, NULL}
};
