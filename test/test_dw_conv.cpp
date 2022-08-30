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

std::string const data_dir("../test/regression_data");

//****************************************************************************
template <typename RealT>
bool run_dw_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "dw_conv",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nDepthwiseConv: input file = " << in_fname << std::endl;

    RealT *input_dc = nullptr;
    uint32_t num_input_elts = read_float_inputs(in_fname, &input_dc);
    TEST_ASSERT(num_input_elts == params.C_i*params.H*params.W);
    TEST_ASSERT(nullptr != input_dc);

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "dw_conv",
                     params,
                     params.C_i*params.k*params.k);
    std::cout << "DepthwiseConv: filter file= " << filter_fname << std::endl;
    RealT *filter_dc = nullptr;
    uint32_t num_filter_elts = read_float_inputs(filter_fname, &filter_dc);
    TEST_ASSERT(num_filter_elts == params.C_i*params.k*params.k);
    TEST_ASSERT(nullptr != filter_dc);

    // Read output regression data
    size_t Ho(compute_output_dim(params.H, params.k, params.s, params.p));
    size_t Wo(compute_output_dim(params.W, params.k, params.s, params.p));
    std::cerr << "Output image dims: " << Ho << ", " << Wo << std::endl;
    std::string out_fname =
        get_pathname(data_dir, "out", "dw_conv",
                     params,
                     params.C_i*Ho*Wo);
    std::cout << "DepthwiseConv: output file= " << out_fname << std::endl;
    RealT *output_dc_answers = nullptr;
    uint32_t num_output_elts = read_float_inputs(out_fname, &output_dc_answers);
    TEST_ASSERT(num_output_elts == params.C_i*Ho*Wo);
    TEST_ASSERT(nullptr != output_dc_answers);

    // Allocate output buffer
    RealT *output_dc = small_alloc<RealT>(num_output_elts);
    TEST_ASSERT(nullptr != output_dc);

    // Compute layer
    DepthwiseConv2D(0,
                    params.k, params.s, params.p,
                    params.C_i, params.H, params.W,
                    input_dc, filter_dc, output_dc);

    // Check answer
    bool passing = true;
    for (size_t ix = 0; ix < num_output_elts; ++ix)
    {
        if (output_dc[ix] != output_dc_answers[ix])
        {
            passing = false;
        }
        //std::cout << ": Maxpool_out(" << ix << ")-->"
        //          << output_dc[ix] << " ?= " << output_dc_answers[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(filter_dc);
    free(output_dc);
    free(output_dc_answers);

    return passing;
}

//****************************************************************************
void test_dw_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {16,  3,  3, 3, 1, 'v', 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3,  8, 3, 1, 'v', 0},
        {16, 30, 30, 3, 1, 'v', 0},

        {96,  3,  8, 3, 1, 'v', 0},
        {96, 30, 30, 3, 1, 'v', 0}
#if 0
      , {16,  3,  3, 3, 1, 'f', 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3,  8, 3, 1, 'f', 0},
        {96, 30, 30, 3, 1, 'f', 0},
        {96,  3,  8, 3, 1, 'f', 0}
#endif
    };
    for (LayerParams const &p: params)
    {
        TEST_CHECK(true == run_dw_config<float>(p));
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"dw_regression_data",     test_dw_regression_data},
    {NULL, NULL}
};
