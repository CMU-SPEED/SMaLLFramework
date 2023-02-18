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
#include <iomanip>
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

    // Pack input data
    RealT *packed_input_dc = small_alloc<RealT>(num_input_elts);
    small::convert_tensor2dc<float>(input_dc,
                                    small::INPUT,
                                    1U, params.C_i, params.H, params.W,
                                    C_ib, C_ob,
                                    packed_input_dc);

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

    // Pack filter data
    RealT *packed_filter_dc = small_alloc<RealT>(num_filter_elts);
    small::convert_tensor2dc<float>(filter_dc,
                                    small::FILTER_CONV,
                                    params.C_i, params.C_o, params.k, params.k,
                                    C_ib, C_ob,
                                    packed_filter_dc);

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

    // Pack output answer data
    RealT *packed_output_dc_answers = small_alloc<RealT>(num_output_elts);
    small::convert_tensor2dc<float>(output_dc_answers,
                                    small::OUTPUT,
                                    1U, params.C_o, Ho, Wo,
                                    C_ib, C_ob,
                                    packed_output_dc_answers);

    // Allocate output buffer
    RealT *packed_output_dc = small_alloc<RealT>(num_output_elts);
    TEST_ASSERT(nullptr != packed_output_dc);

    uint8_t t_pad=0, b_pad=0, l_pad=0, r_pad=0;
    if (params.p == 'f')
    {
        small::calc_padding(params.H, params.k, params.s, t_pad, b_pad);
        small::calc_padding(params.W, params.k, params.s, l_pad, r_pad);
    }

    // Compute layer
    small::Conv2D(params.k, params.s,
                  t_pad, b_pad, l_pad, r_pad,
                  params.C_o, params.C_i, params.H, params.W,
                  packed_input_dc, packed_filter_dc, packed_output_dc);

    // Check answer
    bool passing = true;
    for (size_t ix = 0; ix < num_output_elts; ++ix)
    {
        if ((packed_output_dc[ix] != packed_output_dc_answers[ix]) &&
            !almost_equal(packed_output_dc[ix], packed_output_dc_answers[ix]))
        {
            passing = false;

            std::cout << "FAIL: Conv2D_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc_answers[ix]
                      << std::endl;
        }
    }

    free(input_dc);
    free(packed_input_dc);
    free(filter_dc);
    free(packed_filter_dc);
    free(packed_output_dc);
    free(output_dc_answers);
    free(packed_output_dc_answers);

    if (passing) std::cerr << "Test PASSED\n";
    return passing;
}

//****************************************************************************
void test_conv2d_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {16,  1,  1, 1, 1, 'v', 16},
        {16,  1,  6, 1, 1, 'v', 16},
        {16,  3,  3, 3, 1, 'v', 16},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3,  8, 3, 1, 'v', 16},
        {16, 30, 30, 3, 1, 'v', 16},

        {16,  1,  6, 1, 1, 'v', 96},
        {16,  3,  8, 3, 1, 'v', 96},

        {96,  1,  6, 1, 1, 'v', 16},
        {96,  3,  8, 3, 1, 'v', 16},

        {96, 30, 30, 1, 1, 'v', 96},
        {96, 30, 30, 3, 1, 'v', 96},

#if 1
        {16,  3,  3, 3, 1, 'f', 16},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3,  3, 3, 2, 'f', 16},
        {16,  3,  8, 3, 1, 'f', 16},
        {16,  3,  8, 3, 1, 'f', 96},
        {16,  3, 13, 3, 2, 'f', 16},
        {16,  3, 13, 3, 2, 'f', 96},

        {96,  3,  8, 3, 1, 'f', 16},
        {96,  3, 13, 3, 2, 'f', 16},
        {96, 30, 30, 3, 1, 'f', 96},
        {96, 30, 30, 3, 2, 'f', 96}
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
