//-----------------------------------------------------------------------------
// test/test_relu.cpp:  test cases for xxx
//-----------------------------------------------------------------------------

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

#include <acutest.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

#include <small.h>

#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");

//****************************************************************************
bool run_dw_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "dw_conv",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nDepthwiseConv: input file = " << in_fname << std::endl;

    small::FloatBuffer input_dc = read_float_inputs(in_fname);
    TEST_ASSERT(input_dc.size() == params.C_i*params.H*params.W);

    // Pack input data
    small::FloatBuffer packed_input_dc(input_dc.size());
    small::pack_buffer(input_dc,
                       small::INPUT,
                       1U, params.C_i, params.H, params.W,
                       C_ib, C_ob,
                       packed_input_dc);

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "dw_conv",
                     params,
                     params.C_i*params.k*params.k);
    std::cout << "DepthwiseConv: filter file= " << filter_fname << std::endl;

    small::FloatBuffer filter_dc = read_float_inputs(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k);

    // Pack filter data
    small::FloatBuffer packed_filter_dc(filter_dc.size());
    small::pack_buffer(filter_dc,
                       small::FILTER_DW, //FILTER_CONV,
                       params.C_i, 1 /* params.C_o */, params.k, params.k,
                       C_ib, C_ob,
                       packed_filter_dc);

    // Read output regression data
    size_t Ho(compute_output_dim(params.H, params.k, params.s, params.p));
    size_t Wo(compute_output_dim(params.W, params.k, params.s, params.p));
    std::string out_fname =
        get_pathname(data_dir, "out", "dw_conv",
                     params,
                     params.C_i*Ho*Wo);
    std::cout << "DepthwiseConv: output file= " << out_fname << std::endl;

    small::FloatBuffer output_dc_answers = read_float_inputs(out_fname);
    TEST_ASSERT(output_dc_answers.size() == params.C_i*Ho*Wo);

    // Pack output answer data
    small::FloatBuffer packed_output_dc_answers(output_dc_answers.size());
    small::pack_buffer(output_dc_answers,
                       small::OUTPUT,
                       1U, params.C_i, Ho, Wo,
                       C_ib, C_ob,
                       packed_output_dc_answers);

    // Allocate output buffer
    small::FloatBuffer packed_output_dc(output_dc_answers.size());

    uint8_t t_pad=0, b_pad=0, l_pad=0, r_pad=0;
    if (params.p == 'f')
    {
        small::calc_padding(params.H, params.k, params.s, t_pad, b_pad);
        small::calc_padding(params.W, params.k, params.s, l_pad, r_pad);
    }

    // Compute layer
    small::DepthwiseConv2D(params.k, params.s,
                           t_pad, b_pad, l_pad, r_pad,
                           params.C_i, params.H, params.W,
                           packed_input_dc, packed_filter_dc, packed_output_dc);

    // Check answer
    bool passing = true;
    for (size_t ix = 0; ix < packed_output_dc.size(); ++ix)
    {
        if ((packed_output_dc[ix] != packed_output_dc_answers[ix]) &&
            !almost_equal(packed_output_dc[ix], packed_output_dc_answers[ix]))
        {
            passing = false;

            std::cout << "FAIL: DepthwiseConv_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc_answers[ix]
                      << std::endl;
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
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
        {96, 30, 30, 3, 1, 'v', 0},

        {16,  3,  3, 3, 1, 'f', 0},
        {16,  3,  3, 3, 2, 'f', 0},
        {16,  3,  8, 3, 1, 'f', 0},
        {16,  3, 13, 3, 2, 'f', 0},  //Ci,Hi,Wi,k,s,p,Co
        {96, 30, 30, 3, 1, 'f', 0},
        {96, 30, 30, 3, 2, 'f', 0},
        {96,  3, 13, 3, 2, 'f', 0},
        {96,  3,  8, 3, 1, 'f', 0}
    };
    for (LayerParams const &p: params)
    {
        TEST_CHECK(true == run_dw_config(p));
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"dw_regression_data",     test_dw_regression_data},
    {NULL, NULL}
};
