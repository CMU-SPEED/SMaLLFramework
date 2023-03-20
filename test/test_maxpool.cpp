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
template <class BufferT>
bool run_maxpool_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "pool",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nMaxpool: input file = " << in_fname << std::endl;

    BufferT input_dc = read_inputs<BufferT>(in_fname);
    TEST_ASSERT(input_dc.size() == params.C_i*params.H*params.W);

    // Pack input data
    BufferT packed_input_dc(input_dc.size());
    small::pack_buffer(input_dc,
                       small::INPUT,
                       1U, params.C_i, params.H, params.W,
                       C_ib, C_ob,
                       packed_input_dc);

    // Read output regression data
    size_t Ho(compute_output_dim(params.H, params.k, params.s, params.p));
    size_t Wo(compute_output_dim(params.W, params.k, params.s, params.p));
    std::cerr << "Output image dims: " << Ho << ", " << Wo << std::endl;
    std::string out_fname =
        get_pathname(data_dir, "out", "pool",
                     params,
                     params.C_i*Ho*Wo);
    std::cout << "Maxpool: output file= " << out_fname << std::endl;

    BufferT output_dc_answers = read_inputs<BufferT>(out_fname);
    TEST_ASSERT(output_dc_answers.size() == params.C_i*Ho*Wo);

    // Pack output answer data
    BufferT packed_output_dc_answers(output_dc_answers.size());
    small::pack_buffer(output_dc_answers,
                       small::OUTPUT,
                       1U, params.C_i, Ho, Wo,
                       C_ib, C_ob,
                       packed_output_dc_answers);

    // Allocate output buffer
#if defined(QUANTIZED)
    BufferT packed_output_dc(output_dc_answers.size()*4);  /// @todo HACK hardcoded.
#else
    BufferT packed_output_dc(output_dc_answers.size());
#endif

    uint8_t t_pad=0, b_pad=0, l_pad=0, r_pad=0;
    if (params.p == 'f')
    {
        small::calc_padding(params.H, params.k, params.s, t_pad, b_pad);
        small::calc_padding(params.W, params.k, params.s, l_pad, r_pad);
    }

    // Compute layer
    small::Maxpool2D(params.k, params.s,
                     t_pad, b_pad, l_pad, r_pad,
                     params.C_i, params.H, params.W,
                     packed_input_dc, packed_output_dc);

    // Check answer
    bool passing = true;
    for (size_t ix = 0; ix < packed_output_dc_answers.size(); ++ix)
    {
        if (packed_output_dc[ix] != packed_output_dc_answers[ix])
        {
            passing = false;

            std::cout << "FAIL: Maxpool_out(" << ix << ")--> "
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
void test_maxpool_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {16,  3,  3, 3, 2, 'v', 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3, 13, 3, 2, 'v', 0},

        {16, 30, 30, 3, 2, 'v', 0},
        {96, 30, 30, 3, 2, 'v', 0},
        {96,  3, 13, 3, 2, 'v', 0},

        {16,  3,  3, 3, 2, 'f', 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3, 13, 3, 2, 'f', 0},
        {96, 30, 30, 3, 2, 'f', 0},
        {96,  3, 13, 3, 2, 'f', 0}
    };
    for (LayerParams const &p: params)
    {
#if defined(QUANTIZED)
        TEST_CHECK(true == run_maxpool_config<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(true == run_maxpool_config<small::FloatBuffer>(p));
#endif
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"maxpool_regression_data",          test_maxpool_regression_data},
    {NULL, NULL}
};
