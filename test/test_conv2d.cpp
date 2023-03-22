//-----------------------------------------------------------------------------
// test/test_conv2d.cpp:  test cases for xxx
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
#include <small/Conv2DLayer.hpp>

#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");

//****************************************************************************
template <class BufferT>
bool run_conv2d_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nConv2D: input file = " << in_fname << std::endl;

    BufferT input_dc = read_inputs<BufferT>(in_fname);
    TEST_ASSERT(input_dc.size() == params.C_i*params.H*params.W);

    // Pack input data
    BufferT packed_input_dc(input_dc.size());
    small::pack_buffer(input_dc,
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

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    // Pack filter data
    BufferT packed_filter_dc(filter_dc.size());
    small::pack_buffer(filter_dc,
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

    BufferT output_dc_answers = read_inputs<BufferT>(out_fname);
    TEST_ASSERT(output_dc_answers.size() == params.C_o*Ho*Wo);

    // Pack output answer data
    BufferT packed_output_dc_answers(output_dc_answers.size());
    small::pack_buffer(output_dc_answers,
                       small::OUTPUT,
                       1U, params.C_o, Ho, Wo,
                       C_ib, C_ob,
                       packed_output_dc_answers);

    // Allocate output buffer
#if defined(QUANTIZED)
    BufferT packed_output_dc(output_dc_answers.size()*4);  /// @todo HACK hardcoded.
#else
    BufferT packed_output_dc(output_dc_answers.size());
#endif

    uint8_t t_pad=0, b_pad=0, l_pad=0, r_pad=0;
    if (params.p == small::PADDING_F)
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
    for (size_t ix = 0; ix < packed_output_dc_answers.size(); ++ix)
    {
#if defined(QUANTIZED)
        if (packed_output_dc[ix] != packed_output_dc_answers[ix])
#else
        if ((packed_output_dc[ix] != packed_output_dc_answers[ix]) &&
            !almost_equal(packed_output_dc[ix], packed_output_dc_answers[ix]))
#endif
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

    if (passing) std::cerr << "Test PASSED\n";
    return passing;
}

//****************************************************************************
template <class BufferT>
bool run_conv2d_layer_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "Conv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================
    small::Conv2DLayer<BufferT> conv2d_layer(params.k, params.k,
                                             params.s, params.p,
                                             params.C_i, params.C_o,
                                             params.H, params.W,
                                             filter_dc);
    //=========================================================================

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nConv2D: input file = " << in_fname << std::endl;

    BufferT input_dc = read_inputs<BufferT>(in_fname);

    TEST_ASSERT(input_dc.size() == params.C_i*params.H*params.W);
    TEST_ASSERT(conv2d_layer.input_buffer_size() == params.C_i*params.H*params.W);

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
        get_pathname(data_dir, "out", "conv2d",
                     params,
                     params.C_o*Ho*Wo);
    std::cout << "Conv2D: output file= " << out_fname << std::endl;

    BufferT output_dc_answers = read_inputs<BufferT>(out_fname);
    TEST_ASSERT(output_dc_answers.size() == params.C_o*Ho*Wo);
    TEST_ASSERT(conv2d_layer.output_buffer_size() == params.C_o*Ho*Wo);

    // Pack output answer data
    BufferT packed_output_dc_answers(output_dc_answers.size());
    small::pack_buffer(output_dc_answers,
                       small::OUTPUT,
                       1U, params.C_o, Ho, Wo,
                       C_ib, C_ob,
                       packed_output_dc_answers);

    // Allocate output buffer
#if defined(QUANTIZED)
    BufferT packed_output_dc(output_dc_answers.size()*4);  /// @todo HACK hardcoded.
#else
    BufferT packed_output_dc(output_dc_answers.size());
#endif

    // Compute layer
    conv2d_layer.compute_output(packed_input_dc, packed_output_dc);

    // Check answer
    bool passing = true;
    for (size_t ix = 0; ix < conv2d_layer.output_buffer_size(); ++ix)
    {
#if defined(QUANTIZED)
        if (packed_output_dc[ix] != packed_output_dc_answers[ix])
#else
        if ((packed_output_dc[ix] != packed_output_dc_answers[ix]) &&
            !almost_equal(packed_output_dc[ix], packed_output_dc_answers[ix]))
#endif
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

    if (passing) std::cerr << "Test PASSED\n";
    return passing;
}

//****************************************************************************

//****************************************************************************
void test_conv2d_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {16,  1,  1, 1, 1, small::PADDING_V, 16},
        {16,  1,  6, 1, 1, small::PADDING_V, 16},
        {16,  3,  3, 3, 1, small::PADDING_V, 16},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3,  8, 3, 1, small::PADDING_V, 16},
        {16, 30, 30, 3, 1, small::PADDING_V, 16},

        {16,  1,  6, 1, 1, small::PADDING_V, 96},
        {16,  3,  8, 3, 1, small::PADDING_V, 96},

        {96,  1,  6, 1, 1, small::PADDING_V, 16},
        {96,  3,  8, 3, 1, small::PADDING_V, 16},

        {96, 30, 30, 1, 1, small::PADDING_V, 96},
        {96, 30, 30, 3, 1, small::PADDING_V, 96},

#if 1
        {16,  3,  3, 3, 1, small::PADDING_F, 16},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3,  3, 3, 2, small::PADDING_F, 16},
        {16,  3,  8, 3, 1, small::PADDING_F, 16},
        {16,  3,  8, 3, 1, small::PADDING_F, 96},
        {16,  3, 13, 3, 2, small::PADDING_F, 16},
        {16,  3, 13, 3, 2, small::PADDING_F, 96},

        {96,  3,  8, 3, 1, small::PADDING_F, 16},
        {96,  3, 13, 3, 2, small::PADDING_F, 16},
        {96, 30, 30, 3, 1, small::PADDING_F, 96},
        {96, 30, 30, 3, 2, small::PADDING_F, 96}
#endif
    };
    for (LayerParams const &p: params)
    {
#if defined(QUANTIZED)
        TEST_CHECK(true == run_conv2d_config<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(true == run_conv2d_config<small::FloatBuffer>(p));
#endif
    }
}

//****************************************************************************
void test_conv2d_layer_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {16,  1,  1, 1, 1, small::PADDING_V, 16},
        {16,  1,  6, 1, 1, small::PADDING_V, 16},
        {16,  3,  3, 3, 1, small::PADDING_V, 16},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3,  8, 3, 1, small::PADDING_V, 16},
        {16, 30, 30, 3, 1, small::PADDING_V, 16},

        {16,  1,  6, 1, 1, small::PADDING_V, 96},
        {16,  3,  8, 3, 1, small::PADDING_V, 96},

        {96,  1,  6, 1, 1, small::PADDING_V, 16},
        {96,  3,  8, 3, 1, small::PADDING_V, 16},

        {96, 30, 30, 1, 1, small::PADDING_V, 96},
        {96, 30, 30, 3, 1, small::PADDING_V, 96},

#if 1
        {16,  3,  3, 3, 1, small::PADDING_F, 16},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3,  3, 3, 2, small::PADDING_F, 16},
        {16,  3,  8, 3, 1, small::PADDING_F, 16},
        {16,  3,  8, 3, 1, small::PADDING_F, 96},
        {16,  3, 13, 3, 2, small::PADDING_F, 16},
        {16,  3, 13, 3, 2, small::PADDING_F, 96},

        {96,  3,  8, 3, 1, small::PADDING_F, 16},
        {96,  3, 13, 3, 2, small::PADDING_F, 16},
        {96, 30, 30, 3, 1, small::PADDING_F, 96},
        {96, 30, 30, 3, 2, small::PADDING_F, 96}
#endif
    };
    for (LayerParams const &p: params)
    {
#if defined(QUANTIZED)
        TEST_CHECK(true == run_conv2d_layer_config<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(true == run_conv2d_layer_config<small::FloatBuffer>(p));
#endif
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"conv2d_regression_data",     test_conv2d_regression_data},
    {"conv2d_layer_regression_data",     test_conv2d_layer_regression_data},
    {NULL, NULL}
};
