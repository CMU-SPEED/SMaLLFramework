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

#define PARALLEL 1

#include <acutest.h>
#include <stdlib.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include <small.h>
#include <small/interface_abstract_fused.hpp>
#include <small/utils/Timer.hpp>

#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");

//****************************************************************************
template <class BufferT>
bool run_conv2d_relu_config(LayerParams const &params)
{
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nConv2D_ReLU: input file = " << in_fname << std::endl;

    BufferT input_dc = read_inputs<BufferT>(in_fname);
    TEST_ASSERT(input_dc.size() == params.C_i*params.H*params.W);

    BufferT packed_input_dc(input_dc.size());
    small::pack_buffer(input_dc,
                       small::INPUT,
                       1U, params.C_i, params.H, params.W,
                       BufferT::C_ib, BufferT::C_ob,
                       packed_input_dc);

    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "Conv2D_ReLU: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    BufferT packed_filter_dc(filter_dc.size());
    small::pack_buffer(filter_dc,
                       small::FILTER_CONV,
                       params.C_o, params.C_i, params.k, params.k,
                       BufferT::C_ib, BufferT::C_ob,
                       packed_filter_dc);

    size_t Ho(small::compute_output_dim(params.H, params.k, params.s, params.p));
    size_t Wo(small::compute_output_dim(params.W, params.k, params.s, params.p));
    size_t output_size = params.C_o*Ho*Wo;

    BufferT packed_output_dc(output_size);
    BufferT conv_output_dc(output_size);
    BufferT output_dc_answers(output_size);

    uint8_t t_pad = 0, b_pad = 0, l_pad = 0, r_pad = 0;
    if (params.p == small::PADDING_F)
    {
        small::calc_padding(params.H, params.k, params.s, t_pad, b_pad);
        small::calc_padding(params.W, params.k, params.s, l_pad, r_pad);
    }

    small::Conv2D_ReLU(
        params.k, params.k, params.s,
        t_pad, b_pad, l_pad, r_pad,
        params.C_o, params.C_i,
        params.H, params.W,
        packed_input_dc,
        packed_filter_dc,
        packed_output_dc);

    small::Conv2D(
        params.k, params.k, params.s,
        t_pad, b_pad, l_pad, r_pad,
        params.C_o, params.C_i,
        params.H, params.W,
        packed_input_dc,
        packed_filter_dc,
        conv_output_dc);

    small::ReLUActivation(params.C_o, Ho, Wo,
                          conv_output_dc,
                          output_dc_answers);

    bool passing = true;
    size_t mismatch_count = 0;
    for (size_t ix = 0; ix < output_dc_answers.size(); ++ix)
    {
        if ((packed_output_dc[ix] != output_dc_answers[ix]) &&
            !almost_equal(packed_output_dc[ix], output_dc_answers[ix]))
        {
            passing = false;
            ++mismatch_count;
            if (mismatch_count <= 20)
            {
                std::cout << "FAIL: Conv2D_ReLU_out(" << ix << ")-->"
                          << std::setw(12) << std::setprecision(10)
                          << packed_output_dc[ix] << "(computed) != "
                          << std::setw(12) << std::setprecision(10)
                          << output_dc_answers[ix]
                          << std::endl;
            }
        }
    }
    if (mismatch_count > 20)
    {
        std::cout << "Conv2D_ReLU mismatch count: " << mismatch_count
                  << " (only first 20 shown)." << std::endl;
    }

    if (passing) std::cerr << "Test PASSED\n";
    return passing;
}

//****************************************************************************
template <class BufferT>
bool run_conv2d_bias_relu_config(LayerParams const &params)
{
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nConv2D_Bias_ReLU: input file = " << in_fname << std::endl;

    BufferT input_dc = read_inputs<BufferT>(in_fname);
    TEST_ASSERT(input_dc.size() == params.C_i*params.H*params.W);

    BufferT packed_input_dc(input_dc.size());
    small::pack_buffer(input_dc,
                       small::INPUT,
                       1U, params.C_i, params.H, params.W,
                       BufferT::C_ib, BufferT::C_ob,
                       packed_input_dc);

    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "Conv2D_Bias_ReLU: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    BufferT packed_filter_dc(filter_dc.size());
    small::pack_buffer(filter_dc,
                       small::FILTER_CONV,
                       params.C_o, params.C_i, params.k, params.k,
                       BufferT::C_ib, BufferT::C_ob,
                       packed_filter_dc);

    size_t Ho(small::compute_output_dim(params.H, params.k, params.s, params.p));
    size_t Wo(small::compute_output_dim(params.W, params.k, params.s, params.p));
    size_t output_size = params.C_o*Ho*Wo;

    BufferT bias_dc(params.C_o);
    small::init_ones(bias_dc, bias_dc.size());

    BufferT packed_output_dc(output_size);
    BufferT conv_output_dc(output_size);
    BufferT output_dc_answers(output_size);

    uint8_t t_pad = 0, b_pad = 0, l_pad = 0, r_pad = 0;
    if (params.p == small::PADDING_F)
    {
        small::calc_padding(params.H, params.k, params.s, t_pad, b_pad);
        small::calc_padding(params.W, params.k, params.s, l_pad, r_pad);
    }

    small::Conv2D_Bias_ReLU(
        params.k, params.k, params.s,
        t_pad, b_pad, l_pad, r_pad,
        params.C_o, params.C_i,
        params.H, params.W,
        packed_input_dc,
        packed_filter_dc,
        bias_dc,
        packed_output_dc);

    small::Conv2D(
        params.k, params.k, params.s,
        t_pad, b_pad, l_pad, r_pad,
        params.C_o, params.C_i,
        params.H, params.W,
        packed_input_dc,
        packed_filter_dc,
        conv_output_dc);

    small::PartialBias(params.C_o, Ho, Wo,
                       bias_dc, conv_output_dc);
    small::ReLUActivation(params.C_o, Ho, Wo,
                          conv_output_dc,
                          output_dc_answers);

    bool passing = true;
    size_t mismatch_count = 0;
    for (size_t ix = 0; ix < output_dc_answers.size(); ++ix)
    {
        if ((packed_output_dc[ix] != output_dc_answers[ix]) &&
            !almost_equal(packed_output_dc[ix], output_dc_answers[ix]))
        {
            passing = false;
            ++mismatch_count;
            if (mismatch_count <= 20)
            {
                std::cout << "FAIL: Conv2D_Bias_ReLU_out(" << ix << ")-->"
                          << std::setw(12) << std::setprecision(10)
                          << packed_output_dc[ix] << "(computed) != "
                          << std::setw(12) << std::setprecision(10)
                          << output_dc_answers[ix]
                          << std::endl;
            }
        }
    }
    if (mismatch_count > 20)
    {
        std::cout << "Conv2D_Bias_ReLU mismatch count: " << mismatch_count
                  << " (only first 20 shown)." << std::endl;
    }

    if (passing) std::cerr << "Test PASSED\n";
    return passing;
}

//****************************************************************************
void test_conv2d_relu_regression_data(void)
{
#if defined(QUANTIZED)
    TEST_CHECK(true);
#else
    std::vector<LayerParams> params =
    {
        // {16,  3,  8, 3, 1, small::PADDING_F, 16},
        // {16,  3, 13, 3, 2, small::PADDING_F, 96},
        {96, 30, 30, 3, 2, small::PADDING_F, 96}
    };

    for (LayerParams const &p: params)
    {
        TEST_CHECK(true == run_conv2d_relu_config<small::FloatBuffer>(p));
    }
#endif
}

//****************************************************************************
void test_conv2d_bias_relu_regression_data(void)
{
#if defined(QUANTIZED)
    TEST_CHECK(true);
#else
    std::vector<LayerParams> params =
    {
        {16,  3,  8, 3, 1, small::PADDING_F, 16},
        {16,  3, 13, 3, 2, small::PADDING_F, 96},
        {96, 30, 30, 3, 2, small::PADDING_F, 96}
    };

    for (LayerParams const &p: params)
    {
        TEST_CHECK(true == run_conv2d_bias_relu_config<small::FloatBuffer>(p));
    }
#endif
}

//****************************************************************************
void test_conv2d_relu_ones_input_weights(void)
{
#if defined(QUANTIZED)
    TEST_CHECK(true);
#else
    LayerParams params{96, 30, 30, 3, 2, small::PADDING_F, 96};

    size_t Ho(small::compute_output_dim(params.H, params.k, params.s, params.p));
    size_t Wo(small::compute_output_dim(params.W, params.k, params.s, params.p));
    size_t output_size = params.C_o*Ho*Wo;

    small::FloatBuffer input_dc(params.C_i*params.H*params.W);
    small::FloatBuffer filter_dc(params.C_i*params.k*params.k*params.C_o);
    small::FloatBuffer bias_dc(params.C_o);
    small::init(input_dc, input_dc.size());
    small::init(filter_dc, filter_dc.size());
    small::init_ones(bias_dc, bias_dc.size());


    small::FloatBuffer packed_input_dc(input_dc.size());
    small::pack_buffer(input_dc,
                       small::INPUT,
                       1U, params.C_i, params.H, params.W,
                       small::FloatBuffer::C_ib, small::FloatBuffer::C_ob,
                       packed_input_dc);

    small::FloatBuffer packed_filter_dc(filter_dc.size());
    small::pack_buffer(filter_dc,
                       small::FILTER_CONV,
                       params.C_o, params.C_i, params.k, params.k,
                       small::FloatBuffer::C_ib, small::FloatBuffer::C_ob,
                       packed_filter_dc);

    uint8_t t_pad = 0, b_pad = 0, l_pad = 0, r_pad = 0;
    small::FloatBuffer out_conv(output_size);
    small::FloatBuffer out_ref_relu(output_size);
    small::FloatBuffer out_fused_relu(output_size);
    small::FloatBuffer out_bias_relu(output_size);

    small::Conv2D(
        params.k, params.k, params.s,
        t_pad, b_pad, l_pad, r_pad,
        params.C_o, params.C_i,
        params.H, params.W,
        packed_input_dc,
        packed_filter_dc,
        out_conv);

    small::ReLUActivation(params.C_o, Ho, Wo,
                          out_conv,
                          out_ref_relu);

    small::Conv2D_ReLU(
        params.k, params.k, params.s,
        t_pad, b_pad, l_pad, r_pad,
        params.C_o, params.C_i,
        params.H, params.W,
        packed_input_dc,
        packed_filter_dc,
        out_fused_relu);

    small::Conv2D_Bias_ReLU(
        params.k, params.k, params.s,
        t_pad, b_pad, l_pad, r_pad,
        params.C_o, params.C_i,
        params.H, params.W,
        packed_input_dc,
        packed_filter_dc,
        bias_dc,
        out_bias_relu);

    float expected_relu = static_cast<float>(params.C_i*params.k*params.k);
    float expected_bias_relu = expected_relu + 1.f;

    for (size_t ix = 0; ix < output_size; ++ix)
    {
        // TEST_CHECK(almost_equal(out_ref_relu[ix], expected_relu));
        // TEST_CHECK(almost_equal(out_fused_relu[ix], expected_relu));
        TEST_CHECK(almost_equal(out_fused_relu[ix], out_ref_relu[ix]));
        if(ix < 20)
        {
            printf("%f %f\n",out_fused_relu[ix], out_ref_relu[ix]);
        }
    }
#endif
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"conv2d_relu_regression_data", test_conv2d_relu_regression_data},
    // {"conv2d_bias_relu_regression_data", test_conv2d_bias_relu_regression_data},
    {"conv2d_relu_ones_input_weights", test_conv2d_relu_ones_input_weights},
    {NULL, NULL}
};
