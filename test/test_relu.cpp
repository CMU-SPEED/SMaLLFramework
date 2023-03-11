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
#include <random>

#include <small.h>
#include <small/ReLU.hpp>

#include "test_utils.hpp"


//****************************************************************************
small::FloatBuffer create_relu_data(size_t num_elements)
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution{0.f, 1.f};  // what distribution is Torch::Tensor::randn?

    small::FloatBuffer input_dc(num_elements);

    for (size_t ix = 0; ix < num_elements; ++ix)
    {
        input_dc[ix] = distribution(generator);
    }

    return input_dc;
}

//****************************************************************************
void test_relu_single_element(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1;
    //size_t const kernel_size = 1;
    //size_t const stride = 1;
    //char const type = 'v';
    //size_t const C_o = 16;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;

    small::FloatBuffer input_dc = create_relu_data(num_input_elts);

    //size_t num_output_elts = C_i*H*W;
    small::FloatBuffer output_dc(num_input_elts);

    small::ReLUActivation(C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == 0.f));
        //std::cout << ix << ": ReLU(" << input_dc[ix] << ")-->" << output_dc[ix]
        //          << std::endl;
    }
}

//****************************************************************************
void test_relu_single_tile(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 6;
    //size_t const kernel_size = 1;
    //size_t const stride = 1;
    //char const type = 'v';
    //size_t const C_o = 16;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;

    small::FloatBuffer input_dc = create_relu_data(num_input_elts);

    //size_t num_output_elts = C_i*H*W;
    small::FloatBuffer output_dc(num_input_elts);

    small::ReLUActivation(C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == 0.f));
        //std::cout << ix << ": ReLU(" << input_dc[ix] << ")-->" << output_dc[ix]
        //          << std::endl;
    }
}

//****************************************************************************
void test_relu_large_tile(void)
{
    size_t const C_i = 16;
    size_t const H = 30;
    size_t const W = 30;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;

    small::FloatBuffer input_dc = create_relu_data(num_input_elts);

    //size_t num_output_elts = C_i*H*W;
    small::FloatBuffer output_dc(num_input_elts);

    small::ReLUActivation(C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == 0.f));
        //std::cout << ix << ": ReLU(" << input_dc[ix] << ")-->" << output_dc[ix]
        //          << std::endl;
    }
}

//****************************************************************************

//****************************************************************************
bool run_relu_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    // Read input data
    std::string in_fname =
        get_pathname("../test/regression_data", "in", "relu",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nReLU: input file = " << in_fname << std::endl;

    small::FloatBuffer input_dc = read_float_inputs(in_fname);
    TEST_ASSERT(input_dc.size() == params.C_i*params.H*params.W);

    // Pack input data
    small::FloatBuffer packed_input_dc(input_dc.size());
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
        get_pathname("../test/regression_data", "out", "relu",
                     params,
                     params.C_i*Ho*Wo);
    std::cout << "ReLU: output file= " << out_fname << std::endl;

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

    // Compute layer
    small::ReLUActivation(params.C_i,
                          params.H, params.W,
                          packed_input_dc, packed_output_dc);

    // Check answer
    bool passing = true;
    for (size_t ix = 0; ix < packed_output_dc.size(); ++ix)
    {
        if (packed_output_dc[ix] != packed_output_dc_answers[ix])
        {
            passing = false;
            std::cout << "FAIL: ReLU_out(" << ix << ")-->"
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
bool run_relu_layer_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    // Read input data
    std::string in_fname =
        get_pathname("../test/regression_data", "in", "relu",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nReLU: input file = " << in_fname << std::endl;

    small::ReLU<small::FloatBuffer> relu(params.H, params.W, params.C_i);

    // Allocate the input buffer
    small::FloatBuffer input_dc = read_float_inputs(in_fname);

    TEST_ASSERT(input_dc.size() == relu.input_buffer_size());
    TEST_ASSERT(params.C_i*params.H*params.W == relu.input_buffer_size());

    // Pack input data
    small::FloatBuffer packed_input_dc(input_dc.size());
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
        get_pathname("../test/regression_data", "out", "relu",
                     params,
                     params.C_i*Ho*Wo);
    std::cout << "ReLU: output file= " << out_fname << std::endl;

    small::FloatBuffer output_dc_answers = read_float_inputs(out_fname);
    TEST_ASSERT(relu.output_buffer_size() == params.C_i*Ho*Wo);
    TEST_ASSERT(relu.output_buffer_size() == output_dc_answers.size());

    // Pack output answer data
    small::FloatBuffer packed_output_dc_answers(relu.output_buffer_size());
    small::pack_buffer(output_dc_answers,
                       small::OUTPUT,
                       1U, params.C_i, Ho, Wo,
                       C_ib, C_ob,
                       packed_output_dc_answers);

    // Allocate output buffer
    small::FloatBuffer packed_output_dc(relu.output_buffer_size());

    // Compute layer
    relu.compute_output(packed_input_dc, packed_output_dc);

    // Check answer
    bool passing = true;
    for (size_t ix = 0; ix < relu.output_buffer_size(); ++ix)
    {
        if (packed_output_dc[ix] != packed_output_dc_answers[ix])
        {
            passing = false;
            std::cout << "FAIL: ReLU_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc_answers[ix]
                      << std::endl;
        }
    }

    return passing;
}

//****************************************************************************
void test_relu_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {16,  1,  1, 1, 1, 'v', 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  1,  6, 1, 1, 'v', 0},
        {96,  1,  6, 1, 1, 'v', 0},
        {96, 30, 30, 1, 1, 'v', 0}
    };

    for (LayerParams const &p : params)
    {
        TEST_CHECK(true == run_relu_config(p));
    }
}

//****************************************************************************
void test_relu_layer_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {16,  1,  1, 1, 1, 'v', 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  1,  6, 1, 1, 'v', 0},
        {96,  1,  6, 1, 1, 'v', 0},
        {96, 30, 30, 1, 1, 'v', 0}
    };

    for (LayerParams const &p : params)
    {
        TEST_CHECK(true == run_relu_layer_config(p));
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"relu_single_element",  test_relu_single_element},
    {"relu_single_tile",  test_relu_single_tile},
    {"relu_large_tile",  test_relu_large_tile},
    {"relu_regression_data", test_relu_regression_data},
    {"relu_layer_regression_data", test_relu_layer_regression_data},
    {NULL, NULL}
};
