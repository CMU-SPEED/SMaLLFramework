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

#define PARALLEL 1;

#include <acutest.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>

#include <small.h>
#include <small/ReLULayer.hpp>

#include "test_utils.hpp"
#include "Timer.hpp"

std::string const data_dir("../test/regression_data");

//****************************************************************************
template <class BufferT>
BufferT create_relu_data(size_t num_elements)
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution{0.f, 1.f};  // what distribution is Torch::Tensor::randn?

    BufferT input_dc(num_elements);

    for (size_t ix = 0; ix < num_elements; ++ix)
    {
#if defined(QUANTIZED)
        input_dc[ix] = (typename BufferT::value_type)(64*distribution(generator));
#else
        input_dc[ix] = distribution(generator);
#endif
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

#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_relu_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts);
#else
    small::FloatBuffer  input_dc = create_relu_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer  output_dc(num_input_elts);
#endif

    small::ReLUActivation(C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == 0));
        //std::cout << ix << ": ReLU(" << (int)input_dc[ix] << ")-->"
        //          << (int)output_dc[ix] << std::endl;
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

#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_relu_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts);
#else
    small::FloatBuffer  input_dc = create_relu_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer  output_dc(num_input_elts);
#endif

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

#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_relu_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts);
#else
    small::FloatBuffer  input_dc = create_relu_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer  output_dc(num_input_elts);
#endif

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
template <class BufferT>
bool run_relu_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "relu",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nReLU: input file = " << in_fname << std::endl;

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
        get_pathname(data_dir, "out", "relu",
                     params,
                     params.C_i*Ho*Wo);
    std::cout << "ReLU: output file= " << out_fname << std::endl;

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
    BufferT packed_output_dc(output_dc_answers.size());

    // Compute layer
    small::ReLUActivation(params.C_i,
                          params.H, params.W,
                          packed_input_dc, packed_output_dc);

    // Check answer
    bool passing = true;
    for (size_t ix = 0; ix < packed_output_dc_answers.size(); ++ix)
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
template <class BufferT>
bool run_relu_layer_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    small::ReLULayer<BufferT> relu(params.C_i, params.H, params.W);

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "relu",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nReLU: input file = " << in_fname << std::endl;

    // Allocate the input buffer
    BufferT input_dc = read_inputs<BufferT>(in_fname);

    TEST_ASSERT(input_dc.size() == relu.input_buffer_size());
    TEST_ASSERT(params.C_i*params.H*params.W == relu.input_buffer_size());

    // Pack input data
    BufferT packed_input_dc(input_dc.size());
    small::pack_buffer(input_dc,
                       small::INPUT,
                       1U, params.C_i, params.H, params.W,
                       C_ib, C_ob,
                       packed_input_dc);

    // Read output regression data
    size_t Ho(small::compute_output_dim(params.H, params.k, params.s, params.p));
    size_t Wo(small::compute_output_dim(params.W, params.k, params.s, params.p));
    std::cerr << "Output image dims: " << Ho << ", " << Wo << std::endl;
    std::string out_fname =
        get_pathname(data_dir, "out", "relu",
                     params,
                     params.C_i*Ho*Wo);
    std::cout << "ReLU: output file= " << out_fname << std::endl;

    BufferT output_dc_answers = read_inputs<BufferT>(out_fname);
    TEST_ASSERT(relu.output_buffer_size() == params.C_i*Ho*Wo);
    TEST_ASSERT(relu.output_buffer_size() == output_dc_answers.size());

    // Pack output answer data
    BufferT packed_output_dc_answers(relu.output_buffer_size());
    small::pack_buffer(output_dc_answers,
                       small::OUTPUT,
                       1U, params.C_i, Ho, Wo,
                       C_ib, C_ob,
                       packed_output_dc_answers);

    // Allocate output buffer
    BufferT packed_output_dc(relu.output_buffer_size());

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

    if (passing) std::cerr << "Test PASSED\n";
    return passing;
}

//****************************************************************************
//****************************************************************************
void test_relu_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {16,  1,  1, 1, 1, small::PADDING_V, 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  1,  6, 1, 1, small::PADDING_V, 0},
        {96,  1,  6, 1, 1, small::PADDING_V, 0},
        {96, 30, 30, 1, 1, small::PADDING_V, 0}
    };

    for (LayerParams const &p : params)
    {
#if defined(QUANTIZED)
        TEST_CHECK(true == run_relu_config<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(true == run_relu_config<small::FloatBuffer>(p));
#endif
    }
}

//****************************************************************************
void test_relu_layer_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {16,  1,  1, 1, 1, small::PADDING_V, 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  1,  6, 1, 1, small::PADDING_V, 0},
        {96,  1,  6, 1, 1, small::PADDING_V, 0},
        {96, 30, 30, 1, 1, small::PADDING_V, 0}
    };

    for (LayerParams const &p : params)
    {
#if defined(QUANTIZED)
        TEST_CHECK(true == run_relu_layer_config<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(true == run_relu_layer_config<small::FloatBuffer>(p));
#endif
    }
}

//****************************************************************************
void measure_relu_performance(void)
{
    // C_i,Hi,Wi,k,s,p,C_o
    std::vector<LayerParams> params =
    {
        {  16,   48,  48, 3, 1, small::PADDING_F,   16},
        {  32,   24,  24, 3, 1, small::PADDING_F,   32},
        { 128,    6,   6, 3, 1, small::PADDING_F,  256},
        {  64,   12,  12, 3, 1, small::PADDING_F,  128},
        {  16,   48,  48, 3, 1, small::PADDING_F,   32},
        {  32,   24,  24, 3, 1, small::PADDING_F,   64},
        {  32,   48,  48, 3, 1, small::PADDING_F,   32},
        { 128,   12,  12, 3, 1, small::PADDING_F,  128},
        {  64,   24,  24, 3, 1, small::PADDING_F,   64},
        { 128,   24,  24, 3, 1, small::PADDING_F,  128},
        { 256,   12,  12, 3, 1, small::PADDING_F,  256},
        { 512,   12,  12, 3, 1, small::PADDING_F,  512},
        { 1024,   6,   6, 3, 1, small::PADDING_F, 1024},
        {  32,  208, 208, 3, 1, small::PADDING_F,   64},
        {  64,  104, 104, 3, 1, small::PADDING_F,  128},
        { 128,   52,  52, 3, 1, small::PADDING_F,  256},
        { 256,   26,  26, 3, 1, small::PADDING_F,  512},
        { 512,   13,  13, 3, 1, small::PADDING_F, 1024}
    };

    uint32_t const  num_threads[] = {1, 2, 4};
    char const *str_num_threads[] = {"1", "2", "4"};
    uint32_t const num_runs(100);
    Timer t;

#if defined(QUANTIZED)
    std::string type("quint8");
    using Buffer = small::QUInt8Buffer;
#else
    std::string type("float");
    using Buffer = small::FloatBuffer;
#endif

    std::cout << "\nReLU(" << type << "),\n"
              << "C_i,H,W,k,s,nthd(set/get),runs,t_min,t_max,t_avg\n";

    for (LayerParams const &p: params)
    {
        size_t num_elts(p.C_i*p.H*p.W);

        Buffer  input_dc(num_elts);
        Buffer output_dc(num_elts);
        small::init(input_dc, num_elts);

        small::ReLULayer<Buffer> relu_layer(p.C_i, p.H, p.W);

        for (size_t ix = 0; ix < 3; ++ix)
        {
            setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);
            std::string ont = std::getenv("OMP_NUM_THREADS");
            auto nt = atol(ont.c_str());

            double tx(0.);
            double min_t = std::numeric_limits<double>::max();
            double max_t = 0.;

            // Warmup
            small::ReLUActivation(p.C_i, p.H, p.W, input_dc, output_dc);

            for (size_t iy = 0; iy < num_runs; ++iy)
            {
                t.start();
                small::ReLUActivation(p.C_i, p.H, p.W, input_dc, output_dc);
                t.stop();
                double ts = t.elapsed();
                tx += ts;
                min_t = std::min(min_t, ts);
                max_t = std::max(max_t, ts);
            }

            std::cout << "function," << p.C_i << ","<< p.H << "," << p.W
                      << "," << p.k << "," << p.s
                      << "," << num_threads[ix] << "/" << nt
                      << "," << num_runs
                      << "," << min_t << "," << max_t
                      << "," << (tx/num_runs) << std::endl;
        }

        for (size_t ix = 0; ix < 3; ++ix)
        {
            setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);
            std::string ont = std::getenv("OMP_NUM_THREADS");
            auto nt = atol(ont.c_str());

            double tx(0.);
            double min_t = std::numeric_limits<double>::max();
            double max_t = 0.;

            // Warm up
            relu_layer.compute_output(input_dc, output_dc);

            for (size_t iy = 0; iy < num_runs; ++iy)
            {
                t.start();
                relu_layer.compute_output(input_dc, output_dc);
                t.stop();
                double ts = t.elapsed();
                tx += ts;
                min_t = std::min(min_t, ts);
                max_t = std::max(max_t, ts);
            }

            std::cout << "class," << p.C_i << ","<< p.H << "," << p.W
                      << "," << p.k << "," << p.s
                      << "," << num_threads[ix] << "/" << nt
                      << "," << num_runs
                      << "," << min_t << "," << max_t
                      << "," << (tx/num_runs) << std::endl;
        }
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
    {"relu_performance", measure_relu_performance},
    {NULL, NULL}
};
