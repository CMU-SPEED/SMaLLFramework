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
#include <iostream>
#include <iomanip>
#include <random>

typedef float dtype;

#include <small.h>

#include "test_utils.hpp"
#include "Timer.hpp"

std::string const data_dir("../test/regression_data");

//****************************************************************************
float* create_relu_data(size_t num_elements)
{
    using RealT = float;

    std::default_random_engine generator;
    std::normal_distribution<RealT> distribution{0.f, 1.f};  // what distribution is Torch::Tensor::randn?

    RealT *input_dc = small_alloc<RealT>(num_elements);
    TEST_ASSERT(nullptr != input_dc);

    for (size_t ix = 0; ix < num_elements; ++ix)
    {
        input_dc[ix] = distribution(generator);
    }

    return input_dc;
}

//****************************************************************************
void test_relu_single_element(void)
{
    using RealT = float;

    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1;
    //size_t const kernel_size = 1;
    //size_t const stride = 1;
    //char const type = 'v';
    //size_t const C_o = 16;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;

    RealT *input_dc = create_relu_data(num_input_elts);

    //size_t num_output_elts = C_i*H*W;
    RealT *output_dc = small_alloc<RealT>(num_input_elts);
    TEST_ASSERT(nullptr != output_dc);

    ReLUActivation(0, C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == 0.f));
        //std::cout << ix << ": ReLU(" << input_dc[ix] << ")-->" << output_dc[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(output_dc);
}

//****************************************************************************
void test_relu_single_tile(void)
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

    RealT *input_dc = create_relu_data(num_input_elts);

    //size_t num_output_elts = C_i*H*W;
    RealT *output_dc = small_alloc<RealT>(num_input_elts);
    TEST_ASSERT(nullptr != output_dc);

    ReLUActivation(0, C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == 0.f));
        //std::cout << ix << ": ReLU(" << input_dc[ix] << ")-->" << output_dc[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(output_dc);
}

//****************************************************************************
void test_relu_large_tile(void)
{
    using RealT = float;

    size_t const C_i = 16;
    size_t const H = 30;
    size_t const W = 30;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;

    RealT *input_dc = create_relu_data(num_input_elts);

    //size_t num_output_elts = C_i*H*W;
    RealT *output_dc = small_alloc<RealT>(num_input_elts);
    TEST_ASSERT(nullptr != output_dc);

    ReLUActivation(0, C_i, H, W, input_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == 0.f));
        //std::cout << ix << ": ReLU(" << input_dc[ix] << ")-->" << output_dc[ix]
        //          << std::endl;
    }

    free(input_dc);
    free(output_dc);
}

//****************************************************************************

//****************************************************************************
template <typename RealT>
bool run_relu_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    // Read input data
    std::string in_fname =
        get_pathname("../test/regression_data", "in", "relu",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nReLU: input file = " << in_fname << std::endl;

    RealT *input_dc = nullptr;
    uint32_t num_input_elts = read_float_inputs(in_fname, &input_dc);
    TEST_ASSERT(num_input_elts > 0);
    TEST_ASSERT(nullptr != input_dc);
    TEST_ASSERT(num_input_elts == params.C_i*params.H*params.W);

    // Pack input data
    RealT *packed_input_dc = small_alloc<RealT>(num_input_elts);
    small::convert_tensor2dc<float>(input_dc,
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
    RealT *output_dc_answers = nullptr;
    uint32_t num_output_elts = read_float_inputs(out_fname, &output_dc_answers);
    TEST_ASSERT(num_output_elts > 0);
    TEST_ASSERT(nullptr != output_dc_answers);
    TEST_ASSERT(num_output_elts == params.C_i*Ho*Wo);

    // Pack output answer data
    RealT *packed_output_dc_answers = small_alloc<RealT>(num_output_elts);
    small::convert_tensor2dc<float>(output_dc_answers,
                                    small::OUTPUT,
                                    1U, params.C_i, Ho, Wo,
                                    C_ib, C_ob,
                                    packed_output_dc_answers);

    // Allocate output buffer
    RealT *packed_output_dc = small_alloc<RealT>(num_output_elts);
    TEST_ASSERT(nullptr != packed_output_dc);

    // Compute layer
    ReLUActivation(0,
                   params.C_i,
                   params.H, params.W,
                   packed_input_dc, packed_output_dc);

    // Check answer
    bool passing = true;
    for (size_t ix = 0; ix < num_output_elts; ++ix)
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

    free(input_dc);
    free(packed_input_dc);
    free(packed_output_dc);
    free(output_dc_answers);
    free(packed_output_dc_answers);

    if (passing) std::cerr << "Test PASSED\n";
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
#if 0
        , {16,  1,  1, 1, 1, 'f', 0},
        {16,  1,  6, 1, 1, 'f', 0},
        {96,  1,  6, 1, 1, 'f', 0},
        {96, 30, 30, 1, 1, 'f', 0}
#endif
    };

    for (LayerParams const &p : params)
    {
        TEST_CHECK(true == run_relu_config<float>(p));
    }
}

//****************************************************************************
void measure_relu_performance(void)
{
    // C_i,Hi,Wi,k,s,p,C_o
    std::vector<LayerParams> params =
    {
        {  16,   48,  48, 3, 1, 'f',   16},
        {  32,   24,  24, 3, 1, 'f',   32},

        {  32,   48,  48, 3, 1, 'f',   32},
        {  64,   24,  24, 3, 1, 'f',   64},
        { 128,   12,  12, 3, 1, 'f',  128},

        {  16,   48,  48, 3, 1, 'f',   32},
        {  32,   24,  24, 3, 1, 'f',   64},
        {  64,   12,  12, 3, 1, 'f',  128},
        { 128,    6,   6, 3, 1, 'f',  256},

        { 128,   24,  24, 3, 1, 'f',  128},
        { 256,   12,  12, 3, 1, 'f',  256},

        { 512,   12,  12, 3, 1, 'f',  512},
        {1024,    6,   6, 3, 1, 'f', 1024},

        {  32,  208, 208, 3, 1, 'f',   64},
        {  64,  104, 104, 3, 1, 'f',  128},
        { 128,   52,  52, 3, 1, 'f',  256},
        { 256,   26,  26, 3, 1, 'f',  512},
        { 512,   13,  13, 3, 1, 'f', 1024}
    };

    uint32_t const  num_threads[] = {1, 2, 4};
    char const *str_num_threads[] = {"1", "2", "4"};
    uint32_t const num_runs(100);
    Timer t;

    using RealT = float;

    printf("\nReLU(float)\n");
    printf("\t\tC_i\tH\tW\tk\ts\tnthd(set/get)\truns\tt_min\tt_max\tt_avg\n");

    for (LayerParams const &p: params)
    {
        size_t num_input_elts(p.C_i*p.H*p.W);

        RealT *input_dc = create_relu_data(num_input_elts);
        RealT *output_dc = small_alloc<RealT>(num_input_elts);

        for (size_t ix = 0; ix < 3; ++ix)
        {
            setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);
            std::string ont = std::getenv("OMP_NUM_THREADS"); // read it back
            auto nt = atol(ont.c_str());

            double tx(0.);
            double min_t = std::numeric_limits<double>::max();
            double max_t = 0.;

            // Warmup
            ReLUActivation(0, p.C_i, p.H, p.W, input_dc, output_dc);

            for (size_t iy = 0; iy < num_runs; ++iy)
            {
                t.start();
                ReLUActivation(0, p.C_i, p.H, p.W, input_dc, output_dc);
                t.stop();
                double ts = t.elapsed();
                tx += ts;
                min_t = std::min(min_t, ts);
                max_t = std::max(max_t, ts);
            }

            printf("function\t%ld\t%d\t%d\t%d\t%d\t%d/%ld\t%d\t%lf\t%lf\t%lf\n",
                   p.C_i, p.H, p.W, p.k, p.s,
                   num_threads[ix], nt, num_runs,
                   min_t, max_t, (tx/num_runs));
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
    {"relu_performance", measure_relu_performance},
    {NULL, NULL}
};
