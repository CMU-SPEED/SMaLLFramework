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

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

typedef float dtype;

#include <small.h>

#include "test_utils.hpp"
#include "Timer.hpp"

std::string const data_dir("../test/regression_data");

//****************************************************************************
template <typename RealT>
bool run_maxpool_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "pool",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nMaxpool: input file = " << in_fname << std::endl;

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
        get_pathname(data_dir, "out", "pool",
                     params,
                     params.C_i*Ho*Wo);
    std::cout << "Maxpool: output file= " << out_fname << std::endl;
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

    uint8_t t_pad=0, b_pad=0, l_pad=0, r_pad=0;
    if (params.p == 'f')
    {
        CALC_PADDING(params.H, params.k, params.s, t_pad, b_pad);
        CALC_PADDING(params.W, params.k, params.s, l_pad, r_pad);
    }

    // Compute layer
    Maxpool2D(0,
              params.k, params.s,
              t_pad, b_pad, l_pad, r_pad,
              params.C_i, params.H, params.W,
              packed_input_dc, packed_output_dc);

    // Check answer
    bool passing = true;
    for (size_t ix = 0; ix < num_output_elts; ++ix)
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

    free(input_dc);
    free(packed_input_dc);

    free(packed_output_dc);

    free(output_dc_answers);
    free(packed_output_dc_answers);

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
        TEST_CHECK(true == run_maxpool_config<float>(p));
    }
}

//****************************************************************************
void measure_maxpool_performance(void)
{
    size_t const C_i = 512;
    size_t const H = 192;
    size_t const W = 192;
    size_t const k = 3;
    size_t const s = 1;
    char const p = 'f';

    size_t Ho(compute_output_dim(H, k, s, p));
    size_t Wo(compute_output_dim(W, k, s, p));

    uint8_t t_pad=0, b_pad=0, l_pad=0, r_pad=0;
    CALC_PADDING(H, k, s, t_pad, b_pad);
    CALC_PADDING(W, k, s, l_pad, r_pad);

    size_t num_input_elts(C_i*H*W);
    size_t num_output_elts(C_i*Ho*Wo);

    std::string label("MaxPool2D(float): ");
    using RealT = float;

    RealT *input_dc = small_alloc<RealT>(num_input_elts);
    RealT *output_dc= small_alloc<RealT>(num_output_elts);

    uint32_t const  num_threads[] = {1, 2, 4};
    char const *str_num_threads[] = {"1", "2", "4"};
    uint32_t const num_runs(100);

    std::cout << std::endl;
    Timer t;
    for (size_t ix = 0; ix < 3; ++ix)
    {
        setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);
        //omp_set_num_threads(num_threads[ix]);
        //auto nt = omp_get_num_threads();
        std::string ont = std::getenv("OMP_NUM_THREADS");
        auto nt = atol(ont.c_str());

        double tx(0.);
        double min_t = std::numeric_limits<double>::max();
        double max_t = 0.;

        for (size_t iy = 0; iy < num_runs; ++iy)
        {
            t.start();
            Maxpool2D(0,
                      k, s,
                      t_pad, b_pad, l_pad, r_pad,
                      C_i, H, W,
                      input_dc, output_dc);
            t.stop();
            double ts = t.elapsed();
            tx += ts;
            min_t = std::min(min_t, ts);
            max_t = std::max(max_t, ts);
        }
        std::cout << label << "MaxPool2D(),"
                  << "k/s/C/H/W=" << k << "/" << s << "/" << C_i
                  << "/" << H << "/" << W
                  << ",nthd(set/get)=" << num_threads[ix] << "/" << nt
                  << ",min=" << min_t
                  << ",max=" << max_t
                  << ",avg=" << (tx/num_runs) << std::endl;
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"maxpool_regression_data",          test_maxpool_regression_data},
    {"maxpool_performance", measure_maxpool_performance},
    {NULL, NULL}
};
