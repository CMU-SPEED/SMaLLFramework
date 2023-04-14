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

#include <small.h>
#include <small/MaxPool2DLayer.hpp>

#include "test_utils.hpp"
#include "Timer.hpp"

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
    std::cout << "\nMaxPool: input file = " << in_fname << std::endl;

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
    std::cout << "MaxPool: output file= " << out_fname << std::endl;

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
    if (params.p == small::PADDING_F)
    {
        small::calc_padding(params.H, params.k, params.s, t_pad, b_pad);
        small::calc_padding(params.W, params.k, params.s, l_pad, r_pad);
    }
    //std::cerr << "pad: " << (int)t_pad << "," << (int)b_pad
    //          << "," << (int)l_pad << "," << (int)r_pad << std::endl;

    // Compute layer
    small::MaxPool2D(params.k, params.s,
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

            std::cout << "FAIL: MaxPool_out(" << ix << ")--> "
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
template <typename BufferT>
bool run_maxpool_layer_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    small::MaxPool2DLayer<BufferT> maxpool_layer(params.k, params.k, params.s,
                                                 params.p, params.C_i,
                                                 params.H, params.W);

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "pool",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\nMaxPool: input file = " << in_fname << std::endl;

    // Allocate the input buffer
    BufferT input_dc = read_inputs<BufferT>(in_fname);

    TEST_ASSERT(input_dc.size() == params.C_i*params.H*params.W);
    TEST_ASSERT(maxpool_layer.input_buffer_size() == params.C_i*params.H*params.W);

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
        get_pathname(data_dir, "out", "pool",
                     params,
                     params.C_i*Ho*Wo);
    std::cout << "MaxPool: output file= " << out_fname << std::endl;

    BufferT output_dc_answers = read_inputs<BufferT>(out_fname);
    TEST_ASSERT(output_dc_answers.size() == params.C_i*Ho*Wo);
    TEST_ASSERT(maxpool_layer.output_buffer_size() == params.C_i*Ho*Wo);

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
    maxpool_layer.compute_output(packed_input_dc, packed_output_dc);

    // Check answer
    bool passing = true;
    for (size_t ix = 0; ix < maxpool_layer.output_buffer_size(); ++ix)
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

//****************************************************************************
void test_maxpool_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {16,  3,  3, 3, 2, small::PADDING_V, 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3, 13, 3, 2, small::PADDING_V, 0},

        {16, 30, 30, 3, 2, small::PADDING_V, 0},
        {96, 30, 30, 3, 2, small::PADDING_V, 0},
        {96,  3, 13, 3, 2, small::PADDING_V, 0},

        {16,  3,  3, 3, 2, small::PADDING_F, 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3, 13, 3, 2, small::PADDING_F, 0},
        {96, 30, 30, 3, 2, small::PADDING_F, 0},
        {96,  3, 13, 3, 2, small::PADDING_F, 0}
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
void test_maxpool_layer_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {16,  3,  3, 3, 2, small::PADDING_V, 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3, 13, 3, 2, small::PADDING_V, 0},

        {16, 30, 30, 3, 2, small::PADDING_V, 0},
        {96, 30, 30, 3, 2, small::PADDING_V, 0},
        {96,  3, 13, 3, 2, small::PADDING_V, 0},

        {16,  3,  3, 3, 2, small::PADDING_F, 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3, 13, 3, 2, small::PADDING_F, 0},
        {96, 30, 30, 3, 2, small::PADDING_F, 0},
        {96,  3, 13, 3, 2, small::PADDING_F, 0}
    };
    for (LayerParams const &p: params)
    {
#if defined(QUANTIZED)
        TEST_CHECK(true == run_maxpool_layer_config<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(true == run_maxpool_layer_config<small::FloatBuffer>(p));
#endif
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
    small::PaddingEnum p = small::PADDING_F;

    size_t Ho(small::compute_output_dim(H, k, s, p));
    size_t Wo(small::compute_output_dim(W, k, s, p));

    uint8_t t_pad=0, b_pad=0, l_pad=0, r_pad=0;
    small::calc_padding(H, k, s, t_pad, b_pad);
    small::calc_padding(W, k, s, l_pad, r_pad);

    size_t num_input_elts(C_i*H*W);
    size_t num_output_elts(C_i*Ho*Wo);

#if defined(QUANTIZED)
    std::string label("MaxPool2D(quint8): ");
    using Buffer = small::QUInt8Buffer;
#else
    std::string label("MaxPool2D(float): ");
    using Buffer = small::FloatBuffer;
#endif

    Buffer input_dc(num_input_elts);
    Buffer output_dc(num_output_elts);
    small::MaxPool2DLayer<Buffer>  maxpool_layer(k, k, s, p, C_i, H, W);

    small::init(input_dc, num_input_elts);

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
            small::MaxPool2D(k, s, t_pad, b_pad, l_pad, r_pad,
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
            maxpool_layer.compute_output(input_dc, output_dc);
            t.stop();
            double ts = t.elapsed();
            tx += ts;
            min_t = std::min(min_t, ts);
            max_t = std::max(max_t, ts);
        }
        std::cout << label << "MaxPool2DLayer,"
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
    {"maxpool_regression_data",       test_maxpool_regression_data},
    {"maxpool_layer_regression_data", test_maxpool_layer_regression_data},
    {"maxpool_performance", measure_maxpool_performance},
    {NULL, NULL}
};
