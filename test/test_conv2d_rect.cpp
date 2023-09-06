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

#include <small.h>
#include <small/utils/Timer.hpp>
#include <small/Conv2DLayer.hpp>

#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");

//****************************************************************************
template <class BufferT>
bool run_conv2d_rect_config(LayerParams const &params)
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
                       BufferT::C_ib, BufferT::C_ob,
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
                       params.C_o, params.C_i, params.k, params.k,
                       BufferT::C_ib, BufferT::C_ob,
                       packed_filter_dc);

    // Read output regression data
    size_t Ho(small::compute_output_dim(
                  params.H, params.k, params.s, params.p));
    size_t Wo(small::compute_output_dim(
                  params.W, params.k, params.s, params.p));
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
                       BufferT::C_ib, BufferT::C_ob,
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
    small::Conv2D(params.k, params.k, params.s,
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
//****************************************************************************
void test_conv2d_rect_regression_data(void)
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
        TEST_CHECK(true == run_conv2d_rect_config<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(true == run_conv2d_rect_config<small::FloatBuffer>(p));
#endif
    }
}

//****************************************************************************
void measure_conv2d_rect_performance(void)
{
    // C_i,Hi,Wi,k,s,p,C_o
    std::vector<LayerParams> params =
    {
        {  16,   48,  48, 3, 1, small::PADDING_F,   16},
        {  32,   24,  24, 3, 1, small::PADDING_F,   32},

        {  32,   48,  48, 3, 1, small::PADDING_F,   32},
        {  64,   24,  24, 3, 1, small::PADDING_F,   64},
        { 128,   12,  12, 3, 1, small::PADDING_F,  128},

        {  16,   48,  48, 3, 1, small::PADDING_F,   32},
        {  32,   24,  24, 3, 1, small::PADDING_F,   64},
        {  64,   12,  12, 3, 1, small::PADDING_F,  128},
        { 128,    6,   6, 3, 1, small::PADDING_F,  256},

        { 128,   24,  24, 3, 1, small::PADDING_F,  128},
        { 256,   12,  12, 3, 1, small::PADDING_F,  256},

        { 512,   12,  12, 3, 1, small::PADDING_F,  512},
        {1024,    6,   6, 3, 1, small::PADDING_F, 1024},

        {  32,  208, 208, 3, 1, small::PADDING_F,   64},
        {  64,  104, 104, 3, 1, small::PADDING_F,  128},
        { 128,   52,  52, 3, 1, small::PADDING_F,  256},
        { 256,   26,  26, 3, 1, small::PADDING_F,  512},
        { 512,   13,  13, 3, 1, small::PADDING_F, 1024}
    };

    uint32_t const  num_threads[] = {1, 2, 4};
    char const *str_num_threads[] = {"1", "2", "4"};
    uint32_t const num_runs(10);
    small::Timer t;

#if defined(QUANTIZED)
    std::string type("quint8");
    using Buffer = small::QUInt8Buffer;
#else
    std::string type("float");
    using Buffer = small::FloatBuffer;
#endif

    printf("\nConv2D_rect(%s) func.\n", type.c_str());
    printf("\tC_i\tH\tW\tk\ts\tC_o\tnthd\truns\tt_min\tt_max\tt_avg\n");

    for (LayerParams const &p: params)
    {
        size_t Ho(small::compute_output_dim(p.H, p.k, p.s, p.p));
        size_t Wo(small::compute_output_dim(p.W, p.k, p.s, p.p));

        uint8_t t_pad=0, b_pad=0, l_pad=0, r_pad=0;
        if (p.p == small::PADDING_F)
        {
            small::calc_padding(p.H, p.k, p.s, t_pad, b_pad);
            small::calc_padding(p.W, p.k, p.s, l_pad, r_pad);
        }

        size_t num_input_elts(p.C_i*p.H*p.W);
        size_t num_filter_elts(p.C_i*p.k*p.k*p.C_o);
        size_t num_output_elts(p.C_o*Ho*Wo);

        Buffer input_dc(num_input_elts);
        Buffer filter_dc(num_filter_elts);
        Buffer output_dc(num_output_elts);
        small::init(input_dc, num_input_elts);
        small::init(filter_dc, num_filter_elts);

        for (size_t ix = 0; ix < 3; ++ix)
        {
            setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);
            //std::string ont = std::getenv("OMP_NUM_THREADS"); // read it back
            //auto nt = atol(ont.c_str());

            double tx(0.);
            double min_t = std::numeric_limits<double>::max();
            double max_t = 0.;

            // Warmup
            small::Conv2D(p.k, p.k, p.s, t_pad, b_pad, l_pad, r_pad,
                          p.C_o, p.C_i, p.H, p.W,
                          input_dc, filter_dc, output_dc);

            for (size_t iy = 0; iy < num_runs; ++iy)
            {
                t.start();
                small::Conv2D(p.k, p.k, p.s, t_pad, b_pad, l_pad, r_pad,
                              p.C_o, p.C_i, p.H, p.W,
                              input_dc, filter_dc, output_dc);
                t.stop();
                double ts = t.elapsed();
                tx += ts;
                min_t = std::min(min_t, ts);
                max_t = std::max(max_t, ts);
            }

            printf("function\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%lf\t%lf\t%lf\n",
                   p.C_i, p.H, p.W, p.k, p.s, p.C_o,
                   num_threads[ix], num_runs,
                   min_t, max_t, (tx/num_runs));
        }
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"conv2d_rect_regression_data",       test_conv2d_rect_regression_data},
    // {"conv2d_rect_performance", measure_conv2d_rect_performance},
    {NULL, NULL}
};
