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
#include <small/LeakyReLULayer.hpp>

#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");

//****************************************************************************
template <class BufferT>
BufferT create_leaky_data(size_t num_elements)
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
void test_leaky_single_element(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1;
    int const negative_slope = 0.01;
    //size_t const kernel_size = 1;
    //size_t const stride = 1;
    //char const type = 'v';
    //size_t const C_o = 16;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;

#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_leaky_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts);
#else
    small::FloatBuffer  input_dc = create_leaky_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer  output_dc(num_input_elts);
    small::FloatBuffer  negative_dc(1);
    negative_dc[0] = negative_slope;
#endif

    small::LeakyReLUActivation(
        C_i, H, W,
        input_dc,
        negative_dc,
        output_dc
    );

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {

        TEST_CHECK((input_dc[ix] >= 0) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == negative_slope*input_dc[ix]));
        //std::cout << ix << ": leaky(" << (int)input_dc[ix] << ")-->"
        //          << (int)output_dc[ix] << std::endl;
    }
}

//****************************************************************************
void test_leaky_single_tile(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 6;
    int const negative_slope = 0.01;
    //size_t const kernel_size = 1;
    //size_t const stride = 1;
    //char const type = 'v';
    //size_t const C_o = 16;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;

#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_leaky_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts);
#else
    small::FloatBuffer  input_dc = create_leaky_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer  output_dc(num_input_elts);
    small::FloatBuffer  negative_dc(1);
    negative_dc[0] = negative_slope;
#endif

    small::LeakyReLUActivation(C_i, H, W, input_dc, negative_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == negative_slope*input_dc[ix]));
        //std::cout << ix << ": leaky(" << input_dc[ix] << ")-->" << output_dc[ix]
        //          << std::endl;
    }
}

//****************************************************************************
void test_leaky_large_tile(void)
{
    size_t const C_i = 16;
    size_t const H = 30;
    size_t const W = 30;
    int const negative_slope = 0.01;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;

#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_leaky_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts);
#else
    small::FloatBuffer  input_dc = create_leaky_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer  output_dc(num_input_elts);
    small::FloatBuffer  negative_dc(1);
    negative_dc[0] = negative_slope;
#endif

    small::LeakyReLUActivation(C_i, H, W, input_dc, negative_dc, output_dc);

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK((input_dc[ix] >= 0.f) ?
                   (output_dc[ix] == input_dc[ix]) :
                   (output_dc[ix] == negative_slope*input_dc[ix]));
        //std::cout << ix << ": leaky(" << input_dc[ix] << ")-->" << output_dc[ix]
        //          << std::endl;
    }
}

//****************************************************************************

// //****************************************************************************
// template <class BufferT>
// bool run_leaky_config(LayerParams const &params)
// {
//     /// @todo add smart pointer to buffers
//     // Read input data
//     std::string in_fname =
//         get_pathname(data_dir, "in", "leaky",
//                      params,
//                      params.C_i*params.H*params.W);
//     std::cout << "\nleaky: input file = " << in_fname << std::endl;

//     BufferT input_dc = read_inputs<BufferT>(in_fname);
//     TEST_ASSERT(input_dc.size() == params.C_i*params.H*params.W);

//     // Pack input data
//     BufferT packed_input_dc(input_dc.size());
//     small::pack_buffer(input_dc,
//                        small::INPUT,
//                        1U, params.C_i, params.H, params.W,
//                        BufferT::C_ib, BufferT::C_ob,
//                        packed_input_dc);

//     // Read output regression data
//     size_t Ho(small::compute_output_dim(
//                   params.H, params.k, params.s, params.p));
//     size_t Wo(small::compute_output_dim(
//                   params.W, params.k, params.s, params.p));
//     std::cerr << "Output image dims: " << Ho << ", " << Wo << std::endl;
//     std::string out_fname =
//         get_pathname(data_dir, "out", "leaky",
//                      params,
//                      params.C_i*Ho*Wo);
//     std::cout << "leaky: output file= " << out_fname << std::endl;

//     BufferT output_dc_answers = read_inputs<BufferT>(out_fname);
//     TEST_ASSERT(output_dc_answers.size() == params.C_i*Ho*Wo);

//     // Pack output answer data
//     BufferT packed_output_dc_answers(output_dc_answers.size());
//     small::pack_buffer(output_dc_answers,
//                        small::OUTPUT,
//                        1U, params.C_i, Ho, Wo,
//                        BufferT::C_ib, BufferT::C_ob,
//                        packed_output_dc_answers);

//     // Allocate output buffer
//     BufferT packed_output_dc(output_dc_answers.size());

//     BufferT negative_dc(1);
//     negative_dc[0] = 0.01f;

//     // Compute layer
//     small::LeakyReLUActivation(params.C_i,
//                           params.H, params.W,
//                           packed_input_dc, negative_dc, packed_output_dc);

//     // Check answer
//     bool passing = true;
//     for (size_t ix = 0; ix < packed_output_dc_answers.size(); ++ix)
//     {
//         if (packed_output_dc[ix] != packed_output_dc_answers[ix])
//         {
//             passing = false;
//             std::cout << "FAIL: leaky_out(" << ix << ")-->"
//                       << std::setw(12) << std::setprecision(10)
//                       << packed_output_dc[ix] << "(computed) != "
//                       << std::setw(12) << std::setprecision(10)
//                       << packed_output_dc_answers[ix]
//                       << std::endl;
//         }
//     }

//     if (passing) std::cerr << "Test PASSED\n";
//     return passing;
// }

// //****************************************************************************
// template <class BufferT>
// bool run_leaky_layer_config(LayerParams const &params)
// {
//     /// @todo add smart pointer to buffers
//     //=========================================================================
//     small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
//     size_t input_size = params.C_i*params.H*params.W;
//     small::LeakyReLULayer<BufferT>   leaky_layer(input_size);
//     //=========================================================================

//     // Read input data
//     std::string in_fname =
//         get_pathname(data_dir, "in", "leaky",
//                      params,
//                      input_layer.output_size());
//     std::cout << "\nleaky: input file = " << in_fname << std::endl;

//     // Allocate the input buffer
//     BufferT input_dc = read_inputs<BufferT>(in_fname);

//     TEST_ASSERT(input_dc.size() == input_layer.output_size());

//     // Pack input data
//     BufferT packed_input_dc(input_dc.size());
//     small::pack_buffer(input_dc,
//                        small::INPUT,
//                        1U, params.C_i, params.H, params.W,
//                        BufferT::C_ib, BufferT::C_ob,
//                        packed_input_dc);

//     small::Tensor<BufferT> packed_input_tensor(
//         input_layer.output_shape(),
//         std::move(packed_input_dc));

//     // Read output regression data
//     auto output_shape(leaky_layer.output_shape());
//     size_t output_buffer_size(leaky_layer.output_size());

//     std::cerr << "Output image dims: "
//               << output_shape[small::HEIGHT] << "x" << output_shape[small::WIDTH]
//               << std::endl;
//     std::string out_fname =
//         get_pathname(data_dir, "out", "leaky",
//                      params,
//                      output_buffer_size);
//     std::cout << "leaky: output file= " << out_fname << std::endl;

//     BufferT output_dc_answers = read_inputs<BufferT>(out_fname);
//     TEST_ASSERT(output_dc_answers.size() == output_buffer_size);

//     // Pack output answer data
//     BufferT packed_output_dc_answers(output_dc_answers.size());
//     small::pack_buffer(output_dc_answers,
//                        small::OUTPUT,
//                        1U, output_shape[small::CHANNEL],
//                        output_shape[small::HEIGHT], output_shape[small::WIDTH],
//                        BufferT::C_ib, BufferT::C_ob,
//                        packed_output_dc_answers);

//     // Allocate output buffer
//     BufferT packed_output_dc(leaky_layer.output_size());
//     small::Tensor<BufferT> packed_output_tensor(output_shape,
//                                                 std::move(packed_output_dc));

//     // Compute layer
//     leaky_layer.compute_output({packed_input_tensor}, packed_output_tensor);

//     // Check answer
//     bool passing = true;
//     BufferT &buf(packed_output_tensor.buffer());
//     for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
//     {
//         if (buf[ix] != packed_output_dc_answers[ix])
//         {
//             passing = false;
//             std::cout << "FAIL: leaky_out(" << ix << ")-->"
//                       << std::setw(12) << std::setprecision(10)
//                       << buf[ix] << "(computed) != "
//                       << std::setw(12) << std::setprecision(10)
//                       << packed_output_dc_answers[ix]
//                       << std::endl;
//         }
//     }

//     if (passing) std::cerr << "Test PASSED\n";
//     return passing;
// }

// //****************************************************************************
// //****************************************************************************
// void test_leaky_regression_data(void)
// {
//     std::vector<LayerParams> params =
//     {
//         {16,  1,  1, 1, 1, small::PADDING_V, 0},  //Ci,Hi,Wi,k,s,p,Co
//         {16,  1,  6, 1, 1, small::PADDING_V, 0},
//         {96,  1,  6, 1, 1, small::PADDING_V, 0},
//         {96, 30, 30, 1, 1, small::PADDING_V, 0}
//     };

//     for (LayerParams const &p : params)
//     {
// #if defined(QUANTIZED)
//         TEST_CHECK(true == run_leaky_config<small::QUInt8Buffer>(p));
// #else
//         TEST_CHECK(true == run_leaky_config<small::FloatBuffer>(p));
// #endif
//     }
// }

// //****************************************************************************
// void test_leaky_layer_regression_data(void)
// {
//     std::vector<LayerParams> params =
//     {
//         {16,  1,  1, 1, 1, small::PADDING_V, 0},  //Ci,Hi,Wi,k,s,p,Co
//         {16,  1,  6, 1, 1, small::PADDING_V, 0},
//         {96,  1,  6, 1, 1, small::PADDING_V, 0},
//         {96, 30, 30, 1, 1, small::PADDING_V, 0}
//     };

//     for (LayerParams const &p : params)
//     {
// #if defined(QUANTIZED)
//         TEST_CHECK(true == run_leaky_layer_config<small::QUInt8Buffer>(p));
// #else
//         TEST_CHECK(true == run_leaky_layer_config<small::FloatBuffer>(p));
// #endif
//     }
// }

//****************************************************************************
void measure_leaky_performance(void)
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
    uint32_t const num_runs(100);
    small::Timer t;

#if defined(QUANTIZED)
    std::string type("quint8");
    using Buffer = small::QUInt8Buffer;
#else
    std::string type("float");
    using Buffer = small::FloatBuffer;
#endif

    printf("\nleaky(%s) func.\n", type.c_str());
    printf("\tC_i\tH\tW\tk\ts\tnthd\truns\tt_min\tt_max\tt_avg\n");

    for (LayerParams const &p: params)
    {
        size_t num_input_elts(p.C_i*p.H*p.W);

        Buffer  input_dc(num_input_elts);
        Buffer output_dc(num_input_elts);
        Buffer negative_dc(1);
        negative_dc[0] = 0.01f;
        small::init(input_dc, num_input_elts);

        for (size_t ix = 0; ix < 3; ++ix)
        {
            setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);
            //std::string ont = std::getenv("OMP_NUM_THREADS"); // read it back
            //auto nt = atol(ont.c_str());

            double tx(0.);
            double min_t = std::numeric_limits<double>::max();
            double max_t = 0.;

            // Warmup
            small::LeakyReLUActivation(p.C_i, p.H, p.W, input_dc, negative_dc, output_dc);

            for (size_t iy = 0; iy < num_runs; ++iy)
            {
                t.start();
                small::LeakyReLUActivation(p.C_i, p.H, p.W, input_dc, negative_dc, output_dc);
                t.stop();
                double ts = t.elapsed();
                tx += ts;
                min_t = std::min(min_t, ts);
                max_t = std::max(max_t, ts);
            }

            printf("function\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%lf\t%lf\t%lf\n",
                   p.C_i, p.H, p.W, p.k, p.s,
                   num_threads[ix], num_runs,
                   min_t, max_t, (tx/num_runs));
        }
    }

    printf("\nleaky(%s) class\n", type.c_str());
    printf("\tC_i\tH\tW\tk\ts\tnthd\truns\tt_min\tt_max\tt_avg\n");

    for (LayerParams const &p: params)
    {
        size_t num_input_elts(p.C_i*p.H*p.W);
        size_t num_output_elts(p.C_i*p.H*p.W);

        small::Tensor<Buffer> input_dc({1UL, p.C_i, p.H, p.W});
        small::init(input_dc.buffer(), num_input_elts);

        small::Tensor<Buffer> output_dc(num_output_elts);

        small::shape_type input_shape({1UL, p.C_i, p.H, p.W});
        small::LeakyReLULayer<Buffer> leaky_layer(input_shape);

        for (size_t ix = 0; ix < 3; ++ix)
        {
            setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);
            //std::string ont = std::getenv("OMP_NUM_THREADS");
            //auto nt = atol(ont.c_str());

            double tx(0.);
            double min_t = std::numeric_limits<double>::max();
            double max_t = 0.;

            // Warm up
            leaky_layer.compute_output({&input_dc}, &output_dc);

            for (size_t iy = 0; iy < num_runs; ++iy)
            {
                t.start();
                leaky_layer.compute_output({&input_dc}, &output_dc);
                t.stop();
                double ts = t.elapsed();
                tx += ts;
                min_t = std::min(min_t, ts);
                max_t = std::max(max_t, ts);
            }

            printf("class   \t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%lf\t%lf\t%lf\n",
                   p.C_i, p.H, p.W, p.k, p.s,
                   num_threads[ix], num_runs,
                   min_t, max_t, (tx/num_runs));
        }
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"leaky_single_element",  test_leaky_single_element},
    {"leaky_single_tile",  test_leaky_single_tile},
    {"leaky_large_tile",  test_leaky_large_tile},
    // {"leaky_regression_data", test_leaky_regression_data},
    // {"leaky_layer_regression_data", test_leaky_layer_regression_data},
    // {"leaky_performance", measure_leaky_performance},
    {NULL, NULL}
};
