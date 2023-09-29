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
// #include <small/PartialBiasLayer.hpp>

#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");

//****************************************************************************
// Generate partial_bias output regression data from unpack MaxPool input data.
//****************************************************************************
template <class BufferT>
bool compute_partial_bias_output(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "pool",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\npartial_bias: input file = " << in_fname << std::endl;

    BufferT output_dc = read_inputs<BufferT>(in_fname);
    TEST_ASSERT(output_dc.size() == params.C_i*params.H*params.W);

    // Read output regression data
    size_t Ho(small::compute_output_dim(
                  params.H, 1,1, params.p));
    size_t Wo(small::compute_output_dim(
                  params.W, 1,1, params.p));
    std::cerr << "Output image dims: " << Ho << ", " << Wo << std::endl;
    assert(Ho == params.H && Wo == params.W);
    std::string out_fname =
        get_pathname(data_dir, "out", "partial_bias",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "partial_bias: output file= " << out_fname << std::endl;

    BufferT output_dc_answers(params.C_i*Ho*Wo);

    uint8_t t_pad=0, b_pad=0, l_pad=0, r_pad=0;
    if (params.p == small::PADDING_F)
    {
        small::calc_padding(params.H, 1,1, t_pad, b_pad);
        small::calc_padding(params.W, 1,1, l_pad, r_pad);
    }

    std::cout << "Padding t,b,l,r: " << (int)t_pad << "," << (int)b_pad
              << "," << (int)l_pad << "," << (int)r_pad << std::endl;



    // Compute partial_bias outputs
    size_t num_outputs = 0;

    //Add the per channel bias value to all height and width elements
    for (size_t c = 0; c < params.C_i * params.H * params.W; ++c)
    {
        // Assuming CHW
        float channel_idx = (c)/(params.H*params.W);
        output_dc_answers[c] = output_dc[c] + channel_idx;

        // if(c %(params.W*params.H) == 0)
        // {
        //     printf("%f -> %f \n", output_dc[c], output_dc_answers[c]);
        // }
        num_outputs++;

    }


    std::cerr << "num_outputs = " << num_outputs << std::endl;
    std::cerr << "..should be = " << (params.C_i*Ho*Wo) << std::endl;
    TEST_CHECK(num_outputs == params.C_i*Ho*Wo);
    write_outputs(out_fname, output_dc_answers, num_outputs);

    return true;
}

//****************************************************************************
void test_compute_partial_bias_output(void)
{
    std::vector<LayerParams> params =
    {
        {16,  3,  3, 3, 2, small::PADDING_V, 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3, 13, 3, 2, small::PADDING_V, 0},

        {16, 30, 30, 3, 2, small::PADDING_V, 0},
        {96, 30, 30, 3, 2, small::PADDING_V, 0},
        {96,  3, 13, 3, 2, small::PADDING_V, 0},

        // {16,  3,  3, 3, 2, small::PADDING_F, 0},  //Ci,Hi,Wi,k,s,p,Co
        // {16,  3, 13, 3, 2, small::PADDING_F, 0},
        // {96, 30, 30, 3, 2, small::PADDING_F, 0},
        // {96,  3, 13, 3, 2, small::PADDING_F, 0}
    };
    for (LayerParams const &p: params)
    {
#if defined(QUANTIZED)
        TEST_CHECK(true == compute_partial_bias_output<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(true == compute_partial_bias_output<small::FloatBuffer>(p));
#endif
    }
}

//****************************************************************************
template <class BufferT>
bool run_partial_bias_config(LayerParams const &params)
{
    /// @todo add smart pointer to buffers
    // Read input data

    // Read output regression data
    uint8_t t_pad = 0, b_pad = 0, l_pad = 0, r_pad = 0;
    if (params.p == small::PADDING_F)
    {
        small::calc_padding(params.H, 1, 1, t_pad, b_pad);
        small::calc_padding(params.W, 1, 1, l_pad, r_pad);
    }

    size_t Ho(small::compute_output_dim(
        params.H, 1, 1, params.p));
    size_t Wo(small::compute_output_dim(
        params.W, 1, 1, params.p));
    std::cerr << "Output image dims: " << Ho << ", " << Wo << std::endl;
    std::string out_fname =
        get_pathname(data_dir, "out", "partial_bias",
                     params,
                     params.C_i * Ho * Wo);
    std::cout << "partial_bias: output file= " << out_fname << std::endl;

    std::string in_fname =
        get_pathname(data_dir, "in", "pool",
                     params,
                     params.C_i*params.H*params.W);
    std::cout << "\npartial_bias: input file = " << in_fname << std::endl;

    BufferT output_dc = read_inputs<BufferT>(in_fname);
    TEST_ASSERT(output_dc.size() == params.C_i * Ho * Wo);

    // Load output data to accumulate onto
    BufferT packed_output_dc(output_dc.size());
    small::pack_buffer(output_dc,
                       small::OUTPUT,
                       1U, params.C_i, Ho, Wo,
                       BufferT::C_ib, BufferT::C_ob,
                       packed_output_dc);

    // Bias vector
    BufferT packed_input_dc(params.C_i);
    for(size_t c = 0; c < params.C_i; c++)
    {
        packed_input_dc[c] = float(c);
    }

    // answers
    BufferT output_dc_answers = read_inputs<BufferT>(out_fname);
    TEST_ASSERT(output_dc_answers.size() == params.C_i*Ho*Wo);

    // Pack output answer data
    BufferT packed_output_dc_answers(output_dc_answers.size());
    small::pack_buffer(output_dc_answers,
                       small::OUTPUT,
                       1U, params.C_i, Ho, Wo,
                       BufferT::C_ib, BufferT::C_ob,
                       packed_output_dc_answers);

    // Compute layer
    small::PartialBias(
                     params.C_i, params.H, params.W,
                     packed_input_dc, packed_output_dc);

    // Check answer
    bool passing = true;
    for (size_t ix = 0; ix < packed_output_dc_answers.size(); ++ix)
    {
        //if (packed_output_dc[ix] != packed_output_dc_answers[ix])
        // printf("%f %f\n", packed_output_dc[ix], packed_output_dc_answers[ix]);
        if (!almost_equal(packed_output_dc[ix], packed_output_dc_answers[ix]))
        {
            passing = false;

            std::cout << "FAIL: partial_bias_out(" << ix << ")--> "
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

// template <typename BufferT>
// bool run_partial_bias_layer_config(LayerParams const &params)
// {
//     /// @todo add smart pointer to buffers
//     //=========================================================================
//     small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
//     size_t input_size = params.C_i*params.H*params.W;
//     small::PartialBiasLayer<BufferT> partial_bias_layer(input_shape);
//     //=========================================================================

//     // Read input data
//     std::string in_fname =
//         get_pathname(data_dir, "in", "pool",
//                      params,
//                      input_size);
//     std::cout << "\npartial_bias: input file = " << in_fname << std::endl;

//     // Allocate the input buffer
//     BufferT input_dc = read_inputs<BufferT>(in_fname);

//     TEST_ASSERT(input_dc.size() == input_size);

//     // Pack input data
//     BufferT packed_input_dc(input_dc.size());
//     small::pack_buffer(input_dc,
//                        small::INPUT,
//                        1U, params.C_i, params.H, params.W,
//                        BufferT::C_ib, BufferT::C_ob,
//                        packed_input_dc);

//     small::Tensor<BufferT> packed_input_tensor(
//         input_shape,
//         std::move(packed_input_dc));

//     // Read output regression data
//     auto output_shape(partial_bias_layer.output_shape());
//     size_t output_buffer_size(partial_bias_layer.output_size());

//     std::cerr << "Output image dims: "
//               << output_shape[small::HEIGHT] << "x" << output_shape[small::WIDTH]
//               << std::endl;
//     std::string out_fname =
//         get_pathname(data_dir, "out", "partial_bias",
//                      params,
//                      output_buffer_size);
//     std::cout << "partial_bias: output file= " << out_fname << std::endl;

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
// #if defined(QUANTIZED)
//     BufferT packed_output_dc(output_dc_answers.size()*4);  /// @todo HACK hardcoded.
// #else
//     BufferT packed_output_dc(output_dc_answers.size());
// #endif
//     small::Tensor<BufferT> packed_output_tensor(output_shape,
//                                                 std::move(packed_output_dc));

//     // Compute layer
//     partial_bias_layer.compute_output({&packed_input_tensor}, &packed_output_tensor);

//     // Check answer
//     bool passing = true;
//     BufferT &buf(packed_output_tensor.buffer());
//     for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
//     {
//         //if (buf[ix] != packed_output_dc_answers[ix])
//         if (!almost_equal(buf[ix], packed_output_dc_answers[ix]))
//         {
//             passing = false;
//             std::cout << "FAIL: partial_bias_out(" << ix << ")--> "
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

//****************************************************************************
void test_partial_bias_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {16,  3,  3, 3, 2, small::PADDING_V, 0},  //Ci,Hi,Wi,k,s,p,Co
        {16,  3, 13, 3, 2, small::PADDING_V, 0},

        {16, 30, 30, 3, 2, small::PADDING_V, 0},
        {96, 30, 30, 3, 2, small::PADDING_V, 0},
        {96,  3, 13, 3, 2, small::PADDING_V, 0},

        // {16,  3,  3, 3, 2, small::PADDING_F, 0},  //Ci,Hi,Wi,k,s,p,Co
        // {16,  3, 13, 3, 2, small::PADDING_F, 0},
        // {96, 30, 30, 3, 2, small::PADDING_F, 0},
        // {96,  3, 13, 3, 2, small::PADDING_F, 0}
    };
    for (LayerParams const &p: params)
    {
#if defined(QUANTIZED)
        TEST_CHECK(true == run_partial_bias_config<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(true == run_partial_bias_config<small::FloatBuffer>(p));
#endif
    }
}

//****************************************************************************
// void test_partial_bias_layer_regression_data(void)
// {
//     std::vector<LayerParams> params =
//     {
//         {16,  3,  3, 3, 2, small::PADDING_V, 0},  //Ci,Hi,Wi,k,s,p,Co
//         {16,  3, 13, 3, 2, small::PADDING_V, 0},

//         // {16, 30, 30, 3, 2, small::PADDING_V, 0},
//         // {96, 30, 30, 3, 2, small::PADDING_V, 0},
//         // {96,  3, 13, 3, 2, small::PADDING_V, 0},

//         // {16,  3,  3, 3, 2, small::PADDING_F, 0},  //Ci,Hi,Wi,k,s,p,Co
//         // {16,  3, 13, 3, 2, small::PADDING_F, 0},
//         // {96, 30, 30, 3, 2, small::PADDING_F, 0},
//         // {96,  3, 13, 3, 2, small::PADDING_F, 0}
//     };
//     for (LayerParams const &p: params)
//     {
// #if defined(QUANTIZED)
//         TEST_CHECK(true == run_partial_bias_layer_config<small::QUInt8Buffer>(p));
// #else
//         TEST_CHECK(true == run_partial_bias_layer_config<small::FloatBuffer>(p));
// #endif
//     }
// }

//****************************************************************************
// void measure_partial_bias_performance(void)
// {
//     // C_i,Hi,Wi,k,s,p,C_o
//     std::vector<LayerParams> params =
//     {
//         {  16,   48,  48, 3, 1, small::PADDING_F,   16},
//         {  32,   24,  24, 3, 1, small::PADDING_F,   32},

//         {  32,   48,  48, 3, 1, small::PADDING_F,   32},
//         {  64,   24,  24, 3, 1, small::PADDING_F,   64},
//         { 128,   12,  12, 3, 1, small::PADDING_F,  128},

//         {  16,   48,  48, 3, 1, small::PADDING_F,   32},
//         {  32,   24,  24, 3, 1, small::PADDING_F,   64},
//         {  64,   12,  12, 3, 1, small::PADDING_F,  128},
//         { 128,    6,   6, 3, 1, small::PADDING_F,  256},

//         { 128,   24,  24, 3, 1, small::PADDING_F,  128},
//         { 256,   12,  12, 3, 1, small::PADDING_F,  256},

//         { 512,   12,  12, 3, 1, small::PADDING_F,  512},
//         {1024,    6,   6, 3, 1, small::PADDING_F, 1024},

//         {  32,  208, 208, 3, 1, small::PADDING_F,   64},
//         {  64,  104, 104, 3, 1, small::PADDING_F,  128},
//         { 128,   52,  52, 3, 1, small::PADDING_F,  256},
//         { 256,   26,  26, 3, 1, small::PADDING_F,  512},
//         { 512,   13,  13, 3, 1, small::PADDING_F, 1024}
//     };

//     uint32_t const  num_threads[] = {1, 2, 4};
//     char const *str_num_threads[] = {"1", "2", "4"};
//     uint32_t const num_runs(100);
//     small::Timer t;

// #if defined(QUANTIZED)
//     std::string type("quint8");
//     using Buffer = small::QUInt8Buffer;
// #else
//     std::string type("float");
//     using Buffer = small::FloatBuffer;
// #endif

//     printf("\npartial_bias2D(%s) func.\n", type.c_str());
//     printf("\tC_i\tH\tW\tk\ts\tnthd\truns\tt_min\tt_max\tt_avg\n");

//     for (LayerParams const &p: params)
//     {
//         size_t Ho(small::compute_output_dim(p.H, p.k, p.s, p.p));
//         size_t Wo(small::compute_output_dim(p.W, p.k, p.s, p.p));

//         uint8_t t_pad=0, b_pad=0, l_pad=0, r_pad=0;
//         if (p.p == small::PADDING_F)
//         {
//             small::calc_padding(p.H, p.k, p.s, t_pad, b_pad);
//             small::calc_padding(p.W, p.k, p.s, l_pad, r_pad);
//         }

//         size_t num_input_elts(p.C_i*p.H*p.W);
//         size_t num_output_elts(p.C_i*Ho*Wo);

//         Buffer  input_dc(num_input_elts);
//         Buffer output_dc(num_output_elts);
//         small::init(input_dc, num_input_elts);

//         for (size_t ix = 0; ix < 3; ++ix)
//         {
//             setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);
//             //std::string ont = std::getenv("OMP_NUM_THREADS"); // read it back
//             //auto nt = atol(ont.c_str());

//             double tx(0.);
//             double min_t = std::numeric_limits<double>::max();
//             double max_t = 0.;

//             // Warmup
//             small::PartialBias(
//                              p.C_i, p.H, p.W,
//                              input_dc, output_dc);

//             for (size_t iy = 0; iy < num_runs; ++iy)
//             {
//                 t.start();
//                 small::PartialBias(
//                                  p.C_i, p.H, p.W,
//                                  input_dc, output_dc);
//                 t.stop();
//                 double ts = t.elapsed();
//                 tx += ts;
//                 min_t = std::min(min_t, ts);
//                 max_t = std::max(max_t, ts);
//             }

//             printf("function\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%lf\t%lf\t%lf\n",
//                    p.C_i, p.H, p.W, p.k, p.s,
//                    num_threads[ix], num_runs,
//                    min_t, max_t, (tx/num_runs));
//         }
//     }

//     printf("\npartial_bias2D(%s) class\n", type.c_str());
//     printf("\tC_i\tH\tW\tk\ts\tnthd\truns\tt_min\tt_max\tt_avg\n");

//     for (LayerParams const &p: params)
//     {
//         size_t Ho(small::compute_output_dim(p.H, p.k, p.s, p.p));
//         size_t Wo(small::compute_output_dim(p.W, p.k, p.s, p.p));

//         uint8_t t_pad=0, b_pad=0, l_pad=0, r_pad=0;
//         if (p.p == small::PADDING_F)
//         {
//             small::calc_padding(p.H, p.k, p.s, t_pad, b_pad);
//             small::calc_padding(p.W, p.k, p.s, l_pad, r_pad);
//         }

//         size_t num_input_elts(p.C_i*p.H*p.W);
//         size_t num_output_elts(p.C_i*Ho*Wo);
//         small::shape_type input_shape({1UL, p.C_i, p.H, p.W});

//         small::Tensor<Buffer> input_dc(input_shape);
//         small::init(input_dc.buffer(), num_input_elts);
//         std::vector<small::Tensor<Buffer>*> inputs;
//         inputs.push_back(&input_dc);

//         small::Tensor<Buffer> output_dc(num_output_elts);
//         std::vector<small::Tensor<Buffer>*> outputs;
//         outputs.push_back(&output_dc);

//         small::PartialBiasLayer<Buffer>
//             partial_bias_layer(input_shape);

//         for (size_t ix = 0; ix < 3; ++ix)
//         {
//             setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);
//             //std::string ont = std::getenv("OMP_NUM_THREADS");
//             //auto nt = atol(ont.c_str());

//             double tx(0.);
//             double min_t = std::numeric_limits<double>::max();
//             double max_t = 0.;

//             // Warm up
//             partial_bias_layer.compute_output({&input_dc}, &output_dc);

//             for (size_t iy = 0; iy < num_runs; ++iy)
//             {
//                 t.start();
//                 partial_bias_layer.compute_output({&input_dc}, &output_dc);
//                 t.stop();
//                 double ts = t.elapsed();
//                 tx += ts;
//                 min_t = std::min(min_t, ts);
//                 max_t = std::max(max_t, ts);
//             }

//             printf("class   \t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%lf\t%lf\t%lf\n",
//                    p.C_i, p.H, p.W, p.k, p.s,
//                    num_threads[ix], num_runs,
//                    min_t, max_t, (tx/num_runs));
//         }
//     }
// }

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    //{"compute_output", test_compute_partial_bias_output},
    {"partial_bias_regression_data", test_partial_bias_regression_data},
    // {"partial_bias_layer_regression_data", test_partial_bias_layer_regression_data},
    // {"partial_bias_performance", measure_partial_bias_performance},
    {NULL, NULL}
};
