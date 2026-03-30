//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2026 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126
//****************************************************************************

#include <acutest.h>

#include <small.h>
#include <small/buffers.hpp>

//#define DEBUG_LAYERS 1
#include <small/SequentialLayer.hpp>
#include <small/Conv2DLayer.hpp>
#include <small/ReLULayer.hpp>

#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");

#if defined(QUANTIZED)
using Buffer = small::QUInt8Buffer;
#else
using Buffer = small::FloatBuffer;
#endif

//****************************************************************************
void test_sequential_ctor(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // Build conv2d
    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {96, 30, 30, 3, 2, small::PADDING_F, 96};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "\nConv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================
    BufferT bias(params.C_o);
    float bias_const = 1.0f;

    for (size_t ix = 0; ix < params.C_o; ++ix)
    {
        bias[ix] = bias_const;
    }

    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    //size_t input_size = params.C_i*params.H*params.W;

    std::vector<small::Layer<BufferT> *> layers;
    layers.push_back(new small::Conv2DLayer<BufferT>(input_shape,
                                                     params.k, params.k,
                                                     params.s, params.p,
                                                     params.C_o,
                                                     filter_dc, bias,
                                                     false));

    small::shape_type output_shape(
        {1UL, params.C_o,
         small::compute_output_dim(params.H, params.k, params.s, params.p),
         small::compute_output_dim(params.W, params.k, params.s, params.p)});

    layers.push_back(new small::ReLULayer<BufferT>(output_shape));

    small::SequentialLayer seq(layers);
}

//****************************************************************************
void test_sequential_conv2d_relu_without_bias(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,Hi,Wi,k,s,p,C_o
    // LayerParams params {96, 30, 30, 3, 2, small::PADDING_F, 96};
    LayerParams params {16, 30, 30, 3, 1, small::PADDING_V, 16};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "\nConv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================
    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;

    std::vector<small::Layer<BufferT>*> layers;
    layers.push_back(
        new small::Conv2DLayer<BufferT>(input_shape,
                                        params.k, params.k,
                                        params.s, params.p,
                                        params.C_o,
                                        filter_dc, false));
    layers.push_back(
        new small::ReLULayer<BufferT>(layers[0]->output_shape()));

    small::SequentialLayer<BufferT> seq_conv2d_relu(layers);

    small::shape_type output_shape(seq_conv2d_relu.output_shape());
    size_t output_buffer_size(seq_conv2d_relu.output_size());
    //=========================================================================

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nConv2D: input file = " << in_fname << std::endl;

    BufferT input_dc = read_inputs<BufferT>(in_fname);
    TEST_ASSERT(input_dc.size() == input_size);

    // Pack input data
    BufferT packed_input_dc(input_dc.size());
    small::pack_buffer(input_dc,
                       small::INPUT,
                       1U, params.C_i, params.H, params.W,
                       BufferT::C_ib, BufferT::C_ob,
                       packed_input_dc);

    small::Tensor<BufferT> packed_input_tensor(
        input_shape,
        std::move(packed_input_dc));

    // Read output regression data
    std::cerr << "Output image dims: "
              << output_shape[small::HEIGHT] << "x" << output_shape[small::WIDTH]
              << std::endl;
    std::string out_fname =
        get_pathname(data_dir, "out", "conv2d",
                     params,
                     output_buffer_size);
    std::cout << "Conv2D: output file= " << out_fname << std::endl;

    BufferT output_dc_answers = read_inputs<BufferT>(out_fname);
    TEST_ASSERT(output_dc_answers.size() == output_buffer_size);

    // Pack output answer data
    BufferT packed_output_dc_answers(output_dc_answers.size());
    small::pack_buffer(output_dc_answers,
                       small::OUTPUT,
                       1U, output_shape[small::CHANNEL],
                       output_shape[small::HEIGHT], output_shape[small::WIDTH],
                       BufferT::C_ib, BufferT::C_ob,
                       packed_output_dc_answers);

    // Allocate output buffer
#if defined(QUANTIZED)
    BufferT packed_output_dc(output_dc_answers.size()*4);  /// @todo HACK hardcoded.
#else
    BufferT packed_output_dc(output_dc_answers.size());
#endif
    small::Tensor<BufferT> packed_output_tensor(output_shape,
                                                std::move(packed_output_dc));

    // Compute layer
    seq_conv2d_relu.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == seq_conv2d_relu.output_size());

    // Check answer
    bool passing = true;
    BufferT &buf(packed_output_tensor.buffer());
    for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
    {
        BufferT::value_type answer = packed_output_dc_answers[ix];
        if (answer < 0)
        {
            answer = 0;
        }
#if defined(QUANTIZED)
        if (buf[ix] != answer)
#else
        if ((buf[ix] != answer) && !almost_equal(buf[ix], (answer)))
#endif
        {
            passing = false;

            std::cout << "FAIL: FusedConv2DReLU_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << buf[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << answer
                      << std::endl;
        }
        // else
        // {
        //     std::cout << "PASS: FusedConv2DReLU_out(" << ix << ")-->"
        //               << std::setw(12) << std::setprecision(10)
        //               << buf[ix] << "(computed) == "
        //               << std::setw(12) << std::setprecision(10)
        //               << answer
        //               << std::endl;
        // }
    }

    if (passing)
        std::cerr << "Test PASSED: " << in_fname << std::endl;
    else
        std::cerr << "Test FAILED: " << in_fname << std::endl;

    TEST_ASSERT(passing);
}

//****************************************************************************
template <class BufferT>
bool run_sequential_conv2d_relu_layer_config(LayerParams const &params)
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
    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;
    std::vector<small::Layer<BufferT>*> layers;

    layers.push_back(
        new small::Conv2DLayer<BufferT>(input_shape,
                                        params.k, params.k,
                                        params.s, params.p,
                                        params.C_o,
                                        filter_dc, false));
    layers.push_back(
        new small::ReLULayer<BufferT>(layers[0]->output_shape()));

    small::SequentialLayer<BufferT> seq_conv2d_relu(layers);
    //=========================================================================

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nConv2D: input file = " << in_fname << std::endl;

    // Allocate the input buffer
    BufferT input_dc(read_inputs<BufferT>(in_fname));

    TEST_ASSERT(input_dc.size() == input_size);

    // Pack input data
    BufferT packed_input_dc(input_dc.size());
    small::pack_buffer(input_dc,
                       small::INPUT,
                       1U, params.C_i, params.H, params.W,
                       BufferT::C_ib, BufferT::C_ob,
                       packed_input_dc);

    small::Tensor<BufferT> packed_input_tensor(
        input_shape,
        std::move(packed_input_dc));

    // Read output regression data
    auto output_shape(seq_conv2d_relu.output_shape());
    size_t output_buffer_size(seq_conv2d_relu.output_size());

    std::cerr << "Output image dims: "
              << output_shape[small::HEIGHT] << "x" << output_shape[small::WIDTH]
              << std::endl;
    std::string out_fname =
        get_pathname(data_dir, "out", "conv2d",
                     params,
                     output_buffer_size);
    std::cout << "Conv2D: output file= " << out_fname << std::endl;

    BufferT output_dc_answers = read_inputs<BufferT>(out_fname);
    TEST_ASSERT(output_dc_answers.size() == output_buffer_size);

    // Pack output answer data
    BufferT packed_output_dc_answers(output_dc_answers.size());
    small::pack_buffer(output_dc_answers,
                       small::OUTPUT,
                       1U, output_shape[small::CHANNEL],
                       output_shape[small::HEIGHT], output_shape[small::WIDTH],
                       BufferT::C_ib, BufferT::C_ob,
                       packed_output_dc_answers);

    // Allocate output buffer
#if defined(QUANTIZED)
    BufferT packed_output_dc(output_dc_answers.size()*4);  /// @todo HACK hardcoded.
#else
    BufferT packed_output_dc(output_dc_answers.size());
#endif
    small::Tensor<BufferT> packed_output_tensor(output_shape,
                                                std::move(packed_output_dc));

    // Compute layer
    seq_conv2d_relu.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == seq_conv2d_relu.output_size());

    // Check answer
    bool passing = true;
    BufferT &buf(packed_output_tensor.buffer());
    for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
    {
        typename BufferT::value_type answer = packed_output_dc_answers[ix];
        if (answer < 0)
        {
            answer = 0;
        }
#if defined(QUANTIZED)
        if (buf[ix] != answer)
#else
        if ((buf[ix] != answer) &&
            !almost_equal(buf[ix], answer))
#endif
        {
            passing = false;

            std::cout << "FAIL: Conv2D_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << buf[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << answer
                      << std::endl;
        }
    }

    if (passing)
        std::cerr << "Test PASSED: " << in_fname << std::endl;
    else
        std::cerr << "Test FAILED: " << in_fname << std::endl;

    return passing;
}

//****************************************************************************
//****************************************************************************
void test_sequential_conv2d_relu_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {3,  3,  3, 1, 1, small::PADDING_V, 16},
        {3,  3,  3, 3, 1, small::PADDING_V, 16},
        {3,  1,  1, 1, 1, small::PADDING_F, 16},
        {3,  3,  3, 1, 1, small::PADDING_F, 16},

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
        TEST_CHECK(
            true == run_sequential_conv2d_relu_layer_config<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(
            true == run_sequential_conv2d_relu_layer_config<small::FloatBuffer>(p));
#endif
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"SequentialLayer constructor", test_sequential_ctor},
    {"Seq(Conv2d+ReLU) w/o bias",   test_sequential_conv2d_relu_without_bias},
    {"Seq(Conv2d+ReLU) regression", test_sequential_conv2d_relu_regression_data},
    {NULL, NULL}
};
