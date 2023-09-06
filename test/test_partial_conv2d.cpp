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

#define DEBUG_LAYERS
#define RECORD_CALLS
#define PARALLEL 1

#include <acutest.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>

#include <small.h>
#include <small/utils/Timer.hpp>
#include <small/PartialConv2DLayer.hpp>

#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");

//****************************************************************************
//****************************************************************************
void test_partial_conv2d(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {96, 30, 30, 3, 2, small::PADDING_F, 96};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "PartialConv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================
    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;

    small::PartialConv2DLayer<BufferT> conv2d_layer(input_shape,
                                                    params.k, params.k,
                                                    params.s, params.p,
                                                    params.C_o,
                                                    filter_dc, false);

    small::shape_type output_shape(conv2d_layer.output_shape());
    size_t output_buffer_size(conv2d_layer.output_size());
    //=========================================================================

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nPartialConv2D: input file = " << in_fname << std::endl;

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
    std::cout << "PartialConv2D: output file= " << out_fname << std::endl;

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
    small::init_zeros(packed_output_tensor.buffer(),
                      packed_output_tensor.size());

    // Compute layer
    conv2d_layer.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size());

    // Check answer
    bool passing = true;
    BufferT &buf(packed_output_tensor.buffer());
    for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
    {
#if defined(QUANTIZED)
        if (buf[ix] != packed_output_dc_answers[ix])
#else
        if ((buf[ix] != packed_output_dc_answers[ix]) &&
            !almost_equal(buf[ix], (packed_output_dc_answers[ix])))
#endif
        {
            passing = false;

            std::cout << "FAIL: PartialConv2D_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << buf[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc_answers[ix]
                      << std::endl;
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_partial_conv2d_1s(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {96, 30, 30, 3, 2, small::PADDING_F, 96};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "PartialConv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================
    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;

    small::PartialConv2DLayer<BufferT> conv2d_layer(input_shape,
                                                    params.k, params.k,
                                                    params.s, params.p,
                                                    params.C_o,
                                                    filter_dc, false);

    small::shape_type output_shape(conv2d_layer.output_shape());
    size_t output_buffer_size(conv2d_layer.output_size());
    //=========================================================================

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nPartialConv2D: input file = " << in_fname << std::endl;

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
    std::cout << "PartialConv2D: output file= " << out_fname << std::endl;

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
    small::init_ones(packed_output_tensor.buffer(),
                     packed_output_tensor.size());

    // Compute layer
    conv2d_layer.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size());

    // Check answer
    bool passing = true;
    BufferT &buf(packed_output_tensor.buffer());
    for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
    {
#if defined(QUANTIZED)
        if (buf[ix] != packed_output_dc_answers[ix] + 1)
#else
        if ((buf[ix] != packed_output_dc_answers[ix]) &&
            !almost_equal(buf[ix], (packed_output_dc_answers[ix] + 1)))
#endif
        {
            passing = false;

            std::cout << "FAIL: PartialConv2D_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << buf[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc_answers[ix] + 1
                      << std::endl;
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_partial_conv2d_bias(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {96, 30, 30, 3, 2, small::PADDING_F, 96};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "PartialConv2D: filter file= " << filter_fname << std::endl;

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
    size_t input_size = params.C_i*params.H*params.W;

    small::PartialConv2DLayer<BufferT> conv2d_layer(input_shape,
                                                    params.k, params.k,
                                                    params.s, params.p,
                                                    params.C_o,
                                                    filter_dc, bias, false);

    small::shape_type output_shape(conv2d_layer.output_shape());
    size_t output_buffer_size(conv2d_layer.output_size());

    //=========================================================================

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nPartialConv2D: input file = " << in_fname << std::endl;

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
    std::cout << "PartialConv2D: output file= " << out_fname << std::endl;

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
    small::init_zeros(packed_output_tensor.buffer(),
                      packed_output_tensor.size());

    // Compute layer
    conv2d_layer.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size());

    // Check answer
    bool passing = true;
    BufferT &buf(packed_output_tensor.buffer());
    for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
    {
#if defined(QUANTIZED)
        if (buf[ix] != packed_output_dc_answers[ix] + bias_const)
#else
        if ((buf[ix] != packed_output_dc_answers[ix] + bias_const) &&
            !almost_equal(buf[ix], (packed_output_dc_answers[ix] + bias_const)))
#endif
        {
            passing = false;

            std::cout << "FAIL: PartialConv2D_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << buf[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc_answers[ix] + bias_const
                      << std::endl;
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_partial_conv2d_bias_1s(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {96, 30, 30, 3, 2, small::PADDING_F, 96};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "PartialConv2D: filter file= " << filter_fname << std::endl;

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
    size_t input_size = params.C_i*params.H*params.W;

    small::PartialConv2DLayer<BufferT> conv2d_layer(input_shape,
                                                    params.k, params.k,
                                                    params.s, params.p,
                                                    params.C_o,
                                                    filter_dc, bias, false);

    small::shape_type output_shape(conv2d_layer.output_shape());
    size_t output_buffer_size(conv2d_layer.output_size());

    //=========================================================================

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nPartialConv2D: input file = " << in_fname << std::endl;

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
    std::cout << "PartialConv2D: output file= " << out_fname << std::endl;

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
    small::init_ones(packed_output_tensor.buffer(),
                     packed_output_tensor.size());

    // Compute layer
    conv2d_layer.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size());

    // Check answer
    bool passing = true;
    BufferT &buf(packed_output_tensor.buffer());
    for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
    {
#if defined(QUANTIZED)
        if (buf[ix] != packed_output_dc_answers[ix] + bias_const + 1)
#else
        if ((buf[ix] != packed_output_dc_answers[ix] + bias_const + 1) &&
            !almost_equal(buf[ix], (packed_output_dc_answers[ix] + bias_const + 1)))
#endif
        {
            passing = false;

            std::cout << "FAIL: PartialConv2D_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << buf[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc_answers[ix] + bias_const + 1
                      << std::endl;
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_partial_conv2d_batchnorm_identity(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {96, 30, 30, 3, 2, small::PADDING_F, 96};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "PartialConv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================
    BufferT bn_weight(params.C_o);
    BufferT bn_bias(params.C_o);
    BufferT bn_running_mean(params.C_o);
    BufferT bn_running_variance(params.C_o);
    float   bn_eps = 0.f;
    for (size_t ix = 0; ix < params.C_o; ++ix)
    {
        bn_weight[ix] = 1;
        bn_bias[ix] = 0;
        bn_running_mean[ix] = 0;
        bn_running_variance[ix] = 1;
    }

    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;

    small::PartialConv2DLayer<BufferT> conv2d_layer(input_shape,
                                                    params.k, params.k,
                                                    params.s, params.p,
                                                    params.C_o,
                                                    filter_dc,
                                                    bn_weight, bn_bias,
                                                    bn_running_mean,
                                                    bn_running_variance,
                                                    bn_eps,
                                                    false);

    small::shape_type output_shape(conv2d_layer.output_shape());
    size_t output_buffer_size(conv2d_layer.output_size());

    //=========================================================================

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nPartialConv2D: input file = " << in_fname << std::endl;

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
    std::cout << "PartialConv2D: output file= " << out_fname << std::endl;

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
    small::init_zeros(packed_output_tensor.buffer(),
                      packed_output_tensor.size());

    // Compute layer
    conv2d_layer.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size());

    // Check answer
    bool passing = true;
    BufferT &buf(packed_output_tensor.buffer());
    for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
    {
#if defined(QUANTIZED)
        if (buf[ix] != packed_output_dc_answers[ix])
#else
        if ((buf[ix] != packed_output_dc_answers[ix]) &&
            !almost_equal(buf[ix], packed_output_dc_answers[ix]))
#endif
        {
            passing = false;

            std::cout << "FAIL: PartialConv2D_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << buf[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc_answers[ix]
                      << std::endl;
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_partial_conv2d_batchnorm_identity_1s(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {96, 30, 30, 3, 2, small::PADDING_F, 96};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "PartialConv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================
    BufferT bn_weight(params.C_o);
    BufferT bn_bias(params.C_o);
    BufferT bn_running_mean(params.C_o);
    BufferT bn_running_variance(params.C_o);
    float   bn_eps = 0.f;
    for (size_t ix = 0; ix < params.C_o; ++ix)
    {
        bn_weight[ix] = 1;
        bn_bias[ix] = 0;
        bn_running_mean[ix] = 0;
        bn_running_variance[ix] = 1;
    }

    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;

    small::PartialConv2DLayer<BufferT> conv2d_layer(input_shape,
                                                    params.k, params.k,
                                                    params.s, params.p,
                                                    params.C_o,
                                                    filter_dc,
                                                    bn_weight, bn_bias,
                                                    bn_running_mean,
                                                    bn_running_variance,
                                                    bn_eps,
                                                    false);

    small::shape_type output_shape(conv2d_layer.output_shape());
    size_t output_buffer_size(conv2d_layer.output_size());

    //=========================================================================

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nPartialConv2D: input file = " << in_fname << std::endl;

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
    std::cout << "PartialConv2D: output file= " << out_fname << std::endl;

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
    small::init_ones(packed_output_tensor.buffer(),
                     packed_output_tensor.size());

    // Compute layer
    conv2d_layer.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size());

    // Check answer
    bool passing = true;
    BufferT &buf(packed_output_tensor.buffer());
    for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
    {
#if defined(QUANTIZED)
        if (buf[ix] != packed_output_dc_answers[ix] + 1)
#else
        if ((buf[ix] != packed_output_dc_answers[ix] + 1) &&
            !almost_equal(buf[ix], packed_output_dc_answers[ix] + 1))
#endif
        {
            passing = false;

            std::cout << "FAIL: PartialConv2D_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << buf[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc_answers[ix] + 1
                      << std::endl;
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_partial_conv2d_batchnorm_bias_1(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {96, 30, 30, 3, 2, small::PADDING_F, 96};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "PartialConv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================
    BufferT bn_weight(params.C_o);
    BufferT bn_bias(params.C_o);
    BufferT bn_running_mean(params.C_o);
    BufferT bn_running_variance(params.C_o);
    float   bn_eps = 0.f;
    float   bias = 2.0f;
    for (size_t ix = 0; ix < params.C_o; ++ix)
    {
        bn_weight[ix] = 1;
        bn_bias[ix] = bias;
        bn_running_mean[ix] = 0;
        bn_running_variance[ix] = 1;
    }

    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;

    small::PartialConv2DLayer<BufferT> conv2d_layer(input_shape,
                                                    params.k, params.k,
                                                    params.s, params.p,
                                                    params.C_o,
                                                    filter_dc,
                                                    bn_weight, bn_bias,
                                                    bn_running_mean,
                                                    bn_running_variance,
                                                    bn_eps,
                                                    false);

    small::shape_type output_shape(conv2d_layer.output_shape());
    size_t output_buffer_size(conv2d_layer.output_size());

    //=========================================================================

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nPartialConv2D: input file = " << in_fname << std::endl;

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
    std::cout << "PartialConv2D: output file= " << out_fname << std::endl;

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
    small::init_zeros(packed_output_tensor.buffer(),
                      packed_output_tensor.size());

    // Compute layer
    conv2d_layer.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size());

    // Check answer
    bool passing = true;
    BufferT &buf(packed_output_tensor.buffer());
    for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
    {
#if defined(QUANTIZED)
        if (buf[ix] != packed_output_dc_answers[ix] + bias)
#else
        if ((buf[ix] != packed_output_dc_answers[ix] + bias) &&
            !almost_equal(buf[ix], packed_output_dc_answers[ix] + bias))
#endif
        {
            passing = false;

            std::cout << "FAIL: PartialConv2D_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << buf[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc_answers[ix] + bias
                      << std::endl;
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_partial_conv2d_batchnorm_bias_1_1s(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {96, 30, 30, 3, 2, small::PADDING_F, 96};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "PartialConv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================
    BufferT bn_weight(params.C_o);
    BufferT bn_bias(params.C_o);
    BufferT bn_running_mean(params.C_o);
    BufferT bn_running_variance(params.C_o);
    float   bn_eps = 0.f;
    float   bias = 2.0f;
    for (size_t ix = 0; ix < params.C_o; ++ix)
    {
        bn_weight[ix] = 1;
        bn_bias[ix] = bias;
        bn_running_mean[ix] = 0;
        bn_running_variance[ix] = 1;
    }

    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;

    small::PartialConv2DLayer<BufferT> conv2d_layer(input_shape,
                                                    params.k, params.k,
                                                    params.s, params.p,
                                                    params.C_o,
                                                    filter_dc,
                                                    bn_weight, bn_bias,
                                                    bn_running_mean,
                                                    bn_running_variance,
                                                    bn_eps,
                                                    false);

    small::shape_type output_shape(conv2d_layer.output_shape());
    size_t output_buffer_size(conv2d_layer.output_size());

    //=========================================================================

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nPartialConv2D: input file = " << in_fname << std::endl;

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
    std::cout << "PartialConv2D: output file= " << out_fname << std::endl;

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
    small::init_ones(packed_output_tensor.buffer(),
                     packed_output_tensor.size());

    // Compute layer
    conv2d_layer.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size());

    // Check answer
    bool passing = true;
    BufferT &buf(packed_output_tensor.buffer());
    for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
    {
#if defined(QUANTIZED)
        if (buf[ix] != packed_output_dc_answers[ix] + bias + 1)
#else
        if ((buf[ix] != packed_output_dc_answers[ix] + bias + 1) &&
            !almost_equal(buf[ix], packed_output_dc_answers[ix] + bias + 1))
#endif
        {
            passing = false;

            std::cout << "FAIL: PartialConv2D_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << buf[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << packed_output_dc_answers[ix] + bias + 1
                      << std::endl;
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_partial_conv2d_batchnorm_mean_1(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {16, 3, 3, 3, 1, small::PADDING_F, 16};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "PartialConv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================
    BufferT bn_weight(params.C_o);
    BufferT bn_bias(params.C_o);
    BufferT bn_running_mean(params.C_o);
    BufferT bn_running_variance(params.C_o);
    float   bn_eps = 0.f;
    float   running_mean(10.f);
    for (size_t ix = 0; ix < params.C_o; ++ix)
    {
        bn_weight[ix] = 1;
        bn_bias[ix] = 0;
        bn_running_mean[ix] = running_mean;
        bn_running_variance[ix] = 1;
    }

    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;

    small::PartialConv2DLayer<BufferT> conv2d_layer(input_shape,
                                                    params.k, params.k,
                                                    params.s, params.p,
                                                    params.C_o,
                                                    filter_dc,
                                                    bn_weight, bn_bias,
                                                    bn_running_mean,
                                                    bn_running_variance,
                                                    bn_eps,
                                                    false);

    small::shape_type output_shape(conv2d_layer.output_shape());
    size_t output_buffer_size(conv2d_layer.output_size());

    //=========================================================================

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nPartialConv2D: input file = " << in_fname << std::endl;

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
    std::cout << "PartialConv2D: output file= " << out_fname << std::endl;

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
    small::init_zeros(packed_output_tensor.buffer(),
                      packed_output_tensor.size());

    // Compute layer
    conv2d_layer.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size());

    // Check answer
    bool passing = true;
    BufferT &buf(packed_output_tensor.buffer());
    for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
    {
        // std::cerr << ix << ": computed,answer = "
        //           << buf[ix] << ","
        //           << packed_output_dc_answers[ix]
        //           << std::endl;
#if defined(QUANTIZED)
        if (buf[ix] != (packed_output_dc_answers[ix] - running_mean))
#else
        if ((buf[ix] != (packed_output_dc_answers[ix] - running_mean)) &&
            !almost_equal(buf[ix], packed_output_dc_answers[ix]-running_mean))
#endif
        {
            passing = false;

            std::cout << "FAIL: PartialConv2D_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << buf[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << (packed_output_dc_answers[ix] - running_mean)
                      << std::endl;
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_partial_conv2d_batchnorm_mean_1_1s(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {16, 3, 3, 3, 1, small::PADDING_F, 16};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "PartialConv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================
    BufferT bn_weight(params.C_o);
    BufferT bn_bias(params.C_o);
    BufferT bn_running_mean(params.C_o);
    BufferT bn_running_variance(params.C_o);
    float   bn_eps = 0.f;
    float   running_mean(10.f);
    for (size_t ix = 0; ix < params.C_o; ++ix)
    {
        bn_weight[ix] = 1;
        bn_bias[ix] = 0;
        bn_running_mean[ix] = running_mean;
        bn_running_variance[ix] = 1;
    }

    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;

    small::PartialConv2DLayer<BufferT> conv2d_layer(input_shape,
                                                    params.k, params.k,
                                                    params.s, params.p,
                                                    params.C_o,
                                                    filter_dc,
                                                    bn_weight, bn_bias,
                                                    bn_running_mean,
                                                    bn_running_variance,
                                                    bn_eps,
                                                    false);

    small::shape_type output_shape(conv2d_layer.output_shape());
    size_t output_buffer_size(conv2d_layer.output_size());

    //=========================================================================

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nPartialConv2D: input file = " << in_fname << std::endl;

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
    std::cout << "PartialConv2D: output file= " << out_fname << std::endl;

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
    small::init_ones(packed_output_tensor.buffer(),
                     packed_output_tensor.size());

    // Compute layer
    conv2d_layer.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size());

    // Check answer
    bool passing = true;
    BufferT &buf(packed_output_tensor.buffer());
    for (size_t ix = 0; ix < packed_output_tensor.size(); ++ix)
    {
        // std::cerr << ix << ": computed,answer = "
        //           << buf[ix] << ","
        //           << packed_output_dc_answers[ix]
        //           << std::endl;
#if defined(QUANTIZED)
        if (buf[ix] != (packed_output_dc_answers[ix] - running_mean + 1))
#else
        if ((buf[ix] != (packed_output_dc_answers[ix] - running_mean + 1)) &&
            !almost_equal(buf[ix], packed_output_dc_answers[ix] - running_mean + 1))
#endif
        {
            passing = false;

            std::cout << "FAIL: PartialConv2D_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << buf[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << (packed_output_dc_answers[ix] - running_mean + 1)
                      << std::endl;
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_partial_conv2d_batchnorm_mean_variance_1(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {16, 3, 3, 3, 1, small::PADDING_F, 16};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "PartialConv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================

    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nPartialConv2D: input file = " << in_fname << std::endl;

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

    //=========================================================================
    uint32_t Ho(small::compute_output_dim(
                    params.H, params.k, params.s, params.p));
    uint32_t Wo(small::compute_output_dim(
                    params.W, params.k, params.s, params.p));

    small::shape_type output_shape({1U, params.C_o, Ho, Wo});
    size_t output_buffer_size = params.C_o*Ho*Wo;

    // Read output regression data
    std::cerr << "Output image dims: "
              << output_shape[small::HEIGHT] << "x" << output_shape[small::WIDTH]
              << std::endl;
    std::string out_fname =
        get_pathname(data_dir, "out", "conv2d",
                     params,
                     output_buffer_size);
    std::cout << "PartialConv2D: output file= " << out_fname << std::endl;

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

    //=========================================================================
    // Compute mean and variance by output channel
    //=========================================================================
    BufferT bn_weight(params.C_o);
    BufferT bn_bias(params.C_o);
    BufferT bn_running_mean(params.C_o);
    BufferT bn_running_variance(params.C_o);
    float   bn_eps = 0.f;

    compute_mean_var(output_shape, output_dc_answers,
                     bn_running_mean, bn_running_variance);

    for (size_t ix = 0; ix < params.C_o; ++ix)
    {
        bn_weight[ix] = 1;
        bn_bias[ix] = 0;
    }

    small::PartialConv2DLayer<BufferT> conv2d_layer(input_shape,
                                                    params.k, params.k,
                                                    params.s, params.p,
                                                    params.C_o,
                                                    filter_dc,
                                                    bn_weight, bn_bias,
                                                    bn_running_mean,
                                                    bn_running_variance,
                                                    bn_eps,
                                                    false);

    output_shape = conv2d_layer.output_shape();
    output_buffer_size = conv2d_layer.output_size();
    //=========================================================================

    // Allocate output buffer
#if defined(QUANTIZED)
    BufferT packed_output_dc(output_dc_answers.size()*4);  /// @todo HACK hardcoded.
#else
    BufferT packed_output_dc(output_dc_answers.size());
#endif
    small::Tensor<BufferT> packed_output_tensor(output_shape,
                                                std::move(packed_output_dc));
    small::init_zeros(packed_output_tensor.buffer(),
                      packed_output_tensor.size());

    // Compute layer
    conv2d_layer.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size());

    //=========================================================================
    BufferT unpacked_output_tensor(packed_output_tensor.size());
    small::unpack_buffer(packed_output_tensor.buffer(),
                         small::OUTPUT,
                         1U, output_shape[small::CHANNEL],
                         output_shape[small::HEIGHT], output_shape[small::WIDTH],
                         BufferT::C_ib, BufferT::C_ob,
                         unpacked_output_tensor);

    BufferT output_mean(params.C_o);
    BufferT output_var(params.C_o);

    compute_mean_var(output_shape, unpacked_output_tensor,
                     output_mean, output_var);


    // Check answer
    bool passing = true;

    for (size_t ix = 0; ix < params.C_o; ++ix)
    {
#if defined(QUANTIZED)
        if ((output_mean[ix] != 0) || (output_var[ix] != 1))
#else
        if (!almost_equal(output_mean[ix], 0.f) ||
            !almost_equal(output_var[ix],  1.f))
#endif
        {
            passing = false;
            std::cerr << "FAIL: computed mean,var(" << ix
                      << "): "
                      << output_mean[ix] << ", "
                      << output_var[ix]
                      << std::endl;
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_partial_conv2d_batchnorm_mean_variance_1_1s(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {16, 3, 3, 3, 1, small::PADDING_F, 16};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "PartialConv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================

    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;

    // Read input data
    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nPartialConv2D: input file = " << in_fname << std::endl;

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

    //=========================================================================
    uint32_t Ho(small::compute_output_dim(
                    params.H, params.k, params.s, params.p));
    uint32_t Wo(small::compute_output_dim(
                    params.W, params.k, params.s, params.p));

    small::shape_type output_shape({1U, params.C_o, Ho, Wo});
    size_t output_buffer_size = params.C_o*Ho*Wo;

    // Read output regression data
    std::cerr << "Output image dims: "
              << output_shape[small::HEIGHT] << "x" << output_shape[small::WIDTH]
              << std::endl;
    std::string out_fname =
        get_pathname(data_dir, "out", "conv2d",
                     params,
                     output_buffer_size);
    std::cout << "PartialConv2D: output file= " << out_fname << std::endl;

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

    //=========================================================================
    // Compute mean and variance by output channel
    //=========================================================================
    BufferT bn_weight(params.C_o);
    BufferT bn_bias(params.C_o);
    BufferT bn_running_mean(params.C_o);
    BufferT bn_running_variance(params.C_o);
    float   bn_eps = 0.f;

    compute_mean_var(output_shape, output_dc_answers,
                     bn_running_mean, bn_running_variance);

    for (size_t ix = 0; ix < params.C_o; ++ix)
    {
        bn_weight[ix] = 1;
        bn_bias[ix] = 0;
    }

    small::PartialConv2DLayer<BufferT> conv2d_layer(input_shape,
                                                    params.k, params.k,
                                                    params.s, params.p,
                                                    params.C_o,
                                                    filter_dc,
                                                    bn_weight, bn_bias,
                                                    bn_running_mean,
                                                    bn_running_variance,
                                                    bn_eps,
                                                    false);

    output_shape = conv2d_layer.output_shape();
    output_buffer_size = conv2d_layer.output_size();
    //=========================================================================

    // Allocate output buffer
#if defined(QUANTIZED)
    BufferT packed_output_dc(output_dc_answers.size()*4);  /// @todo HACK hardcoded.
#else
    BufferT packed_output_dc(output_dc_answers.size());
#endif
    small::Tensor<BufferT> packed_output_tensor(output_shape,
                                                std::move(packed_output_dc));
    small::init_ones(packed_output_tensor.buffer(),
                     packed_output_tensor.size());

    // Compute layer
    conv2d_layer.compute_output({&packed_input_tensor}, &packed_output_tensor);
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size());

    //=========================================================================
    BufferT unpacked_output_tensor(packed_output_tensor.size());
    small::unpack_buffer(packed_output_tensor.buffer(),
                         small::OUTPUT,
                         1U, output_shape[small::CHANNEL],
                         output_shape[small::HEIGHT], output_shape[small::WIDTH],
                         BufferT::C_ib, BufferT::C_ob,
                         unpacked_output_tensor);

    BufferT output_mean(params.C_o);
    BufferT output_var(params.C_o);

    compute_mean_var(output_shape, unpacked_output_tensor,
                     output_mean, output_var);


    // Check answer
    bool passing = true;

    for (size_t ix = 0; ix < params.C_o; ++ix)
    {
#if defined(QUANTIZED)
        if ((output_mean[ix] != 1) || (output_var[ix] != 1))
#else
        if (!almost_equal(output_mean[ix], 1.f) ||
            !almost_equal(output_var[ix],  1.f))
#endif
        {
            passing = false;
            std::cerr << "FAIL: computed mean,var(" << ix
                      << "): "
                      << output_mean[ix] << ", "
                      << output_var[ix]
                      << std::endl;
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"partial_conv2d",                    test_partial_conv2d},
    {"partial_conv2d_1s",                 test_partial_conv2d_1s},
    {"partial_conv2d_bias",               test_partial_conv2d_bias},
    {"partial_conv2d_bias_1s",            test_partial_conv2d_bias_1s},
    {"partial_conv2d_batchnorm_identity",    test_partial_conv2d_batchnorm_identity},
    {"partial_conv2d_batchnorm_identity_1s", test_partial_conv2d_batchnorm_identity_1s},
    {"partial_conv2d_batchnorm_bias_1",   test_partial_conv2d_batchnorm_bias_1},
    {"partial_conv2d_batchnorm_bias_1_1s",test_partial_conv2d_batchnorm_bias_1_1s},
    {"partial_conv2d_batchnorm_mean_1",   test_partial_conv2d_batchnorm_mean_1},
    {"partial_conv2d_batchnorm_mean_1_1s",test_partial_conv2d_batchnorm_mean_1_1s},
    {"partial_conv2d_batchnorm_mean_variance_1",   test_partial_conv2d_batchnorm_mean_variance_1},
    {"partial_conv2d_batchnorm_mean_variance_1_1s",test_partial_conv2d_batchnorm_mean_variance_1_1s},
    {NULL, NULL}
};
