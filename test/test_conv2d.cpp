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
void test_conv2d_layer_odd_output_channels(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,H,W,k,s,p,C_o
    LayerParams params {1024, 13, 13, 1, 1, small::PADDING_F, 13};

    // Odd packed buffers should fail
    try
    {
        small::shape_type input_shape{1U, params.C_i, params.H, params.W};
        BufferT filters(params.C_i*params.k*params.k*params.C_o);
        small::Conv2DLayer conv2d(input_shape,
                                  params.k, params.k,
                                  params.s, params.p,
                                  params.C_o,
                                  filters,
                                  true);

        TEST_ASSERT(params.C_o % BufferT::C_ob == 0);
    }
    catch (std::invalid_argument &e_obj)
    {
        TEST_CHECK(params.C_o % BufferT::C_ob != 0);
    }

    // Odd unpacked buffers should not fail
    try
    {
        small::shape_type input_shape{1U, params.C_i, params.H, params.W};

        BufferT filters(params.C_i*params.k*params.k*params.C_o);

        small::Conv2DLayer conv2d(input_shape,
                                  params.k, params.k,
                                  params.s, params.p,
                                  params.C_o,
                                  filters,
                                  false);

        TEST_ASSERT(conv2d.get_effective_output_channels() == params.C_o);
        if (params.C_o % BufferT::C_ob == 0)
        {
            TEST_ASSERT(conv2d.output_shape()[small::CHANNEL] == params.C_o);
        }
        else
        {
            TEST_ASSERT(conv2d.output_shape()[small::CHANNEL] ==
                        (params.C_o + (BufferT::C_ob - params.C_o % BufferT::C_ob)));
        }
    }
    catch (std::invalid_argument &e_obj)
    {
        TEST_CHECK(false);
    }

    // Test with bias
    try
    {
        small::shape_type input_shape{1U, params.C_i, params.H, params.W};

        BufferT filters(params.C_i*params.k*params.k*params.C_o);
        BufferT bias(params.C_o);

        small::Conv2DLayer conv2d(input_shape,
                                  params.k, params.k,
                                  params.s, params.p,
                                  params.C_o,
                                  filters,
                                  bias,
                                  false);

        TEST_ASSERT(conv2d.get_effective_output_channels() == params.C_o);
        if (params.C_o % BufferT::C_ob == 0)
        {
            TEST_ASSERT(conv2d.output_shape()[small::CHANNEL] == params.C_o);
        }
        else
        {
            TEST_ASSERT(conv2d.output_shape()[small::CHANNEL] ==
                        (params.C_o + (BufferT::C_ob - params.C_o % BufferT::C_ob)));
        }
    }
    catch (std::invalid_argument &e_obj)
    {
        TEST_CHECK(false);
    }

    // Test with batchnorm params
    try
    {
        small::shape_type input_shape{1U, params.C_i, params.H, params.W};

        BufferT filters(params.C_i*params.k*params.k*params.C_o);
        BufferT bn_weight(params.C_o);
        BufferT bn_bias(params.C_o);
        BufferT bn_running_mean(params.C_o);
        BufferT bn_running_variance(params.C_o);

        small::Conv2DLayer conv2d(input_shape,
                                  params.k, params.k,
                                  params.s, params.p,
                                  params.C_o,
                                  filters,
                                  bn_weight,
                                  bn_bias,
                                  bn_running_mean,
                                  bn_running_variance,
                                  0.f,
                                  false);

        TEST_ASSERT(conv2d.get_effective_output_channels() == params.C_o);
        if (params.C_o % BufferT::C_ob == 0)
        {
            TEST_ASSERT(conv2d.output_shape()[small::CHANNEL] == params.C_o);
        }
        else
        {
            TEST_ASSERT(conv2d.output_shape()[small::CHANNEL] ==
                        (params.C_o + (BufferT::C_ob - params.C_o % BufferT::C_ob)));
        }
    }
    catch (std::invalid_argument &e_obj)
    {
        std::cerr << "EXCEPTION: " << e_obj.what() << std::endl;
        TEST_CHECK(false);
    }

    // Test extra channel values with random filter and bias
    try
    {
        small::shape_type input_shape{1U, params.C_i, params.H, params.W};

        BufferT filters(params.C_i*params.k*params.k*params.C_o);
        small::init(filters, filters.size());
        BufferT bias(params.C_o);
        small::init(bias, bias.size());

        small::Conv2DLayer conv2d(input_shape,
                                  params.k, params.k,
                                  params.s, params.p,
                                  params.C_o,
                                  filters,
                                  bias,
                                  false);

        small::Tensor<BufferT>  input(input_shape);
        small::Tensor<BufferT> output(conv2d.output_size());

        conv2d.compute_output({&input}, &output);
        for (size_t co = params.C_o;
             co < conv2d.output_shape()[small::CHANNEL]; ++co)
        {
            for (size_t h = 0; h < conv2d.output_shape()[small::HEIGHT]; ++h)
            {
                for (size_t w = 0; w < conv2d.output_shape()[small::WIDTH]; ++w)
                {
                    size_t packed_index =
                        small::packed_buffer_index(
                            conv2d.output_shape()[small::CHANNEL],
                            conv2d.output_shape()[small::HEIGHT],
                            conv2d.output_shape()[small::WIDTH],
                            BufferT::C_ob,
                            co, h, w);
                    TEST_ASSERT(0.f == output.buffer()[packed_index]);
                }
            }
        }
    }
    catch (std::invalid_argument &e_obj)
    {
        std::cerr << "Unexpected exception caught: " << e_obj.what() << std::endl;
        TEST_CHECK(false);
    }
}

//****************************************************************************
void test_conv2d_bias(void)
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
    size_t input_size = params.C_i*params.H*params.W;

    small::Conv2DLayer<BufferT> conv2d_layer(input_shape,
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

            std::cout << "FAIL: Conv2D_out(" << ix << ")-->"
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
void test_conv2d_batchnorm_identity(void)
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
    std::cout << "Conv2D: filter file= " << filter_fname << std::endl;

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

    small::Conv2DLayer<BufferT> conv2d_layer(input_shape,
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

            std::cout << "FAIL: Conv2D_out(" << ix << ")-->"
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
void test_conv2d_batchnorm_bias_1(void)
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
    std::cout << "Conv2D: filter file= " << filter_fname << std::endl;

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

    small::Conv2DLayer<BufferT> conv2d_layer(input_shape,
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

            std::cout << "FAIL: Conv2D_out(" << ix << ")-->"
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
void test_conv2d_batchnorm_mean_1(void)
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
    std::cout << "Conv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================
    BufferT bn_weight(params.C_o);
    BufferT bn_bias(params.C_o);
    BufferT bn_running_mean(params.C_o);
    BufferT bn_running_variance(params.C_o);
    float   bn_eps = 0.f;
    float   running_mean = 10.f;
    for (size_t ix = 0; ix < params.C_o; ++ix)
    {
        bn_weight[ix] = 1;
        bn_bias[ix] = 0;
        bn_running_mean[ix] = running_mean;
        bn_running_variance[ix] = 1;
    }

    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;

    small::Conv2DLayer<BufferT> conv2d_layer(input_shape,
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

            std::cout << "FAIL: Conv2D_out(" << ix << ")-->"
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
void test_conv2d_batchnorm_mean_variance_1(void)
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
    std::cout << "Conv2D: filter file= " << filter_fname << std::endl;

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

    small::Conv2DLayer<BufferT> conv2d_layer(input_shape,
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

    //=========================================================================
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

//*****************************************************************************
void test_conv2d_batchnorm(void) {

    using BufferT = small::FloatBuffer;

    // run a hardcoded test for batchnorm against pytorch
    // this comes from the yolo model
    // [convolutional]
    // batch_normalize=1
    // filters=16
    // size=3
    // stride=1
    // pad=1
    // activation=leaky

    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params = {3, 416, 416, 3, 1, small::PADDING_F, 16};

    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});

    // we are using PADDING:F so output H and W are the same as input
    small::shape_type output_shape({1UL, params.C_o, params.H, params.W});

    size_t input_size = params.C_i*params.H*params.W;
    size_t output_buffer_size = params.C_o*params.H*params.W;
    size_t batch_norm_size = params.C_o*4;
    size_t filter_size = params.C_o*params.C_i*params.k*params.k;

    bool passing = true;

    std::string in_fname =
        get_pathname(data_dir, "in", "conv2d",
                     params,
                     input_size);
    std::cout << "\nConv2D: input file = " << in_fname << std::endl;

    BufferT input(read_inputs<BufferT>(in_fname));
    BufferT input_dc(input.size());
    small::pack_buffer(input, small::INPUT,
                       1U, params.C_i, params.H, params.W,
                       BufferT::C_ib, BufferT::C_ob,
                       input_dc);
    small::Tensor<BufferT> input_tensor(input_shape, input_dc);

    std::string filter_data = data_dir + "/filter__conv2d_bn_Ci3_Co16_H416_W416_k3_s0_f_432.bin2";
    std::cout << "Conv2D: filter file= " << filter_data << std::endl;

    std::string bn_fname =
        get_pathname(data_dir, "bn", "conv2d",
                     params,
                     batch_norm_size);
    BufferT batch_norm_data(read_inputs<BufferT>(bn_fname));

    BufferT bn_bias(params.C_o);
    std::copy(&batch_norm_data[0], &batch_norm_data[params.C_o], &bn_bias[0]);

    BufferT bn_weight(params.C_o);
    std::copy(&batch_norm_data[params.C_o], &batch_norm_data[params.C_o*2], &bn_weight[0]);

    BufferT bn_running_mean(params.C_o);
    std::copy(&batch_norm_data[params.C_o*2], &batch_norm_data[params.C_o*3], &bn_running_mean[0]);

    BufferT bn_running_variance(params.C_o);
    std::copy(&batch_norm_data[params.C_o*3], &batch_norm_data[params.C_o*4], &bn_running_variance[0]);


    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     filter_size);
    BufferT filter(read_inputs<BufferT>(filter_fname));

    small::ActivationType activation = small::ActivationType::LEAKY;
    small::Conv2DLayer<BufferT> conv(
        input_shape,
        params.k, params.k,
        params.s, params.p,
        params.C_o,
        filter,
        bn_weight,
        bn_bias,
        bn_running_mean,
        bn_running_variance,
        1.e-5,
        false,
        activation,
        0.1 // leaky slope from pytorch yolov3 code
    );

    small::Tensor<BufferT> output_tensor_ans(output_shape);
    conv.compute_output({&input_tensor}, &output_tensor_ans);

    small::Tensor<BufferT> output_tensor_ans_unpacked(output_shape);
    small::unpack_buffer(output_tensor_ans.buffer(), small::OUTPUT,
                       1U, params.C_o, params.H, params.W,
                       BufferT::C_ib, BufferT::C_ob,
                       output_tensor_ans_unpacked.buffer());


    std::string out_fname =
        get_pathname(data_dir, "out", "conv2d",
                     params,
                     output_buffer_size);
    std::cout << "Conv2D: output file= " << out_fname << std::endl;

    BufferT output(read_inputs<BufferT>(out_fname));
    small::Tensor<BufferT> output_tensor_ref(output_shape, output);

    // compare output_tensor_ans to output_tensor_ref
    /// @todo revisit accuracy
    size_t fail_cnt = 0;
    for (size_t i = 0; i < output_tensor_ref.size(); ++i) {
        if (!almost_equal(output_tensor_ref.buffer()[i], output_tensor_ans_unpacked.buffer()[i], 5e-03, 1e-04)) {
            fail_cnt++;
            passing = false;
            if(fail_cnt < 10) {
                std::cout << "FAIL: Conv2D_out(" << i << ")-->"
                        << std::setw(12) << std::setprecision(10)
                        << output_tensor_ans_unpacked.buffer()[i] << "(computed) != "
                        << std::setw(12) << std::setprecision(10)
                        << output_tensor_ref.buffer()[i]
                        << std::endl;
            }
        }
    }

    TEST_CHECK(passing);

}


//****************************************************************************
template <class BufferT>
bool run_conv2d_config(LayerParams const &params)
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
template <class BufferT>
bool run_conv2d_layer_config(LayerParams const &params)
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
    small::Conv2DLayer<BufferT> conv2d_layer(input_shape,
                                             params.k, params.k,
                                             params.s, params.p,
                                             params.C_o,
                                             filter_dc, false);
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
    auto output_shape(conv2d_layer.output_shape());
    size_t output_buffer_size(conv2d_layer.output_size());

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

            std::cout << "FAIL: Conv2D_out(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << buf[ix] << "(computed) != "
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
void test_conv2d_regression_data(void)
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
        TEST_CHECK(true == run_conv2d_config<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(true == run_conv2d_config<small::FloatBuffer>(p));
#endif
    }
}

//****************************************************************************
void test_conv2d_layer_regression_data(void)
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
        TEST_CHECK(true == run_conv2d_layer_config<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(true == run_conv2d_layer_config<small::FloatBuffer>(p));
#endif
    }
}

//****************************************************************************
void measure_conv2d_performance(void)
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

    printf("\nConv2D(%s) func.\n", type.c_str());
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

    printf("\nConv2D(%s) class\n", type.c_str());
    printf("\tC_i\tH\tW\tk\ts\tC_o\tnthd\truns\tt_min\tt_max\tt_avg\n");

    for (LayerParams const &p: params)
    {
        uint32_t Ho(small::compute_output_dim(p.H, p.k, p.s, p.p));
        uint32_t Wo(small::compute_output_dim(p.W, p.k, p.s, p.p));

        uint8_t t_pad=0, b_pad=0, l_pad=0, r_pad=0;
        if (p.p == small::PADDING_F)
        {
            small::calc_padding(p.H, p.k, p.s, t_pad, b_pad);
            small::calc_padding(p.W, p.k, p.s, l_pad, r_pad);
        }

        size_t num_input_elts(p.C_i*p.H*p.W);
        size_t num_filter_elts(p.C_i*p.k*p.k*p.C_o);
        size_t num_output_elts(p.C_o*Ho*Wo);
        small::shape_type input_shape({1UL, p.C_i, p.H, p.W});

        Buffer filter_dc(num_filter_elts);
        small::init(filter_dc, num_filter_elts);

        small::Tensor<Buffer> input_dc(input_shape);
        small::init(input_dc.buffer(), num_input_elts);

        small::Tensor<Buffer> output_dc(num_output_elts);

        small::Conv2DLayer<Buffer>
            conv2d_layer(input_shape, p.k, p.k, p.s, p.p,
                         p.C_o, filter_dc, true);

        for (size_t ix = 0; ix < 3; ++ix)
        {
            setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);
            //std::string ont = std::getenv("OMP_NUM_THREADS");
            //auto nt = atol(ont.c_str());

            double tx(0.);
            double min_t = std::numeric_limits<double>::max();
            double max_t = 0.;

            // Warm up
            conv2d_layer.compute_output({&input_dc}, &output_dc);

            for (size_t iy = 0; iy < num_runs; ++iy)
            {
                t.start();
                conv2d_layer.compute_output({&input_dc}, &output_dc);
                t.stop();
                double ts = t.elapsed();
                tx += ts;
                min_t = std::min(min_t, ts);
                max_t = std::max(max_t, ts);
            }

            printf("class   \t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%lf\t%lf\t%lf\n",
                   p.C_i, p.H, p.W, p.k, p.s, p.C_o,
                   num_threads[ix], num_runs,
                   min_t, max_t, (tx/num_runs));
        }
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"conv2d_layer_odd_output_channels",   test_conv2d_layer_odd_output_channels},
    {"conv2d_bias",                  test_conv2d_bias},
    {"conv2d_batchnorm_identity",    test_conv2d_batchnorm_identity},
    {"conv2d_batchnorm_bias_1",      test_conv2d_batchnorm_bias_1},
    {"conv2d_batchnorm_mean_1",      test_conv2d_batchnorm_mean_1},
    {"conv2d_batchnorm_mean_variance_1", test_conv2d_batchnorm_mean_variance_1},
    {"conv2d_batchnorm", test_conv2d_batchnorm},
    {"conv2d_regression_data",       test_conv2d_regression_data},
    {"conv2d_layer_regression_data", test_conv2d_layer_regression_data},
    // {"conv2d_performance", measure_conv2 d_performance},
    {NULL, NULL}
};
