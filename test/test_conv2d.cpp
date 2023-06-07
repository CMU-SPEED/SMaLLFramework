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

//#define DEBUG_LAYERS
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
void test_filter_packing_indices(void)
{
    bool passed = true;

    // uint32_t C_i = 16; //96;
    // uint32_t C_o = 1; //96;
    // uint32_t H = 2; //3;
    // uint32_t W = 2;// 3;

    // C_i,Hi,Wi,k,s,p,C_o
    std::vector<LayerParams> layer_params =
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
        { 512,   13,  13, 3, 1, small::PADDING_F, 1024},

        /// @todo fix for non reference platforms or disable
//        { 3,   13,  13, 3, 1, small::PADDING_F, 16},
//        { 1,   52,  52, 3, 1, small::PADDING_F, 16}
    };

    std::vector<size_t> unpacked_to_packed_mapping;
    for (auto &[C_i, H, W, k, s, p, C_o] : layer_params)
    {
        //std::cerr << "Testing indexing for Ci,H,W,C_o = "
        //          << C_i << "," << H << "," << W << "," << C_o
        //          << std::endl;
        size_t sz = C_i*H*W*C_o;

        unpacked_to_packed_mapping.clear();
        for (uint32_t co = 0; co < C_o; ++co)
            for (uint32_t ci = 0; ci < C_i; ++ci)
                for (uint32_t h = 0; h < H; ++h)
                    for (uint32_t w = 0; w < W; ++w)
                    {
                        size_t packed_index =
                            small::packed_weight_index(C_o, C_i, H, W,
                                                       C_ob, C_ib,
                                                       co, ci, h, w);
                        if (packed_index > sz)
                        {
                            std::cerr << "ERROR: out of range: unpacked("
                                      << unpacked_to_packed_mapping.size()
                                      << "),  packed(macro) "
                                      << packed_index << " >= " << sz
                                      << std::endl;
                        }
                        unpacked_to_packed_mapping.push_back(packed_index);
                    }

        // *** Extracted from convert_tensor2dc ***
        uint32_t _C_ib = C_ib;
        uint32_t _C_ob = C_ob;

        if (C_i < _C_ib) //(dim1 < _C_ob)
        {
            //std::cerr << "HERE: dim1, C_ob: " << H << ", " << _C_ob << std::endl;;
            _C_ib = 3;    /// @todo why is this a 3?
        }

        uint32_t ip_block = _C_ib;
        uint32_t op_block = _C_ob;

        uint32_t offset = 0;
        for (uint32_t g = 0; g < C_o; g += op_block)
        {
            uint32_t g_offset = g * C_i * H * W;
            for (uint32_t h = 0; h < C_i; h += ip_block)
            {
                uint32_t h_offset = h * H * W;
                for (uint32_t i = 0; i < H; i++)
                {
                    uint32_t i_offset = i * W;
                    for (uint32_t j = 0; j < W; j++)
                    {
                        uint32_t j_offset = j;
                        for (uint32_t k = 0; k < ip_block; k++)
                        {
                            uint32_t k_offset = k * H * W;
                            for (uint32_t l = 0; l < op_block; l++)
                            {
                                int l_offset = l * C_i * H * W;
                                //printf("offset: %d\n", offset);fflush(0);
                                //std::cerr << "dst index = " << offset << ", src index = "
                                //          << (g_offset + l_offset +
                                //              h_offset + k_offset +
                                //              i_offset +
                                //              j_offset)
                                //          << std::endl;
                                auto idx = small::packed_weight_index(
                                    C_o, C_i, H, W,
                                    C_ob, C_ib,
                                    g+l, h+k, i, j);

                                size_t unpacked_index = g_offset + l_offset +
                                    h_offset + k_offset +
                                    i_offset + j_offset;
                                size_t packed_index = offset++;

                                if (packed_index != idx)
                                {
                                    passed = false;
                                    std::cerr << "ERROR: packed(macro) != packed(offset), "
                                              << idx << " != " << packed_index
                                              << std::endl;
                                    continue;
                                }

                                if (unpacked_index >= unpacked_to_packed_mapping.size())
                                {
                                    passed = false;
                                    std::cerr << "ERROR: Unpacked index = " << unpacked_index
                                              << " out of bounds (size = "
                                              << unpacked_to_packed_mapping.size()
                                              << ")\n";
                                    continue;
                                }

                                // std::cerr << "unpacked: " << unpacked_index
                                //           << ": packed(macro): "
                                //           << unpacked_to_packed_mapping[unpacked_index]
                                //           << " ?= packed(t2dc): "
                                //           << packed_index
                                //           << " ?= direct(macro): "
                                //           << idx << std::endl;

                                if (unpacked_to_packed_mapping[unpacked_index] !=
                                    packed_index)
                                {
                                    passed = false;
                                    std::cerr << "ERROR: unpacked: " << unpacked_index
                                              << ": packed(macro): "
                                              << unpacked_to_packed_mapping[unpacked_index]
                                              << " ?= packed(t2dc): "
                                              << packed_index << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    TEST_ASSERT(passed);
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
    std::cout << "Conv2D: filter file= " << filter_fname << std::endl;

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

    small::shape_type output_shape(conv2d_layer.output_shape(0));
    size_t output_buffer_size(conv2d_layer.output_size(0));

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
                       C_ib, C_ob,
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
                       C_ib, C_ob,
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
    conv2d_layer.compute_output({&packed_input_tensor}, {&packed_output_tensor});
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size(0));

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
                                             bn_running_mean, bn_running_variance,
                                             bn_eps,
                                             false);

    small::shape_type output_shape(conv2d_layer.output_shape(0));
    size_t output_buffer_size(conv2d_layer.output_size(0));

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
                       C_ib, C_ob,
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
                       C_ib, C_ob,
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
    conv2d_layer.compute_output({&packed_input_tensor}, {&packed_output_tensor});
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size(0));

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
                                             bn_running_mean, bn_running_variance,
                                             bn_eps,
                                             false);

    small::shape_type output_shape(conv2d_layer.output_shape(0));
    size_t output_buffer_size(conv2d_layer.output_size(0));

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
                       C_ib, C_ob,
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
                       C_ib, C_ob,
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
    conv2d_layer.compute_output({&packed_input_tensor}, {&packed_output_tensor});
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size(0));

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

    small::Conv2DLayer<BufferT> conv2d_layer(input_shape,
                                             params.k, params.k,
                                             params.s, params.p,
                                             params.C_o,
                                             filter_dc,
                                             bn_weight, bn_bias,
                                             bn_running_mean, bn_running_variance,
                                             bn_eps,
                                             false);

    small::shape_type output_shape(conv2d_layer.output_shape(0));
    size_t output_buffer_size(conv2d_layer.output_size(0));

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
                       C_ib, C_ob,
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
                       C_ib, C_ob,
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
    conv2d_layer.compute_output({&packed_input_tensor}, {&packed_output_tensor});
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size(0));

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
                       C_ib, C_ob,
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
                       C_ib, C_ob,
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

    // Compute layer
    small::Conv2D(params.k, params.s,
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
                       C_ib, C_ob,
                       packed_input_dc);

    small::Tensor<BufferT> packed_input_tensor(
        input_shape,
        std::move(packed_input_dc));

    // Read output regression data
    auto output_shape(conv2d_layer.output_shape(0));
    size_t output_buffer_size(conv2d_layer.output_size(0));

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
                       C_ib, C_ob,
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
    conv2d_layer.compute_output({&packed_input_tensor}, {&packed_output_tensor});
    TEST_ASSERT(packed_output_tensor.size() == conv2d_layer.output_size(0));

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
            small::Conv2D(p.k, p.s, t_pad, b_pad, l_pad, r_pad,
                          p.C_o, p.C_i, p.H, p.W,
                          input_dc, filter_dc, output_dc);

            for (size_t iy = 0; iy < num_runs; ++iy)
            {
                t.start();
                small::Conv2D(p.k, p.s, t_pad, b_pad, l_pad, r_pad,
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
            conv2d_layer.compute_output({&input_dc}, {&output_dc});

            for (size_t iy = 0; iy < num_runs; ++iy)
            {
                t.start();
                conv2d_layer.compute_output({&input_dc}, {&output_dc});
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
    {"conv2d_filter_packing",        test_filter_packing_indices},
    {"conv2d_bias",                  test_conv2d_bias},
    {"conv2d_batchnorm_identity",    test_conv2d_batchnorm_identity},
    {"conv2d_batchnorm_bias_1",      test_conv2d_batchnorm_bias_1},
    {"conv2d_batchnorm_mean_1",      test_conv2d_batchnorm_mean_1},
    {"conv2d_regression_data",       test_conv2d_regression_data},
    {"conv2d_layer_regression_data", test_conv2d_layer_regression_data},
    {"conv2d_performance",           measure_conv2d_performance},
    {NULL, NULL}
};
