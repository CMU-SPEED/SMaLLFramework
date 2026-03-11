//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2026 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//****************************************************************************

#define PARALLEL 1

#include <acutest.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include <small.h>

#include "test_utils.hpp"

namespace
{

using BufferT = small::FloatBuffer;

struct FusedBlockParams
{
    uint32_t C_i;
    uint32_t H;
    uint32_t W;
    uint32_t conv_k;
    uint32_t conv_s;
    small::PaddingEnum conv_p;
    uint32_t C_o;
    uint32_t block_k;
    uint32_t block_s;
    small::PaddingEnum block_p;
};

enum class ActivationMode
{
    none,
    bias_only,
    relu,
    bias_relu
};

void calc_padding(uint32_t H, uint32_t W, uint32_t k, uint32_t s,
                  small::PaddingEnum p,
                  uint8_t &t_pad, uint8_t &b_pad,
                  uint8_t &l_pad, uint8_t &r_pad)
{
    t_pad = b_pad = l_pad = r_pad = 0;
    if (p == small::PADDING_F)
    {
        small::calc_padding(H, k, s, t_pad, b_pad);
        small::calc_padding(W, k, s, l_pad, r_pad);
    }
}

bool check_buffers_close(BufferT const &expected,
                         BufferT const &actual,
                         char const *label)
{
    TEST_ASSERT(expected.size() == actual.size());

    bool passing = true;
    for (size_t ix = 0; ix < expected.size(); ++ix)
    {
        if ((actual[ix] != expected[ix]) &&
            !almost_equal(actual[ix], expected[ix]))
        {
            passing = false;
            std::cout << "FAIL: " << label << "(" << ix << ")-->"
                      << std::setw(12) << std::setprecision(10)
                      << actual[ix] << "(computed) != "
                      << std::setw(12) << std::setprecision(10)
                      << expected[ix]
                      << std::endl;
        }
    }
    return passing;
}

bool run_conv_dw_fused_config(FusedBlockParams const &params,
                              ActivationMode mode)
{
    size_t input_size = params.C_i * params.H * params.W;
    BufferT input_dc(input_size);
    small::init(input_dc, input_size);

    size_t conv_filter_size =
        params.C_o * params.C_i * params.conv_k * params.conv_k;
    BufferT conv_filter_dc(conv_filter_size);
    small::init(conv_filter_dc, conv_filter_size);

    size_t dw_filter_size = params.C_o * params.block_k * params.block_k;
    BufferT dw_filter_dc(dw_filter_size);
    small::init(dw_filter_dc, dw_filter_size);

    BufferT conv_bias(params.C_o);
    BufferT dw_bias(params.C_o);
    small::init(conv_bias, params.C_o);
    small::init(dw_bias, params.C_o);

    uint8_t conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad;
    calc_padding(params.H, params.W,
                 params.conv_k, params.conv_s, params.conv_p,
                 conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad);

    size_t conv_H = small::compute_output_dim(
        params.H, params.conv_k, params.conv_s, params.conv_p);
    size_t conv_W = small::compute_output_dim(
        params.W, params.conv_k, params.conv_s, params.conv_p);

    uint8_t dw_t_pad, dw_b_pad, dw_l_pad, dw_r_pad;
    calc_padding(conv_H, conv_W,
                 params.block_k, params.block_s, params.block_p,
                 dw_t_pad, dw_b_pad, dw_l_pad, dw_r_pad);

    size_t output_H = small::compute_output_dim(
        conv_H, params.block_k, params.block_s, params.block_p);
    size_t output_W = small::compute_output_dim(
        conv_W, params.block_k, params.block_s, params.block_p);

    size_t intermediate_size = params.C_o * conv_H * conv_W;
    size_t output_size = params.C_o * output_H * output_W;

    BufferT reference_intermediate(intermediate_size);
    BufferT reference_output(output_size);
    BufferT fused_intermediate(intermediate_size);
    BufferT fused_output(output_size);
    std::memset(reference_intermediate.data(), 0, intermediate_size * sizeof(BufferT::value_type));
    std::memset(reference_output.data(), 0, output_size * sizeof(BufferT::value_type));
    std::memset(fused_intermediate.data(), 0, intermediate_size * sizeof(BufferT::value_type));
    std::memset(fused_output.data(), 0, output_size * sizeof(BufferT::value_type));

    switch (mode)
    {
    case ActivationMode::none:
        small::Conv2D(params.conv_k, params.conv_k, params.conv_s,
                      conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
                      params.C_o, params.C_i, params.H, params.W,
                      input_dc, conv_filter_dc,
                      reference_intermediate);
        small::DepthwiseConv2D(params.block_k, params.block_k, params.block_s,
                               dw_t_pad, dw_b_pad, dw_l_pad, dw_r_pad,
                               params.C_o, conv_H, conv_W,
                               reference_intermediate, dw_filter_dc,
                               reference_output);
        small::Conv2D_DepthwiseConv2D(params.conv_k, params.conv_k, params.conv_s,
                                      conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
                                      params.block_k, params.block_k, params.block_s,
                                      dw_t_pad, dw_b_pad, dw_l_pad, dw_r_pad,
                                      params.C_o, params.C_i, params.H, params.W,
                                      input_dc, conv_filter_dc,
                                      fused_intermediate, dw_filter_dc,
                                      fused_output);
        break;

    case ActivationMode::bias_only:
        small::Bias(params.C_o, conv_H, conv_W, conv_bias, reference_intermediate);
        small::PartialConv2D(params.conv_k, params.conv_k, params.conv_s,
                             conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
                             params.C_o, params.C_i, params.H, params.W,
                             input_dc, conv_filter_dc,
                             reference_intermediate);
        small::Bias(params.C_o, output_H, output_W, dw_bias, reference_output);
        small::PartialDepthwiseConv2D(
            params.block_k, params.block_k, params.block_s,
            dw_t_pad, dw_b_pad, dw_l_pad, dw_r_pad,
            params.C_o, conv_H, conv_W,
            reference_intermediate, dw_filter_dc,
            reference_output);
        small::Conv2D_Bias_DepthwiseConv2D_Bias(
            params.conv_k, params.conv_k, params.conv_s,
            conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
            params.block_k, params.block_k, params.block_s,
            dw_t_pad, dw_b_pad, dw_l_pad, dw_r_pad,
            params.C_o, params.C_i, params.H, params.W,
            input_dc, conv_filter_dc, conv_bias,
            fused_intermediate, dw_filter_dc, dw_bias,
            fused_output);
        break;

    case ActivationMode::relu:
        small::Conv2D_ReLU(params.conv_k, params.conv_k, params.conv_s,
                           conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
                           params.C_o, params.C_i, params.H, params.W,
                           input_dc, conv_filter_dc,
                           reference_intermediate);
        small::DepthwiseConv2D_ReLU(params.block_k, params.block_k, params.block_s,
                                    dw_t_pad, dw_b_pad, dw_l_pad, dw_r_pad,
                                    params.C_o, conv_H, conv_W,
                                    reference_intermediate, dw_filter_dc,
                                    reference_output);
        small::Conv2D_ReLU_DepthwiseConv2D_ReLU(
            params.conv_k, params.conv_k, params.conv_s,
            conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
            params.block_k, params.block_k, params.block_s,
            dw_t_pad, dw_b_pad, dw_l_pad, dw_r_pad,
            params.C_o, params.C_i, params.H, params.W,
            input_dc, conv_filter_dc,
            fused_intermediate, dw_filter_dc, fused_output);
        break;

    case ActivationMode::bias_relu:
        small::Conv2D_Bias_ReLU(params.conv_k, params.conv_k, params.conv_s,
                                conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
                                params.C_o, params.C_i, params.H, params.W,
                                input_dc, conv_filter_dc,
                                conv_bias, reference_intermediate);
        small::DepthwiseConv2D_Bias_ReLU(
            params.block_k, params.block_k, params.block_s,
            dw_t_pad, dw_b_pad, dw_l_pad, dw_r_pad,
            params.C_o, conv_H, conv_W,
            reference_intermediate, dw_filter_dc, dw_bias,
            reference_output);
        small::Conv2D_Bias_ReLU_DepthwiseConv2D_Bias_ReLU(
            params.conv_k, params.conv_k, params.conv_s,
            conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
            params.block_k, params.block_k, params.block_s,
            dw_t_pad, dw_b_pad, dw_l_pad, dw_r_pad,
            params.C_o, params.C_i, params.H, params.W,
            input_dc, conv_filter_dc, conv_bias,
            fused_intermediate, dw_filter_dc, dw_bias,
            fused_output);
        break;
    }

    return check_buffers_close(reference_output, fused_output, "ConvDW_out");
}

bool run_conv_maxpool_fused_config(FusedBlockParams const &params,
                                   ActivationMode mode)
{
    size_t input_size = params.C_i * params.H * params.W;
    BufferT input_dc(input_size);
    small::init(input_dc, input_size);

    size_t conv_filter_size =
        params.C_o * params.C_i * params.conv_k * params.conv_k;
    BufferT conv_filter_dc(conv_filter_size);
    small::init(conv_filter_dc, conv_filter_size);

    BufferT conv_bias(params.C_o);
    small::init(conv_bias, params.C_o);

    uint8_t conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad;
    calc_padding(params.H, params.W,
                 params.conv_k, params.conv_s, params.conv_p,
                 conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad);

    size_t conv_H = small::compute_output_dim(
        params.H, params.conv_k, params.conv_s, params.conv_p);
    size_t conv_W = small::compute_output_dim(
        params.W, params.conv_k, params.conv_s, params.conv_p);

    uint8_t pool_t_pad, pool_b_pad, pool_l_pad, pool_r_pad;
    calc_padding(conv_H, conv_W,
                 params.block_k, params.block_s, params.block_p,
                 pool_t_pad, pool_b_pad, pool_l_pad, pool_r_pad);

    size_t output_H = small::compute_output_dim(
        conv_H, params.block_k, params.block_s, params.block_p);
    size_t output_W = small::compute_output_dim(
        conv_W, params.block_k, params.block_s, params.block_p);

    size_t intermediate_size = params.C_o * conv_H * conv_W;
    size_t output_size = params.C_o * output_H * output_W;

    BufferT reference_intermediate(intermediate_size);
    BufferT reference_output(output_size);
    BufferT fused_intermediate(intermediate_size);
    BufferT fused_output(output_size);
    std::memset(reference_intermediate.data(), 0, intermediate_size * sizeof(BufferT::value_type));
    std::memset(reference_output.data(), 0, output_size * sizeof(BufferT::value_type));
    std::memset(fused_intermediate.data(), 0, intermediate_size * sizeof(BufferT::value_type));
    std::memset(fused_output.data(), 0, output_size * sizeof(BufferT::value_type));

    switch (mode)
    {
    case ActivationMode::none:
        small::Conv2D(params.conv_k, params.conv_k, params.conv_s,
                      conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
                      params.C_o, params.C_i, params.H, params.W,
                      input_dc, conv_filter_dc,
                      reference_intermediate);
        small::MaxPool2D(params.block_k, params.block_k, params.block_s,
                         pool_t_pad, pool_b_pad, pool_l_pad, pool_r_pad,
                         params.C_o, conv_H, conv_W,
                         reference_intermediate, reference_output);
        small::Conv2D_Maxpool2D(params.conv_k, params.conv_k, params.conv_s,
                                conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
                                params.block_k, params.block_k, params.block_s,
                                pool_t_pad, pool_b_pad, pool_l_pad, pool_r_pad,
                                params.C_o, params.C_i, params.H, params.W,
                                input_dc, conv_filter_dc,
                                fused_intermediate, fused_output);
        break;

    case ActivationMode::bias_only:
        TEST_ASSERT(false &&
                    "conv+maxpool fused bias-only is not implemented in the public API");
        break;

    case ActivationMode::relu:
        small::Conv2D_ReLU(params.conv_k, params.conv_k, params.conv_s,
                           conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
                           params.C_o, params.C_i, params.H, params.W,
                           input_dc, conv_filter_dc,
                           reference_intermediate);
        small::MaxPool2D(params.block_k, params.block_k, params.block_s,
                         pool_t_pad, pool_b_pad, pool_l_pad, pool_r_pad,
                         params.C_o, conv_H, conv_W,
                         reference_intermediate, reference_output);
        small::Conv2D_ReLU_Maxpool2D(params.conv_k, params.conv_k, params.conv_s,
                                     conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
                                     params.block_k, params.block_k, params.block_s,
                                     pool_t_pad, pool_b_pad, pool_l_pad, pool_r_pad,
                                     params.C_o, params.C_i, params.H, params.W,
                                     input_dc, conv_filter_dc,
                                     fused_intermediate, fused_output);
        break;

    case ActivationMode::bias_relu:
        small::Conv2D_Bias_ReLU(params.conv_k, params.conv_k, params.conv_s,
                                conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
                                params.C_o, params.C_i, params.H, params.W,
                                input_dc, conv_filter_dc,
                                conv_bias, reference_intermediate);
        small::MaxPool2D(params.block_k, params.block_k, params.block_s,
                         pool_t_pad, pool_b_pad, pool_l_pad, pool_r_pad,
                         params.C_o, conv_H, conv_W,
                         reference_intermediate, reference_output);
        small::Conv2D_Bias_ReLU_Maxpool2D(
            params.conv_k, params.conv_k, params.conv_s,
            conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
            params.block_k, params.block_k, params.block_s,
            pool_t_pad, pool_b_pad, pool_l_pad, pool_r_pad,
            params.C_o, params.C_i, params.H, params.W,
            input_dc, conv_filter_dc, conv_bias,
            fused_intermediate, fused_output);
        break;
    }

    return check_buffers_close(reference_output, fused_output, "ConvPool_out");
}

std::vector<FusedBlockParams> const kBlockConfigs = {
    //smallest valid output size
    {16, 3, 3, 3, 1, small::PADDING_F, 16, 3, 1, small::PADDING_V},
    // uses all kernels with and without padding
    {16, 12, 12, 3, 1, small::PADDING_V, 16, 3, 2, small::PADDING_V},
    {16, 12, 12, 3, 1, small::PADDING_F, 16, 3, 2, small::PADDING_F},
    {16, 12, 12, 3, 2, small::PADDING_F, 16, 3, 1, small::PADDING_F},
    {16, 12, 12, 3, 2, small::PADDING_F, 16, 3, 2, small::PADDING_F},
    // more than 1 block of input channels (tests peeling of the final input channel block)
    {96, 12, 12, 3, 1, small::PADDING_F, 16, 3, 2, small::PADDING_F},
    //multiple output channel blocks
    {16, 12, 12, 3, 1, small::PADDING_F, 96, 3, 2, small::PADDING_F},
    {96, 12, 12, 3, 1, small::PADDING_F, 96, 3, 2, small::PADDING_F},
    // //larger problem sizes
    {96, 30, 30, 3, 1, small::PADDING_F, 96, 3, 1, small::PADDING_F},
};

void test_conv_dw_fused_plain(void)
{
    for (auto const &params : kBlockConfigs)
    {
        TEST_CHECK(run_conv_dw_fused_config(params, ActivationMode::none));
    }
}

void test_conv_dw_fused_relu(void)
{
    for (auto const &params : kBlockConfigs)
    {
        TEST_CHECK(run_conv_dw_fused_config(params, ActivationMode::relu));
    }
}

void test_conv_dw_fused_bias_relu(void)
{
    for (auto const &params : kBlockConfigs)
    {
        TEST_CHECK(run_conv_dw_fused_config(params, ActivationMode::bias_relu));
    }
}

void test_conv_maxpool_fused_plain(void)
{
    for (auto const &params : kBlockConfigs)
    {
        TEST_CHECK(run_conv_maxpool_fused_config(params, ActivationMode::none));
    }
}

void test_conv_maxpool_fused_relu(void)
{
    for (auto const &params : kBlockConfigs)
    {
        TEST_CHECK(run_conv_maxpool_fused_config(params, ActivationMode::relu));
    }
}

void test_conv_maxpool_fused_bias_relu(void)
{
    for (auto const &params : kBlockConfigs)
    {
        TEST_CHECK(run_conv_maxpool_fused_config(params, ActivationMode::bias_relu));
    }
}

void test_conv_dw_fused_only_all_ones_sanity(void)
{
    FusedBlockParams const params = {
        16, 3, 3, 3, 1, small::PADDING_F, 16, 3, 1, small::PADDING_V};

    uint8_t conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad;
    calc_padding(params.H, params.W,
                 params.conv_k, params.conv_s, params.conv_p,
                 conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad);

    size_t conv_H = small::compute_output_dim(
        params.H, params.conv_k, params.conv_s, params.conv_p);
    size_t conv_W = small::compute_output_dim(
        params.W, params.conv_k, params.conv_s, params.conv_p);
    size_t output_H = small::compute_output_dim(
        conv_H, params.block_k, params.block_s, params.block_p);
    size_t output_W = small::compute_output_dim(
        conv_W, params.block_k, params.block_s, params.block_p);

    TEST_ASSERT(conv_H == 3);
    TEST_ASSERT(conv_W == 3);
    TEST_ASSERT(output_H == 1);
    TEST_ASSERT(output_W == 1);

    uint8_t dw_t_pad = 0, dw_b_pad = 0, dw_l_pad = 0, dw_r_pad = 0;

    size_t input_size = params.C_i * params.H * params.W;
    size_t conv_filter_size =
        params.C_o * params.C_i * params.conv_k * params.conv_k;
    size_t dw_filter_size = params.C_o * params.block_k * params.block_k;
    size_t intermediate_size = params.C_o * conv_H * conv_W;
    size_t output_size = params.C_o * output_H * output_W;

    BufferT input_dc(input_size);
    BufferT conv_filter_dc(conv_filter_size);
    BufferT dw_filter_dc(dw_filter_size);
    BufferT conv_bias(params.C_o);
    BufferT dw_bias(params.C_o);

    small::init_ones(input_dc, input_size);
    small::init_ones(conv_filter_dc, conv_filter_size);
    small::init_ones(dw_filter_dc, dw_filter_size);
    small::init_ones(conv_bias, params.C_o);
    small::init_ones(dw_bias, params.C_o);

    BufferT reference_intermediate(intermediate_size);
    BufferT reference_output(output_size);
    BufferT fused_intermediate(intermediate_size);
    BufferT fused_output(output_size);
    std::memset(reference_intermediate.data(), 0, intermediate_size * sizeof(BufferT::value_type));
    std::memset(reference_output.data(), 0, output_size * sizeof(BufferT::value_type));
    std::memset(fused_intermediate.data(), 0, intermediate_size * sizeof(BufferT::value_type));
    std::memset(fused_output.data(), 0, output_size * sizeof(BufferT::value_type));

    small::Conv2D(params.conv_k, params.conv_k, params.conv_s,
                         conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
                         params.C_o, params.C_i, params.H, params.W,
                         input_dc, conv_filter_dc,
                         reference_intermediate);
    small::DepthwiseConv2D(
        params.block_k, params.block_k, params.block_s,
        dw_t_pad, dw_b_pad, dw_l_pad, dw_r_pad,
        params.C_o, conv_H, conv_W,
        reference_intermediate, dw_filter_dc,
        reference_output);

    small::Conv2D_DepthwiseConv2D(
        params.conv_k, params.conv_k, params.conv_s,
        conv_t_pad, conv_b_pad, conv_l_pad, conv_r_pad,
        params.block_k, params.block_k, params.block_s,
        dw_t_pad, dw_b_pad, dw_l_pad, dw_r_pad,
        params.C_o, params.C_i, params.H, params.W,
        input_dc, conv_filter_dc, 
        fused_intermediate, dw_filter_dc,
        fused_output);

    TEST_CHECK(check_buffers_close(reference_output, fused_output, "ConvDW_ones"));

    float constexpr expected_scalar = 784.0f;
    for (size_t ix = 0; ix < reference_output.size(); ++ix)
    {
        TEST_CHECK(almost_equal(reference_output[ix], expected_scalar));
        TEST_CHECK(almost_equal(fused_output[ix], expected_scalar));
    }
}

} // namespace

TEST_LIST = {
    {"conv_dw_fused_only_all_ones_sanity",
     test_conv_dw_fused_only_all_ones_sanity},
    {"conv_dw_fused_plain", test_conv_dw_fused_plain},
    {"conv_maxpool_fused_plain", test_conv_maxpool_fused_plain},
    // {"conv_dw_fused_relu", test_conv_dw_fused_relu},
    // {"conv_dw_fused_bias_relu", test_conv_dw_fused_bias_relu},
    // {"conv_maxpool_fused_relu", test_conv_maxpool_fused_relu},
    // {"conv_maxpool_fused_bias_relu", test_conv_maxpool_fused_bias_relu},
    {nullptr, nullptr}
};
