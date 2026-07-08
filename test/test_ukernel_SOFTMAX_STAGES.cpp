//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2026 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//****************************************************************************

#define PARALLEL 1

#include <acutest.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>

#include <small.h>
#include <small/float_detail/abstract_op.hpp>
#include <small/op_type.hpp>

#include "test_performance_utils.hpp"

namespace small {
namespace float_detail {

#if !defined(QUANTIZED)

void compute_softmax_microkernel_stages(FloatBuffer &microkernel_buf,
                                        FloatBuffer &scalar_buf,
                                        dim_t ChannelBlocks,
                                        dim_t height,
                                        dim_t WidthBlocks)
{
    using ScalarT = typename FloatBuffer::value_type;

    dim_t const width = WidthBlocks * FLOAT_W_ob;
    constexpr dim_t step = FLOAT_C_ob;
    constexpr dim_t _UNROLL = FLOAT_C_ob;

    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < WidthBlocks; ++wb)
            {
                ScalarT *tile = row + wb * FLOAT_W_ob * FLOAT_C_ob;

                FLOAT_DEF_TILE_C;
                FLOAT_LOAD_TILE_C(tile);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_EXP, 0,
                                                      scalar_buf.data());
                FLOAT_STORE_TILE_C(tile);
            }
        }
    }

    scalar_buf[0] = 0.0f;
    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT const *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT const *row = block + h * width * FLOAT_C_ob;
            for (dim_t w = 0; w < width; ++w)
            {
                ScalarT const *a_cur = row + w * FLOAT_C_ob;
                FLOAT_ABSTRACT_FULL_REDUCTION_OP_END(
                    step, OP_ADD, 3, a_cur, scalar_buf.data(), nullptr,
                    scalar_buf.data(), 1, 1);
            }
        }
    }

    scalar_buf[0] = 1.0f / scalar_buf[0];

    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < WidthBlocks; ++wb)
            {
                ScalarT *tile = row + wb * FLOAT_W_ob * FLOAT_C_ob;

                FLOAT_DEF_TILE_C;
                FLOAT_LOAD_TILE_C(tile);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_MUL, 0,
                                                      scalar_buf.data());
                FLOAT_STORE_TILE_C(tile);
            }
        }
    }
}




void compute_softmax_microkernel_lazy(FloatBuffer &microkernel_buf,
                                        FloatBuffer &scalar_buf,
                                        dim_t ChannelBlocks,
                                        dim_t height,
                                        dim_t WidthBlocks)
{
    using ScalarT = typename FloatBuffer::value_type;

    dim_t const width = WidthBlocks * FLOAT_W_ob;
    constexpr dim_t step = FLOAT_C_ob;
    constexpr dim_t _UNROLL = FLOAT_C_ob;

    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < WidthBlocks; ++wb)
            {
                ScalarT *tile = row + wb * FLOAT_W_ob * FLOAT_C_ob;

                FLOAT_DEF_TILE_C;
                FLOAT_LOAD_TILE_C(tile);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_EXP, 0,
                                                      scalar_buf.data());
                FLOAT_STORE_TILE_C(tile);
            }
        }
    }

    scalar_buf[0] = 0.0f;
    FloatBuffer reduction_buf(FLOAT_W_ob * FLOAT_C_ob);
    FLOAT_DEF_TILE_C;
    FLOAT_ZERO_TILE_C;
    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT const *block =
        microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT const *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < WidthBlocks; ++wb)
            {
                ScalarT const *a_cur = row + wb* FLOAT_W_ob * FLOAT_C_ob;
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_ADD, 0, a_cur);
            }
        }
    }
    FLOAT_INPLACE_ACCUM_TILE_C;
    FLOAT_STORE_TILE_C(reduction_buf.data());
    for (dim_t c = 0; c < FLOAT_SIMD; ++c)
    {
        scalar_buf[0] += reduction_buf[c];
    }

    scalar_buf[0] = 1.0f / scalar_buf[0];

    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < WidthBlocks; ++wb)
            {
                ScalarT *tile = row + wb * FLOAT_W_ob * FLOAT_C_ob;

                FLOAT_DEF_TILE_C;
                FLOAT_LOAD_TILE_C(tile);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_MUL, 0,
                                                      scalar_buf.data());
                FLOAT_STORE_TILE_C(tile);
            }
        }
    }
}








void compute_fused_softmax_microkernel_lazy(FloatBuffer &microkernel_buf,
                                        FloatBuffer &scalar_buf,
                                        dim_t ChannelBlocks,
                                        dim_t height,
                                        dim_t WidthBlocks)
{
    using ScalarT = typename FloatBuffer::value_type;

    dim_t const width = WidthBlocks * FLOAT_W_ob;
    constexpr dim_t step = FLOAT_C_ob;
    constexpr dim_t _UNROLL = FLOAT_C_ob;
    FloatBuffer reduction_buf(FLOAT_W_ob * FLOAT_C_ob);
    for(dim_t t =0; t < FLOAT_W_ob * FLOAT_C_ob; t++)
    {
        reduction_buf[t] = 0.0f;
    }

    scalar_buf[0] = 0.0f;

    FLOAT_DEF_TILE_C;
    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < WidthBlocks; ++wb)
            {
                ScalarT *tile = row + wb * FLOAT_W_ob * FLOAT_C_ob;

                
                FLOAT_LOAD_TILE_C(tile);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_EXP, 0,
                                                      scalar_buf.data());
                FLOAT_STORE_TILE_C(tile);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_ADD, 0, reduction_buf.data());
                FLOAT_STORE_TILE_C(reduction_buf.data());
            }
        }
    }
    //reduction over width
    FLOAT_INPLACE_ACCUM_TILE_C;
    FLOAT_STORE_TILE_C(reduction_buf.data());
    //reduction over channels
    for (dim_t c = 0; c < FLOAT_SIMD; ++c)
    {
        scalar_buf[0] += reduction_buf[c];
    }

    scalar_buf[0] = 1.0f / scalar_buf[0];

    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < WidthBlocks; ++wb)
            {
                ScalarT *tile = row + wb * FLOAT_W_ob * FLOAT_C_ob;

                FLOAT_DEF_TILE_C;
                FLOAT_LOAD_TILE_C(tile);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_MUL, 0,
                                                      scalar_buf.data());
                FLOAT_STORE_TILE_C(tile);
            }
        }
    }
}





void compute_fused_softmax_microkernel_lazy_bespoke(FloatBuffer &microkernel_buf,
                                        FloatBuffer &scalar_buf,
                                        dim_t ChannelBlocks,
                                        dim_t height,
                                        dim_t WidthBlocks)
{
    using ScalarT = typename FloatBuffer::value_type;

    dim_t const width = WidthBlocks * FLOAT_W_ob;
    constexpr dim_t step = FLOAT_C_ob;
    constexpr dim_t _UNROLL = FLOAT_C_ob;
    FloatBuffer reduction_buf(FLOAT_W_ob * FLOAT_C_ob);
    for(dim_t t =0; t < FLOAT_W_ob * FLOAT_C_ob; t++)
    {
        reduction_buf[t] = 0.0f;
    }

    scalar_buf[0] = 0.0f;

    FLOAT_DEF_TILE_C;
    a_reg = _mm256_setzero_ps();
    c12 = _mm256_setzero_ps();
    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < WidthBlocks; ++wb)
            {
                ScalarT *tile = row + wb * FLOAT_W_ob * FLOAT_C_ob;

                
                FLOAT_LOAD_TILE_C(tile);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_EXP, 0,
                                                      scalar_buf.data());
                FLOAT_STORE_TILE_C(tile);
                // FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_ADD, 0, reduction_buf.data());
                FLOAT_INPLACE_ACCUM_TILE_C;
                a_reg = _mm256_add_ps(c0, a_reg);
                c12 =  _mm256_add_ps(c12, c1);
                // FLOAT_STORE_ROW_C(reduction_buf.data());
            }
        }
    }
    //reduction over width
    // FLOAT_INPLACE_ACCUM_TILE_C;
    c0 = a_reg;
    c1 = c12;
    FLOAT_STORE_ROW_C(reduction_buf.data());
    //reduction over channels
    for (dim_t c = 0; c < FLOAT_SIMD; ++c)
    {
        scalar_buf[0] += reduction_buf[c];
    }

    scalar_buf[0] = 1.0f / scalar_buf[0];

    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < WidthBlocks; ++wb)
            {
                ScalarT *tile = row + wb * FLOAT_W_ob * FLOAT_C_ob;

                FLOAT_DEF_TILE_C;
                FLOAT_LOAD_TILE_C(tile);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_MUL, 0,
                                                      scalar_buf.data());
                FLOAT_STORE_TILE_C(tile);
            }
        }
    }
}



void compute_softmax_microkernel_exp_only(FloatBuffer &microkernel_buf,
                                          FloatBuffer &scalar_buf,
                                          dim_t ChannelBlocks,
                                          dim_t height,
                                          dim_t WidthBlocks)
{
    using ScalarT = typename FloatBuffer::value_type;

    dim_t const width = WidthBlocks * FLOAT_W_ob;
    constexpr dim_t step = FLOAT_C_ob;

    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < WidthBlocks; ++wb)
            {
                ScalarT *tile = row + wb * FLOAT_W_ob * FLOAT_C_ob;

                FLOAT_DEF_TILE_C;
                FLOAT_LOAD_TILE_C(tile);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_EXP, 0,
                                                      scalar_buf.data());
                FLOAT_STORE_TILE_C(tile);
            }
        }
    }
}

void compute_softmax_microkernel_stages_add_only(
    FloatBuffer const &microkernel_buf,
    FloatBuffer &scalar_buf,
    dim_t ChannelBlocks,
    dim_t height,
    dim_t WidthBlocks)
{
    using ScalarT = typename FloatBuffer::value_type;

    dim_t const width = WidthBlocks * FLOAT_W_ob;
    constexpr dim_t step = FLOAT_C_ob;
    constexpr dim_t _UNROLL = FLOAT_C_ob;

    scalar_buf[0] = 0.0f;
    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT const *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT const *row = block + h * width * FLOAT_C_ob;
            for (dim_t w = 0; w < width; ++w)
            {
                ScalarT const *a_cur = row + w * FLOAT_C_ob;
                FLOAT_ABSTRACT_FULL_REDUCTION_OP_END(
                    step, OP_ADD, 3, a_cur, scalar_buf.data(), nullptr,
                    scalar_buf.data(), 1, 1);
            }
        }
    }
}

void compute_softmax_microkernel_lazy_add_only(
    FloatBuffer const &microkernel_buf,
    FloatBuffer &scalar_buf,
    dim_t ChannelBlocks,
    dim_t height,
    dim_t WidthBlocks)
{
    using ScalarT = typename FloatBuffer::value_type;

    dim_t const width = WidthBlocks * FLOAT_W_ob;
    constexpr dim_t step = FLOAT_C_ob;

    scalar_buf[0] = 0.0f;
    FloatBuffer reduction_buf(FLOAT_W_ob * FLOAT_C_ob);
    FLOAT_DEF_TILE_C;
    FLOAT_ZERO_TILE_C;
    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT const *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT const *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < WidthBlocks; ++wb)
            {
                ScalarT const *a_cur = row + wb * FLOAT_W_ob * FLOAT_C_ob;
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_ADD, 0, a_cur);
            }
        }
    }
    FLOAT_INPLACE_ACCUM_TILE_C;
    FLOAT_STORE_TILE_C(reduction_buf.data());
    for (dim_t c = 0; c < FLOAT_SIMD; ++c)
    {
        scalar_buf[0] += reduction_buf[c];
    }
}


void run_softmax_microkernel_stages(dim_t ChannelBlocks, dim_t Height, dim_t WidthBlocks)
{
    using BufferT = FloatBuffer;

    dim_t const channels = ChannelBlocks * FLOAT_C_ob;
    dim_t const height = Height;
    dim_t const width = WidthBlocks * FLOAT_W_ob;
    size_t const num_elts = channels * height * width;

    BufferT input_buf(num_elts);
    BufferT microkernel_buf(num_elts);
    BufferT softmax_buf(num_elts);
    BufferT scalar_buf(1);

    for (size_t ix = 0; ix < num_elts; ++ix)
    {
        input_buf[ix] = -0.5f + 0.01f * static_cast<float>((ix * 7U) % 101U);
    }
    microkernel_buf = input_buf;

    small::SoftMax(channels, height, width, input_buf, softmax_buf);

    compute_softmax_microkernel_stages(microkernel_buf, scalar_buf,
                                       ChannelBlocks, height, WidthBlocks);

    constexpr float atol = 1.0e-6f;
    constexpr float rtol = 1.0e-5f;
    for (size_t ix = 0; ix < num_elts; ++ix)
    {
        float const expected = softmax_buf[ix];
        float const actual = microkernel_buf[ix];
        float const error = std::abs(actual - expected);
        float const limit = atol + rtol * std::abs(expected);

        if (error > limit)
        {
            std::cout << "FAIL: softmax stage output[" << ix << "] = "
                      << std::setw(12) << std::setprecision(10) << actual
                      << ", expected "
                      << std::setw(12) << std::setprecision(10) << expected
                      << ", error " << error
                      << ", limit " << limit << std::endl;
        }
        TEST_CHECK(error <= limit);
    }
}

void run_softmax_microkernel_lazy(dim_t ChannelBlocks, dim_t Height, dim_t WidthBlocks)
{
    dim_t const channels = ChannelBlocks * FLOAT_C_ob;
    dim_t const height = Height;
    dim_t const width = WidthBlocks * FLOAT_W_ob;
    size_t const num_elts = channels * height * width;

    FloatBuffer input_buf(num_elts);
    FloatBuffer microkernel_buf(num_elts);
    FloatBuffer softmax_buf(num_elts);
    FloatBuffer scalar_buf(1);

    for (size_t ix = 0; ix < num_elts; ++ix)
    {
        input_buf[ix] = -0.5f + 0.01f * static_cast<float>((ix * 7U) % 101U);
    }
    microkernel_buf = input_buf;

    small::SoftMax(channels, height, width, input_buf, softmax_buf);
    compute_softmax_microkernel_lazy(microkernel_buf, scalar_buf,
                                     ChannelBlocks, height, WidthBlocks);

    constexpr float atol = 1.0e-6f;
    constexpr float rtol = 1.0e-5f;
    for (size_t ix = 0; ix < num_elts; ++ix)
    {
        float const expected = softmax_buf[ix];
        float const actual = microkernel_buf[ix];
        float const error = std::abs(actual - expected);
        float const limit = atol + rtol * std::abs(expected);

        if (error > limit)
        {
            std::cout << "FAIL: lazy softmax stage output[" << ix << "] = "
                      << std::setw(12) << std::setprecision(10) << actual
                      << ", expected "
                      << std::setw(12) << std::setprecision(10) << expected
                      << ", error " << error
                      << ", limit " << limit << std::endl;
        }
        TEST_CHECK(error <= limit);
    }
}

void run_fused_softmax_microkernel_lazy(dim_t ChannelBlocks,
                                        dim_t Height,
                                        dim_t WidthBlocks)
{
    dim_t const channels = ChannelBlocks * FLOAT_C_ob;
    dim_t const height = Height;
    dim_t const width = WidthBlocks * FLOAT_W_ob;
    size_t const num_elts = channels * height * width;

    FloatBuffer input_buf(num_elts);
    FloatBuffer microkernel_buf(num_elts);
    FloatBuffer softmax_buf(num_elts);
    FloatBuffer scalar_buf(1);

    for (size_t ix = 0; ix < num_elts; ++ix)
    {
        input_buf[ix] = -0.5f + 0.01f * static_cast<float>((ix * 7U) % 101U);
    }
    microkernel_buf = input_buf;

    small::SoftMax(channels, height, width, input_buf, softmax_buf);
    compute_fused_softmax_microkernel_lazy(microkernel_buf, scalar_buf,
                                           ChannelBlocks, height,
                                           WidthBlocks);

    constexpr float atol = 1.0e-6f;
    constexpr float rtol = 1.0e-5f;
    for (size_t ix = 0; ix < num_elts; ++ix)
    {
        float const expected = softmax_buf[ix];
        float const actual = microkernel_buf[ix];
        float const error = std::abs(actual - expected);
        float const limit = atol + rtol * std::abs(expected);

        if (error > limit)
        {
            std::cout << "FAIL: fused lazy softmax stage output[" << ix << "] = "
                      << std::setw(12) << std::setprecision(10) << actual
                      << ", expected "
                      << std::setw(12) << std::setprecision(10) << expected
                      << ", error " << error
                      << ", limit " << limit << std::endl;
        }
        TEST_CHECK(error <= limit);
    }
}


void run_fused_softmax_microkernel_lazy_bespoke(dim_t ChannelBlocks,
                                                dim_t Height,
                                                dim_t WidthBlocks)
{
    dim_t const channels = ChannelBlocks * FLOAT_C_ob;
    dim_t const height = Height;
    dim_t const width = WidthBlocks * FLOAT_W_ob;
    size_t const num_elts = channels * height * width;

    FloatBuffer input_buf(num_elts);
    FloatBuffer microkernel_buf(num_elts);
    FloatBuffer softmax_buf(num_elts);
    FloatBuffer scalar_buf(1);

    for (size_t ix = 0; ix < num_elts; ++ix)
    {
        input_buf[ix] = -0.5f + 0.01f * static_cast<float>((ix * 7U) % 101U);
    }
    microkernel_buf = input_buf;

    small::SoftMax(channels, height, width, input_buf, softmax_buf);
    compute_fused_softmax_microkernel_lazy_bespoke(microkernel_buf, scalar_buf,
                                                   ChannelBlocks, height,
                                                   WidthBlocks);

    constexpr float atol = 1.0e-6f;
    constexpr float rtol = 1.0e-5f;
    for (size_t ix = 0; ix < num_elts; ++ix)
    {
        float const expected = softmax_buf[ix];
        float const actual = microkernel_buf[ix];
        float const error = std::abs(actual - expected);
        float const limit = atol + rtol * std::abs(expected);

        if (error > limit)
        {
            std::cout << "FAIL: fused lazy bespoke softmax stage output[" << ix << "] = "
                      << std::setw(12) << std::setprecision(10) << actual
                      << ", expected "
                      << std::setw(12) << std::setprecision(10) << expected
                      << ", error " << error
                      << ", limit " << limit << std::endl;
        }
        TEST_CHECK(error <= limit);
    }
}

void test_softmax_microkernel_stages_1x1()
{
    run_softmax_microkernel_stages(1, 1,1);
}

void test_softmax_microkernel_stages_2x3()
{
    run_softmax_microkernel_stages(2, 1, 3);
}

struct SoftmaxStageConfig
{
    dim_t channels;
    dim_t height;
    dim_t width;
};

template <typename Callable>
small::test::TimingStats time_softmax_trials(size_t num_trials,
                                             size_t runs_per_trial,
                                             Callable &&run_once)
{
    return small::test::time_trials(num_trials, [&]() {
        for (size_t run = 0; run < runs_per_trial; ++run)
        {
            run_once();
        }
    });
}

void print_softmax_timing_row(SoftmaxStageConfig const &config,
                              size_t num_trials,
                              size_t runs_per_trial,
                              small::test::TimingStats const &stats)
{
    std::cout << "\t" << config.channels
              << "\t" << config.height
              << "\t" << config.width
              << "\t" << num_trials
              << "\t" << runs_per_trial
              << "\t" << stats.min
              << "\t" << stats.max
              << "\t" << stats.trimmed_avg
              << "\t" << stats.trimmed_std_dev
              << std::endl;
}

void test_softmax_microkernel_stages_conv2d_compatible_configs()
{
    SoftmaxStageConfig const configs[] = {
        {16, 1, 6},
        {16, 30, 30},
        {96, 1, 6},
        {96, 30, 30},
        {16, 48, 48},
        {32, 24, 24},
        {32, 48, 48},
        {64, 12, 12},
        {64, 24, 24},
        {128, 6, 6},
        {128, 12, 12},
        {128, 24, 24},
        {256, 12, 12},
        {512, 12, 12},
        {1024, 6, 6},
    };

    for (SoftmaxStageConfig const &config : configs)
    {
        std::cout << "softmax stages Conv2D-compatible config: C="
                  << config.channels << ", H=" << config.height
                  << ", W=" << config.width << std::endl;
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);
        run_softmax_microkernel_stages(config.channels / 16,
                                       config.height,
                                       config.width / 6);
    }
}

void test_softmax_microkernel_lazy_conv2d_compatible_configs()
{
    SoftmaxStageConfig const configs[] = {
        {16, 1, 6},
        {16, 30, 30},
        {96, 1, 6},
        {96, 30, 30},
        {16, 48, 48},
        {32, 24, 24},
        {32, 48, 48},
        {64, 12, 12},
        {64, 24, 24},
        {128, 6, 6},
        {128, 12, 12},
        {128, 24, 24},
        {256, 12, 12},
        {512, 12, 12},
        {1024, 6, 6},
    };

    for (SoftmaxStageConfig const &config : configs)
    {
        std::cout << "lazy softmax stages Conv2D-compatible config: C="
                  << config.channels << ", H=" << config.height
                  << ", W=" << config.width << std::endl;
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);
        run_softmax_microkernel_lazy(config.channels / 16,
                                     config.height,
                                     config.width / 6);
    }
}

void test_fused_softmax_microkernel_lazy_conv2d_compatible_configs()
{
    SoftmaxStageConfig const configs[] = {
        {16, 1, 6},
        {16, 30, 30},
        {96, 1, 6},
        {96, 30, 30},
        {16, 48, 48},
        {32, 24, 24},
        {32, 48, 48},
        {64, 12, 12},
        {64, 24, 24},
        {128, 6, 6},
        {128, 12, 12},
        {128, 24, 24},
        {256, 12, 12},
        {512, 12, 12},
        {1024, 6, 6},
    };

    for (SoftmaxStageConfig const &config : configs)
    {
        std::cout << "fused lazy softmax stages Conv2D-compatible config: C="
                  << config.channels << ", H=" << config.height
                  << ", W=" << config.width << std::endl;
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);
        run_fused_softmax_microkernel_lazy(config.channels / 16,
                                           config.height,
                                           config.width / 6);
    }
}


void test_fused_softmax_microkernel_lazy_bespoke_conv2d_compatible_configs()
{
    SoftmaxStageConfig const configs[] = {
        {16, 1, 6},
        {16, 30, 30},
        {96, 1, 6},
        {96, 30, 30},
        {16, 48, 48},
        {32, 24, 24},
        {32, 48, 48},
        {64, 12, 12},
        {64, 24, 24},
        {128, 6, 6},
        {128, 12, 12},
        {128, 24, 24},
        {256, 12, 12},
        {512, 12, 12},
        {1024, 6, 6},
    };

    for (SoftmaxStageConfig const &config : configs)
    {
        std::cout << "fused lazy bespoke softmax stages Conv2D-compatible config: C="
                  << config.channels << ", H=" << config.height
                  << ", W=" << config.width << std::endl;
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);
        run_fused_softmax_microkernel_lazy_bespoke(config.channels / 16,
                                                   config.height,
                                                   config.width / 6);
    }
}

void measure_softmax_microkernel_stages_performance()
{
    SoftmaxStageConfig const configs[] = {
        {16, 48, 48},
        {32, 24, 24},
        {32, 48, 48},
        {64, 24, 24},
        {128, 12, 12},
        {64, 12, 12},
        {128, 6, 6},
        {128, 24, 24},
        {256, 12, 12},
        {512, 12, 12},
        {1024, 6, 6},
        {16, 12, 12},
        {32, 6, 6},
        {16, 24, 24},
    };

    constexpr size_t num_trials = 10;
    constexpr size_t runs_per_trial = 100;

    std::cout << "\nsoftmax stage ukernel performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns/trial\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n";

    for (SoftmaxStageConfig const &config : configs)
    {
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);

        dim_t const ChannelBlocks = config.channels / 16;
        dim_t const WidthBlocks = config.width / 6;
        size_t const num_elts = config.channels * config.height * config.width;

        FloatBuffer input_buf(num_elts);
        FloatBuffer microkernel_buf(num_elts);
        FloatBuffer scalar_buf(1);

        for (size_t ix = 0; ix < num_elts; ++ix)
        {
            input_buf[ix] = -0.5f + 0.01f * static_cast<float>((ix * 7U) % 101U);
        }

        microkernel_buf = input_buf;
        compute_softmax_microkernel_stages(microkernel_buf, scalar_buf,
                                           ChannelBlocks, config.height,
                                           WidthBlocks);

        small::test::TimingStats const stats =
            time_softmax_trials(num_trials, runs_per_trial, [&]() {
                microkernel_buf = input_buf;
                compute_softmax_microkernel_stages(microkernel_buf, scalar_buf,
                                                   ChannelBlocks,
                                                   config.height,
                                                   WidthBlocks);
            });
        print_softmax_timing_row(config, num_trials, runs_per_trial, stats);
    }
}

void measure_softmax_microkernel_lazy_performance()
{
    SoftmaxStageConfig const configs[] = {
        {16, 48, 48},
        {32, 24, 24},
        {32, 48, 48},
        {64, 24, 24},
        {128, 12, 12},
        {64, 12, 12},
        {128, 6, 6},
        {128, 24, 24},
        {256, 12, 12},
        {512, 12, 12},
        {1024, 6, 6},
        {16, 12, 12},
        {32, 6, 6},
        {16, 24, 24},
    };

    constexpr size_t num_trials = 10;
    constexpr size_t runs_per_trial = 100;

    std::cout << "\nlazy softmax stage ukernel performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns/trial\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n";

    for (SoftmaxStageConfig const &config : configs)
    {
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);

        dim_t const ChannelBlocks = config.channels / 16;
        dim_t const WidthBlocks = config.width / 6;
        size_t const num_elts = config.channels * config.height * config.width;

        FloatBuffer input_buf(num_elts);
        FloatBuffer microkernel_buf(num_elts);
        FloatBuffer scalar_buf(1);

        for (size_t ix = 0; ix < num_elts; ++ix)
        {
            input_buf[ix] = -0.5f + 0.01f * static_cast<float>((ix * 7U) % 101U);
        }

        microkernel_buf = input_buf;
        compute_softmax_microkernel_lazy(microkernel_buf, scalar_buf,
                                         ChannelBlocks, config.height,
                                         WidthBlocks);

        small::test::TimingStats const stats =
            time_softmax_trials(num_trials, runs_per_trial, [&]() {
                microkernel_buf = input_buf;
                compute_softmax_microkernel_lazy(microkernel_buf, scalar_buf,
                                                 ChannelBlocks,
                                                 config.height,
                                                 WidthBlocks);
            });
        print_softmax_timing_row(config, num_trials, runs_per_trial, stats);
    }
}

void measure_fused_softmax_microkernel_lazy_performance()
{
    SoftmaxStageConfig const configs[] = {
        {16, 48, 48},
        {32, 24, 24},
        {32, 48, 48},
        {64, 24, 24},
        {128, 12, 12},
        {64, 12, 12},
        {128, 6, 6},
        {128, 24, 24},
        {256, 12, 12},
        {512, 12, 12},
        {1024, 6, 6},
        {16, 12, 12},
        {32, 6, 6},
        {16, 24, 24},
    };

    constexpr size_t num_trials = 10;
    constexpr size_t runs_per_trial = 100;

    std::cout << "\nfused lazy softmax stage ukernel performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns/trial\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n";

    for (SoftmaxStageConfig const &config : configs)
    {
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);

        dim_t const ChannelBlocks = config.channels / 16;
        dim_t const WidthBlocks = config.width / 6;
        size_t const num_elts = config.channels * config.height * config.width;

        FloatBuffer input_buf(num_elts);
        FloatBuffer microkernel_buf(num_elts);
        FloatBuffer scalar_buf(1);

        for (size_t ix = 0; ix < num_elts; ++ix)
        {
            input_buf[ix] = -0.5f + 0.01f * static_cast<float>((ix * 7U) % 101U);
        }

        microkernel_buf = input_buf;
        compute_fused_softmax_microkernel_lazy(microkernel_buf, scalar_buf,
                                               ChannelBlocks, config.height,
                                               WidthBlocks);

        small::test::TimingStats const stats =
            time_softmax_trials(num_trials, runs_per_trial, [&]() {
                microkernel_buf = input_buf;
                compute_fused_softmax_microkernel_lazy(microkernel_buf,
                                                       scalar_buf,
                                                       ChannelBlocks,
                                                       config.height,
                                                       WidthBlocks);
            });
        print_softmax_timing_row(config, num_trials, runs_per_trial, stats);
    }
}


void measure_fused_softmax_microkernel_lazy_bespoke_performance()
{
    SoftmaxStageConfig const configs[] = {
        {16, 48, 48},
        {32, 24, 24},
        {32, 48, 48},
        {64, 24, 24},
        {128, 12, 12},
        {64, 12, 12},
        {128, 6, 6},
        {128, 24, 24},
        {256, 12, 12},
        {512, 12, 12},
        {1024, 6, 6},
        {16, 12, 12},
        {32, 6, 6},
        {16, 24, 24},
    };

    constexpr size_t num_trials = 10;
    constexpr size_t runs_per_trial = 100;

    std::cout << "\nfused lazy bespoke softmax stage ukernel performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns/trial\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n";

    for (SoftmaxStageConfig const &config : configs)
    {
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);

        dim_t const ChannelBlocks = config.channels / 16;
        dim_t const WidthBlocks = config.width / 6;
        size_t const num_elts = config.channels * config.height * config.width;

        FloatBuffer input_buf(num_elts);
        FloatBuffer microkernel_buf(num_elts);
        FloatBuffer scalar_buf(1);

        for (size_t ix = 0; ix < num_elts; ++ix)
        {
            input_buf[ix] = -0.5f + 0.01f * static_cast<float>((ix * 7U) % 101U);
        }

        microkernel_buf = input_buf;
        compute_fused_softmax_microkernel_lazy_bespoke(
            microkernel_buf, scalar_buf, ChannelBlocks, config.height,
            WidthBlocks);

        small::test::TimingStats const stats =
            time_softmax_trials(num_trials, runs_per_trial, [&]() {
                microkernel_buf = input_buf;
                compute_fused_softmax_microkernel_lazy_bespoke(
                    microkernel_buf, scalar_buf, ChannelBlocks,
                    config.height, WidthBlocks);
            });
        print_softmax_timing_row(config, num_trials, runs_per_trial, stats);
    }
}


void measure_small_softmax_performance()
{
    SoftmaxStageConfig const configs[] = {
        {16, 48, 48},
        {32, 24, 24},
        {32, 48, 48},
        {64, 24, 24},
        {128, 12, 12},
        {64, 12, 12},
        {128, 6, 6},
        {128, 24, 24},
        {256, 12, 12},
        {512, 12, 12},
        {1024, 6, 6},
        {16, 12, 12},
        {32, 6, 6},
        {16, 24, 24},
    };

    constexpr size_t num_trials = 10;
    constexpr size_t runs_per_trial = 100;

    std::cout << "\nsmall::SoftMax performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns/trial"
              << "\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n";

    for (SoftmaxStageConfig const &config : configs)
    {
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);

        size_t const num_elts = config.channels * config.height * config.width;

        FloatBuffer input_buf(num_elts);
        FloatBuffer softmax_buf(num_elts);

        for (size_t ix = 0; ix < num_elts; ++ix)
        {
            input_buf[ix] = -0.5f + 0.01f * static_cast<float>((ix * 7U) % 101U);
        }

        small::SoftMax(config.channels, config.height, config.width,
                       input_buf, softmax_buf);

        small::test::TimingStats const stats =
            time_softmax_trials(num_trials, runs_per_trial, [&]() {
                small::SoftMax(config.channels, config.height, config.width,
                               input_buf, softmax_buf);
            });
        print_softmax_timing_row(config, num_trials, runs_per_trial, stats);
    }
}

void measure_softmax_microkernel_add_only_performance()
{
    SoftmaxStageConfig const configs[] = {
        {16, 48, 48},
        {32, 24, 24},
        {32, 48, 48},
        {64, 24, 24},
        {128, 12, 12},
        {64, 12, 12},
        {128, 6, 6},
        {128, 24, 24},
        {256, 12, 12},
        {512, 12, 12},
        {1024, 6, 6},
        {16, 12, 12},
        {32, 6, 6},
        {16, 24, 24},
    };

    constexpr size_t num_trials = 10;
    constexpr size_t runs_per_trial = 100;

    std::cout << "\nsoftmax OP_ADD-only ukernel performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns/trial"
              << "\tstages_min\tstages_max\tstages_avg_trim\tstages_std_dev_trim"
              << "\tlazy_min\tlazy_max\tlazy_avg_trim\tlazy_std_dev_trim\n";

    for (SoftmaxStageConfig const &config : configs)
    {
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);

        dim_t const ChannelBlocks = config.channels / 16;
        dim_t const WidthBlocks = config.width / 6;
        size_t const num_elts = config.channels * config.height * config.width;

        FloatBuffer input_buf(num_elts);
        FloatBuffer exp_buf(num_elts);
        FloatBuffer scalar_buf(1);

        for (size_t ix = 0; ix < num_elts; ++ix)
        {
            input_buf[ix] = -0.5f + 0.01f * static_cast<float>((ix * 7U) % 101U);
        }

        exp_buf = input_buf;
        compute_softmax_microkernel_exp_only(exp_buf, scalar_buf,
                                             ChannelBlocks, config.height,
                                             WidthBlocks);

        small::test::TimingStats const stage_stats =
            time_softmax_trials(num_trials, runs_per_trial, [&]() {
                compute_softmax_microkernel_stages_add_only(
                    exp_buf, scalar_buf, ChannelBlocks, config.height,
                    WidthBlocks);
            });
        small::test::TimingStats const lazy_stats =
            time_softmax_trials(num_trials, runs_per_trial, [&]() {
                compute_softmax_microkernel_lazy_add_only(
                    exp_buf, scalar_buf, ChannelBlocks, config.height,
                    WidthBlocks);
            });

        std::cout << "\t" << config.channels
                  << "\t" << config.height
                  << "\t" << config.width
                  << "\t" << num_trials
                  << "\t" << runs_per_trial
                  << "\t" << stage_stats.min
                  << "\t" << stage_stats.max
                  << "\t" << stage_stats.trimmed_avg
                  << "\t" << stage_stats.trimmed_std_dev
                  << "\t" << lazy_stats.min
                  << "\t" << lazy_stats.max
                  << "\t" << lazy_stats.trimmed_avg
                  << "\t" << lazy_stats.trimmed_std_dev
                  << std::endl;
    }
}

#endif

} // namespace float_detail
} // namespace small

TEST_LIST = {
#if !defined(QUANTIZED)
    {"softmax microkernel stages 1x1",
     small::float_detail::test_softmax_microkernel_stages_1x1},
    {"softmax microkernel stages 2x3",
     small::float_detail::test_softmax_microkernel_stages_2x3},
    {"softmax microkernel stages Conv2D-compatible configs",
     small::float_detail::test_softmax_microkernel_stages_conv2d_compatible_configs},
    {"lazy softmax microkernel stages Conv2D-compatible configs",
     small::float_detail::test_softmax_microkernel_lazy_conv2d_compatible_configs},
    {"fused lazy softmax microkernel stages Conv2D-compatible configs",
     small::float_detail::test_fused_softmax_microkernel_lazy_conv2d_compatible_configs},
    {"fused lazy bespoke softmax microkernel stages Conv2D-compatible configs",
     small::float_detail::test_fused_softmax_microkernel_lazy_bespoke_conv2d_compatible_configs},
    {"softmax microkernel stages performance",
     small::float_detail::measure_softmax_microkernel_stages_performance},
    {"lazy softmax microkernel stages performance",
     small::float_detail::measure_softmax_microkernel_lazy_performance},
    {"fused lazy softmax microkernel stages performance",
     small::float_detail::measure_fused_softmax_microkernel_lazy_performance},
    {"fused lazy bespoke softmax microkernel stages performance",
     small::float_detail::measure_fused_softmax_microkernel_lazy_bespoke_performance},
    {"small SoftMax performance",
     small::float_detail::measure_small_softmax_performance},
    {"softmax OP_ADD-only microkernel performance",
     small::float_detail::measure_softmax_microkernel_add_only_performance},
#endif
    {NULL, NULL}
};
