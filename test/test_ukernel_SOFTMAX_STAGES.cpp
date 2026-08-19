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

// Simplified version of small::SoftMax
void compute_softmax_microkernel_stages(FloatBuffer &microkernel_buf,
                                        FloatBuffer &scalar_buf,
                                        dim_t ChannelBlocks,
                                        dim_t height,
                                        dim_t WidthBlocks)
{
    using ScalarT = typename FloatBuffer::value_type;
    float recip_fact_table_positive[] = {1.0f/720.0f, 1.0f / 120.0f,1.0f / 24.0f,1.0f / 6.0f,0.5f,1.0f, 1.0};\

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
                FLOAT_INPLACE_EXP_horner_TILE_C(FLOAT_C_ob, tile, recip_fact_table_positive);
                // FLOAT_ABSTRACT_OP(step, OP_EXP, 1, tile, recip_fact_table_positive, tile, tile);
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



// Simplified version of small::SoftMax, with horner's rule for exp (stable for small range of values -1 to +2)
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
    float recip_fact_table_positive[] = {1.0f/720.0f, 1.0f / 120.0f,1.0f / 24.0f,1.0f / 6.0f,0.5f,1.0f, 1.0};\
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
                FLOAT_INPLACE_EXP_horner_TILE_C(FLOAT_C_ob, tile, recip_fact_table_positive);
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
    float recip_fact_table_positive[] = {1.0f/720.0f, 1.0f / 120.0f,1.0f / 24.0f,1.0f / 6.0f,0.5f,1.0f, 1.0};\

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
                FLOAT_INPLACE_EXP_horner_TILE_C(FLOAT_C_ob, tile, recip_fact_table_positive);
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
    float recip_fact_table_positive[] = {1.0f/720.0f, 1.0f / 120.0f,1.0f / 24.0f,1.0f / 6.0f,0.5f,1.0f, 1.0};\

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
                // FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_EXP, 0,
                //                                       scalar_buf.data());
                FLOAT_INPLACE_EXP_horner_TILE_C(FLOAT_C_ob, tile, recip_fact_table_positive);

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


//inplace microkernel_buf becomes exp_buf
void compute_softmax_microkernel_exp_only(FloatBuffer &microkernel_buf,
                                          dim_t ChannelBlocks,
                                          dim_t height,
                                          dim_t WidthBlocks)
{
    using ScalarT = typename FloatBuffer::value_type;
    float recip_fact_table_positive[] = {1.0f/720.0f, 1.0f / 120.0f,1.0f / 24.0f,1.0f / 6.0f,0.5f,1.0f, 1.0};\

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
                FLOAT_INPLACE_EXP_horner_TILE_C(FLOAT_C_ob, tile, recip_fact_table_positive);
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
    a_reg = _mm256_setzero_ps();
    c12 = _mm256_setzero_ps();
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
                // FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_ADD, 0, a_cur);
                FLOAT_LOAD_TILE_C(a_cur);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_ADD, 0, reduction_buf.data());
                FLOAT_STORE_TILE_C(reduction_buf.data());
            }
        }
    }
    // FLOAT_INPLACE_ACCUM_TILE_C;
    // FLOAT_STORE_TILE_C(reduction_buf.data());
    // for (dim_t c = 0; c < FLOAT_SIMD; ++c)
    // {
    //     scalar_buf[0] += reduction_buf[c];
    // }
    FLOAT_ABSTRACT_REDUCTION_OP_TILE(OP_ADD, reduction_buf.data(), scalar_buf.data());
}


void compute_softmax_microkernel_bespoke_add_only(
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
    a_reg = _mm256_setzero_ps();
    c12 = _mm256_setzero_ps();
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
                // FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_ADD, 0, a_cur);
                FLOAT_LOAD_TILE_C(a_cur);
                // FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(step, OP_ADD, 0, reduction_buf.data());
                // FLOAT_STORE_TILE_C(reduction_buf.data());
                FLOAT_INPLACE_ACCUM_TILE_C;
                a_reg = _mm256_add_ps(c0, a_reg);
                c12 =  _mm256_add_ps(c12, c1);
            }
        }
    }
    c0 = a_reg;
    c1 = c12;
    FLOAT_STORE_ROW_C(reduction_buf.data());
    //reduction over channels
    for (dim_t c = 0; c < FLOAT_SIMD; ++c)
    {
        scalar_buf[0] += reduction_buf[c];
    }


}


void compute_softmax_microkernel_mul_only(
    FloatBuffer &microkernel_buf,
    FloatBuffer const &scalar_buf,
    dim_t ChannelBlocks,
    dim_t height,
    dim_t WidthBlocks)
{
    using ScalarT = typename FloatBuffer::value_type;
    dim_t const width = WidthBlocks * FLOAT_W_ob;
    constexpr dim_t step = FLOAT_C_ob;

    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT *block = microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for (dim_t h = 0; h < height; ++h)
        {
            ScalarT *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < WidthBlocks; ++wb)
            {
                ScalarT *tile = row + wb * FLOAT_W_ob * FLOAT_C_ob;
                FLOAT_DEF_TILE_C;
                FLOAT_LOAD_TILE_C(tile);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(
                    step, OP_MUL, 0, scalar_buf.data());
                FLOAT_STORE_TILE_C(tile);
            }
        }
    }
}


void compute_composed_softmax_microkernel_lazy(FloatBuffer &microkernel_buf,
                                        FloatBuffer &scalar_buf,
                                        dim_t ChannelBlocks,
                                        dim_t height,
                                        dim_t WidthBlocks)
{
    using ScalarT = typename FloatBuffer::value_type;

    compute_softmax_microkernel_exp_only(microkernel_buf, ChannelBlocks, height, WidthBlocks);
    compute_softmax_microkernel_lazy_add_only(microkernel_buf, scalar_buf, ChannelBlocks, height, WidthBlocks);

    scalar_buf[0] = 1.0f / scalar_buf[0];
    constexpr dim_t step = FLOAT_C_ob;
    for (dim_t cb = 0; cb < ChannelBlocks; ++cb)
    {
        ScalarT *block =
            microkernel_buf.data() + cb * height * WidthBlocks * FLOAT_W_ob * FLOAT_C_ob;
        for(dim_t h = 0; h < height; ++h)
        {
            ScalarT *row = block + h *  WidthBlocks * FLOAT_W_ob  * FLOAT_C_ob;
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





FloatBuffer create_small_negative_data(size_t num_elts)
{
    FloatBuffer data(num_elts);
    for (size_t ix = 0; ix < num_elts; ++ix)
    {
        data[ix] = -0.5f + static_cast<float>((ix))*1.0/static_cast<float>(num_elts);
    }
    return data;
}

void run_softmax_microkernel_stages(dim_t ChannelBlocks, dim_t Height, dim_t WidthBlocks)
{
    using BufferT = FloatBuffer;

    dim_t const channels = ChannelBlocks * FLOAT_C_ob;
    dim_t const height = Height;
    dim_t const width = WidthBlocks * FLOAT_W_ob;
    size_t const num_elts = channels * height * width;

    BufferT input_buf = create_small_negative_data(num_elts);
    BufferT microkernel_buf(num_elts);
    BufferT softmax_buf(num_elts);
    BufferT scalar_buf(1);

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

    FloatBuffer input_buf = create_small_negative_data(num_elts);
    FloatBuffer microkernel_buf(num_elts);
    FloatBuffer softmax_buf(num_elts);
    FloatBuffer scalar_buf(1);

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

    FloatBuffer input_buf = create_small_negative_data(num_elts);
    FloatBuffer microkernel_buf(num_elts);
    FloatBuffer softmax_buf(num_elts);
    FloatBuffer scalar_buf(1);

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

    FloatBuffer input_buf = create_small_negative_data(num_elts);
    FloatBuffer microkernel_buf(num_elts);
    FloatBuffer softmax_buf(num_elts);
    FloatBuffer scalar_buf(1);

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



struct SoftmaxStageConfig
{
    dim_t channels;
    dim_t height;
    dim_t width;
};

SoftmaxStageConfig const softmax_stage_configs[] = {
    {16, 48, 48},
    {32, 24, 24},
    {32, 48, 48},
    {64, 24, 24},
    {64, 12, 12},
    {128, 12, 12},
    {128, 6, 6},
    {128, 24, 24},
    {256, 12, 12},
    {512, 12, 12},
    {1024, 6, 6},
    {16, 1, 6},
    {16, 36, 36},
    {16, 60, 60},
    {16, 114, 114},

};

template <typename Callable>
small::test::TimingStats time_softmax_trials(size_t num_runs,
                                             size_t num_trials,
                                             Callable &&run_once)
{
    std::vector<double> times;
    times.reserve(num_runs);
    small::Timer timer;

    for (size_t run = 0; run < num_runs; ++run)
    {
        timer.start();
        for (size_t trial = 0; trial < num_trials; ++trial)
        {
            run_once();
        }
        timer.stop();
        times.push_back(timer.elapsed() / static_cast<double>(num_trials));
    }

    return small::test::summarize_times(std::move(times));
}

void print_softmax_timing_row(SoftmaxStageConfig const &config,
                              size_t num_trials,
                              size_t num_runs,
                              small::test::TimingStats const &stats)
{
    std::cout << "\t" << config.channels
              << "\t" << config.height
              << "\t" << config.width
              << "\t" << num_trials
              << "\t" << num_runs
              << "\t" << stats.min
              << "\t" << stats.max
              << "\t" << stats.trimmed_avg
              << "\t" << stats.trimmed_std_dev
              << std::endl;
}



// ==================================================================
// Performance measurement
// ==================================================================
void measure_softmax_microkernel_stages_performance()
{
    constexpr size_t num_trials = 1000;
    constexpr size_t num_runs = 100;

    std::cout << "\nsoftmax stage ukernel performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n";

    for (SoftmaxStageConfig const &config : softmax_stage_configs)
    {
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);

        dim_t const ChannelBlocks = config.channels / 16;
        dim_t const WidthBlocks = config.width / 6;
        run_softmax_microkernel_stages(ChannelBlocks, config.height,
                                       WidthBlocks);

        size_t const num_elts = config.channels * config.height * config.width;

        FloatBuffer input_buf = create_small_negative_data(num_elts);
        FloatBuffer microkernel_buf(num_elts);
        FloatBuffer scalar_buf(1);


        microkernel_buf = input_buf;
        compute_softmax_microkernel_stages(microkernel_buf, scalar_buf,
                                           ChannelBlocks, config.height,
                                           WidthBlocks);

        small::test::TimingStats const stats =
            time_softmax_trials(num_runs, num_trials, [&]() {
                // microkernel_buf = input_buf;
                compute_softmax_microkernel_stages(microkernel_buf, scalar_buf,
                                                   ChannelBlocks,
                                                   config.height,
                                                   WidthBlocks);
            });
        print_softmax_timing_row(config, num_trials, num_runs, stats);
    }
}

void measure_softmax_microkernel_lazy_performance()
{
    constexpr size_t num_trials = 1000;
    constexpr size_t num_runs = 100;

    std::cout << "\nlazy softmax stage ukernel performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n";

    for (SoftmaxStageConfig const &config : softmax_stage_configs)
    {
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);

        dim_t const ChannelBlocks = config.channels / 16;
        dim_t const WidthBlocks = config.width / 6;
        run_softmax_microkernel_lazy(ChannelBlocks, config.height,
                                     WidthBlocks);

        size_t const num_elts = config.channels * config.height * config.width;

        FloatBuffer input_buf = create_small_negative_data(num_elts);
        FloatBuffer microkernel_buf(num_elts);
        FloatBuffer scalar_buf(1);


        microkernel_buf = input_buf;
        compute_softmax_microkernel_lazy(microkernel_buf, scalar_buf,
                                         ChannelBlocks, config.height,
                                         WidthBlocks);

        small::test::TimingStats const stats =
            time_softmax_trials(num_runs, num_trials, [&]() {
                // microkernel_buf = input_buf;
                compute_softmax_microkernel_lazy(microkernel_buf, scalar_buf,
                                                 ChannelBlocks,
                                                 config.height,
                                                 WidthBlocks);
            });
        print_softmax_timing_row(config, num_trials, num_runs, stats);
    }
}

void measure_fused_softmax_microkernel_lazy_performance()
{
    constexpr size_t num_trials = 1000;
    constexpr size_t num_runs = 100;

    std::cout << "\nfused lazy softmax stage ukernel performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n";

    for (SoftmaxStageConfig const &config : softmax_stage_configs)
    {
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);

        dim_t const ChannelBlocks = config.channels / 16;
        dim_t const WidthBlocks = config.width / 6;
        run_fused_softmax_microkernel_lazy(ChannelBlocks, config.height,
                                           WidthBlocks);

        size_t const num_elts = config.channels * config.height * config.width;

        FloatBuffer input_buf = create_small_negative_data(num_elts);
        FloatBuffer microkernel_buf(num_elts);
        FloatBuffer scalar_buf(1);


        microkernel_buf = input_buf;
        compute_fused_softmax_microkernel_lazy(microkernel_buf, scalar_buf,
                                               ChannelBlocks, config.height,
                                               WidthBlocks);

        small::test::TimingStats const stats =
            time_softmax_trials(num_runs, num_trials, [&]() {
                // microkernel_buf = input_buf;
                compute_fused_softmax_microkernel_lazy(microkernel_buf,
                                                       scalar_buf,
                                                       ChannelBlocks,
                                                       config.height,
                                                       WidthBlocks);
            });
        print_softmax_timing_row(config, num_trials, num_runs, stats);
    }
}


void measure_fused_softmax_microkernel_lazy_bespoke_performance()
{
    constexpr size_t num_trials = 1000;
    constexpr size_t num_runs = 100;

    std::cout << "\nfused lazy bespoke softmax stage ukernel performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n";

    for (SoftmaxStageConfig const &config : softmax_stage_configs)
    {
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);

        dim_t const ChannelBlocks = config.channels / 16;
        dim_t const WidthBlocks = config.width / 6;
        run_fused_softmax_microkernel_lazy_bespoke(
            ChannelBlocks, config.height, WidthBlocks);

        size_t const num_elts = config.channels * config.height * config.width;

        FloatBuffer input_buf = create_small_negative_data(num_elts);
        FloatBuffer microkernel_buf(num_elts);
        FloatBuffer scalar_buf(1);


        microkernel_buf = input_buf;
        compute_fused_softmax_microkernel_lazy_bespoke(
            microkernel_buf, scalar_buf, ChannelBlocks, config.height,
            WidthBlocks);

        small::test::TimingStats const stats =
            time_softmax_trials(num_runs, num_trials, [&]() {
                // microkernel_buf = input_buf;
                compute_fused_softmax_microkernel_lazy_bespoke(
                    microkernel_buf, scalar_buf, ChannelBlocks,
                    config.height, WidthBlocks);
            });
        print_softmax_timing_row(config, num_trials, num_runs, stats);
    }
}


void measure_small_softmax_performance()
{
    constexpr size_t num_trials = 1000;
    constexpr size_t num_runs = 100;

    std::cout << "\nsmall::SoftMax performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns"
              << "\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n";

    for (SoftmaxStageConfig const &config : softmax_stage_configs)
    {
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);

        size_t const num_elts = config.channels * config.height * config.width;

        FloatBuffer input_buf = create_small_negative_data(num_elts);
        FloatBuffer softmax_buf(num_elts);


        small::SoftMax(config.channels, config.height, config.width,
                       input_buf, softmax_buf);

        small::test::TimingStats const stats =
            time_softmax_trials(num_runs, num_trials, [&]() {
                small::SoftMax(config.channels, config.height, config.width,
                               input_buf, softmax_buf);
            });
        print_softmax_timing_row(config, num_trials, num_runs, stats);
    }
}

void measure_softmax_microkernel_add_only_performance()
{
    constexpr size_t num_trials = 1000;
    constexpr size_t num_runs = 100;

    std::cout << "\nsoftmax OP_ADD-only ukernel performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns"
              << "\tstages_min\tstages_max\tstages_avg_trim\tstages_std_dev_trim"
              << "\tlazy_min\tlazy_max\tlazy_avg_trim\tlazy_std_dev_trim\n";

    for (SoftmaxStageConfig const &config : softmax_stage_configs)
    {
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);

        dim_t const ChannelBlocks = config.channels / 16;
        dim_t const WidthBlocks = config.width / 6;
        size_t const num_elts = config.channels * config.height * config.width;

        FloatBuffer input_buf = create_small_negative_data(num_elts);
        FloatBuffer exp_buf(num_elts);
        FloatBuffer scalar_buf(1);


        exp_buf = input_buf;
        compute_softmax_microkernel_exp_only(exp_buf,
                                             ChannelBlocks, config.height,
                                             WidthBlocks);

        small::test::TimingStats const stage_stats =
            time_softmax_trials(num_runs, num_trials, [&]() {
                compute_softmax_microkernel_stages_add_only(
                    exp_buf, scalar_buf, ChannelBlocks, config.height,
                    WidthBlocks);
            });
        small::test::TimingStats const lazy_stats =
            time_softmax_trials(num_runs, num_trials, [&]() {
                compute_softmax_microkernel_lazy_add_only(
                    exp_buf, scalar_buf, ChannelBlocks, config.height,
                    WidthBlocks);
            });

        std::cout << "\t" << config.channels
                  << "\t" << config.height
                  << "\t" << config.width
                  << "\t" << num_trials
                  << "\t" << num_runs
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

void measure_softmax_microkernel_lazy_stages_performance()
{
    constexpr size_t num_trials = 1000;
    constexpr size_t num_runs = 100;

    std::cout << "\nlazy softmax microkernel stage performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns"
              << "\texp_avg\tadd_avg\tmul_avg\n";

    for (SoftmaxStageConfig const &config : softmax_stage_configs)
    {
        dim_t const ChannelBlocks = config.channels / FLOAT_C_ob;
        dim_t const WidthBlocks = config.width / FLOAT_W_ob;
        size_t const num_elts = config.channels * config.height * config.width;
        FloatBuffer const input_buf = create_small_negative_data(num_elts);
        FloatBuffer exp_buf = input_buf;
        FloatBuffer scalar_buf(1);

        compute_softmax_microkernel_exp_only(exp_buf,
                                             ChannelBlocks, config.height,
                                             WidthBlocks);

        small::test::TimingStats const exp_stats =
            time_softmax_trials(num_runs, num_trials, [&]() {
                compute_softmax_microkernel_exp_only(
                    exp_buf, ChannelBlocks, config.height, WidthBlocks);
            });

        exp_buf = input_buf;
        compute_softmax_microkernel_exp_only(
            exp_buf, ChannelBlocks, config.height, WidthBlocks);
        small::test::TimingStats const add_stats =
            time_softmax_trials(num_runs, num_trials, [&]() {
                compute_softmax_microkernel_lazy_add_only(
                    exp_buf, scalar_buf, ChannelBlocks, config.height, WidthBlocks);
            });

        compute_softmax_microkernel_lazy_add_only(
            exp_buf, scalar_buf, ChannelBlocks, config.height, WidthBlocks);
        scalar_buf[0] = 1.0f / scalar_buf[0];
        FloatBuffer scaled_buf = exp_buf;
        small::test::TimingStats const mul_stats =
            time_softmax_trials(num_runs, num_trials, [&]() {
                compute_softmax_microkernel_mul_only(
                    scaled_buf, scalar_buf, ChannelBlocks, config.height, WidthBlocks);
            });

        std::cout << "\t" << config.channels
                  << "\t" << config.height
                  << "\t" << config.width
                  << "\t" << num_trials
                  << "\t" << num_runs
                  << "\t" << exp_stats.trimmed_avg
                  << "\t" << add_stats.trimmed_avg
                  << "\t" << mul_stats.trimmed_avg
                  << std::endl;
    }
}

void measure_softmax_microkernel_mul_only_performance()
{
    constexpr size_t num_trials = 1000;
    constexpr size_t num_runs = 100;

    std::cout << "\nsoftmax OP_MUL-only ukernel performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns"
              << "\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n";

    for (SoftmaxStageConfig const &config : softmax_stage_configs)
    {
        dim_t const ChannelBlocks = config.channels / FLOAT_C_ob;
        dim_t const WidthBlocks = config.width / FLOAT_W_ob;
        size_t const num_elts = config.channels * config.height * config.width;
        FloatBuffer const input_buf = create_small_negative_data(num_elts);
        FloatBuffer exp_buf = input_buf;
        FloatBuffer scalar_buf(1);

        compute_softmax_microkernel_exp_only(
            exp_buf, ChannelBlocks, config.height, WidthBlocks);
        compute_softmax_microkernel_lazy_add_only(
            exp_buf, scalar_buf, ChannelBlocks, config.height, WidthBlocks);
        scalar_buf[0] = 1.0f / scalar_buf[0];
        FloatBuffer mul_buf = exp_buf;

        small::test::TimingStats const stats =
            time_softmax_trials(num_runs, num_trials, [&]() {
                compute_softmax_microkernel_mul_only(
                    mul_buf, scalar_buf, ChannelBlocks, config.height, WidthBlocks);
            });
        print_softmax_timing_row(config, num_trials, num_runs, stats);
    }
}

void measure_composed_softmax_microkernel_lazy_performance()
{
    constexpr size_t num_trials = 1000;
    constexpr size_t num_runs = 100;

    std::cout << "\ncomposed lazy softmax microkernel performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns"
              << "\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n";

    for (SoftmaxStageConfig const &config : softmax_stage_configs)
    {
        dim_t const ChannelBlocks = config.channels / FLOAT_C_ob;
        dim_t const WidthBlocks = config.width / FLOAT_W_ob;
        size_t const num_elts = config.channels * config.height * config.width;
        FloatBuffer const input_buf = create_small_negative_data(num_elts);
        FloatBuffer composed_buf = input_buf;
        FloatBuffer scalar_buf(1);

        small::test::TimingStats const stats =
            time_softmax_trials(num_runs, num_trials, [&]() {
                composed_buf = input_buf;
                compute_composed_softmax_microkernel_lazy(
                    composed_buf, scalar_buf, ChannelBlocks, config.height, WidthBlocks);
            });
        print_softmax_timing_row(config, num_trials, num_runs, stats);
    }
}

void measure_softmax_microkernel_exp_only_performance()
{
    constexpr size_t num_trials = 1000;
    constexpr size_t num_runs = 100;

    std::cout << "\nsoftmax OP_EXP-only ukernel performance\n";
    std::cout << "\tC\tH\tW\ttrials\truns"
              << "\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n";

    for (SoftmaxStageConfig const &config : softmax_stage_configs)
    {
        TEST_ASSERT(config.channels % 16 == 0);
        TEST_ASSERT(config.width % 6 == 0);

        dim_t const ChannelBlocks = config.channels / 16;
        dim_t const WidthBlocks = config.width / 6;
        size_t const num_elts = config.channels * config.height * config.width;

        FloatBuffer input_buf = create_small_negative_data(num_elts);
        FloatBuffer exp_buf = input_buf;
        FloatBuffer scalar_buf(1);
        std::vector<double> times;
        times.reserve(num_runs);
        small::Timer timer;

        compute_softmax_microkernel_exp_only(exp_buf,
                                             ChannelBlocks, config.height,
                                             WidthBlocks);

        for (size_t run = 0; run < num_runs; ++run)
        {
            exp_buf = input_buf;
            timer.start();
            for (size_t trial = 0; trial < num_trials; ++trial)
            {
                compute_softmax_microkernel_exp_only(
                    exp_buf, ChannelBlocks, config.height,
                    WidthBlocks);
            }
            timer.stop();
            times.push_back(timer.elapsed() / static_cast<double>(num_trials));
        }

        small::test::TimingStats const stats =
            small::test::summarize_times(std::move(times));
        print_softmax_timing_row(config, num_trials, num_runs, stats);
    }
}

#endif

} // namespace float_detail
} // namespace small

TEST_LIST = {
#if !defined(QUANTIZED)
    // {"small SoftMax performance",
    //  small::float_detail::measure_small_softmax_performance},
    // {"softmax microkernel stages performance",
    //  small::float_detail::measure_softmax_microkernel_stages_performance},
    // {"lazy softmax microkernel three-stage performance",
    //  small::float_detail::measure_softmax_microkernel_lazy_stages_performance},
    // {"composed lazy softmax microkernel performance",
    //  small::float_detail::measure_composed_softmax_microkernel_lazy_performance},
    // {"softmax OP_EXP-only microkernel performance",
    //  small::float_detail::measure_softmax_microkernel_exp_only_performance},
    {"softmax OP_ADD-only microkernel performance",
     small::float_detail::measure_softmax_microkernel_add_only_performance},
    // {"softmax OP_MUL-only microkernel performance",
    //  small::float_detail::measure_softmax_microkernel_mul_only_performance},
    // {"fused lazy softmax microkernel stages performance",
    //  small::float_detail::measure_fused_softmax_microkernel_lazy_performance},
    // {"fused lazy bespoke softmax microkernel stages performance",
    //  small::float_detail::measure_fused_softmax_microkernel_lazy_bespoke_performance},

#endif
    {NULL, NULL}
};
