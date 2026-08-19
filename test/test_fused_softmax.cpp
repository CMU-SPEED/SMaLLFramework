//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2026 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//****************************************************************************

#define PARALLEL 1

#include <acutest.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include <small.h>
#include <small/float_detail/abstract_op.hpp>

namespace small
{
namespace float_detail
{

// Reference implementation used by the microkernel staging test.  Keeping it
// in this dedicated test allows the public fused interface to be tested
// without running the benchmark test executable.
void compute_fused_global_reduction_microkernel_lazy(FloatBuffer &microkernel_buf,
                                            FloatBuffer &scalar_buf,
                                            dim_t channel_blocks,
                                            dim_t height,
                                            dim_t width_blocks)
{
    using ScalarT = typename FloatBuffer::value_type;
    dim_t const width = width_blocks * FLOAT_W_ob;
    constexpr dim_t step = FLOAT_C_ob;
    constexpr dim_t _UNROLL = FLOAT_C_ob;
    FloatBuffer reduction_buf(FLOAT_W_ob * FLOAT_C_ob);
    float recip_fact_table_positive[] = {
        1.0f / 720.0f, 1.0f / 120.0f, 1.0f / 24.0f, 1.0f / 6.0f,
        0.5f, 1.0f, 1.0f};

    for (dim_t ix = 0; ix < FLOAT_W_ob * FLOAT_C_ob; ++ix)
    {
        reduction_buf[ix] = 0.0f;
    }
    scalar_buf[0] = 0.0f;

    FLOAT_DEF_TILE_C;
    for (dim_t cb = 0; cb < channel_blocks; ++cb)
    {
        ScalarT *block = microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for (dim_t h = 0; h < height; ++h)
        {
            ScalarT *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < width_blocks; ++wb)
            {
                ScalarT *tile = row + wb * FLOAT_W_ob * FLOAT_C_ob;
                FLOAT_LOAD_TILE_C(tile);
                FLOAT_INPLACE_EXP_horner_TILE_C(FLOAT_C_ob, tile,
                                                recip_fact_table_positive);
                FLOAT_STORE_TILE_C(tile);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(
                    step, OP_ADD, 0, reduction_buf.data());
                FLOAT_STORE_TILE_C(reduction_buf.data());
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

    for (dim_t cb = 0; cb < channel_blocks; ++cb)
    {
        ScalarT *block = microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for (dim_t h = 0; h < height; ++h)
        {
            ScalarT *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < width_blocks; ++wb)
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

FloatBuffer make_test_input(size_t num_elts)
{
    FloatBuffer input(num_elts);
    for (size_t ix = 0; ix < num_elts; ++ix)
    {
        input[ix] = -0.5f + static_cast<float>(ix) / static_cast<float>(num_elts);
    }
    return input;
}

void test_fused_global_reduction_matches_microkernel_and_softmax()
{
    constexpr dim_t channel_blocks = 2;
    constexpr dim_t height = 2;
    constexpr dim_t width_blocks = 3;
    dim_t const channels = channel_blocks * FLOAT_C_ob;
    dim_t const width = width_blocks * FLOAT_W_ob;
    size_t const num_elts = channels * height * width;

    FloatBuffer input = make_test_input(num_elts);
    FloatBuffer microkernel_output = input;
    FloatBuffer fused_output(num_elts);
    FloatBuffer softmax_output(num_elts);
    FloatBuffer scalar_buf(1);

    compute_fused_global_reduction_microkernel_lazy(
        microkernel_output, scalar_buf, channel_blocks, height, width_blocks);
    small::FusedGlobalReduction<small::OP_FUSED_SOFTMAX>(channels, height, width, input, fused_output);
    small::SoftMax(channels, height, width, input, softmax_output);

    constexpr float atol = 1.0e-6f;
    constexpr float rtol = 1.0e-5f;
    for (size_t ix = 0; ix < num_elts; ++ix)
    {
        float const fused_value = fused_output[ix];
        float const microkernel_value = microkernel_output[ix];
        float const softmax_value = softmax_output[ix];
        TEST_CHECK(std::abs(fused_value - microkernel_value) <=
                   atol + rtol * std::abs(microkernel_value));
        TEST_CHECK(std::abs(fused_value - softmax_value) <=
                   atol + rtol * std::abs(softmax_value));
    }
}

} // namespace float_detail
} // namespace small

TEST_LIST = {
#if !defined(QUANTIZED)
    {"FusedGlobalReduction matches microkernel and SoftMax",
     small::float_detail::test_fused_global_reduction_matches_microkernel_and_softmax},
#endif
    {NULL, NULL}
};
