//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2026 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//****************************************************************************

#include <acutest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <cstddef>
#include <vector>

#include <small.h>
#include <small/float_detail/abstract_op.hpp>
#include <small/float_detail/abstract_layer_fused_global_reduction.hpp>
#include "test_performance_utils.hpp"

namespace small
{
namespace float_detail
{

// ADD-only global-reduction implementations from test_ukernel_SOFTMAX_STAGES.cpp.
void compute_add_stages(
    FloatBuffer const &microkernel_buf, FloatBuffer &scalar_buf,
    dim_t channel_blocks, dim_t height, dim_t width_blocks)
{
    using ScalarT = typename FloatBuffer::value_type;
    dim_t const width = width_blocks * FLOAT_W_ob;
    constexpr dim_t step = FLOAT_C_ob;
    constexpr dim_t _UNROLL = FLOAT_C_ob;

    scalar_buf[0] = 0.0f;
    for (dim_t cb = 0; cb < channel_blocks; ++cb)
    {
        ScalarT const *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for (dim_t h = 0; h < height; ++h)
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

void compute_max_stages(
    FloatBuffer const &microkernel_buf, FloatBuffer &scalar_buf,
    dim_t channel_blocks, dim_t height, dim_t width_blocks)
{
    using ScalarT = typename FloatBuffer::value_type;
    dim_t const width = width_blocks * FLOAT_W_ob;
    constexpr dim_t step = FLOAT_C_ob;
    constexpr dim_t _UNROLL = FLOAT_C_ob;

    scalar_buf[0] = microkernel_buf[0];
    for (dim_t cb = 0; cb < channel_blocks; ++cb)
    {
        ScalarT const *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for (dim_t h = 0; h < height; ++h)
        {
            ScalarT const *row = block + h * width * FLOAT_C_ob;
            for (dim_t w = 0; w < width; ++w)
            {
                ScalarT const *a_cur = row + w * FLOAT_C_ob;
                FLOAT_ABSTRACT_FULL_REDUCTION_OP_END(
                    step, OP_MAX_POOL, 3, a_cur, scalar_buf.data(), nullptr,
                    scalar_buf.data(), 1, 1);
            }
        }
    }
}



void compute_add_lazy(
    FloatBuffer const &microkernel_buf, FloatBuffer &scalar_buf,
    dim_t channel_blocks, dim_t height, dim_t width_blocks)
{
    using ScalarT = typename FloatBuffer::value_type;
    dim_t const width = width_blocks * FLOAT_W_ob;
    constexpr dim_t step = FLOAT_C_ob;

    scalar_buf[0] = 0.0f;
    FloatBuffer reduction_buf(FLOAT_W_ob * FLOAT_C_ob);
    FLOAT_DEF_TILE_C;
    FLOAT_ZERO_TILE_C;
    for (dim_t cb = 0; cb < channel_blocks; ++cb)
    {
        ScalarT const *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for (dim_t h = 0; h < height; ++h)
        {
            ScalarT const *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < width_blocks; ++wb)
            {
                ScalarT const *a_cur = row + wb * FLOAT_W_ob * FLOAT_C_ob;
                FLOAT_LOAD_TILE_C(a_cur);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(
                    step, OP_ADD, 0, reduction_buf.data());
                FLOAT_STORE_TILE_C(reduction_buf.data());
            }
        }
    }
    FLOAT_ABSTRACT_REDUCTION_OP_TILE(
        OP_ADD, reduction_buf.data(), scalar_buf.data());
}

void compute_max_lazy(
    FloatBuffer const &microkernel_buf, FloatBuffer &scalar_buf,
    dim_t channel_blocks, dim_t height, dim_t width_blocks)
{
    using ScalarT = typename FloatBuffer::value_type;
    dim_t const width = width_blocks * FLOAT_W_ob;
    constexpr dim_t step = FLOAT_C_ob;

    scalar_buf[0] = 0.0f;
    FloatBuffer reduction_buf(FLOAT_W_ob * FLOAT_C_ob);
    FLOAT_DEF_TILE_C;
    // Max: Identity max(x, x) = x
    FLOAT_LOAD_TILE_C(microkernel_buf.data());
    FLOAT_STORE_TILE_C(reduction_buf.data());
    for (dim_t cb = 0; cb < channel_blocks; ++cb)
    {
        ScalarT const *block =
            microkernel_buf.data() + cb * height * width * FLOAT_C_ob;
        for (dim_t h = 0; h < height; ++h)
        {
            ScalarT const *row = block + h * width * FLOAT_C_ob;
            for (dim_t wb = 0; wb < width_blocks; ++wb)
            {
                ScalarT const *a_cur = row + wb * FLOAT_W_ob * FLOAT_C_ob;
                FLOAT_LOAD_TILE_C(a_cur);
                FLOAT_ABSTRACT_SINGLE_ELEMENT_OP_TILE(
                    step, OP_MAX_POOL, 0, reduction_buf.data());
                FLOAT_STORE_TILE_C(reduction_buf.data());
    
            }
        }
    }

    FLOAT_ABSTRACT_REDUCTION_OP_TILE(
        OP_MAX_POOL, reduction_buf.data(), scalar_buf.data());

}



FloatBuffer make_reduction_input(size_t elements)
{
    FloatBuffer input(elements);
    for (size_t ix = 0; ix < elements; ++ix)
    {
        // input[ix] = static_cast<float>(ix)*(2.0/(elements*1.0)) - 1.0f;
        // random init in a range - 1 to 1
        input[ix] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f));
    }
    return input;
}

float reference_sum(FloatBuffer const &input)
{
    float sum = 0.0f;
    for (size_t ix = 0; ix < input.size(); ++ix)
    {
        sum += input[ix];
    }
    return sum;
}

float reference_max(FloatBuffer const &input)
{
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t ix = 0; ix < input.size(); ++ix)
    {
        maximum = std::max(maximum, input[ix]);
    }
    return maximum;
}

FloatBuffer make_reduction_input(size_t total_elements,
                                 size_t actual_elements,
                                 float pad_value = 0.0f)
{
    FloatBuffer input(total_elements);
    size_t ix = 0;
    for (; ix < actual_elements; ++ix)
    {
        input[ix] = -1.0f + static_cast<float>(rand()) /
                   (static_cast<float>(RAND_MAX / 2.0f));
    }
    for (; ix < total_elements; ++ix)
    {
        input[ix] = pad_value;
    }
    return input;
}

template <typename Callable>
small::test::TimingStats time_global_reduction_trials(
    size_t num_runs,
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
        // std::chrono::duration<double> elapsed = stop - start;
        times.push_back(timer.elapsed() / static_cast<double>(num_trials));
    }

    return small::test::summarize_times(std::move(times));
}

template <OpType OP>
void run_fused_global_reduction(dim_t ChannelBlocks, dim_t Height, dim_t WidthBlocks)
{
    dim_t const channels = ChannelBlocks * FLOAT_C_ob;
    dim_t const height = Height;
    dim_t const width = WidthBlocks * FLOAT_W_ob;
    size_t const num_elts = static_cast<size_t>(channels) * height * width;

    FloatBuffer input = make_reduction_input(num_elts);
    FloatBuffer output(1);

    fused_global_reduction_abstract_layer<FloatBuffer, FLOAT_C_ob, FLOAT_W_ob, OP>(
        channels, height, width, &input, &output);

    if constexpr (OP == OP_ADD)
    {
        float const expected = reference_sum(input);
        float const actual = output[0];
        float const abs_error = std::abs(actual - expected);
        float const abs_tolerance = 1.0e-5f;
        float const rel_error = abs_error / (std::abs(expected) + 1.0e-12f);
        bool const close = (abs_error < abs_tolerance) || (rel_error < 1.0e-5f);
        TEST_CHECK(close == 1);
    }
    else if constexpr (OP == OP_MAX_POOL)
    {
        float const expected = reference_max(input);
        float const actual = output[0];
        TEST_CHECK(actual == expected);
    }
}

void measure_global_reduction_performance()
{
    struct ProblemSize { dim_t channels; dim_t height; dim_t width; };

    std::vector<ProblemSize> const params =
    {
        {  16,   48,  48},
        {  32,   24,  24},
        {  32,   48,  48},
        {  64,   24,  24},
        { 128,   12,  12},
        {  16,   48,  48},
        {  32,   24,  24},
        {  64,   12,  12},
        { 128,    6,   6},
        { 128,   24,  24},
        { 256,   12,  12},
        { 512,   12,  12},
        {1024,    6,   6},
        {16, 1, 6}
    };

    constexpr size_t num_runs = 100;
    constexpr size_t num_trials = 1000;

    printf("\nglobal reduction (float) add stages\n");
    printf("\tC\tH\tW\ttrials\truns\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n");
    for (ProblemSize const &p : params)
    {
        dim_t const ChannelBlocks = p.channels / FLOAT_C_ob;
        dim_t const WidthBlocks = (p.width + FLOAT_W_ob - 1) / FLOAT_W_ob;
        size_t const padded_width = WidthBlocks * FLOAT_W_ob;
        size_t const total_elements =
            static_cast<size_t>(p.channels) * p.height * padded_width;
        size_t const actual_elements =
            static_cast<size_t>(p.channels) * p.height * p.width;

        FloatBuffer input = make_reduction_input(total_elements,
                                                actual_elements,
                                                0.0f);
        FloatBuffer output(1);

        compute_max_stages(input, output, ChannelBlocks, p.height, WidthBlocks);

        small::test::TimingStats const stats =
            time_global_reduction_trials(num_runs, num_trials, [&]() {
                compute_max_stages(input, output, ChannelBlocks, p.height,
                                   WidthBlocks);
            });

        printf("add_stages\t%u\t%u\t%u\t%zu\t%zu\t%lf\t%lf\t%lf\t%lf\n",
               static_cast<unsigned>(p.channels),
               static_cast<unsigned>(p.height),
               static_cast<unsigned>(p.width),
               num_trials,
               num_runs,
               stats.min,
               stats.max,
               stats.trimmed_avg,
               stats.trimmed_std_dev);
    }

    printf("\nglobal reduction (float) add lazy\n");
    printf("\tC\tH\tW\ttrials\truns\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n");
    for (ProblemSize const &p : params)
    {
        dim_t const ChannelBlocks = p.channels / FLOAT_C_ob;
        dim_t const WidthBlocks = (p.width + FLOAT_W_ob - 1) / FLOAT_W_ob;
        size_t const padded_width = WidthBlocks * FLOAT_W_ob;
        size_t const total_elements =
            static_cast<size_t>(p.channels) * p.height * padded_width;
        size_t const actual_elements =
            static_cast<size_t>(p.channels) * p.height * p.width;

        FloatBuffer input = make_reduction_input(total_elements,
                                                actual_elements,
                                                0.0f);
        FloatBuffer output(1);

        compute_max_lazy(input, output, ChannelBlocks, p.height, WidthBlocks);

        small::test::TimingStats const stats =
            time_global_reduction_trials(num_runs, num_trials, [&]() {
                compute_max_lazy(input, output, ChannelBlocks, p.height,
                                 WidthBlocks);
            });

        printf("add_lazy\t%u\t%u\t%u\t%zu\t%zu\t%lf\t%lf\t%lf\t%lf\n",
               static_cast<unsigned>(p.channels),
               static_cast<unsigned>(p.height),
               static_cast<unsigned>(p.width),
               num_trials,
               num_runs,
               stats.min,
               stats.max,
               stats.trimmed_avg,
               stats.trimmed_std_dev);
    }
}

void measure_fused_global_reduction_performance()
{
    struct ProblemSize { dim_t channels; dim_t height; dim_t width; };

    std::vector<ProblemSize> const params =
    {
        {  16,   48,  48},
        {  32,   24,  24},
        {  32,   48,  48},
        {  64,   24,  24},
        { 128,   12,  12},
        {  16,   48,  48},
        {  32,   24,  24},
        {  64,   12,  12},
        { 128,    6,   6},
        { 128,   24,  24},
        { 256,   12,  12},
        { 512,   12,  12},
        {1024,    6,   6},
    };

    constexpr size_t num_runs = 100;
    constexpr size_t num_trials = 1000;

    printf("\nfused global reduction (float) performance\n");
    printf("\top\tC\tH\tW\ttrials\truns\tt_min\tt_max\tt_avg_trim\tstd_dev_trim\n");

    for (ProblemSize const &p : params)
    {
        TEST_ASSERT(p.channels % FLOAT_C_ob == 0);
        TEST_ASSERT(p.width % FLOAT_W_ob == 0);

        dim_t const ChannelBlocks = p.channels / FLOAT_C_ob;
        dim_t const WidthBlocks = p.width / FLOAT_W_ob;
        dim_t const channels = p.channels;
        dim_t const height = p.height;
        dim_t const width = p.width;
        size_t const num_elts = static_cast<size_t>(channels) * height * width;

        // Correctness checks
        run_fused_global_reduction<OP_ADD>(ChannelBlocks, height, WidthBlocks);
        run_fused_global_reduction<OP_MAX_POOL>(ChannelBlocks, height, WidthBlocks);

        // Prepare buffers for timing
        FloatBuffer input = make_reduction_input(num_elts);
        FloatBuffer output(1);

        // Time ADD
        small::test::TimingStats const add_stats =
            time_global_reduction_trials(num_runs, num_trials, [&]() {
                fused_global_reduction_abstract_layer<FloatBuffer, FLOAT_C_ob, FLOAT_W_ob, OP_ADD>(
                    channels, height, width, &input, &output);
            });

        printf("ADD\t%u\t%u\t%u\t%zu\t%zu\t%lf\t%lf\t%lf\t%lf\n",
               static_cast<unsigned>(p.channels),
               static_cast<unsigned>(p.height),
               static_cast<unsigned>(p.width),
               num_trials,
               num_runs,
               add_stats.min,
               add_stats.max,
               add_stats.trimmed_avg,
               add_stats.trimmed_std_dev);

        // Time MAX
        small::test::TimingStats const max_stats =
            time_global_reduction_trials(num_runs, num_trials, [&]() {
                fused_global_reduction_abstract_layer<FloatBuffer, FLOAT_C_ob, FLOAT_W_ob, OP_MAX_POOL>(
                    channels, height, width, &input, &output);
            });

        printf("MAX\t%u\t%u\t%u\t%zu\t%zu\t%lf\t%lf\t%lf\t%lf\n",
               static_cast<unsigned>(p.channels),
               static_cast<unsigned>(p.height),
               static_cast<unsigned>(p.width),
               num_trials,
               num_runs,
               max_stats.min,
               max_stats.max,
               max_stats.trimmed_avg,
               max_stats.trimmed_std_dev);
    }
}

void test_lazy_max_reduction()
{
    constexpr dim_t channel_blocks = 2;
    constexpr dim_t height = 3;
    constexpr dim_t width_blocks = 2;
    size_t const elements = channel_blocks * FLOAT_C_ob * height *
                            width_blocks * FLOAT_W_ob;
    FloatBuffer input = make_reduction_input(elements);
    FloatBuffer output(1);

    compute_max_lazy(
        input, output, channel_blocks, height, width_blocks);

    TEST_CHECK(output[0] == reference_max(input));
}
void test_staged_max_reduction()
{
    constexpr dim_t channel_blocks = 2;
    constexpr dim_t height = 3;
    constexpr dim_t width_blocks = 2;
    size_t const elements = channel_blocks * FLOAT_C_ob * height *
                            width_blocks * FLOAT_W_ob;
    FloatBuffer input = make_reduction_input(elements);
    FloatBuffer output(1);

    compute_max_stages(
        input, output, channel_blocks, height, width_blocks);

    TEST_CHECK(output[0] == reference_max(input));
}


void test_staged_add_only_global_reduction()
{
    constexpr dim_t channel_blocks = 2;
    constexpr dim_t height = 3;
    constexpr dim_t width_blocks = 2;
    size_t const elements = channel_blocks * FLOAT_C_ob * height *
                            width_blocks * FLOAT_W_ob;
    FloatBuffer input = make_reduction_input(elements);
    FloatBuffer output(1);

    compute_add_stages(
        input, output, channel_blocks, height, width_blocks);

    TEST_CHECK(std::abs(output[0] - reference_sum(input)) <= 1.0e-5f);
}

void test_lazy_add_only_global_reduction()
{
    constexpr dim_t channel_blocks = 2;
    constexpr dim_t height = 3;
    constexpr dim_t width_blocks = 2;
    size_t const elements = channel_blocks * FLOAT_C_ob * height *
                            width_blocks * FLOAT_W_ob;
    FloatBuffer input = make_reduction_input(elements);
    FloatBuffer output(1);

    compute_add_lazy(
        input, output, channel_blocks, height, width_blocks);

    float const expected = output[0];
    float const abs_tolerance = 1.0e-5f;
    float const rel_tolerance = 1.0e-5f;
    float const actual = reference_sum(input);
    float const abs_error = std::abs(actual - expected);
    float const rel_error = abs_error / (std::abs(expected) + 1.0e-12f);
    bool const close = (abs_error < abs_tolerance) ||
                        (rel_error < rel_tolerance);
    TEST_CHECK(close == 1);
}

void test_fused_global_reduction_add()
{
    constexpr dim_t channels = 32;
    constexpr dim_t height = 4;
    constexpr dim_t width = 12;
    size_t const elements = channels * height * width;

    FloatBuffer input = make_reduction_input(elements);
    FloatBuffer output(1);

    // Call fused global reduction with ADD operation
    small::float_detail::fused_global_reduction_abstract_layer<
        FloatBuffer, FLOAT_C_ob, FLOAT_W_ob, OP_ADD>(
        channels, height, width, &input, &output);

    float const expected = reference_sum(input);
    float const actual = output[0];
    float const abs_error = std::abs(actual - expected);
    float const abs_tolerance = 1.0e-5f;
    float const rel_tolerance = 1.0e-5f;
    float const rel_error = abs_error / (std::abs(expected) + 1.0e-12f);
    bool const close = (abs_error < abs_tolerance) ||
                       (rel_error < rel_tolerance);
    TEST_CHECK(close == 1);
}

void test_fused_global_reduction_max()
{
    constexpr dim_t channels = 32;
    constexpr dim_t height = 4;
    constexpr dim_t width = 12;
    size_t const elements = channels * height * width;

    FloatBuffer input = make_reduction_input(elements);
    FloatBuffer output(1);

    // Call fused global reduction with MAX operation
    small::float_detail::fused_global_reduction_abstract_layer<
        FloatBuffer, FLOAT_C_ob, FLOAT_W_ob, OP_MAX_POOL>(
        channels, height, width, &input, &output);

    float const expected = reference_max(input);
    float const actual = output[0];
    TEST_CHECK(actual == expected);
}

} // namespace float_detail
} // namespace small

TEST_LIST = {
#if !defined(QUANTIZED)
    {"Staged MAX reduction (temporary ADD-only path)",
     small::float_detail::test_staged_max_reduction},
    {"Lazy MAX reduction (temporary ADD-only path)",
     small::float_detail::test_lazy_max_reduction},
    {"staged ADD-only global reduction",
     small::float_detail::test_staged_add_only_global_reduction},
    {"lazy ADD-only global reduction",
     small::float_detail::test_lazy_add_only_global_reduction},
    // {"fused global reduction ADD",
    //  small::float_detail::test_fused_global_reduction_add},
    // {"fused global reduction MAX",
    //  small::float_detail::test_fused_global_reduction_max},
    //  {"fused global reduction performance",
    //   small::float_detail::measure_fused_global_reduction_performance},
#endif
    {"global reduction performance",
     small::float_detail::measure_global_reduction_performance},
    {NULL, NULL}
};
