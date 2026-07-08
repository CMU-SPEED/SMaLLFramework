//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2026 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//****************************************************************************

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <immintrin.h>
#include <small.h>
#include <intrinsics.h>

#include "test_performance_utils.hpp"

using small::float_detail::c_tile_t;

namespace
{

volatile float g_sink = 0.0f;
constexpr std::size_t INPUT_TILES = 8;
constexpr std::size_t TILE_ELEMENTS = FLOAT_W_ob * FLOAT_C_ob;

using AlignedBuffer = small::FloatBuffer;

void fill_pattern(AlignedBuffer &buffer)
{
    for (std::size_t ix = 0; ix < buffer.size(); ++ix)
    {
        buffer[ix] = static_cast<float>((ix % 17U) + 1U) * 0.03125f;
    }
}

void fill(AlignedBuffer &buffer, float value)
{
    for (std::size_t ix = 0; ix < buffer.size(); ++ix)
    {
        buffer[ix] = value;
    }
}

float const *select_input_tile(float const *input, std::size_t run)
{
    return input + ((run & (INPUT_TILES - 1)) * TILE_ELEMENTS);
}

float checksum(float const *data, std::size_t size)
{
    float sum = 0.0f;
    for (std::size_t ix = 0; ix < size; ++ix)
    {
        sum += data[ix];
    }
    return sum;
}

__attribute__((noinline))
void run_accum_tile(float const *input, float *output, std::size_t runs)
{
    constexpr uint32_t step = FLOAT_C_ob;

    FLOAT_DEF_TILE_C;
    FLOAT_ZERO_TILE_C;
    for (std::size_t run = 0; run < runs; ++run)
    {
        FLOAT_ACCUM_TILE_C(step, input + runs *(FLOAT_W_ob * FLOAT_C_ob));
    }
    FLOAT_STORE_TILE_C(output);

    g_sink += output[0];
}

__attribute__((noinline))
void run_accum_row(float const *input, float *output, std::size_t runs)
{
    constexpr uint32_t step = FLOAT_C_ob;

    FLOAT_DEF_TILE_C;
    c0 = _mm256_setzero_ps();
    c1 = _mm256_setzero_ps();
    for (std::size_t run = 0; run < runs; ++run)
    {
        FLOAT_ACCUM_ROW_C(step, input + runs *(FLOAT_W_ob * FLOAT_C_ob));
    }
    FLOAT_STORE_ROW_C(output);

    g_sink += output[0];
}

__attribute__((noinline))
void run_accum_column(float const *input, float *output, std::size_t runs)
{
    constexpr uint32_t step = FLOAT_C_ob;

    FLOAT_DEF_TILE_C;
    FLOAT_ZERO_TILE_C;
    for (std::size_t run = 0; run < runs; ++run)
    {
        FLOAT_ACCUM_COLUMN_C(step, input + runs *(FLOAT_W_ob * FLOAT_C_ob));
    }
    _mm256_store_ps(output + 0 * FLOAT_SIMD, c0);
    _mm256_store_ps(output + 1 * FLOAT_SIMD, c2);
    _mm256_store_ps(output + 2 * FLOAT_SIMD, c4);
    _mm256_store_ps(output + 3 * FLOAT_SIMD, c6);
    _mm256_store_ps(output + 4 * FLOAT_SIMD, c8);
    _mm256_store_ps(output + 5 * FLOAT_SIMD, c10);
    g_sink += output[0];
}

__attribute__((noinline))
void run_accum_end(float const *input, float *output, std::size_t runs)
{
    constexpr uint32_t step = FLOAT_C_ob;
    constexpr uint32_t _UNROLL = FLOAT_UNROLL;

    FLOAT_DEF_END_C;
    for (std::size_t run = 0; run < runs; ++run)
    {
#if FLOAT_SIMD_EPILOGUE == 1
        float const *cur_input = select_input_tile(input, run);
        FLOAT_ACCUM_END_C(step, cur_input, output, FLOAT_W_ob, FLOAT_C_ob);
#else
        float const *cur_input = select_input_tile(input, run);
        FLOAT_ACCUM_END_C(step, cur_input, reinterpret_cast<__m256 *>(output),
                          FLOAT_W_ob, FLOAT_C_ob);
#endif
    }
    g_sink += output[0];
    (void)_UNROLL;
}

__attribute__((noinline))
void run_inplace_accum_tile(float const *input, float *output,
                            std::size_t runs)
{
    FLOAT_DEF_TILE_C;
    __m256 acc = _mm256_setzero_ps();
    for (std::size_t run = 0; run < runs; ++run)
    {
        float const *cur_input = select_input_tile(input, run);
        FLOAT_LOAD_TILE_C(cur_input);
        FLOAT_INPLACE_ACCUM_TILE_C;
        FLOAT_STORE_TILE_C(output);
        acc = _mm256_add_ps(acc, c0);
    }
    _mm256_store_ps(output, acc);
    g_sink += output[0];
}

__attribute__((noinline))
void run_inplace_accum_row(float const *input, float *output,
                           std::size_t runs)
{
    FLOAT_DEF_TILE_C;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (std::size_t run = 0; run < runs; ++run)
    {
        float const *cur_input = select_input_tile(input, run);
        FLOAT_LOAD_TILE_C(cur_input);
        FLOAT_INPLACE_ACCUM_ROW_C;
        FLOAT_STORE_ROW_C(output);
        acc0 = _mm256_add_ps(acc0, c0);
        acc1 = _mm256_add_ps(acc1, c1);
    }
    _mm256_store_ps(output, acc0);
    _mm256_store_ps(output + FLOAT_SIMD, acc1);
    g_sink += output[0];
}

__attribute__((noinline))
void run_accum_upsample_tile(float const *input, float *output,
                             std::size_t runs)
{
    constexpr uint32_t factor = 2;

    FLOAT_DEF_TILE_C;
    FLOAT_ZERO_TILE_C;
    for (std::size_t run = 0; run < runs; ++run)
    {
        float const *cur_input = select_input_tile(input, run);
        FLOAT_ACCUM_TILE_C_upsample(cur_input, factor);
        FLOAT_STORE_TILE_C(output);
    }
    g_sink += output[0];
}

struct Benchmark
{
    char const *name;
    char const *work;
    void (*fn)(float const *, float *, std::size_t);
};

struct Result
{
    char const *name;
    char const *work;
    double min_ns_per_call;
    double max_ns_per_call;
    double trimmed_avg_ns_per_call;
    double trimmed_std_dev_ns_per_call;
    double calls_per_second;
    float sum;
};

Result measure(Benchmark const &benchmark,
               AlignedBuffer const &input,
               AlignedBuffer &output,
               std::size_t runs,
               std::size_t trials)
{
    small::test::TimingStats const stats =
        small::test::time_trials(trials, [&]() {
            fill(output, 0.0f);
            benchmark.fn(input.data(), output.data(), runs);
        });

    double const min_ns_per_call = stats.min / static_cast<double>(runs);
    double const max_ns_per_call = stats.max / static_cast<double>(runs);
    double const trimmed_avg_ns_per_call =
        stats.trimmed_avg / static_cast<double>(runs);
    double const trimmed_std_dev_ns_per_call =
        stats.trimmed_std_dev / static_cast<double>(runs);

    return {benchmark.name,
            benchmark.work,
            min_ns_per_call,
            max_ns_per_call,
            trimmed_avg_ns_per_call,
            trimmed_std_dev_ns_per_call,
            1.0e9 / trimmed_avg_ns_per_call,
            checksum(output.data(), output.size())};
}

} // namespace

int main(int argc, char **argv)
{
    std::size_t runs = 10000000UL;
    std::size_t trials = 9UL;

    if (argc > 1)
    {
        runs = std::stoull(argv[1]);
    }
    if (argc > 2)
    {
        trials = std::stoull(argv[2]);
    }
    if (runs == 0 || trials == 0)
    {
        std::cerr << "runs and trials must be positive\n";
        return 1;
    }

    AlignedBuffer input(INPUT_TILES * TILE_ELEMENTS);
    AlignedBuffer output(FLOAT_W_ob * FLOAT_C_ob);
    fill_pattern(input);
    fill(output, 0.0f);

    std::vector<Benchmark> const benchmarks = {
        {"FLOAT_ACCUM_TILE_C", "accumulate full tile in registers",
         run_accum_tile},
        {"FLOAT_ACCUM_ROW_C", "reduce W rows into one register row",
         run_accum_row},
        {"FLOAT_ACCUM_COLUMN_C", "reduce SIMD halves for each row",
         run_accum_column},
        {"FLOAT_ACCUM_END_C", "accumulate full tile into memory",
         run_accum_end},
        {"FLOAT_INPLACE_ACCUM_TILE_C", "reduce loaded tile to one SIMD vector",
         run_inplace_accum_tile},
        {"FLOAT_INPLACE_ACCUM_ROW_C", "reduce loaded tile to one row",
         run_inplace_accum_row},
        {"FLOAT_ACCUM_TILE_C_upsample", "accumulate 2x upsampled tile",
         run_accum_upsample_tile},
    };

    std::vector<Result> results;
    results.reserve(benchmarks.size());
    for (Benchmark const &benchmark : benchmarks)
    {
        results.push_back(measure(benchmark, input, output, runs, trials));
    }

    double best_ns = std::numeric_limits<double>::max();
    for (Result const &result : results)
    {
        best_ns = std::min(best_ns, result.trimmed_avg_ns_per_call);
    }

    std::cout << "Zen2 ACCUM microkernel relative performance\n"
              << "  runs/trial:    " << runs << '\n'
              << "  trials:        " << trials;
    if (trials > 2)
    {
        std::cout << " (trimmed fastest and slowest for avg/stddev)";
    }
    std::cout << "\n"
              << "  tile:          W_ob=" << FLOAT_W_ob
              << ", C_ob=" << FLOAT_C_ob
              << ", SIMD=" << FLOAT_SIMD << "\n\n";

    std::cout << std::left << std::setw(30) << "microkernel"
              << std::right << std::setw(14) << "min ns/call"
              << std::setw(14) << "max ns/call"
              << std::setw(14) << "avg ns/call"
              << std::setw(14) << "stddev"
              << std::setw(16) << "M calls/s"
              << std::setw(12) << "rel"
              << "  work\n";

    for (Result const &result : results)
    {
        std::cout << std::left << std::setw(30) << result.name
                  << std::right << std::setw(14) << std::fixed
                  << std::setprecision(3) << result.min_ns_per_call
                  << std::setw(14) << std::setprecision(3)
                  << result.max_ns_per_call
                  << std::setw(14) << std::setprecision(3)
                  << result.trimmed_avg_ns_per_call
                  << std::setw(14) << std::setprecision(3)
                  << result.trimmed_std_dev_ns_per_call
                  << std::setw(16) << std::setprecision(1)
                  << result.calls_per_second * 1.0e-6
                  << std::setw(12) << std::setprecision(2)
                  << result.trimmed_avg_ns_per_call / best_ns << "x  "
                  << result.work << '\n';
    }

    std::cout << "\nchecksum sink: " << g_sink << '\n';

    return 0;
}
