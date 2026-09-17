//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2026 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//****************************************************************************

#include <acutest.h>

#include <cmath>
#include <cstddef>

#include <small.h>

namespace small {
namespace float_detail {
namespace {

constexpr std::size_t tile_elements = FLOAT_W_ob * FLOAT_C_ob;

void exp_horner(float *input, float *output)
{
    float const reciprocal_factorials[] = {
        1.0f / 720.0f, 1.0f / 120.0f, 1.0f / 24.0f, 1.0f / 6.0f,
        0.5f, 1.0f, 1.0f};

    FLOAT_DEF_TILE_C;
    FLOAT_LOAD_TILE_C(input);
    FLOAT_INPLACE_EXP_horner_TILE_C(FLOAT_C_ob, input, reciprocal_factorials);
    FLOAT_STORE_TILE_C(output);
}

void exp_negative_horner(float *input, float *output)
{
    float const reciprocal_factorials[] = {
        1.0f / 720.0f, 1.0f / 120.0f, 1.0f / 24.0f, 1.0f / 6.0f,
        0.5f, 1.0f, 1.0f};

    FLOAT_DEF_TILE_C;
    FLOAT_LOAD_TILE_C(input);
    FLOAT_INPLACE_EXP_negative_horner_TILE_C(FLOAT_C_ob, input,
                                              reciprocal_factorials);
    FLOAT_STORE_TILE_C(output);
}

void exp_range_reduced(float *input, float *output)
{
    FLOAT_DEF_TILE_C;
    FLOAT_LOAD_TILE_C(input);
    FLOAT_EXP_RR_TILE_C(input);
    FLOAT_STORE_TILE_C(output);
}

void exp_general(float *input, float *output)
{
    FLOAT_DEF_TILE_C;
    FLOAT_LOAD_TILE_C(input);
    FLOAT_EXP_GENERAL_TILE_C(input);
    FLOAT_STORE_TILE_C(output);
}

void exp_general_positive(float *input, float *output)
{
    FLOAT_DEF_TILE_C;
    FLOAT_LOAD_TILE_C(input);
    FLOAT_EXP_GENERAL_POS_TILE_C(input);
    FLOAT_STORE_TILE_C(output);
}

void exp_general_negative(float *input, float *output)
{
    FLOAT_DEF_TILE_C;
    FLOAT_LOAD_TILE_C(input);
    FLOAT_EXP_GENERAL_NEG_TILE_C(input);
    FLOAT_STORE_TILE_C(output);
}

using ExpMicrokernel = void (*)(float *, float *);

void check_exp_microkernel(char const *name,
                           ExpMicrokernel microkernel,
                           float const *values,
                           std::size_t value_count,
                           float relative_tolerance)
{
    FloatBuffer input(tile_elements);
    FloatBuffer output(tile_elements);
    for (std::size_t ix = 0; ix < tile_elements; ++ix)
    {
        input[ix] = values[ix % value_count];
    }

    microkernel(input.data(), output.data());

    for (std::size_t ix = 0; ix < tile_elements; ++ix)
    {
        float const expected = std::exp(input[ix]);
        float const relative_error =
            std::abs(output[ix] - expected) / expected;
        TEST_CHECK_(relative_error <= relative_tolerance,
                    "%s tile[%zu], input %.9g: expected %.9g, got %.9g "
                    "(relative error %.9g)",
                    name, ix, input[ix], expected, output[ix], relative_error);
    }
}

void test_exp_horner()
{
    constexpr float values[] = {-0.5f, -0.25f, -0.125f, -0.001f, -0.00001f, 0.0f,
                                0.00001f, 0.001f, 0.125f, 0.25f, 0.5f};
    check_exp_microkernel("horner", exp_horner, values,
                          sizeof(values) / sizeof(values[0]), 2.0e-5f);
}

void test_exp_negative_horner()
{
    constexpr float values[] = {-0.5f, -0.25f, -0.125f, -0.01f, -0.001f, -0.00001f, -0.0f};
    check_exp_microkernel("negative horner", exp_negative_horner, values,
                          sizeof(values) / sizeof(values[0]), 3.0e-4f);
}

void test_exp_range_reduced()
{
    constexpr float values[] = {0.0f, 0.00001f, 0.001f, 0.125f, 0.5f, 1.0f,
                                2.0f, 5.0f, 10.0f, 20.0f};
    check_exp_microkernel("range reduced", exp_range_reduced, values,
                          sizeof(values) / sizeof(values[0]), 5.0e-4f);
}

void test_exp_general()
{
    constexpr float values[] = {-20.0f, -10.0f, -1.0f, -0.5f, -0.001f, -0.00001f, 0.0f,
                                0.00001f, 0.001f, 0.5f, 1.0f, 10.0f, 20.0f};
    check_exp_microkernel("general", exp_general, values,
                          sizeof(values) / sizeof(values[0]), 5.0e-4f);
}

void test_exp_general_positive()
{
    constexpr float values[] = {0.0f, 0.00001f, 0.001f, 0.125f, 0.5f, 1.0f,
                                2.0f, 5.0f, 10.0f, 20.0f};
    check_exp_microkernel("general positive", exp_general_positive, values,
                          sizeof(values) / sizeof(values[0]), 5.0e-4f);
}

void test_exp_general_negative()
{
    constexpr float values[] = {-20.0f, -10.0f, -5.0f, -2.0f, -1.0f, -0.5f, -0.125f,
                                -0.001f, -0.00001f, -0.0f};
    check_exp_microkernel("general negative", exp_general_negative, values,
                          sizeof(values) / sizeof(values[0]), 5.0e-4f);
}

} // namespace
} // namespace float_detail
} // namespace small

TEST_LIST = {
    {"EXP Horner microkernel", small::float_detail::test_exp_horner},
    {"EXP negative Horner microkernel", small::float_detail::test_exp_negative_horner},
    {"EXP range-reduced microkernel", small::float_detail::test_exp_range_reduced},
    {"EXP general microkernel", small::float_detail::test_exp_general},
    {"EXP general positive microkernel", small::float_detail::test_exp_general_positive},
    {"EXP general negative microkernel", small::float_detail::test_exp_general_negative},
    {NULL, NULL},
};
