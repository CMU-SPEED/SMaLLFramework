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

#include <acutest.h>

#include <small.h>
#include <small/utils/Timer.hpp>

namespace small {
namespace float_detail {

void test_correctness_FLOAT_SOFTSIGN_TILE(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    using BufferT = FloatBuffer;
    using ScalarT = typename BufferT::value_type;

    size_t const INPUT_SIZE = FLOAT_W_ob * FLOAT_C_ib;
    BufferT input_buf(INPUT_SIZE);
    for (size_t ix = 0; ix < INPUT_SIZE; ++ix) input_buf[ix] = 2.0 * ((float)rand() / RAND_MAX) - 1;

    size_t const OUTPUT_SIZE = FLOAT_W_ob * FLOAT_C_ob;
    BufferT output_buf(OUTPUT_SIZE);
    for (size_t ix = 0; ix < OUTPUT_SIZE; ++ix) output_buf[ix] = 0.0f;

    std::cout << std::endl;

    ScalarT *a_cur = input_buf.data();
    constexpr dim_t _stride = 1U;
    constexpr dim_t step = FLOAT_C_ob * _stride;
    constexpr uint32_t _UNROLL = FLOAT_UNROLL;  /// @todo move to template

    //==================================================
    FLOAT_DEF_TILE_C(FLOAT_W_ob, FLOAT_C_ob);

    FLOAT_ZERO_TILE_C(FLOAT_W_ob, FLOAT_C_ob);
    // FLOAT_LOAD_TILE_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);

    FLOAT_SOFTSIGN_TILE_C(step, a_cur, FLOAT_W_ob, FLOAT_C_ob);

    FLOAT_STORE_TILE_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);
    //==================================================

    for (dim_t ii = 0; ii < FLOAT_W_ob; ++ii)
    {
        for (dim_t jj = 0; jj < FLOAT_C_ob; ++jj)
        {
            size_t ix = ii*FLOAT_C_ob + jj;
            //std::cout << output_buf[ix] << " " << (float)(input_buf[ix]/(1.0f + std::abs(input_buf[ix]))) << " " << input_buf[ix] << " " << ix << std::endl;
            TEST_CHECK(output_buf[ix] == input_buf[ix]/(1.0f + std::abs(input_buf[ix])));
        }
    }
#endif 
}

void test_correctness_FLOAT_FUSED_SOFTSIGN_TILE(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    using BufferT = FloatBuffer;
    using ScalarT = typename BufferT::value_type;

    size_t const INPUT_SIZE = FLOAT_W_ob * FLOAT_C_ib;
    BufferT input_buf(INPUT_SIZE);
    for (size_t ix = 0; ix < INPUT_SIZE; ++ix) input_buf[ix] = 2.0 * ((float)rand() / RAND_MAX) - 1;

    size_t const OUTPUT_SIZE = FLOAT_W_ob * FLOAT_C_ob;
    BufferT output_buf(OUTPUT_SIZE);
    for (size_t ix = 0; ix < OUTPUT_SIZE; ++ix) output_buf[ix] = input_buf[ix];

    std::cout << std::endl;

    ScalarT *a_cur = input_buf.data();
    constexpr dim_t _stride = 1U;
    constexpr dim_t step = FLOAT_C_ob * _stride;
    constexpr uint32_t _UNROLL = FLOAT_UNROLL;  /// @todo move to template

    //==================================================
    FLOAT_DEF_TILE_C(FLOAT_W_ob, FLOAT_C_ob);

    FLOAT_LOAD_TILE_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);

    FLOAT_FUSED_SOFTSIGN_TILE_C(FLOAT_W_ob, FLOAT_C_ob);

    FLOAT_STORE_TILE_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);
    //==================================================

    for (dim_t ii = 0; ii < FLOAT_W_ob; ++ii)
    {
        for (dim_t jj = 0; jj < FLOAT_C_ob; ++jj)
        {
            size_t ix = ii*FLOAT_C_ob + jj;
            //std::cout << output_buf[ix] << " " << (float)(input_buf[ix]/(1.0f + std::abs(input_buf[ix]))) << " " << input_buf[ix] << " " << ix << std::endl;
            TEST_CHECK(output_buf[ix] == input_buf[ix]/(1.0f + std::abs(input_buf[ix])));
        }
    }
#endif 
}


}
}

TEST_LIST = {
    {"correctness FLOAT_SOFTSIGN_TILE",
     small::float_detail::test_correctness_FLOAT_SOFTSIGN_TILE},
    {"correctness FLOAT_FUSED_SOFTSIGN_TILE",
     small::float_detail::test_correctness_FLOAT_FUSED_SOFTSIGN_TILE},
    {NULL, NULL}
};
