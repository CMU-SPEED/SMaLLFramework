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

//****************************************************************************
// detail::abstract_layer<FloatBuffer,
// _G_b = 1,
// _K_b = FLOAT_C_ob,
// _F_cb = FLOAT_C_ib,
// _O_wb = FLOAT_W_ob,
// _stride = 1 or 2,
// _UNROLL = FLOAT_UNROLL,
// op_type = OP_CONV,
// op_class = 2,
// rewrite_output = 1>
//(
// G = 1
// K = num_output_channels = 16
// F_c = num_input_channels = 16
// I_h = input_height = 32
// I_w = input_width = 32
// F_h = kernel_height = 3
// F_w = kernel_width = 3
// pad t, l, r, b = ????
// I
// F
// O
// )

// kernel<ScalarT,
//        AccumT,
//        _G_b, <-- _G_b <-- 1
//        _K_b, <-- _K_b <-- FLOAT_C_ob
//        _F_cb, <-- _F_cb <-- FLOAT_C_ib
//        _O_wb, <-- _O_wb <-- FLOAT_W_ob
//        _stride, <-- 1 or 2 (choose 1)
//        _UNROLL,
//        op_type,
//        op_class>(
//           first = first,
//           F_h = F_h,
//           F_w = F_w,
//           input_col_stride = I_w * _C_ib,
//           I = I_col,
//           F = F_col,
//           O = O_col,
//           H_lb = 0,
//           H_ub = 0,
//           W_lb = 0,
//           W_ub = 0);
// _C_ob = _G_b *_K_b; <-- 1 * FLOAT_C_ob
// _C_ib = _G_b * _F_cb; <-- 1 * FLOAT_C_ib
// step = _stride * _C_ib; <-- 1 * FLOAT_C_ib
//
// FLOAT_DEF_TILE_C(_O_wb, _C_ob); <-- (FLOAT_W_ob, FLOAT_C_ob)
// first: FLOAT_ZERO_TILE_C(_O_wb, _C_ob);
// FLOAT_LOAD_TILE_C(O, _O_wb, _C_ob);
//
// *b_cur <-- *F + ??
// *a_cur <-- *I + ??
// FLOAT_ABSTRACT_OP(op_type, op_class, a_cur, b_cur, _O_wb, _C_ob);
//
//    FLOAT_CONV_TILE_C(step, a_cur, b_cur,

// NOTE We need to put the unit tests in the small::detail namespace so that
// typedefs use in macros are in the same namespace without qualification.
namespace small {
namespace detail {
//--------------------------

//***********************************
void test_correctness_CONV_TILE(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    using BufferT = FloatBuffer;
    using ScalarT = typename BufferT::value_type;

    ScalarT const filter_val = 0.5f;
    ScalarT const input_val  = 1.0f;

    dim_t const num_input_channels = 32;
    dim_t const num_output_channels = 32;
    dim_t const img_height = 16;
    dim_t const img_width = 16;
    dim_t const filter_height = 3;
    dim_t const filter_width = 3;

    size_t const FILTER_SIZE =
        num_input_channels*num_output_channels*filter_height*filter_width;
    BufferT filter_buf(FILTER_SIZE);
    for (size_t ix = 0; ix < FILTER_SIZE; ++ix) filter_buf[ix] = filter_val;

    size_t const INPUT_SIZE = num_input_channels*img_height*img_width;
    BufferT input_buf(INPUT_SIZE);
    for (size_t ix = 0; ix <  INPUT_SIZE; ++ix)  input_buf[ix] = input_val;

    size_t const OUTPUT_SIZE = num_output_channels*img_height*img_width;
    BufferT output_buf(OUTPUT_SIZE);
    for (size_t ix = 0; ix < OUTPUT_SIZE; ++ix) output_buf[ix] = 0.0f;

    std::cout << std::endl;

    ScalarT *a_cur = input_buf.data();
    ScalarT *b_cur = filter_buf.data();
    constexpr dim_t _stride = 1U;
    constexpr dim_t step = _stride * FLOAT_C_ib;
    constexpr uint32_t _UNROLL = FLOAT_UNROLL;  /// @todo move to template

    //==================================================
    FLOAT_DEF_TILE_C(FLOAT_W_ob, FLOAT_C_ob);

    // WHICH SHOULD I DO...DOES IT MATTER?
    FLOAT_ZERO_TILE_C(FLOAT_W_ob, FLOAT_C_ob); // ???
    // FLOAT_LOAD_TILE_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);

    size_t const num_trials = 1UL;
    FLOAT_CONV_TILE_C(step, a_cur, b_cur, FLOAT_W_ob, FLOAT_C_ob);

    // a cross platform way to move results to the output buffer
    FLOAT_STORE_TILE_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);
    //==================================================

    // where STORE writes is different for every platform
    for (dim_t ii = 0; ii < FLOAT_W_ob; ++ii)
    {
        for (dim_t jj = 0; jj < FLOAT_C_ob; ++jj)
        {
            size_t ix = ii*FLOAT_C_ob + jj;
            // std::cout << "CONV_TILE ix=" << ix << " = " << output_buf[ix] << std::endl;
            TEST_CHECK(output_buf[ix] == input_val*filter_val*_UNROLL*num_trials);
        }
    }
#endif

#if defined(SMALL_HAS_QUINT8_SUPPORT)
    // PUT THE EQUIVALIENT TEST FOR QUINT8 HERE
#endif
}

//****************************************************************************
void test_performance_CONV_TILE(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    using BufferT = FloatBuffer;
    using ScalarT = typename BufferT::value_type;

    ScalarT const filter_val = 0.5f;
    ScalarT const input_val  = 1.0f;

    //dim_t const num_input_channels = 64;
    //dim_t const num_output_channels = 32;
    //dim_t const img_height = 16;
    //dim_t const img_width = 16;
    //dim_t const filter_height = 3;
    //dim_t const filter_width = 3;


    uint32_t constexpr _UNROLL = FLOAT_UNROLL;
    constexpr dim_t _stride = 1U;
    constexpr dim_t step = _stride * FLOAT_C_ib;
    constexpr dim_t input_step = _UNROLL;
    constexpr dim_t input_width = FLOAT_W_ob;
    size_t const num_trials = 128; //INPUT_SIZE/input_step; (number of times the kernels is called)

    //Assume input is row major (num_trials*input_step) first, then input_width
    size_t const INPUT_SIZE = input_width * (num_trials * _UNROLL);
    // size_t const INPUT_SIZE = num_input_channels*img_height*img_width;
    BufferT input_buf(INPUT_SIZE);
    for (size_t ix = 0; ix <  INPUT_SIZE; ++ix)  input_buf[ix] = input_val;

    // Assume filter is row major (FLOAT_C_ob) first, then (num_trials*_UNROLL)
    constexpr dim_t filter_step = _UNROLL * FLOAT_C_ob;
    size_t const FILTER_SIZE = (num_trials * _UNROLL) * FLOAT_C_ob; // FLOAT_SIMD*2;;
    //size_t const FILTER_SIZE =
    //    num_input_channels*num_output_channels*filter_height*filter_width;
    BufferT filter_buf(FILTER_SIZE);
    for (size_t ix = 0; ix < FILTER_SIZE; ++ix) filter_buf[ix] = filter_val;

    //size_t const OUTPUT_SIZE = num_output_channels*img_height*img_width;
    size_t const OUTPUT_SIZE = FLOAT_W_ob*FLOAT_C_ob;
    BufferT output_buf(OUTPUT_SIZE);
    for (size_t ix = 0; ix < OUTPUT_SIZE; ++ix) output_buf[ix] = 0.0f;

    std::cout << std::endl;

    small::Timer my_timer;
    std::cout << "Num trials: " << num_trials
              << ", block size: " << input_step
              << ", input_size: " << INPUT_SIZE
              << ", output_size: " << OUTPUT_SIZE
              << ", step,W_ob,C_ob (ukernel): " << step << "," << FLOAT_W_ob
              << "," << FLOAT_C_ob
              << std::endl;

    //==================================================
    ScalarT *b_cur = filter_buf.data();

    FLOAT_DEF_TILE_C(FLOAT_W_ob, FLOAT_C_ob);

    // WHICH SHOULD I DO...DOES IT MATTER?
    FLOAT_ZERO_TILE_C(FLOAT_W_ob, FLOAT_C_ob); // ???
    //FLOAT_LOAD_TILE_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);

    //size_t a_offset = 0;

    my_timer.start();
    for (size_t iy = 0; iy < 1; ++iy)
    {
        ScalarT *a_cur = input_buf.data();
        size_t offset = 0;
        for (size_t ix = 0; ix < num_trials; ++ix)
        {
            FLOAT_CONV_TILE_C(step, a_cur, b_cur, FLOAT_W_ob, FLOAT_C_ob);
            a_cur += input_step;
            offset += input_step;
            b_cur += filter_step;
        }
    }
    my_timer.stop();
    auto elapsed = my_timer.elapsed();

    // a cross platform way to move results to the output buffer
    FLOAT_STORE_TILE_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);
    //==================================================

    std::cout << "Elapsed time: " << elapsed << " ns." << std::endl;
    // where STORE writes is different for every platform
    // std::cout << "CONV_TILE O[0] = " << output_buf[0] << std::endl;
    // std::cout << "CONV_TILE O[n] = " << output_buf[FLOAT_W_ob*FLOAT_C_ob - 1] << std::endl;
    for (dim_t ii = 0; ii < FLOAT_W_ob; ++ii)
    {
        for (dim_t jj = 0; jj < FLOAT_C_ob; ++jj)
        {
            size_t ix = ii*FLOAT_C_ob + jj;
            // std::cout << "CONV_TILE ix=" << ix << " = " << output_buf[ix] << std::endl;
            TEST_CHECK(output_buf[ix] == input_val*filter_val*num_trials*_UNROLL);
        }
    }

#endif

#if defined(SMALL_HAS_QUINT8_SUPPORT)
    // PUT THE EQUIVALIENT TEST FOR QUINT8 HERE
#endif
}

//--------------------------
} // detail
} // small

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"correctness CONV_TILE", small::detail::test_correctness_CONV_TILE},
    {"performance CONV_TILE", small::detail::test_performance_CONV_TILE},
    {NULL, NULL}
};
