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
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>

#include <small.h>
#include <small/utils/Timer.hpp>
#include <small/SoftSignLayer.hpp>

#include "test_utils.hpp"

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
            // std::cout << output_buf[ix] << " " << (float)(input_buf[ix]/(1.0f + std::abs(input_buf[ix]))) << " " << input_buf[ix] << " " << ix << std::endl;
            TEST_CHECK(output_buf[ix] == input_buf[ix]/(1.0f + std::abs(input_buf[ix])));
        }
    }
#endif 
}

void test_correctness_FLOAT_FUSED_SOFTSIGN_TILE(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    using BufferT = FloatBuffer;

    size_t const INPUT_SIZE = FLOAT_W_ob * FLOAT_C_ib;
    BufferT input_buf(INPUT_SIZE);
    for (size_t ix = 0; ix < INPUT_SIZE; ++ix) input_buf[ix] = 2.0 * ((float)rand() / RAND_MAX) - 1;

    size_t const OUTPUT_SIZE = FLOAT_W_ob * FLOAT_C_ob;
    BufferT output_buf(OUTPUT_SIZE);
    for (size_t ix = 0; ix < OUTPUT_SIZE; ++ix) output_buf[ix] = input_buf[ix];

    std::cout << std::endl;


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

void test_performance_FLOAT_SOFTSIGN_TILE(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    using BufferT = FloatBuffer;
    using ScalarT = typename BufferT::value_type;

    volatile size_t const num_trails = 65536 * 4;
    size_t const INPUT_SIZE = FLOAT_W_ob * FLOAT_C_ib * num_trails;
    BufferT input_buf(INPUT_SIZE);
    for (size_t ix = 0; ix < INPUT_SIZE; ++ix) input_buf[ix] = 2.0 * ((float)rand() / RAND_MAX) - 1;

    size_t const OUTPUT_SIZE = FLOAT_W_ob * FLOAT_C_ob;
    BufferT output_buf(OUTPUT_SIZE);
    for (size_t ix = 0; ix < OUTPUT_SIZE; ++ix) output_buf[ix] = 0.0f;

    std::cout << std::endl;

    ScalarT *a_cur = input_buf.data();
    constexpr dim_t _stride = 1U;
    constexpr dim_t step = FLOAT_C_ob * _stride;

    //==================================================
    FLOAT_DEF_TILE_C(FLOAT_W_ob, FLOAT_C_ob);

    FLOAT_ZERO_TILE_C(FLOAT_W_ob, FLOAT_C_ob);

    
    double tx(0.);
    double min_t = std::numeric_limits<double>::max();
    double max_t = 0.;
    for (volatile size_t iy = 0; iy < 10; ++iy)
    {
        Timer my_timer;
        my_timer.start();
        #pragma GCC unroll 16
        for (size_t ix = 0; ix < num_trails; ++ix)
        {
            
            FLOAT_SOFTSIGN_TILE_C(step, a_cur, FLOAT_W_ob, FLOAT_C_ob);
            a_cur += FLOAT_W_ob * FLOAT_C_ib;
            
        }
        my_timer.stop();

        auto elapsed = my_timer.elapsed();

        tx += elapsed;
        min_t = std::min(min_t, elapsed);
        max_t = std::max(max_t, elapsed);
    }
        
    std::cout << "Min time: " << min_t << " ns." << std::endl;
    std::cout << "Max time: " << max_t << " ns." << std::endl;
    std::cout << "Avg time: " << (tx / num_trails / 10) << " ns." << std::endl;

    // a cross platform way to move results to the output buffer
    FLOAT_STORE_TILE_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);
    //==================================================

#endif
}

template <typename BufferT>
BufferT create_softsign_data(size_t num_elements)
{
    std::default_random_engine generator;
    std::normal_distribution<float> distribution{0.f, 1.f};  // what distribution is Torch::Tensor::randn?

    BufferT input_buf(num_elements);

    for (size_t ix = 0; ix < num_elements; ++ix)
    {
#if defined(QUANTIZED)
        input_buf[ix] = (typename BufferT::value_type)(64*distribution(generator));
#else
        input_buf[ix] = distribution(generator);
#endif
    }

    return input_buf;
}

void test_softsign_single_element(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1;
    //size_t const kernel_size = 1;
    //size_t const stride = 1;
    //char const type = 'v';
    //size_t const C_o = 16;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;
#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_softsign_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts);
#else
    small::FloatBuffer  input_dc = create_softsign_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer  output_dc(num_input_elts);
#endif

    small::SoftSign(
        C_i, H, W,
        input_dc,
        output_dc
    );

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK(output_dc[ix] == input_dc[ix]/(1.0f + std::abs(input_dc[ix])));
        // float ref = input_dc[ix]/(1.0f + std::abs(input_dc[ix]));
        // uint32_t* ptr = reinterpret_cast<uint32_t*>(&output_dc[ix]);
        // uint32_t* ptr_ref = reinterpret_cast<uint32_t*>(&ref);
        // std::cout << ix << ": softsign(" << input_dc[ix] << ")-->"
        //          << output_dc[ix] << " " << input_dc[ix]/(1.0f + std::abs(input_dc[ix])) << std::endl;
        //          std::cout << "ptr: " << std::hex << *ptr << " ptr_ref: " << *ptr_ref << std::endl;
    }
}

void test_softsign_single_tile(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 6;
    //size_t const kernel_size = 1;
    //size_t const stride = 1;
    //char const type = 'v';
    //size_t const C_o = 16;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;

#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_softsign_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts);
#else
    small::FloatBuffer  input_dc = create_softsign_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer  output_dc(num_input_elts);
#endif

    small::SoftSign(
        C_i, H, W,
        input_dc,
        output_dc
    );

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK(output_dc[ix] == input_dc[ix]/(1.0f + std::abs(input_dc[ix])));
        // std::cout << ix << ": softsign(" << input_dc[ix] << ")-->"
        //          << output_dc[ix] << " " << input_dc[ix]/(1.0f + std::abs(input_dc[ix])) << std::endl;
    }
}


void test_softsign_large_tile(void)
{
    size_t const C_i = 16;
    size_t const H = 30;
    size_t const W = 30;
    //size_t const kernel_size = 1;
    //size_t const stride = 1;
    //char const type = 'v';
    //size_t const C_o = 16;

    //TEST_CHECK(C_i == C_o);
    size_t const num_input_elts = C_i*H*W;
#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_softsign_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts);
#else
    small::FloatBuffer  input_dc = create_softsign_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer  output_dc(num_input_elts);
#endif

    small::SoftSign(
        C_i, H, W,
        input_dc,
        output_dc
    );

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK(output_dc[ix] == input_dc[ix]/(1.0f + std::abs(input_dc[ix])));
        // std::cout << ix << ": softsign(" << input_dc[ix] << ")-->"
        //          << output_dc[ix] << " " << input_dc[ix]/(1.0f + std::abs(input_dc[ix])) << std::endl;
    }
}

void measure_softsign_performance(void)
{
    // C_i,Hi,Wi,k,s,p,C_o
    std::vector<LayerParams> params =
    {
        {  16,   48,  48, 1, 1, small::PADDING_F,   16},
        {  32,   24,  24, 1, 1, small::PADDING_F,   32},

        {  32,   48,  48, 1, 1, small::PADDING_F,   32},
        {  64,   24,  24, 1, 1, small::PADDING_F,   64},
        { 128,   12,  12, 1, 1, small::PADDING_F,  128},

        {  16,   48,  48, 1, 1, small::PADDING_F,   32},
        {  32,   24,  24, 1, 1, small::PADDING_F,   64},
        {  64,   12,  12, 1, 1, small::PADDING_F,  128},
        { 128,    6,   6, 1, 1, small::PADDING_F,  256},

        { 128,   24,  24, 1, 1, small::PADDING_F,  128},
        { 256,   12,  12, 1, 1, small::PADDING_F,  256},

        { 512,   12,  12, 1, 1, small::PADDING_F,  512},
        {1024,    6,   6, 1, 1, small::PADDING_F, 1024},

        {  32,  208, 208, 1, 1, small::PADDING_F,   64},
        {  64,  104, 104, 1, 1, small::PADDING_F,  128},
        { 128,   52,  52, 1, 1, small::PADDING_F,  256},
        { 256,   26,  26, 1, 1, small::PADDING_F,  512},
        { 512,   13,  13, 1, 1, small::PADDING_F, 1024}
    };

    uint32_t const num_threads[] = {1, 2, 4};
    char const *str_num_threads[] = {"1", "2", "4"};
    uint32_t const num_runs(100);
    small::Timer t;

#if defined(QUANTIZED)
    std::string type("quint8");
    using Buffer = small::QUInt8Buffer;
#else
    std::string type("float");
    using Buffer = small::FloatBuffer;
#endif

    printf("\nsoftsign(%s) func.\n", type.c_str());
    printf("\tC_i\tH\tW\tk\ts\tnthd\truns\tt_min\tt_max\tt_avg\n");
    
    for (LayerParams const &p : params)
    {
        size_t num_input_elts(p.C_i*p.H*p.W);

        Buffer  input_dc(num_input_elts);
        Buffer output_dc(num_input_elts);
        small::init(input_dc, num_input_elts);

        for (size_t ix = 0; ix < 3; ++ix)
        {
            setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);
            //std::string ont = std::getenv("OMP_NUM_THREADS"); // read it back
            //auto nt = atol(ont.c_str());

            double tx(0.);
            double min_t = std::numeric_limits<double>::max();
            double max_t = 0.;

            // Warmup
            small::SoftSign(p.C_i, p.H, p.W, input_dc, output_dc);

            for (size_t iy = 0; iy < num_runs; ++iy)
            {
                t.start();
                small::SoftSign(p.C_i, p.H, p.W, input_dc, output_dc);
                t.stop();
                double ts = t.elapsed();
                tx += ts;
                min_t = std::min(min_t, ts);
                max_t = std::max(max_t, ts);
            }

            printf("\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.0lf\t%.0lf\t%.0lf\n",
                   p.C_i, p.H, p.W, p.k, p.s,
                   num_threads[ix], num_runs,
                   min_t, max_t, (tx/num_runs));
        }
    }


    printf("\nsoftsign(%s) class\n", type.c_str());
    printf("\tC_i\tH\tW\tk\ts\tnthd\truns\tt_min\tt_max\tt_avg\n");

    for (LayerParams const &p: params)
    {
        size_t num_input_elts(p.C_i*p.H*p.W);
        size_t num_output_elts(p.C_i*p.H*p.W);

        small::Tensor<Buffer> input_dc({1UL, p.C_i, p.H, p.W});
        small::init(input_dc.buffer(), num_input_elts);

        small::Tensor<Buffer> output_dc(num_output_elts);

        small::shape_type input_shape({1UL, p.C_i, p.H, p.W});
        small::SoftSignLayer<Buffer> softsign_layer(input_shape);

        for (size_t ix = 0; ix < 3; ++ix)
        {
            setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);
            //std::string ont = std::getenv("OMP_NUM_THREADS");
            //auto nt = atol(ont.c_str());

            double tx(0.);
            double min_t = std::numeric_limits<double>::max();
            double max_t = 0.;

            // Warm up
            softsign_layer.compute_output({&input_dc}, &output_dc);

            for (size_t iy = 0; iy < num_runs; ++iy)
            {
                t.start();
                softsign_layer.compute_output({&input_dc}, &output_dc);
                t.stop();
                double ts = t.elapsed();
                tx += ts;
                min_t = std::min(min_t, ts);
                max_t = std::max(max_t, ts);
            }

            printf("\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.0lf\t%.0lf\t%.0lf\n",
                p.C_i, p.H, p.W, p.k, p.s,
                num_threads[ix], num_runs,
                min_t, max_t, (tx/num_runs));
        }
    }
    
}

}
}

TEST_LIST = {
    {"correctness FLOAT_SOFTSIGN_TILE",
     small::float_detail::test_correctness_FLOAT_SOFTSIGN_TILE},
    // {"correctness FLOAT_FUSED_SOFTSIGN_TILE",
    //  small::float_detail::test_correctness_FLOAT_FUSED_SOFTSIGN_TILE},
    {"performance FLOAT_SOFTSIGN_TILE",
     small::float_detail::test_performance_FLOAT_SOFTSIGN_TILE},
    // {"softsign single element",
    //  small::float_detail::test_softsign_single_element},
    // {"softsign single tile",
    //  small::float_detail::test_softsign_single_tile},
    // {"softsign large tile",
    //  small::float_detail::test_softsign_large_tile},
    {"softsign performance",
     small::float_detail::measure_softsign_performance},
     {NULL, NULL}
};
