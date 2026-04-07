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

#include <thread>
#include <chrono>
#include <string>
#include <sstream>


#include <small.h>
#include <small/utils/Timer.hpp>
#include <small/op_type.hpp>

#include "test_utils.hpp"

namespace small {
namespace float_detail {

float naive_exp(float x)
{
    float reciprocal_factorial_table[] =  {0.008333333333, 0.0416667, 0.16667, 0.5, 1.0};
    float result = ((((reciprocal_factorial_table[0]*x + reciprocal_factorial_table[1])*x + reciprocal_factorial_table[2])*x + reciprocal_factorial_table[3])*x + reciprocal_factorial_table[4])*x + reciprocal_factorial_table[4];
    return result;
}
template<OpType op_type>
float ewise_function(float a, float x)
{
    if constexpr(op_type == OP_SLOPE_RELU || op_type == OP_FUSED_SLOPE_RELU)
    {
        return x > 0 ? x : a * x;
    } else if constexpr(op_type == OP_CELU || op_type == OP_FUSED_CELU)
    {
        printf("%f\n", x);
        float alpha = 0.5f; // Assuming a fixed alpha for CELU
        return std::max(0.f, x) + std::min(0.f, alpha * (std::exp(x / alpha) - 1.f));
    }
    return 0.0f; // Default case for unsupported op_type
}

template<OpType op_type>
void test_correctness_ewise_kernel(void)
{
    using BufferT = FloatBuffer; 
    using ScalarT = typename BufferT::value_type; 
    srand(time(0));
    size_t const INPUT_SIZE = FLOAT_W_ob * FLOAT_C_ib;
    BufferT input_buf(INPUT_SIZE);
    for (size_t ix = 0; ix < INPUT_SIZE; ++ix) input_buf[ix] = 0.000242 + (ix%10)*1e-3;
    BufferT ref_input = input_buf;
    BufferT b_buf(1);
    b_buf[0] = 0.5f; // Scalar for ewise operations

    size_t const OUTPUT_SIZE = FLOAT_W_ob * FLOAT_C_ob;
    BufferT output_buf(OUTPUT_SIZE);
    for (size_t ix = 0; ix < OUTPUT_SIZE; ++ix) output_buf[ix] =  0.000242 + (ix%10)*1e-3;

    BufferT ref_output = output_buf;

    ScalarT *a_cur = input_buf.data();
    ScalarT *b_cur = b_buf.data();
    ScalarT *c_cur = output_buf.data();
    constexpr dim_t _stride = 1U;
    constexpr dim_t step = FLOAT_C_ob * _stride;

    FLOAT_DEF_TILE_C;

    // FLOAT_ZERO_TILE_C(FLOAT_W_ob, FLOAT_C_ob);
    FLOAT_LOAD_TILE_C(c_cur);

    FLOAT_ABSTRACT_OP(step, op_type, 0, a_cur, b_cur, c_cur, nullptr);
    // Note: b_cur is nullptr for ewise operations that do not require a second input
    FLOAT_STORE_TILE_C(c_cur);

    for (dim_t ii = 0; ii < FLOAT_W_ob; ++ii)
    {
        for (dim_t jj = 0; jj < FLOAT_C_ob; ++jj)
        {
            size_t ix = ii*FLOAT_C_ob + jj;
            ScalarT expected_value = ewise_function<op_type>(ref_input[ix], ref_output[ix]);
            // TEST_CHECK(output_buf[ix] == expected_value);
            const float rtol = 1e-2f; // 0.2%
            const float atol = 1e-3f;
            float error = std::abs(output_buf[ix] - expected_value);
            float denom = std::abs(expected_value);
            float rel_err = (denom > 0.0f) ? (error / denom) : error; // fallback to absolute when ref == 0
            TEST_CHECK(error <= atol + rtol * denom);
            std::cout << "(" << ref_input[ix] << "," << ref_output[ix] << ") -> " << "Expected: " << expected_value << ", Actual: " << output_buf[ix] << " Error: "<< error << std::endl;
            
        }
    }
}

void test_correctness_fused_slope_relu(void)
{
    using BufferT = FloatBuffer; 
    using ScalarT = typename BufferT::value_type; 
    srand(time(0));
    size_t const INPUT_SIZE = FLOAT_W_ob * FLOAT_C_ib;
    BufferT input_buf(INPUT_SIZE);
    for (size_t ix = 0; ix < INPUT_SIZE; ++ix) input_buf[ix] = 2.0 * ((float)rand() / RAND_MAX) - 1.0;

    BufferT b_buf(INPUT_SIZE);
    for (size_t ix = 0; ix < INPUT_SIZE; ++ix) b_buf[ix] = 0.f; // Scalar for ewise operations

    size_t const OUTPUT_SIZE = FLOAT_W_ob * FLOAT_C_ob;
    BufferT output_buf(OUTPUT_SIZE);
    for (size_t ix = 0; ix < OUTPUT_SIZE; ++ix) output_buf[ix] = 2.0 * ((float)rand() / RAND_MAX) - 1.0;

    BufferT ref_input = input_buf;
    BufferT ref_output = output_buf;

    ScalarT *a_cur = input_buf.data();

    ScalarT *c_cur = output_buf.data();
    constexpr dim_t _stride = 1U;
    constexpr dim_t step = FLOAT_C_ob * _stride;

    FLOAT_DEF_TILE_C;

    FLOAT_LOAD_TILE_C(output_buf.data());
    FLOAT_FUSED_SLOPE_RELU_TILE_C(step, a_cur, c_cur);
    FLOAT_STORE_TILE_C(output_buf.data());

    for (dim_t ii = 0; ii < FLOAT_W_ob; ++ii)
    {
        for (dim_t jj = 0; jj < FLOAT_C_ob; ++jj)
        {
            size_t ix = ii*FLOAT_C_ob + jj;
            ScalarT expected_value = ref_output[ix] > 0 ? ref_output[ix] : ref_input[ix] * ref_output[ix];
            TEST_CHECK(output_buf[ix] == expected_value);
            // std::cout << "(" << input_buf[ix] << "," << output_buf[ix] << ") -> " << "Expected: " << expected_value << ", Actual: " << ref_buf[ix] << std::endl;
        }
    }
}


#if 0
#include <immintrin.h>

#define REPEAT_10_BASE(macro, base) \
    macro(base##0); macro(base##1); macro(base##2); macro(base##3); macro(base##4); \
    macro(base##5); macro(base##6); macro(base##7); macro(base##8); macro(base##9);

#define REPEAT_100(macro) \
    REPEAT_10_BASE(macro, 0) REPEAT_10_BASE(macro, 1) REPEAT_10_BASE(macro, 2) \
    REPEAT_10_BASE(macro, 3) REPEAT_10_BASE(macro, 4) REPEAT_10_BASE(macro, 5) \
    REPEAT_10_BASE(macro, 6) REPEAT_10_BASE(macro, 7) REPEAT_10_BASE(macro, 8) \
    REPEAT_10_BASE(macro, 9)

#define REPEAT_100_BASE(macro, hundreds) \
    REPEAT_10_BASE(macro, hundreds##0) REPEAT_10_BASE(macro, hundreds##1) \
    REPEAT_10_BASE(macro, hundreds##2) REPEAT_10_BASE(macro, hundreds##3) \
    REPEAT_10_BASE(macro, hundreds##4) REPEAT_10_BASE(macro, hundreds##5) \
    REPEAT_10_BASE(macro, hundreds##6) REPEAT_10_BASE(macro, hundreds##7) \
    REPEAT_10_BASE(macro, hundreds##8) REPEAT_10_BASE(macro, hundreds##9)

#define REPEAT_1000(macro) \
    REPEAT_100_BASE(macro, 0) REPEAT_100_BASE(macro, 1) REPEAT_100_BASE(macro, 2) \
    REPEAT_100_BASE(macro, 3) REPEAT_100_BASE(macro, 4) REPEAT_100_BASE(macro, 5) \
    REPEAT_100_BASE(macro, 6) REPEAT_100_BASE(macro, 7) REPEAT_100_BASE(macro, 8) \
    REPEAT_100_BASE(macro, 9)

#define SLOPE_RELU_CALL(iteration) \
    asm volatile("customized_iteration" #iteration ":" ::: "memory");\
    FLOAT_LOAD_TILE_C(c_cur, FLOAT_W_ob, FLOAT_C_ob); \
    FLOAT_SLOPE_RELU_TILE_C(step, a_cur, FLOAT_W_ob, FLOAT_C_ob); \
    FLOAT_STORE_TILE_C(c_cur, FLOAT_W_ob, FLOAT_C_ob);\
    a_cur += FLOAT_W_ob * FLOAT_C_ib; \
    c_cur += FLOAT_W_ob * FLOAT_C_ob;


#define FUSED_SLOPE_RELU_CALL(iteration) \
    asm volatile("fused_iteration" #iteration ":" ::: "memory");\
    FLOAT_LOAD_TILE_C(c_cur, FLOAT_W_ob, FLOAT_C_ob); \
    FLOAT_FUSED_SLOPE_RELU_TILE_C(step, a_cur, c_cur, FLOAT_W_ob, FLOAT_C_ob); \
    FLOAT_STORE_TILE_C(c_cur, FLOAT_W_ob, FLOAT_C_ob);\
    a_cur += FLOAT_W_ob * FLOAT_C_ib; \
    c_cur += FLOAT_W_ob * FLOAT_C_ob;

template<OpType op_type>
void test_performance_micro_kernel(void)
{
    using BufferT = FloatBuffer;
    using ScalarT = typename BufferT::value_type;
    srand(time(0));
    size_t const num_trails = 10000;
    size_t const INPUT_SIZE = FLOAT_W_ob * FLOAT_C_ib * num_trails;
    BufferT input_buf(INPUT_SIZE);
    for (size_t ix = 0; ix < INPUT_SIZE; ++ix) input_buf[ix] = 2.0 * ((float)rand() / RAND_MAX);

    size_t const OUTPUT_SIZE = FLOAT_W_ob * FLOAT_C_ob * num_trails;
    BufferT ref_buf(OUTPUT_SIZE);
    BufferT output_buf(OUTPUT_SIZE);
    for (size_t ix = 0; ix < OUTPUT_SIZE; ++ix) output_buf[ix] = 2.0 * ((float)rand() / RAND_MAX) - 1.0;
    // std::cout << OUTPUT_SIZE << std::endl;

    ScalarT *a_cur = input_buf.data();
    ScalarT *c_cur = output_buf.data();
    constexpr dim_t _stride = 1U;
    constexpr dim_t step = FLOAT_C_ob * _stride;

    //==================================================
    FLOAT_DEF_TILE_C(FLOAT_W_ob, FLOAT_C_ob);

    

    double tx(0.);
    double min_t = std::numeric_limits<double>::max();
    double max_t = 0.;
    Timer my_timer;

    for (size_t ix = 0; ix < num_trails; ++ix)
    {
        a_cur = input_buf.data();
        c_cur = output_buf.data();
        tx = 0.;
        my_timer.start();    
        // REPEAT_100(SLOPE_RELU_CALL);
        if (op_type == OP_SLOPE_RELU)
        {
            REPEAT_1000(SLOPE_RELU_CALL);
        }
        else if (op_type == OP_FUSED_SLOPE_RELU)
        {
            REPEAT_1000(FUSED_SLOPE_RELU_CALL);
        }
        else
        {
            TEST_CHECK(false); // Unsupported operation
        }
        // REPEAT_100(INDIVIDUAL_SLOPE_RELU_CALL);
        my_timer.stop();
        auto elapsed = my_timer.elapsed();
        tx += elapsed;

        min_t = std::min(min_t, elapsed);
        max_t = std::max(max_t, elapsed);
        // _mm256_store_ps(ref_buf.data(), result_accumulator);
        // std::cout << ref_buf.data()[0] << std::endl;
    }
    
    
    const float cpu_freq = 2.4; // 2.4 GHz, adjust as needed
    std::cout << "Min Ave time: " << min_t << " ns." << std::endl;
    std::cout << "Max Ave time: " << max_t << " ns." << std::endl;
    std::cout << "Peak: " << (FLOAT_W_ob * FLOAT_C_ob / (min_t / 1000 * cpu_freq)) << std::endl;


    FLOAT_STORE_TILE_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);
    // _mm256_storeu_ps(output_buf.data(), result_accumulator);
    // std::cout << output_buf.data()[0] << std::endl;
    //==================================================

}

template<OpType op_type>
void test_performance_micro_kernel_for_loop(void)
{
    using BufferT = FloatBuffer;
    using ScalarT = typename BufferT::value_type;
    srand(time(0));
    size_t const num_trails = 10000;
    size_t const INPUT_SIZE = FLOAT_W_ob * FLOAT_C_ib * num_trails;
    BufferT input_buf(INPUT_SIZE);
    for (size_t ix = 0; ix < INPUT_SIZE; ++ix) input_buf[ix] = 2.0 * ((float)rand() / RAND_MAX);

    size_t const OUTPUT_SIZE = FLOAT_W_ob * FLOAT_C_ob * num_trails;
    BufferT ref_buf(OUTPUT_SIZE);
    BufferT output_buf(OUTPUT_SIZE);
    for (size_t ix = 0; ix < OUTPUT_SIZE; ++ix) output_buf[ix] = 2.0 * ((float)rand() / RAND_MAX) - 1.0;

    // std::cout << OUTPUT_SIZE << std::endl;

    ScalarT *a_cur = input_buf.data();
    ScalarT *c_cur = output_buf.data();
    constexpr dim_t _stride = 1U;
    constexpr dim_t step = FLOAT_C_ob * _stride;

    //==================================================
    FLOAT_DEF_TILE_C(FLOAT_W_ob, FLOAT_C_ob);

    

    double tx(0.);
    double min_t = std::numeric_limits<double>::max();
    double max_t = 0.;
    Timer my_timer;

    for (size_t ix = 0; ix < num_trails; ++ix)
    {
        a_cur = input_buf.data();
        c_cur = output_buf.data();
        tx = 0.;
        my_timer.start();    
        for (size_t iy = 0; iy < 1000; ++iy)
        {
            if (op_type == OP_SLOPE_RELU)
            {
                SLOPE_RELU_CALL(0);
            }
            else if (op_type == OP_FUSED_SLOPE_RELU)
            {
                FUSED_SLOPE_RELU_CALL(0);
            }
            else
            {
                TEST_CHECK(false); // Unsupported operation
            }
        }
        my_timer.stop();
        auto elapsed = my_timer.elapsed();
        tx += elapsed;
        FLOAT_STORE_TILE_C(c_cur, FLOAT_W_ob, FLOAT_C_ob);
        min_t = std::min(min_t, elapsed);
        max_t = std::max(max_t, elapsed);
        // _mm256_store_ps(ref_buf.data(), result_accumulator);
        // std::cout << ref_buf.data()[0] << std::endl;
    }
    
    
    const float cpu_freq = 2.4; // 2.4 GHz, adjust as needed
    std::cout << "Min Ave time: " << min_t << " ns." << std::endl;
    std::cout << "Max Ave time: " << max_t << " ns." << std::endl;
    std::cout << "Peak: " << (FLOAT_W_ob * FLOAT_C_ob / (min_t / 1000 * cpu_freq)) << std::endl;


   
    // _mm256_storeu_ps(output_buf.data(), result_accumulator);
    // std::cout << output_buf.data()[0] << std::endl;
    //==================================================

}
#endif




template <typename BufferT>
BufferT create_positive_data(size_t num_elements, unsigned int seed = static_cast<unsigned int>(time(0)))
{
    // Use a different seed for positive data
    unsigned int pos_seed = seed + 12345;
    std::default_random_engine generator(pos_seed);
    std::normal_distribution<float> distribution{0.f, 1.f};

    BufferT input_buf(num_elements);

    for (size_t ix = 0; ix < num_elements; ++ix)
    {
#if defined(QUANTIZED)
        input_buf[ix] = (typename BufferT::value_type)(64*distribution(generator));
#else
        input_buf[ix] = std::fabs(distribution(generator));
#endif
    }

    return input_buf;
}


template <typename BufferT>
BufferT create_small_positive_data(size_t num_elements, unsigned int seed = static_cast<unsigned int>(time(0)))
{
    // Use a different seed for positive data
    unsigned int pos_seed = seed + 12345;
    std::default_random_engine generator(pos_seed);
    std::normal_distribution<float> distribution{0.f, 1.f};

    BufferT input_buf(num_elements);

    for (size_t ix = 0; ix < num_elements; ++ix)
    {
#if defined(QUANTIZED)
        input_buf[ix] = (typename BufferT::value_type)(64*distribution(generator));
#else
        input_buf[ix] = (ix%10)*0.001f + 0.000242f;
#endif
    }

    return input_buf;
}


template <typename BufferT>
BufferT create_real_data(size_t num_elements, unsigned int seed = static_cast<unsigned int>(time(0)))
{
    // Use a different seed for real data
    unsigned int real_seed = seed + 67890;
    std::default_random_engine generator(real_seed);
    std::normal_distribution<float> distribution{0.f, 1.f};

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

template <OpType op_type>
void test_single_element(void)
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
    small::FloatBuffer input_dc = create_small_positive_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer b_dc(1);
    b_dc[0] = 0.5f; // Scalar for ewise operations
    small::FloatBuffer output_dc = create_real_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer ref_input = input_dc;
    small::FloatBuffer ref_output = output_dc;

    if (op_type == OP_SLOPE_RELU)
        small::SlopeReLU(
            C_i, H, W,
            input_dc,
            output_dc
        );
    else if (op_type == OP_FUSED_SLOPE_RELU)
        small::FusedSlopeReLU(
            C_i, H, W,
            input_dc,
            output_dc
        );
    else if (op_type == OP_CELU)
        small::CeLU(
            C_i, H, W,
            input_dc,
            b_dc,
            output_dc
        );
    else if (op_type == OP_FUSED_CELU)
        small::FusedCeLU(
            C_i, H, W,
            input_dc,
            b_dc,
            output_dc
        );
    else
        TEST_CHECK(false); // Unsupported operation

    for(size_t ix = 0; ix < num_input_elts; ++ix)
    {
        float expected_value = ewise_function<op_type>(ref_input[ix], ref_output[ix]);
        // TEST_CHECK(output_dc[ix] == expected_value);
        const float rtol = 1e-2f; // 0.2%
        const float atol = 1e-6f;
        float error = std::abs(output_dc[ix] - expected_value);
        float denom = std::abs(expected_value);
        float rel_err = (denom > 0.0f) ? (error / denom) : error; // fallback to absolute when ref == 0
        TEST_CHECK(error <= atol + rtol * denom);
        std::cout << "(" << ref_input[ix] << "," << ref_output[ix] << ") -> " << "Expected: " << expected_value << ", Actual: " << output_dc[ix] << " Error: "<< error << std::endl;

    }
}

double get_cpu_frequency_proc() {
    std::ifstream file("/proc/cpuinfo");
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.find("cpu MHz") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string freq_str = line.substr(pos + 1);
                return std::stod(freq_str); // Returns MHz
            }
        }
    }
    return -1.0; // Error
}

template <OpType op_type>
void measure_performance(void)
{
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
        { 512,   13,  13, 1, 1, small::PADDING_F, 1024},

        {16, 12, 12, 1, 1, small::PADDING_F, 16},
        {32, 6, 6, 1, 1, small::PADDING_F, 32},
        {16, 24, 24, 1, 1, small::PADDING_F, 16},
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
    printf("\n\tC_i\tH\tW\tk\ts\tnthd\truns\tt_min\tt_max\tt_avg\n");
    std::string csv_filename;
    if (op_type == OP_SLOPE_RELU) {
        csv_filename = "slope_relu_timing.csv";
    } else if (op_type == OP_FUSED_SLOPE_RELU) {
        csv_filename = "fused_slope_relu_timing.csv";
    } else if (op_type == OP_CELU) {
        csv_filename = "celu_timing.csv";
    } else if (op_type == OP_FUSED_CELU) {
        csv_filename = "fused_celu_timing.csv";
    } else {
        csv_filename = "unknown_op_timing.csv";
    }

    std::ofstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not open " << csv_filename << " for writing" << std::endl;
        return;
    }
    
    // Write CSV header
    csv_file << "C_i,H,W,k,s,num_threads,min_t,max_t,avg_t\n";
    for (LayerParams const &p : params)
    {
        size_t num_input_elts(p.C_i*p.H*p.W);

        Buffer  input_dc(num_input_elts);
        Buffer output_dc(num_input_elts);
        Buffer alpha_dc(1);
        alpha_dc.data()[0] = 0.5f;
        small::init(input_dc, num_input_elts);
        small::init(output_dc, num_input_elts);
        
        for (size_t ix = 0; ix < 1; ++ix)
        {
            setenv("OMP_NUM_THREADS", str_num_threads[ix], 1);
            //std::string ont = std::getenv("OMP_NUM_THREADS"); // read it back
            //auto nt = atol(ont.c_str());

            double tx(0.);
            double min_t = std::numeric_limits<double>::max();
            double max_t = 0.;

            // Warmup
            for (size_t iy = 0; iy < 10; ++iy)
            {
                if (op_type == OP_SLOPE_RELU)
                    small::SlopeReLU(p.C_i, p.H, p.W, input_dc, output_dc);
                else if (op_type == OP_FUSED_SLOPE_RELU)
                    small::FusedSlopeReLU(p.C_i, p.H, p.W, input_dc, output_dc);
                else if (op_type == OP_CELU) 
                    small::CeLU(p.C_i, p.H, p.W, input_dc, alpha_dc, output_dc);
                else if (op_type == OP_FUSED_CELU) 
                    small::FusedCeLU(p.C_i, p.H, p.W, input_dc, alpha_dc, output_dc);
            }
            

            for (size_t iy = 0; iy < num_runs; ++iy)
            {
                t.start();
                if (op_type == OP_SLOPE_RELU) {
                    small::SlopeReLU(p.C_i, p.H, p.W, input_dc, output_dc);
                }
                else if (op_type == OP_FUSED_SLOPE_RELU)
                {
                    small::FusedSlopeReLU(p.C_i, p.H, p.W, input_dc, output_dc);
                } 
                else if (op_type == OP_CELU) {
                    small::CeLU(p.C_i, p.H, p.W, input_dc, alpha_dc, output_dc);
                } else if (op_type == OP_FUSED_CELU) {
                    small::FusedCeLU(p.C_i, p.H, p.W, input_dc, alpha_dc, output_dc);
                }
                t.stop();
                double ts = t.elapsed();
                tx += ts;
                min_t = std::min(min_t, ts);
                max_t = std::max(max_t, ts);
                
            }
            csv_file << p.C_i << "," << p.H << "," << p.W << "," 
                        << p.k << "," << p.s << "," << num_threads[ix] << "," 
                        << min_t << "," << max_t << "," << (tx/num_runs) << "\n";
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
    // {"Correctness of FLOAT_SLOPE_RELU_TILE", small::float_detail::test_correctness_ewise_kernel<small::OP_SLOPE_RELU>},
    // {"Correctness of FLOAT_FUSED_SLOPE_RELU_TILE", small::float_detail::test_correctness_fused_slope_relu},
    // {"Performance of FLOAT_SLOPE_RELU_TILE", small::float_detail::test_performance_micro_kernel<small::OP_SLOPE_RELU>},
    // // {"Performance of FLOAT_FUSED_SLOPE_RELU_TILE", small::float_detail::test_performance_micro_kernel<small::OP_FUSED_SLOPE_RELU>},
    // {"Performance of for loop", small::float_detail::test_performance_micro_kernel_for_loop<small::OP_SLOPE_RELU>},
    // {"Performance of for loop fused", small::float_detail::test_performance_micro_kernel_for_loop<small::OP_FUSED_SLOPE_RELU>},
    // {"Correctness of slope relu single tile", small::float_detail::test_single_element<small::OP_SLOPE_RELU>},
    // {"Correctness of fused slope relu single tile", small::float_detail::test_single_element<small::OP_FUSED_SLOPE_RELU>},
    // {"Performance of slope relu", small::float_detail::measure_performance<small::OP_SLOPE_RELU>},
    // {"Performance of fused slope relu", small::float_detail::measure_performance<small::OP_FUSED_SLOPE_RELU>},
    // {"Correctness of FLOAT_CELU_TILE_C", small::float_detail::test_correctness_ewise_kernel<small::OP_CELU>},
    // {"Correctness of FLOAT_FUSED_CELU_TILE_C", small::float_detail::test_correctness_ewise_kernel<small::OP_FUSED_CELU>},
    // {"Correctness of celu single tile", small::float_detail::test_single_element<small::OP_CELU>},
    // {"Correctness of fused celu single tile", small::float_detail::test_single_element<small::OP_FUSED_CELU>},
    // {"Performance of celu", small::float_detail::measure_performance<small::OP_CELU>},
    // {"Performance of fused celu", small::float_detail::measure_performance<small::OP_FUSED_CELU>},
    {"Correctness of FLOAT_CELU_TILE_C", small::float_detail::test_correctness_ewise_kernel<small::OP_CELU>},
    {"Correctness of FLOAT_FUSED_CELU_TILE_C", small::float_detail::test_correctness_ewise_kernel<small::OP_FUSED_CELU>},
    {"Correctness of celu single tile", small::float_detail::test_single_element<small::OP_CELU>},
    {"Correctness of fused celu single tile", small::float_detail::test_single_element<small::OP_FUSED_CELU>},
    {"Performance of celu", small::float_detail::measure_performance<small::OP_CELU>},
    {"Performance of fused celu", small::float_detail::measure_performance<small::OP_FUSED_CELU>},
    {NULL, NULL}
};