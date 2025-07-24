#include <acutest.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>

#include <small.h>
#include <small/utils/Timer.hpp>


#include "test_utils.hpp"

namespace small {
namespace float_detail {
    
void test_correctness_FLOAT_AVERAGE_TILE(void)
{
    #if defined(SMALL_HAS_FLOAT_SUPPORT)
        using BufferT = FloatBuffer;
        using ScalarT = typename BufferT::value_type;
        srand(time(0));
        size_t const INPUT_SIZE = FLOAT_W_ob * FLOAT_C_ib; 
        BufferT input0_buf(INPUT_SIZE); 
        BufferT input1_buf(INPUT_SIZE);
        for (size_t ix = 0; ix < INPUT_SIZE; ++ix) 
        {
            input0_buf[ix] = 2.0 * ((float)rand() / RAND_MAX) - 1;
            input1_buf[ix] = 2.0 * ((float)rand() / RAND_MAX) - 1;
        }
        
        size_t const OUTPUT_SIZE = FLOAT_W_ob * FLOAT_C_ob;
        BufferT output_buf(OUTPUT_SIZE);
        for (size_t ix = 0; ix < OUTPUT_SIZE; ++ix) 
            output_buf[ix] = input1_buf[ix]; 
        
        ScalarT *a_cur = input0_buf.data();
        // ScalarT *b_cur = input1_buf.data();
        constexpr dim_t _stride = 1U; 
        constexpr dim_t step = FLOAT_C_ob * _stride;
        
        //=====================================================
        FLOAT_DEF_TILE_C(FLOAT_W_ob, FLOAT_C_ob);

        FLOAT_LOAD_TILE_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);

        FLOAT_HALFSUM_TILE_C(step, a_cur, FLOAT_W_ob, FLOAT_C_ob);

        FLOAT_STORE_TILE_C(output_buf.data(), FLOAT_W_ob, FLOAT_C_ob);
        
        //=====================================================
        for (dim_t ii = 0; ii < FLOAT_W_ob; ++ii)
        {
            for (dim_t jj = 0; jj < FLOAT_C_ob; ++jj)
            {
                size_t ix = ii*FLOAT_C_ob + jj;
                std::cout << output_buf[ix] << " " << (float)((input0_buf[ix] + input1_buf[ix]) * .5f) << " " << input0_buf[ix] << " " << input1_buf[ix] << " " << ix << std::endl;
                TEST_CHECK(output_buf[ix] == (input0_buf[ix] + input1_buf[ix]) * .5f);
            }
        }

    #endif
}
}

template <typename BufferT>
BufferT create_data(size_t num_elements)
{
    using T = typename BufferT::value_type;
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<T> distribution{.0, .1};  // what distribution is Torch::Tensor::randn?

    BufferT input_buf(num_elements);

    for (size_t ix = 0; ix < num_elements; ++ix)
    {
#if defined(QUANTIZED)
        input_buf[ix] = (typename BufferT::value_type)(64*distribution(generator));
#else
        input_buf[ix] = abs(distribution(generator));
#endif
    }

    return input_buf;
}

template <typename BufferT, typename T>
void test_upwind(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1; 

    size_t const num_input_elts = C_i * H * W;
    
    size_t const dim = 3; 
    size_t const dir = 1;
#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts);
#else
    BufferT  input_low_dc = create_data<BufferT>(num_input_elts * (dim + 2));
    BufferT  input_high_dc = create_data<BufferT>(num_input_elts * (dim + 2));
    BufferT  output_dc(num_input_elts * (dim + 2));
    BufferT  valid_dc(num_input_elts * (dim + 2));
    BufferT  gamma_buf(1);
    gamma_buf[0] = (T) 0.5; 
#endif

    small::Upwind<BufferT>(
        C_i, H, W,
        dim, dir, gamma_buf,
        input_low_dc,
        input_high_dc,
        output_dc
    );

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        const T& rhol = input_low_dc[ix];
        const T& rhor = input_high_dc[ix];
        const T& ul = input_low_dc[ix + (dir + 1) * num_input_elts];
        const T& ur = input_high_dc[ix + (dir + 1) * num_input_elts];
        const T& pl = input_low_dc[ix + (dim + 1) * num_input_elts];
        const T& pr = input_high_dc[ix + (dim + 1) * num_input_elts];
        T gamma = gamma_buf[0];
        T rhobar = (rhol + rhor)*(T).5;
        T pbar = (pl + pr)*(T).5;
        T ubar = (ul + ur)*(T).5;
        T cbar = sqrt(gamma*pbar/rhobar);
        T pstar = (pl + pr)*(T).5 + rhobar*cbar*(ul - ur)*(T).5;
        T ustar = (ul + ur)*(T).5 + (pl - pr)/((T)2.0*rhobar*cbar);
        int sign;
        if (ustar > 0) 
        {
            sign = -1;
            valid_dc[ix] = input_low_dc[ix];
            for (size_t icomp = 0;icomp < dim;icomp++)
            {
                valid_dc[ix + (icomp + 1) * num_input_elts] = input_low_dc[ix + (icomp + 1) * num_input_elts];
            }
            valid_dc[ix + (dim + 1) * num_input_elts] = input_low_dc[ix + (dim + 1) * num_input_elts];
        }
        else
        {
            sign = 1;
            valid_dc[ix] = input_high_dc[ix];
            for (size_t icomp = 0;icomp < dim;icomp++)
            {
                valid_dc[ix + (icomp + 1) * num_input_elts] = input_high_dc[ix + (icomp + 1) * num_input_elts];
            }
            valid_dc[ix + (dim + 1) * num_input_elts] = input_high_dc[ix + (dim + 1) * num_input_elts];
        }

        T outval = valid_dc[ix] + (pstar - valid_dc[ix + (dim + 1) * num_input_elts])/(cbar*cbar);
        if (cbar + sign * ubar > 0)
        {
            valid_dc[ix] = outval; 
            valid_dc[ix + (dir + 1) * num_input_elts] = ustar; 
            valid_dc[ix + (dim + 1) * num_input_elts] = pstar;
        }

    }

    for (size_t ix = 0; ix < num_input_elts * (dim + 2); ++ix)
    {
        TEST_CHECK(output_dc[ix] == valid_dc[ix]);
        std::cout << ix << ": upwind(" << input_low_dc[ix] << ", " 
        << input_high_dc[ix] << ")-->"
                 << output_dc[ix] << " " << valid_dc[ix] << std::endl;
    }
}

void test_upwind_float(void)
{
    test_upwind<FloatBuffer, float>();
}

void test_upwind_double(void)
{
    test_upwind<DoubleBuffer, double>();
}

void test_consToPrim(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1; 

    size_t const num_input_elts = C_i * H * W;
    
    size_t const dim = 3; 
#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts);
#else
    small::FloatBuffer  input_dc = create_data<small::FloatBuffer>(num_input_elts * (dim + 2));
    small::FloatBuffer  output_dc(num_input_elts * (dim + 2));
    small::FloatBuffer  valid_dc(num_input_elts * (dim + 2));
    small::FloatBuffer  gamma_buf(1);
    gamma_buf[0] = 1.4f; 
#endif
    small::ConsToPrim(
        C_i, H, W,
        dim, gamma_buf,
        input_dc,
        output_dc
    );

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        float rho = input_dc[ix];
        valid_dc[ix] = rho;
        float v2 = 0.0f;
        for (size_t icomp = 0; icomp < dim; ++icomp)
        {
            valid_dc[ix + (icomp + 1) * num_input_elts] = input_dc[ix + (icomp + 1) * num_input_elts] / rho;
            v2 += valid_dc[ix + (icomp + 1) * num_input_elts] *
                  valid_dc[ix + (icomp + 1) * num_input_elts];
        }
        valid_dc[ix + (dim + 1) * num_input_elts] = v2;
        //valid_dc[ix + (dim + 1) * num_input_elts] = (input_dc[ix + (dim + 1) * num_input_elts] - .5f * rho * v2) * (gamma_buf[0] - 1.0f);
    }

    for (size_t ix = 0; ix < num_input_elts * (dim + 2); ++ix)
    {
        TEST_CHECK(output_dc[ix] == valid_dc[ix]);
        std::cout << ix << ": consToPrim(" << input_dc[ix] << ", " 
         << ")-->"
                 << output_dc[ix] << " " << valid_dc[ix] << std::endl;
    }
}

void test_getFlux(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1; 

    size_t const num_input_elts = C_i * H * W;
    size_t const dir = 1; // 0 for x, 1 for y, 2 for z
    size_t const dim = 2;
#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts * (dim + 2));
#else
    small::FloatBuffer  input_dc = create_data<small::FloatBuffer>(num_input_elts * (dim + 2));
    small::FloatBuffer  output_dc(num_input_elts * (dim + 2));
    small::FloatBuffer  valid_dc(num_input_elts * (dim + 2));
    small::FloatBuffer  gamma_buf(1);
    gamma_buf[0] = 1.4f; 
    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        input_dc[ix] = 1.409075f;
        input_dc[ix + 1 * num_input_elts] = 0.006462581f;
        input_dc[ix + 2 * num_input_elts] = 0.0f;
        input_dc[ix + 3 * num_input_elts] = 1.009089f;
    }
#endif
    small::GetFlux(
        C_i, H, W,
        dim, dir, gamma_buf,
        input_dc,
        output_dc
    );

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        float F0 = input_dc[ix] * input_dc[ix + (dir + 1) * num_input_elts];
        float W2 = 0.0f;
        float gamma = gamma_buf[0];
        valid_dc[ix] = F0;
        for (size_t icomp = 0; icomp < dim; ++icomp)
        {
            float Wd = input_dc[ix + (icomp + 1) * num_input_elts];
            valid_dc[ix + (icomp + 1) * num_input_elts] = Wd * F0;
            W2 += Wd * Wd;
        }
        valid_dc[ix + (dir + 1) * num_input_elts] += input_dc[ix + (dim + 1) * num_input_elts];
        valid_dc[ix + (dim + 1) * num_input_elts] = 
            gamma / (gamma - 1.0f) * input_dc[ix + (dir + 1) * num_input_elts] * input_dc[ix + (dim + 1) * num_input_elts] + 0.5f * F0 * W2; 
        for (size_t icomp = 0; icomp < size_t(dim + 2); ++icomp)
        {
            valid_dc[ix + icomp  * num_input_elts] = -valid_dc[ix + icomp * num_input_elts];
        }
    }

    for (size_t ix = 0; ix < num_input_elts * (dim + 2); ++ix)
    {
        TEST_CHECK(output_dc[ix] == valid_dc[ix]);
        std::cout << ix << ": getFlux(" << input_dc[ix] << ", " 
         << ")-->"
           << std::fixed << std::setprecision(6)      << output_dc[ix] << " " << valid_dc[ix] << std::endl;
    }
}

void test_waveSpeedBound(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1; 

    size_t const num_input_elts = C_i * H * W;
    size_t const dim = 2;
#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts * (dim + 2));
#else
    small::FloatBuffer  input_dc = create_data<small::FloatBuffer>(num_input_elts * (dim + 2));
    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        input_dc[ix] = 1.40907f;
        input_dc[ix + 1 * num_input_elts] = 0.00646258f;
        input_dc[ix + 2 * num_input_elts] = 0.0f;
        input_dc[ix + 3 * num_input_elts] = 1.00909f;
    }
    small::FloatBuffer  output_dc(num_input_elts);
    small::FloatBuffer  valid_dc(num_input_elts);
    small::FloatBuffer  gamma_buf(1);
    gamma_buf[0] = 1.4f; 
#endif
    small::WaveSpeedBound(
        C_i, H, W,
        dim, gamma_buf,
        input_dc,
        output_dc
    );

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        valid_dc[ix] = dim * std::sqrt(gamma_buf[0] * input_dc[ix + (dim + 1) * num_input_elts] / input_dc[ix]);
        for (size_t icomp = 0; icomp < dim; ++icomp)
        {
            valid_dc[ix] += std::fabs(input_dc[ix + (icomp + 1) * num_input_elts]);
        }
    }

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK(output_dc[ix] == valid_dc[ix]);
        std::cout << ix << ": waveSpeedBound(" << input_dc[ix] << ", " 
         << ")-->"
                 << output_dc[ix] << " " << valid_dc[ix] << std::endl;
    }
}


}

TEST_LIST = {
    {"correctness FLOAT_AVERAGE_TILE",
     small::float_detail::test_correctness_FLOAT_AVERAGE_TILE},
    {"upwind float",
     small::test_upwind_float},
    {"upwind double",
     small::test_upwind_double},
    {"consToPrim",
     small::test_consToPrim},
    {"getFlux",
     small::test_getFlux},
    {"waveSpeedBound",
     small::test_waveSpeedBound},
    {NULL, NULL}
};

