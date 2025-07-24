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
        input_buf[ix] = abs(distribution(generator));
    }

    return input_buf;
}

//****************************************************************************
// Test Upwind
//****************************************************************************

template <typename BufferT, typename T>
void test_upwind(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1; 

    size_t const num_input_elts = C_i * H * W;
    
    size_t const dim = 3; 
    size_t const dir = 1;

    BufferT  input_low_dc = create_data<BufferT>(num_input_elts * (dim + 2));
    BufferT  input_high_dc = create_data<BufferT>(num_input_elts * (dim + 2));
    BufferT  output_dc(num_input_elts * (dim + 2));
    BufferT  valid_dc(num_input_elts * (dim + 2));
    BufferT  gamma_buf(1);
    gamma_buf[0] = (T) 0.5; 

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
        // std::cout << ix << ": upwind(" << input_low_dc[ix] << ", " 
        // << input_high_dc[ix] << ")-->"
        //          << output_dc[ix] << " " << valid_dc[ix] << std::endl;
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

//****************************************************************************
// Test ConsToPrim
//****************************************************************************

template <typename BufferT, typename T>
void test_consToPrim(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1; 

    size_t const num_input_elts = C_i * H * W;
    
    size_t const dim = 3; 

    BufferT input_dc = create_data<BufferT>(num_input_elts * (dim + 2));
    BufferT output_dc(num_input_elts * (dim + 2));
    BufferT valid_dc(num_input_elts * (dim + 2));
    BufferT gamma_buf(1);
    gamma_buf[0] = (T) 1.4; 

    small::ConsToPrim<BufferT>(
        C_i, H, W,
        dim, gamma_buf,
        input_dc,
        output_dc
    );

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        T rho = input_dc[ix];
        valid_dc[ix] = rho;
        T v2 = (T)0.0;
        for (size_t icomp = 0; icomp < dim; ++icomp)
        {
            valid_dc[ix + (icomp + 1) * num_input_elts] = input_dc[ix + (icomp + 1) * num_input_elts] / rho;
            v2 += valid_dc[ix + (icomp + 1) * num_input_elts] *
                  valid_dc[ix + (icomp + 1) * num_input_elts];
        }
        valid_dc[ix + (dim + 1) * num_input_elts] = (input_dc[ix + (dim + 1) * num_input_elts] - (T)0.5 * rho * v2) * (gamma_buf[0] - (T)1.0);
    }

    for (size_t ix = 0; ix < num_input_elts * (dim + 2); ++ix)
    {
        TEST_CHECK(output_dc[ix] == valid_dc[ix]);
        // std::cout << ix << ": consToPrim(" << input_dc[ix] << ", " 
        //  << ")-->"
        //          << output_dc[ix] << " " << valid_dc[ix] << std::endl;
    }
}

void test_consToPrim_float(void)
{
    test_consToPrim<FloatBuffer, float>();
}

void test_consToPrim_double(void)
{
    test_consToPrim<DoubleBuffer, double>();
}

//****************************************************************************
// Test GetFlux
//****************************************************************************

template <typename BufferT, typename T>
void test_getFlux(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1; 

    size_t const num_input_elts = C_i * H * W;
    size_t const dir = 1; // 0 for x, 1 for y, 2 for z
    size_t const dim = 2;

    BufferT input_dc = create_data<BufferT>(num_input_elts * (dim + 2));
    BufferT output_dc(num_input_elts * (dim + 2));
    BufferT valid_dc(num_input_elts * (dim + 2));
    BufferT gamma_buf(1);
    gamma_buf[0] = (T) 1.4;

    small::GetFlux(
        C_i, H, W,
        dim, dir, gamma_buf,
        input_dc,
        output_dc
    );

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        T F0 = input_dc[ix] * input_dc[ix + (dir + 1) * num_input_elts];
        T W2 = (T)0.0;
        T gamma = gamma_buf[0];
        valid_dc[ix] = F0;
        for (size_t icomp = 0; icomp < dim; ++icomp)
        {
            T Wd = input_dc[ix + (icomp + 1) * num_input_elts];
            valid_dc[ix + (icomp + 1) * num_input_elts] = Wd * F0;
            W2 += Wd * Wd;
        }
        valid_dc[ix + (dir + 1) * num_input_elts] += input_dc[ix + (dim + 1) * num_input_elts];
        valid_dc[ix + (dim + 1) * num_input_elts] = 
            gamma / (gamma - (T)1.0) * input_dc[ix + (dir + 1) * num_input_elts] * input_dc[ix + (dim + 1) * num_input_elts] + (T)0.5 * F0 * W2; 
        for (size_t icomp = 0; icomp < size_t(dim + 2); ++icomp)
        {
            valid_dc[ix + icomp  * num_input_elts] = -valid_dc[ix + icomp * num_input_elts];
        }
    }

    for (size_t ix = 0; ix < num_input_elts * (dim + 2); ++ix)
    {
        TEST_CHECK(output_dc[ix] == valid_dc[ix]);
        // std::cout << ix << ": getFlux(" << input_dc[ix] << ", " 
        //  << ")-->"
        //    << std::fixed << std::setprecision(6)      << output_dc[ix] << " " << valid_dc[ix] << std::endl;
    }
}

void test_getFlux_float(void)
{
    test_getFlux<FloatBuffer, float>();
}

void test_getFlux_double(void)
{
    test_getFlux<DoubleBuffer, double>();
}

//****************************************************************************
// Test WaveSpeedBound
//****************************************************************************

template <typename BufferT, typename T>
void test_waveSpeedBound(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1; 

    size_t const num_input_elts = C_i * H * W;
    size_t const dim = 2;

    BufferT input_dc = create_data<BufferT>(num_input_elts * (dim + 2));
    BufferT output_dc(num_input_elts);
    BufferT valid_dc(num_input_elts);
    BufferT gamma_buf(1);
    gamma_buf[0] = (T)1.4;

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
        // std::cout << ix << ": waveSpeedBound(" << input_dc[ix] << ", " 
        //  << ")-->"
        //          << output_dc[ix] << " " << valid_dc[ix] << std::endl;
    }
}

void test_waveSpeedBound_float(void)
{
    test_waveSpeedBound<FloatBuffer, float>();
}

void test_waveSpeedBound_double(void)
{
    test_waveSpeedBound<DoubleBuffer, double>();
}

}

TEST_LIST = {
    {"upwind float",
     small::test_upwind_float},
    {"upwind double",
     small::test_upwind_double},
    {"consToPrim float",
     small::test_consToPrim_float},
    {"consToPrim double",
     small::test_consToPrim_double},
    {"getFlux float",
     small::test_getFlux_float},
    {"getFlux double",
     small::test_getFlux_double},
    {"waveSpeedBound float",
     small::test_waveSpeedBound_float},
    {"waveSpeedBound double",
     small::test_waveSpeedBound_double},
    {NULL, NULL}
};

