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
    
template <typename BufferT>
BufferT create_data(size_t num_elements)
{
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution{0.f, 1.f};  // what distribution is Torch::Tensor::randn?

    BufferT input_buf(num_elements);

    for (size_t ix = 0; ix < num_elements; ++ix)
    {
#if defined(QUANTIZED)
        input_buf[ix] = (typename BufferT::value_type)(64*distribution(generator));
#else
        input_buf[ix] = std::abs(distribution(generator));
#endif
    }

    return input_buf;
}

void test_stencil_interpolationL(void)
{
    size_t const C_i = 5;
    size_t const H = 8;
    size_t const W = 8; 

    size_t const num_input_elts = C_i * H * W;
    
    
#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts);
#else
    small::FloatBuffer input_dc = create_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer output_dc(num_input_elts - H * C_i);
    small::FloatBuffer valid_dc(num_input_elts);
#endif
    size_t filter_size = C_i * 2;
    FloatBuffer filter_buf(filter_size);
    for (size_t ix = 0; ix < filter_size; ++ix)
    {
        filter_buf.data()[ix] = 0.5f;
    }
    small::DepthwiseConv2D(
        1, 2, 1,
        0, 0, 0, 0,
        C_i, H, W,
        input_dc,
        filter_buf,
        output_dc
    );
    float const *output_data = output_dc.data();
    size_t const size = output_dc.size();
    std::ofstream csvfile("stencil_interpolationL.csv");
    for (size_t ix = 0; ix < size; ++ix) {
        csvfile << output_data[ix] << "\n";
    }
    
    // small::Stencil(
    //     C_i, H, W,
    //     1, 0,
    //     input_dc, 
    //     output_dc
    // );
    size_t ix = 0; 
    size_t iy = 0;
    for (size_t c = 0; c < C_i; ++c)
    {
        for (size_t h = 0; h < H; ++h)
        {
            ix++;
            for (size_t w = 1; w < W; ++w)
            {
                valid_dc[iy] = (input_dc[ix - 1] + input_dc[ix]) * 0.5f;
                iy++; ix++;
            }
        }
    }

    for (size_t ix = 0; ix < num_input_elts - H * C_i; ++ix)
    {
        TEST_CHECK(output_dc[ix] == valid_dc[ix]);
        std::cout << ix << ": stencil("
        << input_dc[ix] << ")-->"
                 << output_dc[ix] << " " << valid_dc[ix] << std::endl;
    }
}

void test_stencil_interpolationR(void)
{
    size_t const C_i = 1;
    size_t const H = 5;
    size_t const W = 16; 

    size_t const num_input_elts = C_i * H * W;
    
    
#if defined(QUANTIZED)
    small::QUInt8Buffer input_dc = create_data<small::QUInt8Buffer>(num_input_elts);
    small::QUInt8Buffer output_dc(num_input_elts);
#else
    small::FloatBuffer input_dc = create_data<small::FloatBuffer>(num_input_elts);
    small::FloatBuffer output_dc(num_input_elts);
    small::FloatBuffer valid_dc(num_input_elts);
#endif
    small::Stencil(
        C_i, H, W,
        0, 1,
        input_dc, 
        output_dc
    );

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        if ((ix + 1) % W == 0)
        {
            valid_dc[ix] = input_dc[ix] * 0.5f;
        }
        else
        {
        valid_dc[ix] = (input_dc[ix] + input_dc[ix + 1]) * 0.5f;
        }
    }

    for (size_t ix = 0; ix < num_input_elts; ++ix)
    {
        TEST_CHECK(output_dc[ix] == valid_dc[ix]);
        std::cout << ix << ": stencil("
        << input_dc[ix] << ")-->"
                 << output_dc[ix] << " " << valid_dc[ix] << std::endl;
    }
}

// void test_stencil_upwind()
// {
//     size_t const C_i = 1;
//     size_t const H = 5;
//     size_t const W = 16; 

//     size_t const num_input_elts = C_i * H * W;
    
    
// #if defined(QUANTIZED)
//     small::QUInt8Buffer input_dc = create_data<small::QUInt8Buffer>(num_input_elts);
//     small::QUInt8Buffer output_dc(num_input_elts);
// #else
//     small::FloatBuffer input_dc(num_input_elts);
//     for (size_t ix = 0; ix < W; ++ix)
//     {
//         if (ix % 4 < 2)
//         {
//             input_dc[ix] = 1.40907f;
//         } else {
//             input_dc[ix] = 1.39093f;
//         }
//     }

//     for (size_t ix = W; ix < 2 * W; ++ix)
//     {
//         if (ix % 4 < 2)
//         {
//             input_dc[ix] = 0.00911646f;
//         } else {
//             input_dc[ix] = -0.00903246f;
//         }
//     }
//     for (size_t ix = 4 * W; ix < 5 * W; ++ix)
//     {
//         if (ix % 4 < 2)
//         {
//             input_dc[ix] = 2.52276f;
//         } else {
//             input_dc[ix] = 2.47738f;
//         }
//     }
//     small::FloatBuffer low_dc(num_input_elts);
//     small::FloatBuffer high_dc(num_input_elts);
//     small::FloatBuffer output_dc(num_input_elts);
//     small::FloatBuffer valid_dc(num_input_elts);
// #endif
//     small::Stencil(
//         C_i, H, W,
//         1, 0,
//         input_dc, 
//         low_dc
//     );
//     small::Stencil(
//         C_i, H, W,
//         0, 1,
//         input_dc, 
//         high_dc
//     );
//     size_t const dim = 3;
//     size_t const dir = 1;
//     size_t const C_i_upwind = 16;
//     size_t const H_upwind = 1; 
//     size_t const W_upwind = 1;
//     size_t const num_input_elts_upwind = C_i_upwind * H_upwind * W_upwind;
//     small::FloatBuffer input_low_rho_dc(num_input_elts_upwind);
//     std::copy(low_dc.data(), low_dc.data() + num_input_elts_upwind, input_low_rho_dc.data());
//     small::FloatBuffer input_low_G_dc(num_input_elts_upwind * dim);
//     std::copy(low_dc.data() + num_input_elts_upwind, low_dc.data() + num_input_elts_upwind * (dim + 1), input_low_G_dc.data());
//     small::FloatBuffer input_low_E_dc(num_input_elts_upwind);
//     std::copy(low_dc.data() + num_input_elts_upwind * (dim + 1), low_dc.data() + num_input_elts_upwind * (dim + 2), input_low_E_dc.data());
//     small::FloatBuffer input_high_rho_dc(num_input_elts_upwind);
//     std::copy(high_dc.data(), high_dc.data() + num_input_elts_upwind, input_high_rho_dc.data());
//     small::FloatBuffer input_high_G_dc(num_input_elts_upwind * dim);
//     std::copy(high_dc.data() + num_input_elts_upwind, high_dc.data() + num_input_elts_upwind * (dim + 1), input_high_G_dc.data());
//     small::FloatBuffer input_high_E_dc(num_input_elts_upwind);
//     std::copy(high_dc.data() + num_input_elts_upwind * (dim + 1), high_dc.data() + num_input_elts_upwind * (dim + 2), input_high_E_dc.data());
//     small::FloatBuffer output_rho_dc(num_input_elts_upwind);
//     small::FloatBuffer output_G_dc(num_input_elts_upwind * dim);
//     small::FloatBuffer output_E_dc(num_input_elts_upwind);
//     small::FloatBuffer valid_rho_dc(num_input_elts_upwind);
//     small::FloatBuffer valid_G_dc(num_input_elts_upwind * dim);
//     small::FloatBuffer valid_E_dc(num_input_elts_upwind);
//     small::FloatBuffer gamma_buf(1);
//     gamma_buf[0] = 0.5f;

//     small::Upwind(
//         C_i_upwind, H_upwind, W_upwind,
//         dim, dir, gamma_buf,
//         input_low_rho_dc,
//         input_low_G_dc,
//         input_low_E_dc,
//         input_high_rho_dc,
//         input_high_G_dc,
//         input_high_E_dc,
//         output_rho_dc,
//         output_G_dc,
//         output_E_dc
//     );

//     for (size_t ix = 0; ix < num_input_elts_upwind; ++ix)
//     {
//         const float& rhol = input_low_rho_dc[ix];
//         const float& rhor = input_high_rho_dc[ix];
//         const float& ul = input_low_G_dc[ix + dir * num_input_elts_upwind];
//         const float& ur = input_high_G_dc[ix + dir * num_input_elts_upwind];
//         const float& pl = input_low_E_dc[ix];
//         const float& pr = input_high_E_dc[ix];
//         float gamma = gamma_buf[0];
//         float rhobar = (rhol + rhor)*.5f;
//         float pbar = (pl + pr)*.5f;
//         float ubar = (ul + ur)*.5f;
//         float cbar = sqrt(gamma*pbar/rhobar);
//         float pstar = (pl + pr)*.5f + rhobar*cbar*(ul - ur)*.5f;
//         float ustar = (ul + ur)*.5f + (pl - pr)/(2.0f*rhobar*cbar);
//         int sign;
//         if (ustar > 0) 
//         {
//             sign = -1;
//             valid_rho_dc[ix] = input_low_rho_dc[ix];
//             for (int icomp = 0;icomp < dim;icomp++)
//             {
//                 valid_G_dc[ix + icomp * num_input_elts_upwind] = input_low_G_dc[ix + icomp * num_input_elts_upwind];
//             }
//             valid_E_dc[ix] = input_low_E_dc[ix];
//         }
//         else
//         {
//             sign = 1;
//             valid_rho_dc[ix] = input_high_rho_dc[ix];
//             for (int icomp = 0;icomp < dim;icomp++)
//             {
//                 valid_G_dc[ix + icomp * num_input_elts_upwind] = input_high_G_dc[ix + icomp * num_input_elts_upwind];
//             }
//             valid_E_dc[ix] = input_high_E_dc[ix];
//         }

//         float outval = valid_rho_dc[ix] + (pstar - valid_E_dc[ix])/(cbar*cbar);
//         if (cbar + sign * ubar > 0)
//         {
//             valid_rho_dc[ix] = outval; 
//             valid_G_dc[ix + dir * num_input_elts_upwind] = ustar; 
//             valid_E_dc[ix] = pstar;
//         }

//     }

//     for (size_t ix = 0; ix < num_input_elts_upwind; ++ix)
//     {
//         TEST_CHECK(output_rho_dc[ix] == valid_rho_dc[ix]);
//         std::cout << ix << ": upwind(" << input_low_rho_dc[ix] << ", " 
//         << input_high_rho_dc[ix] << ")-->"
//                  << output_rho_dc[ix] << " " << valid_rho_dc[ix] << std::endl;
//     }

//     for (size_t iy = 0; iy < dim; ++iy)
//     {
//         for (size_t ix = 0; ix < num_input_elts_upwind; ++ix)
//         {
//             TEST_CHECK(output_G_dc[ix + iy * num_input_elts_upwind] == valid_G_dc[ix + iy * num_input_elts_upwind]);
//             std::cout << ix << ": upwind(" << input_low_G_dc[ix + iy * num_input_elts_upwind] << ", " 
//             << input_high_G_dc[ix + iy * num_input_elts_upwind] << ")-->"
//                     << output_G_dc[ix + iy * num_input_elts_upwind] << " " << valid_G_dc[ix + iy * num_input_elts_upwind] << std::endl;
//         }
//     }

//     for (size_t ix = 0; ix < num_input_elts_upwind; ++ix)
//     {
//         TEST_CHECK(output_E_dc[ix] == valid_E_dc[ix]);
//         std::cout << ix << ": upwind(" << input_low_E_dc[ix] << ", " 
//         << input_high_E_dc[ix] << ")-->"
//                  << output_E_dc[ix] << " " << valid_E_dc[ix] << std::endl;
//     }
// }

// template <typename BufferT>
// void f_consToPrim(const size_t C_i, 
//                   const size_t H,
//                   const size_t W,
//                   BufferT const& input_dc, 
//                   BufferT & output_dc, 
//                   const size_t dim,
//                   const float gamma)
// {
//     if ((C_i * H * W) % (dim + 2) != 0)
//     {
//         throw std::runtime_error("Input data size does not match expected size for conversion to primitive variables.");
//     }
//     C_i /= (dim + 2);
//     size_t const num_input_elts = C_i * H * W;
    
//     // output_rho = input_rho
//     std::copy(input_dc.data(), input_dc.data() + num_input_elts, output_dc.data());
    
//     FloatBuffer v2_dc(num_input_elts);
//     // output_G = input_G / input_rho
//     for (size_t const icomp = 0; icomp < dim; ++icomp)
//     {
//         FloatBuffer input_G_dc(num_input_elts);
//         std::copy(input_dc.data() + (icomp + 1) * num_input_elts,
//                   input_dc.data() + (icomp + 2) * num_input_elts,
//                   input_G_dc.data());
//         FloatBuffer output_G_dc(num_input_elts);
//         std::copy(output_dc.data(), output_dc.data() + num_input_elts, 
//                     output_G_dc.data());
//         small::Div(C_i, H, W,
//                    input_G_dc, output_G_dc, gamma);
//         std::copy(output_G_dc.data(), output_G_dc.data() + num_input_elts, 
//                     output_dc.data() + (icomp + 1) * num_input_elts);

//         FloatBuffer tmp_dc(num_input_elts);
//         std::copy(output_G_dc.data(), output_G_dc.data() + num_input_elts, tmp_dc.data());
//         small::Mul(C_i, H, W,
//                    output_G_dc, tmp_dc);
//         small::Accum(C_i, H, W, 
//                     tmp_dc, v2_dc);
//     }

//     // a_W(NUMCOMPS-1) = (a_U(NUMCOMPS-1) - .5 * rho * v2) * (a_gamma - 1.0);
//     FloatBuffer input_E_dc(num_input_elts);
//     std::copy(input_dc.data() + (dim + 1) * num_input_elts,
//               input_dc.data() + (dim + 2) * num_input_elts,
//               input_E_dc.data());
//     FloatBuffer output_E_dc(num_input_elts);
//     std::copy(output_dc.data(), output_dc.data() + num_input_elts,
//               output_E_dc.data());
//     small::Mul(C_i, H, W,
//                output_E_dc, v2_dc);
//     FloatBuffer scalar_dc(1);
//     scalar_dc[0] = .5f;
//     small::MulScalar(C_i, H, W,
//                    v2_dc, scalar_dc, v2_dc);

    
    
// }


}
}

TEST_LIST = {
    {"stencil interpolation L",
     small::float_detail::test_stencil_interpolationL},
    {"stencil interpolation R",
     small::float_detail::test_stencil_interpolationR},
    // {"stencil upwind",
    //  small::float_detail::test_stencil_upwind},
    {NULL, NULL}
};

