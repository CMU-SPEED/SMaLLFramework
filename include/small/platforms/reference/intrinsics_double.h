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

#pragma once

#include <DoubleBuffer.hpp>
#include <cmath>

// scalar versions of all the double microkernels for platform portability
// Use the DOUBLE_ prefix for all macros in this file.

#define DOUBLE_SIMD_EPILOGUE 1

namespace small
{
    namespace double_detail
    {

        /// @todo both pairs of typedefs should not be needed.
        typedef small::DoubleBuffer::value_type dtype;

        typedef small::DoubleBuffer::value_type c_tile_t;

    }
}

// The macros in this file are intended to be used in a shared scope.
//****************************************************************************
//****************************************************************************
// Kernel Structure
//****************************************************************************
//****************************************************************************
// Steady State Kernel
//****************************************************************************
// Each kernel will
// 1) define a tile (size known at run-time) of C,
// 2) initialize it,
// 3) perform some computation in a loop,
// 4) and (optionally) store the result.

// The macros used in this kernel have the _TILE_C suffix.
// Pseduocode below
/*
    template <W_ob, C_ob, step>
    kernel(a, b, c)
    {
        DEF_TILE_C(W_ob, C_ob)

        if(ZERO)
        {
            ZERO_TILE_C(W_ob, C_ob)
        } else {
            LOAD_TILE_C(a, step, W_ob, C_ob)
        }

        computation loop
        {
            <COMPUTE>_TILE_C(step, a, b, W_ob, C_ob)
        }

        STORE_TILE_C(c, W_ob, C_ob)
    }

*/
//****************************************************************************
// Edge-case kernel
//****************************************************************************
// Each kernel will
// 1) define a tile (size known at run-time) of C,
// 2) initialize it,
// 3) perform some computation in a loop,
// 4) and (optionally) store the result.


// Macros have the _END_C suffix.
// Pseduocode below
/*
    template <step>
    kernel_end(a, b, c, W_ob, C_ob)
    {
        DEF_END_C(W_ob, C_ob)

        if(ZERO)
        {
            ZERO_END_C(W_ob, C_ob)
        }
        else
        {
            LOAD_END_C(a, step, W_ob, C_ob)
        }

        computation loop
        {
            <COMPUTE>_END_C(step, a, b, W_ob, C_ob)
        }

        STORE_END_C(c, W_ob, C_ob)
    }

*/

//****************************************************************************
// Initializations
//****************************************************************************

/**
 * @brief Macro to define a tile of C matrix with given dimensions.
 *
 * @param W_ob The width of the tile.              // Constant at compile time
 * @param C_ob The number of channels in the tile. // Constant at compile time
 */
#define DOUBLE_DEF_TILE_C(W_ob, C_ob) \
    c_tile_t c_tile[W_ob * C_ob];

/**
 * @brief Macro to define a tile of C matrix with given dimensions.
 *
 * @param W_ob The width of the tile.             // Variable, determined at runtime
 * @param C_ob The number of channels in the tile.// Variable, determined at runtime
 */
#define DOUBLE_DEF_END_C(W_ob, C_ob) \
    c_tile_t c_tile[W_ob * C_ob];

/**
 * @brief Macro to zero-initialize a (previously defined) tile of C matrix with given dimensions.
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 */
#define DOUBLE_ZERO_TILE_C(W_ob, C_ob)          \
    for (uint32_t kk = 0; kk < W_ob; kk++)     \
    {                                          \
        for (uint32_t jj = 0; jj < C_ob; jj++) \
        {                                      \
            c_tile[kk * C_ob + jj] = 0.0;      \
        }                                      \
    }

/**x
 * @brief Macro to zero-initialize the end of a tile of C matrix with given dimensions.
 *
 * @param _W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 */
#define DOUBLE_ZERO_END_C(W_ob, C_ob)           \
    for (uint32_t kk = 0; kk < W_ob; kk++)     \
    {                                          \
        for (uint32_t jj = 0; jj < C_ob; jj++) \
        {                                      \
            c_tile[kk * C_ob + jj] = 0.0;      \
        }                                      \
    }

//****************************************************************************
// Loads
//****************************************************************************

#define DOUBLE_LOAD_TILE_C(O, W_ob, C_ob)                \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
        }                                               \
    }

//  c_tile_t c_tile[W_ob * C_ob];
#define DOUBLE_LOAD_END_C(O, W_ob, C_ob)                 \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
        }                                               \
    }


//****************************************************************************
// Pooling Loads
//****************************************************************************

// strided loads
#define DOUBLE_LOAD_TILE_C_strided(O, step, W_ob, C_ob)         \
    for (uint32_t kk = 0; kk < W_ob; kk++)                     \
    {                                                          \
        for (uint32_t jj = 0; jj < C_ob; jj++)                 \
        {                                                      \
            c_tile[kk * C_ob + jj] = O[kk * step + jj];        \
        }                                                      \
    }

//  c_tile_t c_tile[W_ob * C_ob];
#define DOUBLE_LOAD_END_C_strided(O, step, W_ob, C_ob)   \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = O[kk * step + jj]; \
        }                                               \
    }


//****************************************************************************
// Upsampling loads (stride < 1)
//****************************************************************************

#define DOUBLE_LOAD_TILE_C_upsample(I, stride, _C_ib, W_ob, C_ob)        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            c_tile[kk * C_ob + jj] = I[(kk / stride) * (_C_ib) + jj];   \
        }                                                               \
    }

#define DOUBLE_LOAD_END_C_upsample(I, stride, _C_ib, W_ob, C_ob)         \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            c_tile[kk * C_ob + jj] = I[(kk / stride) * (_C_ib) + jj];   \
        }                                                               \
    }


//****************************************************************************
// Stores
//****************************************************************************

#define DOUBLE_STORE_TILE_C(O, W_ob, C_ob)               \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
        }                                               \
    }

#define DOUBLE_STORE_END_C(O, W_ob, C_ob)                \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
        }                                               \
    }


//****************************************************************************
// Convolution
//****************************************************************************

#define DOUBLE_CONV_TILE_C(step, a, b, W_ob, C_ob)       \
    c_tile_t *c_pixel = c_tile;                         \
    c_tile_t const *a_channel = a;                      \
    size_t a_offset = 0;                                \
    /*size_t c_offset = 0;*/                            \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        c_tile_t a_val = *(a_channel);                  \
        c_tile_t * c_channel = c_pixel;                 \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile_t b_val = *(b + jj);                 \
            *(c_channel) += a_val * b_val;              \
            c_channel++;                                \
            /*std::cout << "I_idx=" << a_offset << ", F_idx=" << jj << "--> C_idx=" << c_offset++ << std::endl;*/ \
        }                                               \
        a_channel += step;                              \
        a_offset += step;                               \
        c_pixel += C_ob;                                \
    }

#define DOUBLE_CONV_END_C(step, a, b, c_cur, W_ob, C_ob)         \
    c_tile_t *c_pixel = c_cur;                                  \
    c_tile_t const *a_channel = a;                              \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        c_tile_t a_val = *(a_channel);                          \
        c_tile_t * c_channel = c_pixel;                         \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            c_tile_t b_val = *(b + jj);                         \
            *(c_channel) += a_val * b_val;                      \
            c_channel++;                                        \
        }                                                       \
        a_channel += step;                                      \
        c_pixel += C_ob;                                        \
    }


//****************************************************************************
// Pooling
//   Max pooling
//****************************************************************************

#define DOUBLE_MAX_TILE_C(step, a, W_ob, C_ob)                           \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (*(a_channel) > *(c_channel)) ? *(a_channel) : *(c_channel); \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_MAX_END_C(step, a, c_cur, W_last, C_ob)                   \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_last; kk++)                            \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (*(a_channel) > *(c_channel)) ? *(a_channel) : *(c_channel); \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }


//****************************************************************************
// DW Convolution
//****************************************************************************

#define DOUBLE_DW_TILE_C(step, a, b, W_ob, C_ob)                 \
    {                                                           \
        c_tile_t *c_pixel = c_tile;                             \
        c_tile_t const *a_pixel = a;                            \
        for (uint32_t kk = 0; kk < W_ob; kk++)                  \
        {                                                       \
            c_tile_t *c_channel = c_pixel;                      \
            c_tile_t const *a_channel = a_pixel;                \
            c_tile_t const *b_channel = b;                      \
            for (uint32_t jj = 0; jj < C_ob; jj++)              \
            {                                                   \
                *(c_channel) += (*(a_channel) * *(b_channel));  \
                c_channel++;                                    \
                b_channel++;                                    \
                a_channel++;                                    \
            }                                                   \
            a_pixel += step;                                    \
            c_pixel += C_ob;                                    \
        }                                                       \
    }

#define DOUBLE_DW_END_C(step, a, b, c_cur, W_ob, C_ob)           \
    {                                                           \
        c_tile_t *c_pixel = c_cur;                              \
        c_tile_t const *a_pixel = a;                            \
        for (uint32_t kk = 0; kk < W_ob; kk++)                  \
        {                                                       \
            c_tile_t *c_channel = c_pixel;                      \
            c_tile_t const *a_channel = a_pixel;                \
            c_tile_t const *b_channel = b;                      \
            for (uint32_t jj = 0; jj < C_ob; jj++)              \
            {                                                   \
                *(c_channel) += (*(a_channel) * *(b_channel));  \
                c_channel++;                                    \
                b_channel++;                                    \
                a_channel++;                                    \
            }                                                   \
            a_pixel += step;                                    \
            c_pixel += C_ob;                                    \
        }                                                       \
    }

//****************************************************************************
// ReLU Activation
//****************************************************************************

// Same kernel as Pooling, set to zero to start.

// When Fused, compare with a register of zeros
#define DOUBLE_FUSED_RELU_TILE_C(W_ob, C_ob)                           \
    double *c_pixel = c_tile;                                          \
    for (uint32_t kk = 0; kk < W_ob; kk++)                            \
    {                                                                 \
        double *c_channel = c_pixel;                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                        \
        {                                                             \
            *(c_channel) = (0.0 > *(c_channel)) ? 0.0 : *(c_channel); \
            c_channel++;                                              \
        }                                                             \
        c_pixel += C_ob;                                              \
    }

#define DOUBLE_FUSED_RELU_END_C(c_cur, W_last, C_ob)                   \
    double *c_pixel = c_cur;                                           \
    for (uint32_t kk = 0; kk < W_last; kk++)                          \
    {                                                                 \
        double *c_channel = c_pixel;                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                        \
        {                                                             \
            *(c_channel) = (0.0 > *(c_channel)) ? 0.0 : *(c_channel); \
            c_channel++;                                              \
        }                                                             \
        c_pixel += C_ob;                                              \
    }
//****************************************************************************
// Leaky ReLU activation
//****************************************************************************

#define DOUBLE_COND_SCALE_TILE_C(step, a, b, W_ob, C_ob)                 \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    c_tile_t scale = b[0];                                              \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (*(a_channel) > *(c_channel)) ? *(a_channel) : (*(a_channel) * (scale)); \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_COND_SCALE_END_C(step, a, b, c_cur, W_last, C_ob)         \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t const *a_pixel = a;                                        \
    c_tile_t scale = b[0];                                              \
    for (uint32_t kk = 0; kk < W_last; kk++)                            \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (*(a_channel) > *(c_channel)) ? *(a_channel) : (*(a_channel) * (scale)); \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_FUSED_COND_SCALE_TILE_C(b, W_ob, C_ob)                                       \
    double *c_pixel = c_tile;                                                               \
    double scale = b[0];                                                                    \
    for (uint32_t kk = 0; kk < W_ob; kk++)                                                 \
    {                                                                                      \
        double *c_channel = c_pixel;                                                        \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                             \
        {                                                                                  \
            *(c_channel) = (0.0 > *(c_channel)) ? (*(c_channel) * (scale)) : *(c_channel); \
            c_channel++;                                                                   \
        }                                                                                  \
        c_pixel += C_ob;                                                                   \
    }

#define DOUBLE_FUSED_COND_SCALE_END_C(b, c_cur, W_last, C_ob)                               \
    double *c_pixel = c_cur;                                                                \
    double scale = b[0];                                                                    \
    for (uint32_t kk = 0; kk < W_last; kk++)                                               \
    {                                                                                      \
        double *c_channel = c_pixel;                                                        \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                             \
        {                                                                                  \
            *(c_channel) = (0.0 > *(c_channel)) ? (*(c_channel) * (scale)) : *(c_channel); \
            c_channel++;                                                                   \
        }                                                                                  \
        c_pixel += C_ob;                                                                   \
    }

//****************************************************************************
// Accumulation kernels
//****************************************************************************

#define DOUBLE_ACCUM_TILE_C(step, a, W_ob, C_ob) \
    double *c_pixel = c_tile;                    \
    double const *a_pixel = a;                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)      \
    {                                           \
        double *c_channel = c_pixel;             \
        double const *a_channel = a_pixel;       \
        for (uint32_t jj = 0; jj < C_ob; jj++)  \
        {                                       \
            *(c_channel) += *(a_channel);       \
            c_channel++;                        \
            a_channel++;                        \
        }                                       \
        a_pixel += step;                        \
        c_pixel += C_ob;                        \
    }

#define DOUBLE_ACCUM_END_C(step, a, c_cur, W_last, C_ob) \
    double const * a_in_channel = a;                     \
    for(uint32_t u =0 ; u < _UNROLL; u++)               \
    {                                                   \
        double *c_pixel = c_cur;                         \
        double const *a_pixel = a_in_channel;            \
        for (uint32_t kk = 0; kk < W_last; kk++)        \
        {                                               \
            double *c_channel = c_pixel;                 \
            double const *a_channel = a_pixel;           \
            for (uint32_t jj = 0; jj < C_ob; jj++)      \
            {                                           \
                *(c_channel) += *(a_channel);           \
                c_channel++;                            \
                a_channel++;                            \
            }                                           \
            a_pixel += step;                            \
            c_pixel += C_ob;                            \
        }                                               \
        a_in_channel++;                                 \
    }

//****************************************************************************
// Broadcast multiplication kernels
//****************************************************************************

#define DOUBLE_DIV_TILE_C(norm, W_ob, C_ob)     \
    double *c_pixel = c_tile;                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)     \
    {                                          \
        double *c_channel = c_pixel;            \
        for (uint32_t jj = 0; jj < C_ob; jj++) \
        {                                      \
            *(c_channel) *= norm;              \
            c_channel++;                       \
        }                                      \
        c_pixel += C_ob;                       \
    }

#define DOUBLE_DIV_END_C(c_cur, norm, W_last, C_ob)    \
    double *c_pixel = c_cur;                           \
    for (uint32_t kk = 0; kk < W_last; kk++)          \
    {                                                 \
        double *c_channel = c_pixel;                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)        \
        {                                             \
            *(c_channel) *= norm;                     \
            c_channel++;                              \
        }                                             \
        c_pixel += C_ob;                              \
    }

#define DOUBLE_EWISE_ADD_SCALAR_TILE_C(scalar, W_ob, C_ob)     \
    double *c_pixel = c_tile;                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)     \
    {                                          \
        double *c_channel = c_pixel;            \
        for (uint32_t jj = 0; jj < C_ob; jj++) \
        {                                      \
            *(c_channel) += scalar;              \
            c_channel++;                       \
        }                                      \
        c_pixel += C_ob;                       \
    }

#define DOUBLE_EWISE_ADD_SCALAR_END_C(c_cur, scalar, W_last, C_ob)    \
    double *c_pixel = c_cur;                           \
    for (uint32_t kk = 0; kk < W_last; kk++)          \
    {                                                 \
        double *c_channel = c_pixel;                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)        \
        {                                             \
            *(c_channel) += scalar;                     \
            c_channel++;                              \
        }                                             \
        c_pixel += C_ob;                              \
    }

//****************************************************************************
// Accumulate upsampling
//****************************************************************************
#define DOUBLE_ACCUM_TILE_C_upsample(I, stride, _C_ib, _W_ob, C_ob)     \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                            \
    {                                                                  \
        for (uint32_t jj = 0; jj < C_ob; jj++)                         \
        {                                                              \
            c_tile[kk * C_ob + jj] += I[(kk / stride) * (_C_ib) + jj]; \
        }                                                              \
    }

#define DOUBLE_ACCUM_END_C_upsample(I, stride, _C_ib, _W_ob, C_ob)      \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                            \
    {                                                                  \
        for (uint32_t jj = 0; jj < C_ob; jj++)                         \
        {                                                              \
            c_tile[kk * C_ob + jj] += I[(kk / stride) * (_C_ib) + jj]; \
        }                                                              \
    }

//****************************************************************************
// Accumulate channel dimension
//****************************************************************************

#define DOUBLE_REDUCE_CHANNEL_END_C(O_w_left, C_ob)                      \
    if constexpr (C_ob == 1 && C_ob != DOUBLE_SIMD_EPILOGUE)             \
    {                                                                   \
        double c_tile_array[DOUBLE_C_ob];                                 \
        for (uint32_t kk = 0; kk < O_w_left; kk++)                      \
        {                                                               \
            double *c_channel_v = c_tile + kk * (DOUBLE_C_ob);            \
            for (uint32_t jj = 1; jj < DOUBLE_C_ob; jj++)                \
            {                                                           \
                c_channel_v[0] += c_channel_v[jj];                      \
                c_channel_v[jj] = 0;                                    \
            }                                                           \
        }                                                               \
    }

//****************************************************************************
// Reduce kernels??
//****************************************************************************

#define DOUBLE_REDUCE_div_C(O, d, W_ob_g, C_ob)          \
    {                                                   \
        c_tile_t *c_pixel = c_tile;                     \
        c_tile_t *O_channel = O;                        \
        c_tile_t *c_channel = c_pixel;                  \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)        \
        {                                               \
            c_tile_t *O_channel = O;                    \
            c_tile_t *c_channel = c_pixel;              \
            for (uint32_t kk = 0; kk < C_ob; kk++)      \
            {                                           \
                *O_channel += *c_channel;               \
                O_channel++;                            \
                c_channel++;                            \
            }                                           \
            c_pixel += C_ob;                            \
        }                                               \
        O_channel = O;                                  \
        for (uint32_t kk = 0; kk < C_ob; kk++)          \
        {                                               \
            *O_channel *= d;                            \
            O_channel++;                                \
        }                                               \
    }

#define DOUBLE_REDUCE_C(O, W_ob_g, C_ob)                 \
    {                                                   \
        c_tile_t *c_pixel = c_tile;                     \
        c_tile_t *O_channel = O;                        \
        c_tile_t *c_channel = c_pixel;                  \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)        \
        {                                               \
            c_tile_t *O_channel = O;                    \
            c_tile_t *c_channel = c_pixel;              \
            for (uint32_t kk = 0; kk < C_ob; kk++)      \
            {                                           \
                *O_channel += *c_channel;               \
                O_channel++;                            \
                c_channel++;                            \
            }                                           \
            c_pixel += C_ob;                            \
        }                                               \
    }

#define DOUBLE_REDUCE_C_last(O, W_last, C_ob)            \
    {                                                   \
        c_tile_t *c_pixel = c_tile;                     \
        c_tile_t *O_channel = O;                        \
        c_tile_t *c_channel = c_pixel;                  \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)        \
        {                                               \
            c_tile_t *O_channel = O;                    \
            c_tile_t *c_channel = c_pixel;              \
            for (uint32_t kk = 0; kk < C_ob; kk++)      \
            {                                           \
                *O_channel += *c_channel;               \
                O_channel++;                            \
                c_channel++;                            \
            }                                           \
            c_pixel += C_ob;                            \
        }                                               \
    }

//****************************************************************************
// Softmax  (Ewise exponentiation)
//****************************************************************************

#define DOUBLE_EXP_TILE_C(step, a, W_ob, C_ob)                           \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = std::exp(*a_channel);                        \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_EXP_END_C(step, a, c_cur, W_last, C_ob) \
    c_tile_t *c_pixel = c_cur;                        \
    c_tile_t const *a_pixel = a;                      \
    for (uint32_t kk = 0; kk < W_last; kk++)          \
    {                                                 \
        c_tile_t *c_channel = c_pixel;                \
        c_tile_t const *a_channel = a_pixel;          \
        for (uint32_t jj = 0; jj < C_ob; jj++)        \
        {                                             \
            *(c_channel) = std::exp(*a_channel);      \
            c_channel++;                              \
            a_channel++;                              \
        }                                             \
        a_pixel += step;                              \
        c_pixel += C_ob;                              \
    }

#define DOUBLE_FUSED_EXP_TILE_C(W_ob, C_ob)       \
    c_tile_t *c_pixel = c_tile;                  \
    for (uint32_t kk = 0; kk < W_ob; kk++)       \
    {                                            \
        c_tile_t *c_channel = c_pixel;           \
        for (uint32_t jj = 0; jj < C_ob; jj++)   \
        {                                        \
            *(c_channel) = std::exp(*c_channel); \
            c_channel++;                         \
        }                                        \
        c_pixel += C_ob;                         \
    }

#define DOUBLE_FUSED_EXP_END_C(c_cur, W_last, C_ob) \
    c_tile_t *c_pixel = c_cur;                     \
    for (uint32_t kk = 0; kk < W_last; kk++)       \
    {                                              \
        c_tile_t *c_channel = c_pixel;             \
        for (uint32_t jj = 0; jj < C_ob; jj++)     \
        {                                          \
            *(c_channel) = std::exp(*c_channel);   \
            c_channel++;                           \
        }                                          \
        c_pixel += C_ob;                           \
    }


//****************************************************************************
// Softsign (Single-elementwise activation)
//****************************************************************************

#define DOUBLE_SOFTSIGN_TILE_C(step, a, W_ob, C_ob)                       \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = *(a_channel) / (1.0 + std::abs(*a_channel)); \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_SOFTSIGN_END_C(step, a, c_cur, W_last, C_ob) \
    c_tile_t *c_pixel = c_cur;                                  \
    c_tile_t const *a_pixel = a;                                \
    for (uint32_t kk = 0; kk < W_last; kk++)                    \
    {                                                           \
        c_tile_t *c_channel = c_pixel;                          \
        c_tile_t const *a_channel = a_pixel;                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            *(c_channel) = *(a_channel) / (1.0 + std::abs(*a_channel)); \
            c_channel++;                                        \
            a_channel++;                                        \
        }                                                       \
        a_pixel += step;                                        \
        c_pixel += C_ob;                                        \
    }

#define DOUBLE_FUSED_SOFTSIGN_TILE_C(W_ob, C_ob)       \
    c_tile_t *c_pixel = c_tile;                  \
    for (uint32_t kk = 0; kk < W_ob; kk++)       \
    {                                            \
        c_tile_t *c_channel = c_pixel;           \
        for (uint32_t jj = 0; jj < C_ob; jj++)   \
        {                                        \
            *(c_channel) = *(c_channel) / (1.0 + std::abs(*(c_channel))); \
            c_channel++;                         \
        }                                        \
        c_pixel += C_ob;                         \
    }

#define DOUBLE_FUSED_SOFTSIGN_END_C(c_cur, W_last, C_ob) \
    c_tile_t *c_pixel = c_cur;                     \
    for (uint32_t kk = 0; kk < W_last; kk++)       \
    {                                              \
        c_tile_t *c_channel = c_pixel;             \
        for (uint32_t jj = 0; jj < C_ob; jj++)     \
        {                                          \
            *(c_channel) = *(c_channel) / (1.0 + std::abs(*(c_channel))); \
            c_channel++;                           \
        }                                          \
        c_pixel += C_ob;                           \
    }

#define DOUBLE_ABS_TILE_C(step, a, W_ob, C_ob)                       \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = std::abs(*a_channel);                        \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_ABS_END_C(step, a, c_cur, W_last, C_ob) \
    c_tile_t *c_pixel = c_cur;                                  \
    c_tile_t const *a_pixel = a;                                \
    for (uint32_t kk = 0; kk < W_last; kk++)                    \
    {                                                           \
        c_tile_t *c_channel = c_pixel;                          \
        c_tile_t const *a_channel = a_pixel;                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            *(c_channel) = std::abs(*a_channel);                \
            c_channel++;                                        \
            a_channel++;                                        \
        }                                                       \
        a_pixel += step;                                        \
        c_pixel += C_ob;                                        \
    }

#define DOUBLE_FUSED_DIV_TILE_C(step, a, W_ob, C_ob) \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = *(a_channel) / *(c_channel);                 \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_FUSED_DIV_END_C(step, a, c_cur, W_last, C_ob) \
    c_tile_t *c_pixel = c_cur;                                  \
    c_tile_t const *a_pixel = a;                                \
    for (uint32_t kk = 0; kk < W_last; kk++)                    \
    {                                                           \
        c_tile_t *c_channel = c_pixel;                          \
        c_tile_t const *a_channel = a_pixel;                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            *(c_channel) = *(a_channel) / *(c_channel);         \
            c_channel++;                                        \
            a_channel++;                                        \
        }                                                       \
        a_pixel += step;                                        \
        c_pixel += C_ob;                                        \
    }

#define DOUBLE_SQRT_TILE_C(step, a, W_ob, C_ob)                       \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = std::sqrt(*(a_channel));                     \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_SQRT_END_C(step, a, c_cur, W_last, C_ob)               \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_last; kk++)                            \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = std::sqrt(*(a_channel));                     \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_MUL_TILE_C(step, a, W_ob, C_ob)                       \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) *= *(a_channel);                               \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_MUL_END_C(step, a, c_cur, W_last, C_ob) \
    c_tile_t *c_pixel = c_cur;                                  \
    c_tile_t const *a_pixel = a;                                \
    for (uint32_t kk = 0; kk < W_last; kk++)                    \
    {                                                           \
        c_tile_t *c_channel = c_pixel;                          \
        c_tile_t const *a_channel = a_pixel;                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            *(c_channel) *= *(a_channel);                       \
            c_channel++;                                        \
            a_channel++;                                        \
        }                                                       \
        a_pixel += step;                                        \
        c_pixel += C_ob;                                        \
    }

#define DOUBLE_SIGN_TILE_C(step, a, W_ob, C_ob)                          \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = *(c_channel) > 0 ? *(a_channel) : 0;         \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_SIGN_END_C(step, a, c_cur, W_last, C_ob)                  \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_last; kk++)                            \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = *(c_channel) > 0 ? *(a_channel) : 0;         \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_NSIGN_TILE_C(step, a, W_ob, C_ob)                         \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = *(c_channel) <= 0 ? *(a_channel) : 0;        \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_NSIGN_END_C(step, a, c_cur, W_last, C_ob)                 \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_last; kk++)                            \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = *(c_channel) <= 0 ? *(a_channel) : 0;        \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }


#define DOUBLE_SUB_TILE_C(step, a, W_ob, C_ob) \
    double *c_pixel = c_tile;                    \
    double const *a_pixel = a;                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)      \
    {                                           \
        double *c_channel = c_pixel;             \
        double const *a_channel = a_pixel;       \
        for (uint32_t jj = 0; jj < C_ob; jj++)  \
        {                                       \
            *(c_channel) -= *(a_channel);       \
            c_channel++;                        \
            a_channel++;                        \
        }                                       \
        a_pixel += step;                        \
        c_pixel += C_ob;                        \
    }

#define DOUBLE_SUB_END_C(step, a, c_cur, W_last, C_ob) \
    double const * a_in_channel = a;                     \
    for(uint32_t u =0 ; u < _UNROLL; u++)               \
    {                                                   \
        double *c_pixel = c_cur;                         \
        double const *a_pixel = a_in_channel;            \
        for (uint32_t kk = 0; kk < W_last; kk++)        \
        {                                               \
            double *c_channel = c_pixel;                 \
            double const *a_channel = a_pixel;           \
            for (uint32_t jj = 0; jj < C_ob; jj++)      \
            {                                           \
                *(c_channel) -= *(a_channel);           \
                c_channel++;                            \
                a_channel++;                            \
            }                                           \
            a_pixel += step;                            \
            c_pixel += C_ob;                            \
        }                                               \
        a_in_channel++;                                 \
    }

#define DOUBLE_COND_NSIGN_TILE_C(step, a, W_ob, C_ob)                     \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = *(c_channel) > 0 ? -*(a_channel) : *(a_channel);         \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define DOUBLE_COND_NSIGN_END_C(step, a, c_cur, W_last, C_ob)             \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_last; kk++)                            \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = *(c_channel) > 0 ? -*(a_channel) : *(a_channel);         \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }
