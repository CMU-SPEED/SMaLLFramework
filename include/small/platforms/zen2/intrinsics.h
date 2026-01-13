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

#include <params.h>
#include <Buffer.hpp>

#include <immintrin.h>
#include "avx_mathfun.h"

// scalar versions of all the float microkernels for platform portability
// Use the FLOAT_ prefix for all macros in this file.

#define FLOAT_SIMD_EPILOGUE 1
// #define FLOAT_SIMD_EPILOGUE 8

namespace small
{
    namespace float_detail
    {
#if FLOAT_SIMD_EPILOGUE == 1
        typedef small::FloatBuffer::value_type  c_tile_t;
#else
        typedef __m256 c_tile_t;
#endif
    }
}

/// @todo Remove a_i, b_i's from DEF_END/TILE_C and move as needed to kernels
///       and rename to I_i and W_i.


//****************************************************************************
// Definitions
//****************************************************************************

#define FLOAT_DEF_TILE_C                                                \
    __m256 a_reg, b0, b1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12;

/// @todo VERIFY this. Args are W_ob, C_ob but does not use C_ob
#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_DEF_END_C(W_ob, C_ob)             \
    c_tile_t c_tile[W_ob * FLOAT_C_ob];
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_DEF_END_C(W_ob, C_ob)                     \
    __m256 a_0, a_1, a_2, a_3, b_0, b_1;                \
    __m256 c_tile[W_ob * FLOAT_C_ob/FLOAT_SIMD];
#endif

//****************************************************************************
// Initializations
//****************************************************************************

#define FLOAT_ZERO_TILE_C                 \
        c0 = _mm256_setzero_ps();         \
        c1 = _mm256_setzero_ps();         \
        c2 = _mm256_setzero_ps();         \
        c3 = _mm256_setzero_ps();         \
        c4 = _mm256_setzero_ps();         \
        c5 = _mm256_setzero_ps();         \
        c6 = _mm256_setzero_ps();         \
        c7 = _mm256_setzero_ps();         \
        c8 = _mm256_setzero_ps();         \
        c9 = _mm256_setzero_ps();         \
        c10 = _mm256_setzero_ps();        \
        c11 = _mm256_setzero_ps();

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_ZERO_END_C(W_ob, C_ob)                \
    for (uint32_t kk = 0; kk < W_ob; kk++)          \
    {                                               \
        for (uint32_t jj = 0; jj < C_ob; jj++)      \
        {                                           \
            c_tile[kk * C_ob + jj] = 0.f;           \
        }                                           \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_ZERO_END_C(W_ob, C_ob)                 \
    for (uint32_t kk = 0; kk < W_ob; kk++)           \
    {                                                \
        c_tile[kk * C_ob + 0] = _mm256_setzero_ps(); \
        c_tile[kk * C_ob + 1] = _mm256_setzero_ps(); \
    }
#endif


//****************************************************************************
// Loads
//****************************************************************************

#define FLOAT_LOAD_TILE_C(I)                                      \
    {                                                             \
        c0 =  _mm256_load_ps(I + (0 * FLOAT_C_ob));               \
        c1 =  _mm256_load_ps(I + (0 * FLOAT_C_ob) + FLOAT_SIMD);  \
        c2 =  _mm256_load_ps(I + (1 * FLOAT_C_ob));               \
        c3 =  _mm256_load_ps(I + (1 * FLOAT_C_ob) + FLOAT_SIMD);  \
        c4 =  _mm256_load_ps(I + (2 * FLOAT_C_ob));               \
        c5 =  _mm256_load_ps(I + (2 * FLOAT_C_ob) + FLOAT_SIMD);  \
        c6 =  _mm256_load_ps(I + (3 * FLOAT_C_ob));               \
        c7 =  _mm256_load_ps(I + (3 * FLOAT_C_ob) + FLOAT_SIMD);  \
        c8 =  _mm256_load_ps(I + (4 * FLOAT_C_ob));               \
        c9 =  _mm256_load_ps(I + (4 * FLOAT_C_ob) + FLOAT_SIMD);  \
        c10 = _mm256_load_ps(I + (5 * FLOAT_C_ob));               \
        c11 = _mm256_load_ps(I + (5 * FLOAT_C_ob) + FLOAT_SIMD);  \
    }

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_LOAD_END_C(I, W_ob, C_ob)                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)                \
    {                                                     \
        for (uint32_t jj = 0; jj < C_ob; jj++)            \
        {                                                 \
            c_tile[kk * C_ob + jj] = I[kk * C_ob + jj];   \
        }                                                 \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_LOAD_END_C(I, W_ob, C_ob)                                 \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile[kk * C_ob + 0] = _mm256_load_ps(I + kk * C_ob + 0);      \
        c_tile[kk * C_ob + 1] = _mm256_load_ps(I + kk * C_ob + FLOAT_SIMD); \
    }
#endif
/// @todo missing a #else

//****************************************************************************
// Pooling Loads
//****************************************************************************

//  strided loads
// @todo: make this asm
#define FLOAT_LOAD_TILE_C_strided(I, step)                \
    c0 =  _mm256_load_ps(I + (0 * step));                 \
    c1 =  _mm256_load_ps(I + (0 * step) + FLOAT_SIMD);    \
    c2 =  _mm256_load_ps(I + (1 * step));                 \
    c3 =  _mm256_load_ps(I + (1 * step) + FLOAT_SIMD);    \
    c4 =  _mm256_load_ps(I + (2 * step));                 \
    c5 =  _mm256_load_ps(I + (2 * step) + FLOAT_SIMD);    \
    c6 =  _mm256_load_ps(I + (3 * step));                 \
    c7 =  _mm256_load_ps(I + (3 * step) + FLOAT_SIMD);    \
    c8 =  _mm256_load_ps(I + (4 * step));                 \
    c9 =  _mm256_load_ps(I + (4 * step) + FLOAT_SIMD);    \
    c10 = _mm256_load_ps(I + (5 * step));                 \
    c11 = _mm256_load_ps(I + (5 * step) + FLOAT_SIMD);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_LOAD_END_C_strided(I, step, W_ob, C_ob)   \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = I[kk * step + jj]; \
        }                                               \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_LOAD_END_C_strided(I, step, W_ob, C_ob)              \
    for (uint32_t kk = 0; kk < W_ob; kk++)                         \
    {                                                              \
        c_tile[kk * C_ob + 0] = _mm256_load_ps(I + kk * step + 0); \
        c_tile[kk * C_ob + 1] = _mm256_load_ps(I + kk * step + FLOAT_SIMD); \
    }
#endif

//****************************************************************************
// Upsampling loads (stride < 1, factor = 1/stride)
//****************************************************************************

#define FLOAT_LOAD_TILE_C_upsample(I, factor)                             \
    c0 =  _mm256_load_ps(I + ((0 / factor) * (FLOAT_C_ob)));              \
    c1 =  _mm256_load_ps(I + ((0 / factor) * (FLOAT_C_ob) + FLOAT_SIMD)); \
    c2 =  _mm256_load_ps(I + ((1 / factor) * (FLOAT_C_ob)));              \
    c3 =  _mm256_load_ps(I + ((1 / factor) * (FLOAT_C_ob) + FLOAT_SIMD)); \
    c4 =  _mm256_load_ps(I + ((2 / factor) * (FLOAT_C_ob)));              \
    c5 =  _mm256_load_ps(I + ((2 / factor) * (FLOAT_C_ob) + FLOAT_SIMD)); \
    c6 =  _mm256_load_ps(I + ((3 / factor) * (FLOAT_C_ob)));              \
    c7 =  _mm256_load_ps(I + ((3 / factor) * (FLOAT_C_ob) + FLOAT_SIMD)); \
    c8 =  _mm256_load_ps(I + ((4 / factor) * (FLOAT_C_ob)));              \
    c9 =  _mm256_load_ps(I + ((4 / factor) * (FLOAT_C_ob) + FLOAT_SIMD)); \
    c10 = _mm256_load_ps(I + ((5 / factor) * (FLOAT_C_ob)));              \
    c11 = _mm256_load_ps(I + ((5 / factor) * (FLOAT_C_ob) + FLOAT_SIMD));


#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_LOAD_END_C_upsample(I, factor, W_ob, C_ob)              \
    for (uint32_t kk = 0; kk < W_ob; kk++)                            \
    {                                                                 \
        for (uint32_t jj = 0; jj < C_ob; jj++)                        \
        {                                                             \
            c_tile[kk * C_ob + jj] = I[(kk / factor) * (C_ob) + jj];  \
        }                                                             \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_LOAD_END_C_upsample(I, factor, W_ob, C_ob)                \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile[kk * C_ob + 0] = _mm256_load_ps(I + (kk/factor) * (C_ob) + 0); \
        c_tile[kk * C_ob + 1] = _mm256_load_ps(I + (kk/factor) * (C_ob) + FLOAT_SIMD); \
    }
#endif


//****************************************************************************
// Stores
//****************************************************************************

#define FLOAT_STORE_TILE_C(O)                                    \
    {                                                            \
        _mm256_store_ps(O + (0 * FLOAT_C_ob), c0);               \
        _mm256_store_ps(O + (0 * FLOAT_C_ob) + FLOAT_SIMD, c1);  \
        _mm256_store_ps(O + (1 * FLOAT_C_ob), c2);               \
        _mm256_store_ps(O + (1 * FLOAT_C_ob + FLOAT_SIMD), c3);  \
        _mm256_store_ps(O + (2 * FLOAT_C_ob), c4);               \
        _mm256_store_ps(O + (2 * FLOAT_C_ob + FLOAT_SIMD), c5);  \
        _mm256_store_ps(O + (3 * FLOAT_C_ob), c6);               \
        _mm256_store_ps(O + (3 * FLOAT_C_ob + FLOAT_SIMD), c7);  \
        _mm256_store_ps(O + (4 * FLOAT_C_ob), c8);               \
        _mm256_store_ps(O + (4 * FLOAT_C_ob + FLOAT_SIMD), c9);  \
        _mm256_store_ps(O + (5 * FLOAT_C_ob), c10);              \
        _mm256_store_ps(O + (5 * FLOAT_C_ob + FLOAT_SIMD), c11); \
    }

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_STORE_END_C(O, W_ob, C_ob)                 \
    for (uint32_t kk = 0; kk < W_ob; kk++)               \
    {                                                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)           \
        {                                                \
            O[kk * C_ob + jj] = c_tile[kk * C_ob + jj];  \
        }                                                \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_STORE_END_C(O, W_ob, C_ob)                           \
    for (uint32_t kk = 0; kk < W_ob; kk++)                         \
    {                                                              \
        _mm256_store_ps(O + kk * C_ob + 0, c_tile[kk * C_ob + 0]); \
        _mm256_store_ps(O + kk * C_ob + FLOAT_SIMD, c_tile[kk * C_ob + 1]); \
    }
#endif

//****************************************************************************
/// @todo These strided kernels is not in reference
//****************************************************************************
#define FLOAT_STORE_TILE_C_strided(step, O)    \
    {                                                      \
        _mm256_store_ps(O + (0 * step), c0);               \
        _mm256_store_ps(O + (0 * step) + FLOAT_SIMD, c1);  \
        _mm256_store_ps(O + (1 * step), c2);               \
        _mm256_store_ps(O + (1 * step + FLOAT_SIMD), c3);  \
        _mm256_store_ps(O + (2 * step), c4);               \
        _mm256_store_ps(O + (2 * step + FLOAT_SIMD), c5);  \
        _mm256_store_ps(O + (3 * step), c6);               \
        _mm256_store_ps(O + (3 * step + FLOAT_SIMD), c7);  \
        _mm256_store_ps(O + (4 * step), c8);               \
        _mm256_store_ps(O + (4 * step + FLOAT_SIMD), c9);  \
        _mm256_store_ps(O + (5 * step), c10);              \
        _mm256_store_ps(O + (5 * step + FLOAT_SIMD), c11); \
    }

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_STORE_END_C_strided(step, O, W_ob, C_ob)  \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            O[kk * step + jj] = c_tile[kk * C_ob + jj]; \
        }                                               \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_STORE_END_C_strided(step, O, W_ob, C_ob)   \
    for (uint32_t kk = 0; kk < W_ob; kk++)               \
    {                                                    \
        _mm256_store_ps(O + kk * step + 0, c_tile[kk * C_ob + 0]); \
        _mm256_store_ps(O + kk * step + FLOAT_SIMD, c_tile[kk * C_ob + 1]); \
    }
#endif

//****************************************************************************
// Convolution Computation (Strided GEMM)
//****************************************************************************

#define FLOAT_CONV_TILE_C(step, I, W)                   \
    float const * I_ptr = I;                            \
    a_reg = _mm256_broadcast_ss(I_ptr + 0 * step);      \
    b0 = _mm256_load_ps(W);                             \
    b1 = _mm256_load_ps(W + FLOAT_SIMD);                \
    c12 = _mm256_broadcast_ss(I_ptr + 1 * step);        \
    c0 = _mm256_fmadd_ps(a_reg, b0, c0);                \
    c1 = _mm256_fmadd_ps(a_reg, b1, c1);                \
    a_reg = _mm256_broadcast_ss(I_ptr + 2 * step);      \
    I_ptr += 3*step;                                    \
    c2 = _mm256_fmadd_ps(c12, b0, c2);                  \
    c3 = _mm256_fmadd_ps(c12, b1, c3);                  \
    c12 = _mm256_broadcast_ss(I_ptr + 0 * step);        \
                                                        \
    c4 = _mm256_fmadd_ps(a_reg, b0, c4);                \
    c5 = _mm256_fmadd_ps(a_reg, b1, c5);                \
    a_reg = _mm256_broadcast_ss(I_ptr + 1 * step);      \
                                                        \
    c6 = _mm256_fmadd_ps(c12, b0, c6);                  \
    c7 = _mm256_fmadd_ps(c12, b1, c7);                  \
    c12 = _mm256_broadcast_ss(I_ptr + 2 * step);        \
                                                        \
    c8 = _mm256_fmadd_ps(a_reg, b0, c8);                \
    c9 = _mm256_fmadd_ps(a_reg, b1, c9);                \
                                                        \
    c10 = _mm256_fmadd_ps(c12, b0, c10);                \
    c11 = _mm256_fmadd_ps(c12, b1, c11);

/// @todo This implementation is different than REF
#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_CONV_END_C(step, I, W, c_cur, W_ob, C_ob)  \
    float *c_pixel;                                      \
    float const *I_channel = I;                          \
    for (uint32_t kk = 0; kk < W_ob; kk++)               \
    {                                                    \
        float I_val = *(I_channel);                      \
        c_pixel = c_cur + kk * C_ob;                     \
        for (uint32_t jj = 0; jj < C_ob; jj++)           \
        {                                                \
            float W_val = *(W + jj);                     \
            *(c_pixel + jj) += I_val * W_val;            \
        }                                                \
        I_channel += step;                               \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_CONV_END_C(step, I, W, c_cur, W_ob, C_ob)                 \
    b_0 = _mm256_load_ps(W);                                            \
    b_1 = _mm256_load_ps(W + FLOAT_SIMD);                               \
    __m256 c_pixel = c_cur;                                             \
    switch (W_ob)                                                       \
    {                                                                   \
    case 5:                                                             \
        a_0 = _mm256_broadcast_ss(I + 4 * step);                        \
        c_pixel[4 * (FLOAT_C_ob / FLOAT_SIMD) + 0] = _mm256_fmadd_ps(a_0, b_0, c_pixel[4 * (FLOAT_C_ob / FLOAT_SIMD) + 0]); \
        c_pixel[4 * (FLOAT_C_ob / FLOAT_SIMD) + 1] = _mm256_fmadd_ps(a_0, b_1, c_pixel[4 * (FLOAT_C_ob / FLOAT_SIMD) + 0]); \
    case 4:                                                             \
        a_3 = _mm256_broadcast_ss(I + 3 * step);                        \
        c_pixel[3 * (FLOAT_C_ob / FLOAT_SIMD) + 0] = _mm256_fmadd_ps(a_0, b_0, c_pixel[3 * (FLOAT_C_ob / FLOAT_SIMD) + 0]); \
        c_pixel[3 * (FLOAT_C_ob / FLOAT_SIMD) + 1] = _mm256_fmadd_ps(a_0, b_1, c_pixel[3 * (FLOAT_C_ob / FLOAT_SIMD) + 0]); \
    case 3:                                                             \
        a_2 = _mm256_broadcast_ss(I + 2 * step);                        \
        c_pixel[2 * (FLOAT_C_ob / FLOAT_SIMD) + 0] = _mm256_fmadd_ps(a_0, b_0, c_pixel[2 * (FLOAT_C_ob / FLOAT_SIMD) + 0]); \
        c_pixel[2 * (FLOAT_C_ob / FLOAT_SIMD) + 1] = _mm256_fmadd_ps(a_0, b_1, c_pixel[2 * (FLOAT_C_ob / FLOAT_SIMD) + 0]); \
    case 2:                                                             \
        a_1 = _mm256_broadcast_ss(I + 1 * step);                        \
        c_pixel[1 * (FLOAT_C_ob / FLOAT_SIMD) + 0] = _mm256_fmadd_ps(a_0, b_0, c_pixel[1 * (FLOAT_C_ob / FLOAT_SIMD) + 0]); \
        c_pixel[1 * (FLOAT_C_ob / FLOAT_SIMD) + 1] = _mm256_fmadd_ps(a_0, b_1, c_pixel[1 * (FLOAT_C_ob / FLOAT_SIMD) + 0]); \
    case 1:                                                             \
        a_0 = _mm256_broadcast_ss(I + 0 * step);                        \
        c_pixel[0 * (FLOAT_C_ob / FLOAT_SIMD) + 0] = _mm256_fmadd_ps(a_0, b_0, c_pixel[0 * (FLOAT_C_ob / FLOAT_SIMD) + 0]); \
        c_pixel[0 * (FLOAT_C_ob / FLOAT_SIMD) + 1] = _mm256_fmadd_ps(a_0, b_1, c_pixel[0 * (FLOAT_C_ob / FLOAT_SIMD) + 0]); \
    }
#endif

//****************************************************************************
// Pooling
//   Max pooling
//****************************************************************************

#define FLOAT_MAX_TILE_C(step, I)                      \
    b0 = _mm256_load_ps(I + (0 * step));               \
    b1 = _mm256_load_ps(I + (0 * step) + FLOAT_SIMD);  \
    c0 = _mm256_max_ps(b0, c0);                        \
    c1 = _mm256_max_ps(b1, c1);                        \
    a_reg = _mm256_load_ps(I + (1 * step));            \
    c12 = _mm256_load_ps(I + (1 * step) + FLOAT_SIMD); \
    c2 = _mm256_max_ps(a_reg, c2);                     \
    c3 = _mm256_max_ps(c12, c3);                       \
    b0 = _mm256_load_ps(I + (2 * step));               \
    b1 = _mm256_load_ps(I + (2 * step) + FLOAT_SIMD);  \
    c4 = _mm256_max_ps(b0, c4);                        \
    c5 = _mm256_max_ps(b1, c5);                        \
    a_reg = _mm256_load_ps(I + (3 * step));            \
    c12 = _mm256_load_ps(I + (3 * step) + FLOAT_SIMD); \
    c6 = _mm256_max_ps(a_reg, c6);                     \
    c7 = _mm256_max_ps(c12, c7);                       \
    b0 = _mm256_load_ps(I + (4 * step));               \
    b1 = _mm256_load_ps(I + (4 * step) + FLOAT_SIMD);  \
    c8 = _mm256_max_ps(b0, c8);                        \
    c9 = _mm256_max_ps(b1, c9);                        \
    a_reg = _mm256_load_ps(I + (5 * step));            \
    c12 = _mm256_load_ps(I + (5 * step) + FLOAT_SIMD); \
    c10 = _mm256_max_ps(a_reg, c10);                   \
    c11 = _mm256_max_ps(c12, c11);

/// @todo Replace float* with c_tile_t*?
#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_MAX_END_C(step, I, c_cur, W_ob, C_ob)                     \
    float *c_pixel = c_cur;                                             \
    float const *I_pixel = I;                                           \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        float *c_channel = c_pixel;                                     \
        float const *I_channel = I_pixel;                               \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (*(I_channel) > *(c_channel)) ? *(I_channel) : *(c_channel); \
            c_channel++;                                                \
            I_channel++;                                                \
        }                                                               \
        I_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_MAX_END_C(step, I, c_cur, W_ob, C_ob) \
for (uint32_t kk = 0; kk < W_ob; kk++)             \
{                                                     \
    __m256 *c_pixel = c_cur + kk * C_ob/FLOAT_SIMD;           \
    __m256 a_0 = _mm256_load_ps(I + kk * step + 0);  \
    __m256 a_1 = _mm256_load_ps(I + kk * step + FLOAT_SIMD); \
    c_pixel[0] = _mm256_max_ps(a_0, c_pixel[0]);     \
    c_pixel[1] = _mm256_max_ps(a_1, c_pixel[1]);     \
}
#endif

//****************************************************************************
// DW Convolution
//****************************************************************************

#define FLOAT_DW_TILE_C(step, I, W)                      \
    b0 = _mm256_load_ps(W);                              \
    b1 = _mm256_load_ps(W + FLOAT_SIMD);                 \
    c12 = _mm256_load_ps(I + (0 * step));                \
    a_reg = _mm256_load_ps(I + (0 * step) + FLOAT_SIMD); \
    c0 = _mm256_fmadd_ps(b0, c12, c0);                   \
    c1 = _mm256_fmadd_ps(b1, a_reg, c1);                 \
    c12 = _mm256_load_ps(I + (1 * step));                \
    a_reg = _mm256_load_ps(I + (1 * step) + FLOAT_SIMD); \
    c2 = _mm256_fmadd_ps(b0, c12, c2);                   \
    c3 = _mm256_fmadd_ps(b1, a_reg, c3);                 \
    c12 = _mm256_load_ps(I + (2 * step));                \
    a_reg = _mm256_load_ps(I + (2 * step) + FLOAT_SIMD); \
    c4 = _mm256_fmadd_ps(b0, c12, c4);                   \
    c5 = _mm256_fmadd_ps(b1, a_reg, c5);                 \
    c12 = _mm256_load_ps(I + (3 * step));                \
    a_reg = _mm256_load_ps(I + (3 * step) + FLOAT_SIMD); \
    c6 = _mm256_fmadd_ps(b0, c12, c6);                   \
    c7 = _mm256_fmadd_ps(b1, a_reg, c7);                 \
    c12 = _mm256_load_ps(I + (4 * step));                \
    a_reg = _mm256_load_ps(I + (4 * step) + FLOAT_SIMD); \
    c8 = _mm256_fmadd_ps(b0, c12, c8);                   \
    c9 = _mm256_fmadd_ps(b1, a_reg, c9);                 \
    c12 = _mm256_load_ps(I + (5 * step));                \
    a_reg = _mm256_load_ps(I + (5 * step) + FLOAT_SIMD); \
    c10 = _mm256_fmadd_ps(b0, c12, c10);                 \
    c11 = _mm256_fmadd_ps(b1, a_reg, c11);

/// @todo Replace float* with c_tile_t*?
#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_DW_END_C(step, I, W, c_cur, W_ob, C_ob)          \
    {                                                          \
        float *c_pixel = c_cur;                                \
        float const *I_pixel = I;                              \
        for (uint32_t kk = 0; kk < W_ob; kk++)                 \
        {                                                      \
            float *c_channel = c_pixel;                        \
            float const *I_channel = I_pixel;                  \
            float const *W_channel = W;                        \
            for (uint32_t jj = 0; jj < C_ob; jj++)             \
            {                                                  \
                *(c_channel) += (*(I_channel) * *(W_channel)); \
                c_channel++;                                   \
                W_channel++;                                   \
                I_channel++;                                   \
            }                                                  \
            I_pixel += step;                                   \
            c_pixel += C_ob;                                   \
        }                                                      \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_DW_END_C(step, I, W, c_cur, W_ob, C_ob)        \
    b_0 = _mm256_load_ps(W + 0);                             \
    b_1 = _mm256_load_ps(W + FLOAT_SIMD);                    \
    switch(W_ob)                                             \
    {                                                        \
        case 5:                                              \
            a_0 = _mm256_load_ps(I + 4 * step + 0);          \
            a_1 = _mm256_load_ps(I + 4 * step + FLOAT_SIMD); \
            c_cur[4 * (C_ob / FLOAT_SIMD) + 0] = _mm256_fmadd_ps(a_0, b_0, c_cur[4 * (C_ob / FLOAT_SIMD) + 0]);\
            c_cur[4 * (C_ob / FLOAT_SIMD) + 1] = _mm256_fmadd_ps(a_1, b_1, c_cur[4 * (C_ob / FLOAT_SIMD) + 1]);\
        case 4:                                              \
            a_2 = _mm256_load_ps(I + 3 * step + 0);          \
            a_3 = _mm256_load_ps(I + 3 * step + FLOAT_SIMD); \
            c_cur[3 * (C_ob / FLOAT_SIMD) + 0] = _mm256_fmadd_ps(a_2, b_0, c_cur[3 * (C_ob / FLOAT_SIMD) + 0]);\
            c_cur[3 * (C_ob / FLOAT_SIMD) + 1] = _mm256_fmadd_ps(a_3, b_1, c_cur[3 * (C_ob / FLOAT_SIMD) + 1]);\
        case 3:                                              \
            a_0 = _mm256_load_ps(I + 2 * step + 0);          \
            a_1 = _mm256_load_ps(I + 2 * step + FLOAT_SIMD); \
            c_cur[2 * (C_ob / FLOAT_SIMD) + 0] = _mm256_fmadd_ps(a_0, b_0, c_cur[2 * (C_ob / FLOAT_SIMD) + 0]);\
            c_cur[2 * (C_ob / FLOAT_SIMD) + 1] = _mm256_fmadd_ps(a_1, b_1, c_cur[2 * (C_ob / FLOAT_SIMD) + 1]);\
        case 2:                                              \
            a_2 = _mm256_load_ps(I + 1 * step + 0);          \
            a_3 = _mm256_load_ps(I + 1 * step + FLOAT_SIMD); \
            c_cur[1 * (C_ob / FLOAT_SIMD) + 0] = _mm256_fmadd_ps(a_2, b_0, c_cur[1 * (C_ob / FLOAT_SIMD) + 0]);\
            c_cur[1 * (C_ob / FLOAT_SIMD) + 1] = _mm256_fmadd_ps(a_3, b_1, c_cur[1 * (C_ob / FLOAT_SIMD) + 1]);\
        case 1:                                              \
            a_0 = _mm256_load_ps(I + 0 * step + 0);          \
            a_1 = _mm256_load_ps(I + 0 * step + FLOAT_SIMD); \
            c_cur[0 * (C_ob / FLOAT_SIMD) + 0] = _mm256_fmadd_ps(a_0, b_0, c_cur[0 * (C_ob / FLOAT_SIMD) + 0]);\
            c_cur[0 * (C_ob / FLOAT_SIMD) + 1] = _mm256_fmadd_ps(a_1, b_1, c_cur[0 * (C_ob / FLOAT_SIMD) + 1]);\
    }
#endif

//****************************************************************************
// ReLU Activation
//****************************************************************************

// Same kernel as Pooling, set to zero to start.

// When Fused, compare with a register of zeros
#define FLOAT_INPLACE_RELU_TILE_C           \
    a_reg = _mm256_setzero_ps();            \
    c0 = _mm256_max_ps(c0, a_reg);          \
    c1 = _mm256_max_ps(c1, a_reg);          \
    c2 = _mm256_max_ps(c2, a_reg);          \
    c3 = _mm256_max_ps(c3, a_reg);          \
    c4 = _mm256_max_ps(c4, a_reg);          \
    c5 = _mm256_max_ps(c5, a_reg);          \
    c6 = _mm256_max_ps(c6, a_reg);          \
    c7 = _mm256_max_ps(c7, a_reg);          \
    c8 = _mm256_max_ps(c8, a_reg);          \
    c9 = _mm256_max_ps(c9, a_reg);          \
    c10 = _mm256_max_ps(c10, a_reg);        \
    c11 = _mm256_max_ps(c11, a_reg);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_INPLACE_RELU_END_C(c_cur, W_ob, C_ob)                   \
    float *c_pixel = c_cur;                                           \
    for (uint32_t kk = 0; kk < W_ob; kk++)                            \
    {                                                                 \
        float *c_channel = c_pixel;                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                        \
        {                                                             \
            *(c_channel) = (0.0 > *(c_channel)) ? 0.0 : *(c_channel); \
            c_channel++;                                              \
        }                                                             \
        c_pixel += C_ob;                                              \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_INPLACE_RELU_END_C(c_cur, W_ob, C_ob)     \
    a_0 = _mm256_setzero_ps();                          \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        c_cur[kk * C_ob/FLOAT_SIMD + 0] = _mm256_max_ps(c_cur[kk * C_ob/FLOAT_SIMD + 0], a_0); \
        c_cur[kk * C_ob/FLOAT_SIMD + 1] = _mm256_max_ps(c_cur[kk * C_ob/FLOAT_SIMD + 1], a_0); \
    }
#endif

//****************************************************************************
// Leaky ReLU activation
//****************************************************************************
//Leaky relu is really a composite operator
#define FLOAT_COMPOSITE_COND_SCALE_TILE_C(step, I, W)             \
    FLOAT_COND_SCALE_TILE_C(step, I, W)

#define FLOAT_COND_SCALE_TILE_C(step, I, W)             \
    c12 = _mm256_broadcast_ss(W);                       \
    b0 = _mm256_load_ps(I + (0 * step));                \
    c0 = _mm256_max_ps(b0, c0);                         \
    a_reg = _mm256_cmp_ps(b0, c0, _CMP_LT_OS);          \
    b0 = _mm256_mul_ps(b0, c12);                        \
    b0 = _mm256_and_ps(a_reg, b0);                      \
    c0 = _mm256_add_ps(b0, c0);                         \
    /**/                                                \
    b1 = _mm256_load_ps(I + (0 * step) + FLOAT_SIMD);   \
    c1 = _mm256_max_ps(b1, c1);                         \
    a_reg = _mm256_cmp_ps(b1, c1, _CMP_LT_OS);          \
    b1 = _mm256_mul_ps(b1, c12);                        \
    b1 = _mm256_and_ps(a_reg, b1);                      \
    c1 = _mm256_add_ps(b1, c1);                         \
    /**/                                                \
    b0 = _mm256_load_ps(I + (1 * step));                \
    c2 = _mm256_max_ps(b0, c2);                         \
    a_reg = _mm256_cmp_ps(b0, c2, _CMP_LT_OS);          \
    b0 = _mm256_mul_ps(b0, c12);                        \
    b0 = _mm256_and_ps(a_reg, b0);                      \
    c2 = _mm256_add_ps(b0, c2);                         \
    /**/                                                \
    b1 = _mm256_load_ps(I + (1 * step) + FLOAT_SIMD);   \
    c3 = _mm256_max_ps(b1, c3);                         \
    a_reg = _mm256_cmp_ps(b1, c3, _CMP_LT_OS);          \
    b1 = _mm256_mul_ps(b1, c12);                        \
    b1 = _mm256_and_ps(a_reg, b1);                      \
    c3 = _mm256_add_ps(b1, c3);                         \
    /**/                                                \
    b0 = _mm256_load_ps(I + (2 * step));                \
    c4 = _mm256_max_ps(b0, c4);                         \
    a_reg = _mm256_cmp_ps(b0, c4, _CMP_LT_OS);          \
    b0 = _mm256_mul_ps(b0, c12);                        \
    b0 = _mm256_and_ps(a_reg, b0);                      \
    c4 = _mm256_add_ps(b0, c4);                         \
    /**/                                                \
    b1 = _mm256_load_ps(I + (2 * step) + FLOAT_SIMD);   \
    c5 = _mm256_max_ps(b1, c5);                         \
    a_reg = _mm256_cmp_ps(b1, c5, _CMP_LT_OS);          \
    b1 = _mm256_mul_ps(b1, c12);                        \
    b1 = _mm256_and_ps(a_reg, b1);                      \
    c5 = _mm256_add_ps(b1, c5);                         \
    /**/                                                \
    b0 = _mm256_load_ps(I + (3 * step));                \
    c6 = _mm256_max_ps(b0, c6);                         \
    a_reg = _mm256_cmp_ps(b0, c6, _CMP_LT_OS);          \
    b0 = _mm256_mul_ps(b0, c12);                        \
    b0 = _mm256_and_ps(a_reg, b0);                      \
    c6 = _mm256_add_ps(b0, c6);                         \
    /**/                                                \
    b1 = _mm256_load_ps(I + (3 * step) + FLOAT_SIMD);   \
    c7 = _mm256_max_ps(b1, c7);                         \
    a_reg = _mm256_cmp_ps(b1, c7, _CMP_LT_OS);          \
    b1 = _mm256_mul_ps(b1, c12);                        \
    b1 = _mm256_and_ps(a_reg, b1);                      \
    c7 = _mm256_add_ps(b1, c7);                         \
    /**/                                                \
    b0 = _mm256_load_ps(I + (4 * step));                \
    c8 = _mm256_max_ps(b0, c8);                         \
    a_reg = _mm256_cmp_ps(b0, c8, _CMP_LT_OS);          \
    b0 = _mm256_mul_ps(b0, c12);                        \
    b0 = _mm256_and_ps(a_reg, b0);                      \
    c8 = _mm256_add_ps(b0, c8);                         \
    /**/                                                \
    b1 = _mm256_load_ps(I + (4 * step) + FLOAT_SIMD);   \
    c9 = _mm256_max_ps(b1, c9);                         \
    a_reg = _mm256_cmp_ps(b1, c9, _CMP_LT_OS);          \
    b1 = _mm256_mul_ps(b1, c12);                        \
    b1 = _mm256_and_ps(a_reg, b1);                      \
    c9 = _mm256_add_ps(b1, c9);                         \
    /**/                                                \
    b0 = _mm256_load_ps(I + (5 * step));                \
    c10 = _mm256_max_ps(b0, c10);                       \
    a_reg = _mm256_cmp_ps(b0, c10, _CMP_LT_OS);         \
    b0 = _mm256_mul_ps(b0, c12);                        \
    b0 = _mm256_and_ps(a_reg, b0);                      \
    c10 = _mm256_add_ps(b0, c10);                       \
    /**/                                                \
    b1 = _mm256_load_ps(I + (5 * step) + FLOAT_SIMD);   \
    c11 = _mm256_max_ps(b1, c11);                       \
    a_reg = _mm256_cmp_ps(b1, c11, _CMP_LT_OS);         \
    b1 = _mm256_mul_ps(b1, c12);                        \
    b1 = _mm256_and_ps(a_reg, b1);                      \
    c11 = _mm256_add_ps(b1, c11);


#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_COND_SCALE_END_C(step, I, W, c_cur, W_ob, C_ob)                                       \
    float *c_pixel = c_cur;                                                                         \
    float const *I_pixel = I;                                                                       \
    float scale = W[0];                                                                             \
    for (uint32_t kk = 0; kk < W_ob; kk++)                                                          \
    {                                                                                               \
        float *c_channel = c_pixel;                                                                 \
        float const *I_channel = I_pixel;                                                           \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                                      \
        {                                                                                           \
            *(c_channel) = (*(I_channel) > *(c_channel)) ? *(I_channel) : (*(I_channel) * (scale)); \
            c_channel++;                                                                            \
            I_channel++;                                                                            \
        }                                                                                           \
        I_pixel += step;                                                                            \
        c_pixel += C_ob;                                                                            \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_COND_SCALE_END_C(step, I, W, c_cur, W_ob, C_ob)         \
    b_0 = _mm256_broadcast_ss(W);                                     \
    __m256 *c_pixel = c_cur;                                          \
    for (uint32_t kk = 0; kk < W_ob; kk++)                            \
    {                                                                 \
        a_0 = _mm256_load_ps(I + (kk * step));                        \
        c_pixel[0] = _mm256_max_ps(a_0, c_pixel[0]);                  \
        a_2 = _mm256_cmp_ps(a_0, c_pixel[0], _CMP_LT_OS);             \
        a_0 = _mm256_mul_ps(a_0, b_0);                                \
        a_0 = _mm256_and_ps(a_2, a_0);                                \
        c_pixel[0] = _mm256_add_ps(a_0, c_pixel[0]);                  \
        a_1 = _mm256_load_ps(I + (kk * step) + FLOAT_SIMD);           \
        c_pixel[1] = _mm256_max_ps(a_1, c_pixel[1]);                  \
        a_3 = _mm256_cmp_ps(a_1, c_pixel[1], _CMP_LT_OS);             \
        a_1 = _mm256_mul_ps(a_1, b_0);                                \
        a_1 = _mm256_and_ps(a_3, a_1);                                \
        c_pixel[1] = _mm256_add_ps(a_1, c_pixel[1]);                  \
        c_pixel += (C_ob/FLOAT_SIMD);                                 \
    }
#endif

#define FLOAT_INPLACE_COND_SCALE_TILE_C(W)           \
    c12 = _mm256_broadcast_ss(W); /*scale*/          \
    b0 = _mm256_setzero_ps();                        \
    /**/                                             \
    a_reg = _mm256_cmp_ps(c0, b0, _CMP_LT_OS);       \
    b1 = _mm256_cmp_ps(c1, b0, _CMP_LT_OS);          \
    a_reg = _mm256_and_ps(a_reg, c0);                \
    b1 = _mm256_and_ps(b1, c1);                      \
    a_reg = _mm256_mul_ps(a_reg, c12);               \
    b1 = _mm256_mul_ps(b1, c12);                     \
    c0 = _mm256_max_ps(b0, c0);                      \
    c1 = _mm256_max_ps(b0, c1);                      \
    c0 = _mm256_add_ps(a_reg, c0);                   \
    c1 = _mm256_add_ps(b1, c1);                      \
    /**/                                             \
    a_reg = _mm256_cmp_ps(c2, b0, _CMP_LT_OS);       \
    b1 = _mm256_cmp_ps(c3, b0, _CMP_LT_OS);          \
    a_reg = _mm256_and_ps(a_reg, c2);                \
    b1 = _mm256_and_ps(b1, c3);                      \
    a_reg = _mm256_mul_ps(a_reg, c12);               \
    b1 = _mm256_mul_ps(b1, c12);                     \
    c2 = _mm256_max_ps(b0, c2);                      \
    c3 = _mm256_max_ps(b0, c3);                      \
    c2 = _mm256_add_ps(a_reg, c2);                   \
    c3 = _mm256_add_ps(b1, c3);                      \
    /**/                                             \
    a_reg = _mm256_cmp_ps(c4, b0, _CMP_LT_OS);       \
    b1 = _mm256_cmp_ps(c5, b0, _CMP_LT_OS);          \
    a_reg = _mm256_and_ps(a_reg, c4);                \
    b1 = _mm256_and_ps(b1, c5);                      \
    a_reg = _mm256_mul_ps(a_reg, c12);               \
    b1 = _mm256_mul_ps(b1, c12);                     \
    c4 = _mm256_max_ps(b0, c4);                      \
    c5 = _mm256_max_ps(b0, c5);                      \
    c4 = _mm256_add_ps(a_reg, c4);                   \
    c5 = _mm256_add_ps(b1, c5);                      \
    /**/                                             \
    a_reg = _mm256_cmp_ps(c6, b0, _CMP_LT_OS);       \
    b1 = _mm256_cmp_ps(c7, b0, _CMP_LT_OS);          \
    a_reg = _mm256_and_ps(a_reg, c6);                \
    b1 = _mm256_and_ps(b1, c7);                      \
    a_reg = _mm256_mul_ps(a_reg, c12);               \
    b1 = _mm256_mul_ps(b1, c12);                     \
    c6 = _mm256_max_ps(b0, c6);                      \
    c7 = _mm256_max_ps(b0, c7);                      \
    c6 = _mm256_add_ps(a_reg, c6);                   \
    c7 = _mm256_add_ps(b1, c7);                      \
    /**/                                             \
    a_reg = _mm256_cmp_ps(c8, b0, _CMP_LT_OS);       \
    b1 = _mm256_cmp_ps(c9, b0, _CMP_LT_OS);          \
    a_reg = _mm256_and_ps(a_reg, c8);                \
    b1 = _mm256_and_ps(b1, c9);                      \
    a_reg = _mm256_mul_ps(a_reg, c12);               \
    b1 = _mm256_mul_ps(b1, c12);                     \
    c8 = _mm256_max_ps(b0, c8);                      \
    c9 = _mm256_max_ps(b0, c9);                      \
    c8 = _mm256_add_ps(a_reg, c8);                   \
    c9 = _mm256_add_ps(b1, c9);                      \
    /**/                                             \
    a_reg = _mm256_cmp_ps(c10, b0, _CMP_LT_OS);      \
    b1 = _mm256_cmp_ps(c11, b0, _CMP_LT_OS);         \
    a_reg = _mm256_and_ps(a_reg, c10);               \
    b1 = _mm256_and_ps(b1, c11);                     \
    a_reg = _mm256_mul_ps(a_reg, c12);               \
    b1 = _mm256_mul_ps(b1, c12);                     \
    c10 = _mm256_max_ps(b0, c10);                    \
    c11 = _mm256_max_ps(b0, c11);                    \
    c10 = _mm256_add_ps(a_reg, c10);                 \
    c11 = _mm256_add_ps(b1, c11);


#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_INPLACE_COND_SCALE_END_C(W, c_cur, W_ob, C_ob)            \
    float *c_pixel = c_cur;                                             \
    float scale = W[0];                                                 \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        float *c_channel = c_pixel;                                     \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (0.0 > *(c_channel)) ? (*(c_channel) * (scale)) : *(c_channel); \
            c_channel++;                                                \
        }                                                               \
        c_pixel += C_ob;                                                \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_INPLACE_COND_SCALE_END_C(W, c_cur, W_ob, C_ob)        \
    b_0 = _mm256_broadcast_ss(W);                                   \
    b_1 = _mm256_setzero_ps();                                      \
    __m256 *c_pixel = c_cur;                                        \
    for (uint32_t kk = 0; kk < W_last; kk++)                        \
    {                                                               \
        a_0 = _mm256_cmp_ps(c_pixel[0], b_1, _CMP_LT_OS);           \
        a_1 = _mm256_cmp_ps(c_pixel[1], b_1, _CMP_LT_OS);           \
        a_0 = _mm256_and_ps(a_0, c_pixel[0]);                       \
        a_1 = _mm256_and_ps(a_1, c_pixel[1]);                       \
        a_0 = _mm256_mul_ps(a_0, b_0);                              \
        a_1 = _mm256_mul_ps(a_1, b_0);                              \
        c_pixel[0] = _mm256_max_ps(b_1, c_pixel[0]);                \
        c_pixel[1] = _mm256_max_ps(b_1, c_pixel[1]);                \
        c_pixel[0] = _mm256_add_ps(a_0, c_pixel[0]);                \
        c_pixel[1] = _mm256_add_ps(a_1, c_pixel[1]);                \
        c_pixel += (C_ob/FLOAT_SIMD);                               \
    }
#endif

#define FLOAT_ACCUM_TILE_C(step, I)                    \
    b0 = _mm256_load_ps(I + (0 * step));               \
    b1 = _mm256_load_ps(I + (0 * step) + FLOAT_SIMD);  \
    c0 = _mm256_add_ps(b0, c0);                        \
    c1 = _mm256_add_ps(b1, c1);                        \
    a_reg = _mm256_load_ps(I + (1 * step));            \
    c12 = _mm256_load_ps(I + (1 * step) + FLOAT_SIMD); \
    c2 = _mm256_add_ps(a_reg, c2);                     \
    c3 = _mm256_add_ps(c12, c3);                       \
    b0 = _mm256_load_ps(I + (2 * step));               \
    b1 = _mm256_load_ps(I + (2 * step) + FLOAT_SIMD);  \
    c4 = _mm256_add_ps(b0, c4);                        \
    c5 = _mm256_add_ps(b1, c5);                        \
    a_reg = _mm256_load_ps(I + (3 * step));            \
    c12 = _mm256_load_ps(I + (3 * step) + FLOAT_SIMD); \
    c6 = _mm256_add_ps(a_reg, c6);                     \
    c7 = _mm256_add_ps(c12, c7);                       \
    b0 = _mm256_load_ps(I + (4 * step));               \
    b1 = _mm256_load_ps(I + (4 * step) + FLOAT_SIMD);  \
    c8 = _mm256_add_ps(b0, c8);                        \
    c9 = _mm256_add_ps(b1, c9);                        \
    a_reg = _mm256_load_ps(I + (5 * step));            \
    c12 = _mm256_load_ps(I + (5 * step) + FLOAT_SIMD); \
    c10 = _mm256_add_ps(a_reg, c10);                   \
    c11 = _mm256_add_ps(c12, c11);


#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_ACCUM_END_C(step, I, c_cur, W_ob, C_ob)   \
    float const *I_in_channel = I;                      \
    for (uint32_t u = 0; u < _UNROLL; u++)              \
    {                                                   \
        float *c_pixel = c_cur;                         \
        float const *I_pixel = I_in_channel;            \
        for (uint32_t kk = 0; kk < W_ob; kk++)          \
        {                                               \
            float *c_channel = c_pixel;                 \
            float const *I_channel = I_pixel;           \
            for (uint32_t jj = 0; jj < C_ob; jj++)      \
            {                                           \
                *(c_channel) += *(I_channel);           \
                c_channel++;                            \
                I_channel++;                            \
            }                                           \
            I_pixel += step;                            \
            c_pixel += C_ob;                            \
        }                                               \
        I_in_channel++;                                 \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_ACCUM_END_C(step, I, c_cur, W_ob, C_ob)       \
    __m256 *c_pixel = c_cur;                                \
    for (uint32_t kk = 0; kk < W_ob; kk++)                  \
    {                                                       \
        a_0 = _mm256_load_ps(I + (kk * step));              \
        c_pixel[0] = _mm256_add_ps(a_0, c_pixel[0]);        \
        a_1 = _mm256_load_ps(I + (kk * step) + FLOAT_SIMD); \
        c_pixel[1] = _mm256_add_ps(a_1, c_pixel[1]);        \
        c_pixel += (C_ob/FLOAT_SIMD);                       \
    }
#endif


/// @todo This is not in reference
#define FLOAT_INPLACE_ACCUM_TILE_C                   \
    c10 = _mm256_add_ps(c10, c11);                   \
    c9 = _mm256_add_ps(c9, c10);                     \
    c8 = _mm256_add_ps(c8, c9);                      \
    c7 = _mm256_add_ps(c7, c8);                      \
    c6 = _mm256_add_ps(c6, c7);                      \
    c5 = _mm256_add_ps(c5, c6);                      \
    c4 = _mm256_add_ps(c4, c5);                      \
    c3 = _mm256_add_ps(c3, c4);                      \
    c2 = _mm256_add_ps(c2, c3);                      \
    c1 = _mm256_add_ps(c1, c2);                      \
    c0 = _mm256_add_ps(c0, c1);


//****************************************************************************
// Broadcast multiplication kernels
//****************************************************************************

#define FLOAT_INPLACE_MUL_SCALAR_TILE_C(scale)  \
    b0 = _mm256_broadcast_ss(&scale);      \
    c0 = _mm256_mul_ps(b0, c0);            \
    c1 = _mm256_mul_ps(b0, c1);            \
    c2 = _mm256_mul_ps(b0, c2);            \
    c3 = _mm256_mul_ps(b0, c3);            \
    c4 = _mm256_mul_ps(b0, c4);            \
    c5 = _mm256_mul_ps(b0, c5);            \
    c6 = _mm256_mul_ps(b0, c6);            \
    c7 = _mm256_mul_ps(b0, c7);            \
    c8 = _mm256_mul_ps(b0, c8);            \
    c9 = _mm256_mul_ps(b0, c9);            \
    c10 = _mm256_mul_ps(b0, c10);          \
    c11 = _mm256_mul_ps(b0, c11);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_INPLACE_MUL_SCALAR_END_C(c_cur, scale, W_ob, C_ob)  \
    float *c_pixel = c_cur;                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)         \
    {                                              \
        float *c_channel = c_pixel;                \
        for (uint32_t jj = 0; jj < C_ob; jj++)     \
        {                                          \
            *(c_channel) *= scale;                 \
            c_channel++;                           \
        }                                          \
        c_pixel += C_ob;                           \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_INPLACE_MUL_SCALAR_END_C(c_cur, scale, W_ob, C_ob)    \
    b_0 = _mm256_broadcast_ss(&scale);               \
    __m256 *c_pixel = c_cur;                         \
    for (uint32_t kk = 0; kk < W_ob; kk++)           \
    {                                                \
        c_pixel[0] = _mm256_mul_ps(b_0, c_pixel[0]); \
        c_pixel[1] = _mm256_mul_ps(b_0, c_pixel[1]); \
        c_pixel += (C_ob/FLOAT_SIMD);                \
    }
#endif

//****************************************************************************
// Broadcast Addition kernels
//****************************************************************************

#define FLOAT_INPLACE_ADD_SCALAR_TILE_C(scalar) \
    b0 = _mm256_broadcast_ss(&scalar);        \
    c0 = _mm256_add_ps(b0, c0);               \
    c1 = _mm256_add_ps(b0, c1);               \
    c2 = _mm256_add_ps(b0, c2);               \
    c3 = _mm256_add_ps(b0, c3);               \
    c4 = _mm256_add_ps(b0, c4);               \
    c5 = _mm256_add_ps(b0, c5);               \
    c6 = _mm256_add_ps(b0, c6);               \
    c7 = _mm256_add_ps(b0, c7);               \
    c8 = _mm256_add_ps(b0, c8);               \
    c9 = _mm256_add_ps(b0, c9);               \
    c10 = _mm256_add_ps(b0, c10);             \
    c11 = _mm256_add_ps(b0, c11);


#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_INPLACE_ADD_SCALAR_END_C(c_cur, scalar, W_ob, C_ob) \
    float *c_pixel = c_cur;                                     \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        float *c_channel = c_pixel;                             \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            *(c_channel) += scalar;                             \
            c_channel++;                                        \
        }                                                       \
        c_pixel += C_ob;                                        \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_INPLACE_ADD_SCALAR_END_C(c_cur, scalar, W_ob, C_ob) \
    b_0 = _mm256_broadcast_ss(&scalar);                         \
    __m256 *c_pixel = c_cur;                                    \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        c_pixel[0] = _mm256_add_ps(b_0, c_pixel[0]);            \
        c_pixel[1] = _mm256_add_ps(b_0, c_pixel[1]);            \
        c_pixel += (C_ob/FLOAT_SIMD);                           \
    }
#endif

//****************************************************************************
// Accumulate upsampling
//****************************************************************************

#define FLOAT_ACCUM_TILE_C_upsample(I, factor)                            \
    a_reg = _mm256_load_ps(I + ((0 / factor) * (FLOAT_C_ob)));            \
    c12 = _mm256_load_ps(I + ((0 / factor) * (FLOAT_C_ob) + FLOAT_SIMD)); \
    b0 = _mm256_load_ps(I + ((1 / factor) * (FLOAT_C_ob)));               \
    b1 = _mm256_load_ps(I + ((1 / factor) * (FLOAT_C_ob) + FLOAT_SIMD));  \
    c0 = _mm256_add_ps(c0, a_reg);                                        \
    a_reg = _mm256_load_ps(I + ((2 / factor) * (FLOAT_C_ob)));            \
    c1 = _mm256_add_ps(c1, c12);                                          \
    c12 = _mm256_load_ps(I + ((2 / factor) * (FLOAT_C_ob) + FLOAT_SIMD)); \
    c2 = _mm256_add_ps(c2, b0);                                           \
    b0 = _mm256_load_ps(I + ((3 / factor) * (FLOAT_C_ob)));               \
    c3 = _mm256_add_ps(c3, b1);                                           \
    b1 = _mm256_load_ps(I + ((3 / factor) * (FLOAT_C_ob) + FLOAT_SIMD));  \
    c4 = _mm256_add_ps(c4, a_reg);                                        \
    a_reg = _mm256_load_ps(I + ((4 / factor) * (FLOAT_C_ob)));            \
    c5 = _mm256_add_ps(c5, c12);                                          \
    c12 = _mm256_load_ps(I + ((4 / factor) * (FLOAT_C_ob) + FLOAT_SIMD)); \
    c6 = _mm256_add_ps(c6, b0);                                           \
    b0 = _mm256_load_ps(I + ((5 / factor) * (FLOAT_C_ob)));               \
    c7 = _mm256_add_ps(c7, b1);                                           \
    b1 = _mm256_load_ps(I + ((5 / factor) * (FLOAT_C_ob) + FLOAT_SIMD));  \
    c8 = _mm256_add_ps(c8, a_reg);                                        \
    c9 = _mm256_add_ps(c9, c12);                                          \
    c10 = _mm256_add_ps(c10, b0);                                         \
    c11 = _mm256_add_ps(c11, b1);


#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_ACCUM_END_C_upsample(I, factor, W_ob, C_ob)              \
    for (uint32_t kk = 0; kk < W_ob; kk++)                             \
    {                                                                  \
        for (uint32_t jj = 0; jj < C_ob; jj++)                         \
        {                                                              \
            c_tile[kk * C_ob + jj] += I[(kk / factor) * (C_ob) + jj];  \
        }                                                              \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_ACCUM_END_C_upsample(I, factor, W_ob, C_ob)               \
    __m256 *c_pixel = c_tile;                                           \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        a_0 = _mm256_load_ps(I + ((kk / factor) * (C_ob)));             \
        c_pixel[0] = _mm256_add_ps(a_0, c_pixel[0]);                    \
        a_1 = _mm256_load_ps(I + ((kk / factor) * (C_ob) + FLOAT_SIMD));\
        c_pixel[1] = _mm256_add_ps(a_1, c_pixel[1]);                    \
        c_pixel += (C_ob/FLOAT_SIMD);                                   \
    }
#endif

//****************************************************************************
// Accumulate channel dimension
//****************************************************************************

/// @todo Code changes made in second macro added brace

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_REDUCE_CHANNEL_END_C(O_w_left, C_ob)            \
    if constexpr (C_ob == 1 && C_ob != FLOAT_SIMD_EPILOGUE)   \
    {                                                         \
        float c_tile_array[FLOAT_C_ob];                       \
        for (uint32_t kk = 0; kk < O_w_left; kk++)            \
        {                                                     \
            float *c_channel_v = c_tile + kk * (FLOAT_C_ob);  \
            for (uint32_t jj = 1; jj < FLOAT_C_ob; jj++)      \
            {                                                 \
                c_channel_v[0] += c_channel_v[jj];            \
                c_channel_v[jj] = 0;                          \
            }                                                 \
        }                                                     \
    }
#elif FLOAT_SIMD_EPILOGUE == 8
#define FLOAT_REDUCE_CHANNEL_END_C(O_w_left, C_ob)                      \
    if constexpr(C_ob == 1 && C_ob != FLOAT_SIMD_EPILOGUE)              \
    {                                                                   \
        float c_tile_array[FLOAT_C_ob];                                 \
        for (uint32_t kk = 0; kk < O_w_left; kk++)                      \
        {                                                               \
            __m256 *c_channel_v = c_tile + kk * (FLOAT_C_ob / FLOAT_SIMD); \
            c_channel_v[0] = _mm256_add_ps(c_channel_v[0], c_channel_v[1]); \
                                                                        \
            _mm256_storeu_ps(c_tile_array, c_channel_v[0]);             \
            for (uint32_t jj = 1; jj < FLOAT_SIMD; jj++)                \
            {                                                           \
                c_tile_array[0] += c_tile_array[jj];                    \
                c_tile_array[jj] = 0;                                   \
            }                                                           \
                                                                        \
            c_channel_v[0] = _mm256_loadu_ps(c_tile_array);             \
            c_channel_v[1] = _mm256_broadcast_ss(0.0);                  \
        }                                                               \
    }
#endif

//****************************************************************************
// Ewise exponentiation (Softmax)
//****************************************************************************
// conversion to and from vector register is very costly
// @todo replacce with a range reduction implementation of EXP microkernel
#define FLOAT_EXP_TILE_C(step, I)                                    \
    c_tile_t c_tile[FLOAT_W_ob * FLOAT_C_ob];                        \
    c_tile_t *c_pixel = c_tile;                                      \
    c_tile_t const *I_pixel = I;                                     \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                     \
    {                                                                \
        c_tile_t *c_channel = c_pixel;                               \
        c_tile_t const *I_channel = I_pixel;                         \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                 \
        {                                                            \
            *(c_channel) = std::exp(*I_channel);                     \
            c_channel++;                                             \
            I_channel++;                                             \
        }                                                            \
        I_pixel += step;                                             \
        c_pixel += FLOAT_C_ob;                                       \
    }                                                                \
    c0 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c2 = _mm256_loadu_ps(c_tile + 1 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c3 = _mm256_loadu_ps(c_tile + 1 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c4 = _mm256_loadu_ps(c_tile + 2 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c5 = _mm256_loadu_ps(c_tile + 2 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c6 = _mm256_loadu_ps(c_tile + 3 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c7 = _mm256_loadu_ps(c_tile + 3 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c8 = _mm256_loadu_ps(c_tile + 4 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c9 = _mm256_loadu_ps(c_tile + 4 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c10 = _mm256_loadu_ps(c_tile + 5 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    c11 = _mm256_loadu_ps(c_tile + 5 * FLOAT_C_ob + 1 * FLOAT_SIMD);

#define FLOAT_EXP_END_C(step, I, c_cur, W_ob, C_ob)   \
    c_tile_t *c_pixel = c_cur;                        \
    c_tile_t const *I_pixel = I;                      \
    for (uint32_t kk = 0; kk < W_ob; kk++)            \
    {                                                 \
        c_tile_t *c_channel = c_pixel;                \
        c_tile_t const *I_channel = I_pixel;          \
        for (uint32_t jj = 0; jj < C_ob; jj++)        \
        {                                             \
            *(c_channel) = std::exp(*I_channel);      \
            c_channel++;                              \
            I_channel++;                              \
        }                                             \
        I_pixel += step;                              \
        c_pixel += C_ob;                              \
    }

#define FLOAT_INPLACE_EXP_TILE_C                                     \
    c_tile_t c_tile[FLOAT_W_ob * FLOAT_C_ob];                        \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, c0);  \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 1 * FLOAT_SIMD, c1);  \
    _mm256_storeu_ps(c_tile + 1 * FLOAT_C_ob + 0 * FLOAT_SIMD, c2);  \
    _mm256_storeu_ps(c_tile + 1 * FLOAT_C_ob + 1 * FLOAT_SIMD, c3);  \
    _mm256_storeu_ps(c_tile + 2 * FLOAT_C_ob + 0 * FLOAT_SIMD, c4);  \
    _mm256_storeu_ps(c_tile + 2 * FLOAT_C_ob + 1 * FLOAT_SIMD, c5);  \
    _mm256_storeu_ps(c_tile + 3 * FLOAT_C_ob + 0 * FLOAT_SIMD, c6);  \
    _mm256_storeu_ps(c_tile + 3 * FLOAT_C_ob + 1 * FLOAT_SIMD, c7);  \
    _mm256_storeu_ps(c_tile + 4 * FLOAT_C_ob + 0 * FLOAT_SIMD, c8);  \
    _mm256_storeu_ps(c_tile + 4 * FLOAT_C_ob + 1 * FLOAT_SIMD, c9);  \
    _mm256_storeu_ps(c_tile + 5 * FLOAT_C_ob + 0 * FLOAT_SIMD, c10); \
    _mm256_storeu_ps(c_tile + 5 * FLOAT_C_ob + 1 * FLOAT_SIMD, c11); \
    c_tile_t *c_pixel = c_tile;                                      \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                     \
    {                                                                \
        c_tile_t *c_channel = c_pixel;                               \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                 \
        {                                                            \
            *(c_channel) = std::exp(*c_channel);                     \
            c_channel++;                                             \
        }                                                            \
        c_pixel += FLOAT_C_ob;                                       \
    }                                                                \
    c0 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c2 = _mm256_loadu_ps(c_tile + 1 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c3 = _mm256_loadu_ps(c_tile + 1 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c4 = _mm256_loadu_ps(c_tile + 2 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c5 = _mm256_loadu_ps(c_tile + 2 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c6 = _mm256_loadu_ps(c_tile + 3 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c7 = _mm256_loadu_ps(c_tile + 3 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c8 = _mm256_loadu_ps(c_tile + 4 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c9 = _mm256_loadu_ps(c_tile + 4 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c10 = _mm256_loadu_ps(c_tile + 5 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    c11 = _mm256_loadu_ps(c_tile + 5 * FLOAT_C_ob + 1 * FLOAT_SIMD);

#define FLOAT_INPLACE_EXP_END_C(c_cur, W_ob, C_ob) \
    c_tile_t *c_pixel = c_cur;                     \
    for (uint32_t kk = 0; kk < W_ob; kk++)         \
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
// Ewise logarithm
//****************************************************************************

// implementation copied from softmax, exponential above

#define FLOAT_LOG_TILE_C(step, I)                                    \
    c_tile_t c_tile[FLOAT_W_ob * FLOAT_C_ob];                        \
    c_tile_t *c_pixel = c_tile;                                      \
    c_tile_t const *I_pixel = I;                                     \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                     \
    {                                                                \
        c_tile_t *c_channel = c_pixel;                               \
        c_tile_t const *I_channel = I_pixel;                         \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                 \
        {                                                            \
            *(c_channel) = std::log(*I_channel);                     \
            c_channel++;                                             \
            I_channel++;                                             \
        }                                                            \
        I_pixel += step;                                             \
        c_pixel += FLOAT_C_ob;                                       \
    }                                                                \
    c0 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c2 = _mm256_loadu_ps(c_tile + 1 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c3 = _mm256_loadu_ps(c_tile + 1 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c4 = _mm256_loadu_ps(c_tile + 2 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c5 = _mm256_loadu_ps(c_tile + 2 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c6 = _mm256_loadu_ps(c_tile + 3 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c7 = _mm256_loadu_ps(c_tile + 3 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c8 = _mm256_loadu_ps(c_tile + 4 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c9 = _mm256_loadu_ps(c_tile + 4 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c10 = _mm256_loadu_ps(c_tile + 5 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    c11 = _mm256_loadu_ps(c_tile + 5 * FLOAT_C_ob + 1 * FLOAT_SIMD);

#define FLOAT_LOG_END_C(step, I, c_cur, W_ob, C_ob)   \
    c_tile_t *c_pixel = c_cur;                        \
    c_tile_t const *I_pixel = I;                      \
    for (uint32_t kk = 0; kk < W_ob; kk++)            \
    {                                                 \
        c_tile_t *c_channel = c_pixel;                \
        c_tile_t const *I_channel = I_pixel;          \
        for (uint32_t jj = 0; jj < C_ob; jj++)        \
        {                                             \
            *(c_channel) = std::log(*I_channel);      \
            c_channel++;                              \
            I_channel++;                              \
        }                                             \
        I_pixel += step;                              \
        c_pixel += C_ob;                              \
    }

#define FLOAT_INPLACE_LOG_TILE_C                                     \
    c_tile_t c_tile[FLOAT_W_ob * FLOAT_C_ob];                        \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD, c0);  \
    _mm256_storeu_ps(c_tile + 0 * FLOAT_C_ob + 1 * FLOAT_SIMD, c1);  \
    _mm256_storeu_ps(c_tile + 1 * FLOAT_C_ob + 0 * FLOAT_SIMD, c2);  \
    _mm256_storeu_ps(c_tile + 1 * FLOAT_C_ob + 1 * FLOAT_SIMD, c3);  \
    _mm256_storeu_ps(c_tile + 2 * FLOAT_C_ob + 0 * FLOAT_SIMD, c4);  \
    _mm256_storeu_ps(c_tile + 2 * FLOAT_C_ob + 1 * FLOAT_SIMD, c5);  \
    _mm256_storeu_ps(c_tile + 3 * FLOAT_C_ob + 0 * FLOAT_SIMD, c6);  \
    _mm256_storeu_ps(c_tile + 3 * FLOAT_C_ob + 1 * FLOAT_SIMD, c7);  \
    _mm256_storeu_ps(c_tile + 4 * FLOAT_C_ob + 0 * FLOAT_SIMD, c8);  \
    _mm256_storeu_ps(c_tile + 4 * FLOAT_C_ob + 1 * FLOAT_SIMD, c9);  \
    _mm256_storeu_ps(c_tile + 5 * FLOAT_C_ob + 0 * FLOAT_SIMD, c10); \
    _mm256_storeu_ps(c_tile + 5 * FLOAT_C_ob + 1 * FLOAT_SIMD, c11); \
    c_tile_t *c_pixel = c_tile;                                      \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                     \
    {                                                                \
        c_tile_t *c_channel = c_pixel;                               \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                 \
        {                                                            \
            *(c_channel) = std::log(*c_channel);                     \
            c_channel++;                                             \
        }                                                            \
        c_pixel += FLOAT_C_ob;                                       \
    }                                                                \
    c0 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c1 = _mm256_loadu_ps(c_tile + 0 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c2 = _mm256_loadu_ps(c_tile + 1 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c3 = _mm256_loadu_ps(c_tile + 1 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c4 = _mm256_loadu_ps(c_tile + 2 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c5 = _mm256_loadu_ps(c_tile + 2 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c6 = _mm256_loadu_ps(c_tile + 3 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c7 = _mm256_loadu_ps(c_tile + 3 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c8 = _mm256_loadu_ps(c_tile + 4 * FLOAT_C_ob + 0 * FLOAT_SIMD);  \
    c9 = _mm256_loadu_ps(c_tile + 4 * FLOAT_C_ob + 1 * FLOAT_SIMD);  \
    c10 = _mm256_loadu_ps(c_tile + 5 * FLOAT_C_ob + 0 * FLOAT_SIMD); \
    c11 = _mm256_loadu_ps(c_tile + 5 * FLOAT_C_ob + 1 * FLOAT_SIMD);

#define FLOAT_INPLACE_LOG_END_C(c_cur, W_ob, C_ob) \
    c_tile_t *c_pixel = c_cur;                     \
    for (uint32_t kk = 0; kk < W_ob; kk++)         \
    {                                              \
        c_tile_t *c_channel = c_pixel;             \
        for (uint32_t jj = 0; jj < C_ob; jj++)     \
        {                                          \
            *(c_channel) = std::log(*c_channel);   \
            c_channel++;                           \
        }                                          \
        c_pixel += C_ob;                           \
    }

//****************************************************************************
// Ewise Softsign
//****************************************************************************

#define FLOAT_SOFTSIGN_TILE_C(step, a) \
    {FLOAT_ABS_TILE_C(step, a);} \
    {const float scalar = 1.0f; FLOAT_EWISE_ADD_SCALAR_TILE_C(scalar);} \
    {FLOAT_FUSED_DIV_TILE_C(step, a);} 

#if 0
#define FLOAT_SOFTSIGN_TILE_C(step, a, W_ob, C_ob)          \
    c12 = _mm256_set1_ps(-0.0f);                            \
    a_reg = _mm256_set1_ps(1.0f);                           \
    b0 = _mm256_load_ps(I + (0 * step));                    \
    c0 = _mm256_andnot_ps(c12, b0);                         \
    c0 = _mm256_add_ps(a_reg, c0);                          \
    c0 = _mm256_div_ps(b0, c0);                             \
    b1 = _mm256_load_ps(I + (0 * step) + FLOAT_SIMD);       \
    c1 = _mm256_andnot_ps(c12, b1);                         \
    c1 = _mm256_add_ps(a_reg, c1);                          \
    c1 = _mm256_div_ps(b1, c1);                             \
    b0 = _mm256_load_ps(I + (1 * step));                    \
    c2 = _mm256_andnot_ps(c12, b0);                         \
    c2 = _mm256_add_ps(a_reg, c2);                          \
    c2 = _mm256_div_ps(b0, c2);                             \
    b1 = _mm256_load_ps(I + (1 * step) + FLOAT_SIMD);       \
    c3 = _mm256_andnot_ps(c12, b1);                         \
    c3 = _mm256_add_ps(a_reg, c3);                          \
    c3 = _mm256_div_ps(b1, c3);                             \
    b0 = _mm256_load_ps(I + (2 * step));                    \
    c4 = _mm256_andnot_ps(c12, b0);                         \
    c4 = _mm256_add_ps(a_reg, c4);                          \
    c4 = _mm256_div_ps(b0, c4);                             \
    b1 = _mm256_load_ps(I + (2 * step) + FLOAT_SIMD);       \
    c5 = _mm256_andnot_ps(c12, b1);                         \
    c5 = _mm256_add_ps(a_reg, c5);                          \
    c5 = _mm256_div_ps(b1, c5);                             \
    b0 = _mm256_load_ps(I + (3 * step));                    \
    c6 = _mm256_andnot_ps(c12, b0);                         \
    c6 = _mm256_add_ps(a_reg, c6);                          \
    c6 = _mm256_div_ps(b0, c6);                             \
    b1 = _mm256_load_ps(I + (3 * step) + FLOAT_SIMD);       \
    c7 = _mm256_andnot_ps(c12, b1);                         \
    c7 = _mm256_add_ps(a_reg, c7);                          \
    c7 = _mm256_div_ps(b1, c7);                             \
    b0 = _mm256_load_ps(I + (4 * step));                    \
    c8 = _mm256_andnot_ps(c12, b0);                         \
    c8 = _mm256_add_ps(a_reg, c8);                          \
    c8 = _mm256_div_ps(b0, c8);                             \
    b1 = _mm256_load_ps(I + (4 * step) + FLOAT_SIMD);       \
    c9 = _mm256_andnot_ps(c12, b1);                         \
    c9 = _mm256_add_ps(a_reg, c9);                          \
    c9 = _mm256_div_ps(b1, c9);                             \
    b0 = _mm256_load_ps(I + (5 * step));                    \
    c10 = _mm256_andnot_ps(c12, b0);                        \
    c10 = _mm256_add_ps(a_reg, c10);                        \
    c10 = _mm256_div_ps(b0, c10);                           \
    b1 = _mm256_load_ps(I + (5 * step) + FLOAT_SIMD);       \
    c11 = _mm256_andnot_ps(c12, b1);                        \
    c11 = _mm256_add_ps(a_reg, c11);                        \
    c11 = _mm256_div_ps(b1, c11);
#endif


#define FLOAT_SOFTSIGN_END_C(step, I, c_cur, W_ob, C_ob)        \
    c_tile_t *c_pixel = c_cur;                                  \
    c_tile_t const *I_pixel = I;                                \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        c_tile_t *c_channel = c_pixel;                          \
        c_tile_t const *I_channel = I_pixel;                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            *(c_channel) = *(I_channel) / (1.0f + std::abs(*I_channel)); \
            c_channel++;                                        \
            I_channel++;                                        \
        }                                                       \
        I_pixel += step;                                        \
        c_pixel += C_ob;                                        \
    }

#define FLOAT_INPLACE_SOFTSIGN_TILE_C             \
    c12 = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));  \
    a_reg = _mm256_set1_ps(1.0f);                 \
    b0 = _mm256_and_ps(c12, c0);                  \
    b0 = _mm256_add_ps(a_reg, b0);                \
    c0 = _mm256_div_ps(c0, b0);                   \
    b1 = _mm256_and_ps(c12, c1);                  \
    b1 = _mm256_add_ps(a_reg, b1);                \
    c1 = _mm256_div_ps(c1, b1);                   \
    b0 = _mm256_and_ps(c12, c2);                  \
    b0 = _mm256_add_ps(a_reg, b0);                \
    c2 = _mm256_div_ps(c2, b0);                   \
    b1 = _mm256_and_ps(c12, c3);                  \
    b1 = _mm256_add_ps(a_reg, b1);                \
    c3 = _mm256_div_ps(c3, b1);                   \
    b0 = _mm256_and_ps(c12, c4);                  \
    b0 = _mm256_add_ps(a_reg, b0);                \
    c4 = _mm256_div_ps(c4, b0);                   \
    b1 = _mm256_and_ps(c12, c5);                  \
    b1 = _mm256_add_ps(a_reg, b1);                \
    c5 = _mm256_div_ps(c5, b1);                   \
    b0 = _mm256_and_ps(c12, c6);                  \
    b0 = _mm256_add_ps(a_reg, b0);                \
    c6 = _mm256_div_ps(c6, b0);                   \
    b1 = _mm256_and_ps(c12, c7);                  \
    b1 = _mm256_add_ps(a_reg, b1);                \
    c7 = _mm256_div_ps(c7, b1);                   \
    b0 = _mm256_and_ps(c12, c8);                  \
    b0 = _mm256_add_ps(a_reg, b0);                \
    c8 = _mm256_div_ps(c8, b0);                   \
    b1 = _mm256_and_ps(c12, c9);                  \
    b1 = _mm256_add_ps(a_reg, b1);                \
    c9 = _mm256_div_ps(c9, b1);                   \
    b0 = _mm256_and_ps(c12, c10);                 \
    b0 = _mm256_add_ps(a_reg, b0);                \
    c10 = _mm256_div_ps(c10, b0);                 \
    b1 = _mm256_and_ps(c12, c11);                 \
    b1 = _mm256_add_ps(a_reg, b1);                \
    c11 = _mm256_div_ps(c11, b1);


#define FLOAT_INPLACE_SOFTSIGN_END_C(c_cur, W_ob, C_ob) \
    c_tile_t *c_pixel = c_cur;                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)            \
    {                                                 \
        c_tile_t *c_channel = c_pixel;                \
        for (uint32_t jj = 0; jj < C_ob; jj++)        \
        {                                             \
            *(c_channel) = *(c_channel) / (1.0f + std::abs(*(c_channel))); \
            c_channel++;                              \
        }                                             \
        c_pixel += C_ob;                              \
    }

//****************************************************************************
// Ewise abs
//****************************************************************************

#define FLOAT_ABS_TILE_C(step, I)                                       \
    c12 = _mm256_set1_ps(-0.0f);                                        \
    b0 = _mm256_load_ps(I + (0 * step));                                \
    c0 = _mm256_andnot_ps(c12, b0);                                     \
    b1 = _mm256_load_ps(I + (0 * step) + FLOAT_SIMD);                   \
    c1 = _mm256_andnot_ps(c12, b1);                                     \
    a_reg = _mm256_load_ps(I + (1 * step));                             \
    c2 = _mm256_andnot_ps(c12, a_reg);                                  \
    b0 = _mm256_load_ps(I + (1 * step) + FLOAT_SIMD);                   \
    c3 = _mm256_andnot_ps(c12, b0);                                     \
    b1 = _mm256_load_ps(I + (2 * step));                                \
    c4 = _mm256_andnot_ps(c12, b1);                                     \
    a_reg = _mm256_load_ps(I + (2 * step) + FLOAT_SIMD);                \
    c5 = _mm256_andnot_ps(c12, a_reg);                                  \
    b0 = _mm256_load_ps(I + (3 * step));                                \
    c6 = _mm256_andnot_ps(c12, b0);                                     \
    b1 = _mm256_load_ps(I + (3 * step) + FLOAT_SIMD);                   \
    c7 = _mm256_andnot_ps(c12, b1);                                     \
    a_reg = _mm256_load_ps(I + (4 * step));                             \
    c8 = _mm256_andnot_ps(c12, a_reg);                                  \
    b0 = _mm256_load_ps(I + (4 * step) + FLOAT_SIMD);                   \
    c9 = _mm256_andnot_ps(c12, b0);                                     \
    b1 = _mm256_load_ps(I + (5 * step));                                \
    c10 = _mm256_andnot_ps(c12, b1);                                    \
    a_reg = _mm256_load_ps(I + (5 * step) + FLOAT_SIMD);                \
    c11 = _mm256_andnot_ps(c12, a_reg);

#define FLOAT_ABS_END_C(step, I, c_cur, W_ob, C_ob)             \
    c_tile_t *c_pixel = c_cur;                                  \
    c_tile_t const *I_pixel = I;                                \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        c_tile_t *c_channel = c_pixel;                          \
        c_tile_t const *I_channel = I_pixel;                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            *(c_channel) = std::abs(*I_channel);                \
            c_channel++;                                        \
            I_channel++;                                        \
        }                                                       \
        I_pixel += step;                                        \
        c_pixel += C_ob;                                        \
    }

//****************************************************************************
// Ewise div
//****************************************************************************

#define FLOAT_INPLACE_DIV_TILE_C(step, I)                               \
    b0 = _mm256_load_ps(I + (0 * step));                                \
    c0 = _mm256_div_ps(b0, c0);                                         \
    b1 = _mm256_load_ps(I + (0 * step) + FLOAT_SIMD);                   \
    c1 = _mm256_div_ps(b1, c1);                                         \
    a_reg = _mm256_load_ps(I + (1 * step));                             \
    c2 = _mm256_div_ps(a_reg, c2);                                      \
    c12 = _mm256_load_ps(I + (1 * step) + FLOAT_SIMD);                  \
    c3 = _mm256_div_ps(c12, c3);                                        \
    b0 = _mm256_load_ps(I + (2 * step));                                \
    c4 = _mm256_div_ps(b0, c4);                                         \
    b1 = _mm256_load_ps(I + (2 * step) + FLOAT_SIMD);                   \
    c5 = _mm256_div_ps(b1, c5);                                         \
    a_reg = _mm256_load_ps(I + (3 * step));                             \
    c6 = _mm256_div_ps(a_reg, c6);                                      \
    c12 = _mm256_load_ps(I + (3 * step) + FLOAT_SIMD);                  \
    c7 = _mm256_div_ps(c12, c7);                                        \
    b0 = _mm256_load_ps(I + (4 * step));                                \
    c8 = _mm256_div_ps(b0, c8);                                         \
    b1 = _mm256_load_ps(I + (4 * step) + FLOAT_SIMD);                   \
    c9 = _mm256_div_ps(b1, c9);                                         \
    a_reg = _mm256_load_ps(I + (5 * step));                             \
    c10 = _mm256_div_ps(a_reg, c10);                                    \
    c12 = _mm256_load_ps(I + (5 * step) + FLOAT_SIMD);                  \
    c11 = _mm256_div_ps(c12, c11);

#define FLOAT_INPLACE_DIV_END_C(step, I, c_cur, W_ob, C_ob)     \
    c_tile_t *c_pixel = c_cur;                                  \
    c_tile_t const *I_pixel = I;                                \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        c_tile_t *c_channel = c_pixel;                          \
        c_tile_t const *I_channel = I_pixel;                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            *(c_channel) = *(I_channel) / *(c_channel);         \
            c_channel++;                                        \
            I_channel++;                                        \
        }                                                       \
        I_pixel += step;                                        \
        c_pixel += C_ob;                                        \
    }

//****************************************************************************
// Slope ReLu activation
//****************************************************************************

#define FLOAT_SLOPE_RELU_TILE_C(step, a_cur, c_cur) \
    FLOAT_MIN_SCALAR_TILE_C; \
    FLOAT_MUL_TILE_C(step, a_cur);  \
    FLOAT_STORE_TILE_C(c_tile); \
    FLOAT_LOAD_TILE_C(c_cur); \
    FLOAT_FUSED_RELU_TILE_C;  \
    FLOAT_ACCUM_TILE_C(step, c_tile);


#define FLOAT_SLOPE_RELU_END_C(step, a, c_cur, W_ob, C_ob)                                     \
    float *c_pixel = c_cur;                                                                         \
    float const *a_pixel = a;                                                                       \
    for (uint32_t kk = 0; kk < W_ob; kk++)                                                        \
    {                                                                                               \
        float *c_channel = c_pixel;                                                                 \
        float const *a_channel = a_pixel;                                                           \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                                      \
        {                                                                                           \
            *(c_channel) = *(c_channel) > 0 ? *(c_channel) : (*(a_channel) * *(c_channel)); \
            c_channel++;                                                                            \
            a_channel++;                                                                            \
        }                                                                                           \
        a_pixel += step;                                                                            \
        c_pixel += C_ob;                                                                            \
    }

//****************************************************************************
// Fused Slope ReLU activation
//****************************************************************************
#define FLOAT_MIN_SCALAR_TILE_C \
    c12 = _mm256_setzero_ps();   \
    c0 = _mm256_min_ps(c0, c12);    \
    c1 = _mm256_min_ps(c1, c12);    \
    c2 = _mm256_min_ps(c2, c12);    \
    c3 = _mm256_min_ps(c3, c12);    \
    c4 = _mm256_min_ps(c4, c12);    \
    c5 = _mm256_min_ps(c5, c12);    \
    c6 = _mm256_min_ps(c6, c12);    \
    c7 = _mm256_min_ps(c7, c12);    \
    c8 = _mm256_min_ps(c8, c12);    \
    c9 = _mm256_min_ps(c9, c12);    \
    c10 = _mm256_min_ps(c10, c12);  \
    c11 = _mm256_min_ps(c11, c12);

#define FLOAT_MIN_SCALAR_END_C(b, c_cur, W_ob, C_ob) \
    c_tile_t *c_pixel = c_cur;                                  \
    for (uint32_t kk = 0; kk < W_ob; kk++)                    \
    {                                                           \
        c_tile_t *c_channel = c_pixel;                          \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            *(c_channel) = std::min(*(c_channel), b[0]);        \
            c_channel++;                                        \
        }                                                       \
        c_pixel += C_ob;                                        \
    }

#define FLOAT_MUL_TILE_C(step, a)        \
    b0 = _mm256_load_ps(a + (0 * step));               \
    c0 = _mm256_mul_ps(b0, c0);                        \
    b1 = _mm256_load_ps(a + (0 * step) + FLOAT_SIMD);  \
    c1 = _mm256_mul_ps(b1, c1);                        \
    a_reg = _mm256_load_ps(a + (1 * step));            \
    c2 = _mm256_mul_ps(a_reg, c2);                     \
    c12 = _mm256_load_ps(a + (1 * step) + FLOAT_SIMD); \
    c3 = _mm256_mul_ps(c12, c3);                       \
    b0 = _mm256_load_ps(a + (2 * step));               \
    c4 = _mm256_mul_ps(b0, c4);                        \
    b1 = _mm256_load_ps(a + (2 * step) + FLOAT_SIMD);  \
    c5 = _mm256_mul_ps(b1, c5);                        \
    a_reg = _mm256_load_ps(a + (3 * step));            \
    c6 = _mm256_mul_ps(a_reg, c6);                     \
    c12 = _mm256_load_ps(a + (3 * step) + FLOAT_SIMD); \
    c7 = _mm256_mul_ps(c12, c7);                       \
    b0 = _mm256_load_ps(a + (4 * step));               \
    c8 = _mm256_mul_ps(b0, c8);                        \
    b1 = _mm256_load_ps(a + (4 * step) + FLOAT_SIMD);  \
    c9 = _mm256_mul_ps(b1, c9);                        \
    a_reg = _mm256_load_ps(a + (5 * step));            \
    c10 = _mm256_mul_ps(a_reg, c10);                   \
    c12 = _mm256_load_ps(a + (5 * step) + FLOAT_SIMD); \
    c11 = _mm256_mul_ps(c12, c11);

#define FLOAT_FMA_TILE_C(step, a, c)           \
    b0 = _mm256_load_ps(a + (0 * step));                \
    b1 = _mm256_load_ps(c + (0 * step));                \
    c0 = _mm256_fmadd_ps(b0, b1, c0);                   \
    c12 = _mm256_load_ps(a + (0 * step) + FLOAT_SIMD); \
    a_reg = _mm256_load_ps(c + (0 * step) + FLOAT_SIMD); \
    c1 = _mm256_fmadd_ps(c12, a_reg, c1);                 \
    b0 = _mm256_load_ps(a + (1 * step));                \
    b1 = _mm256_load_ps(c + (1 * step));                \
    c2 = _mm256_fmadd_ps(b0, b1, c2);                   \
    c12 = _mm256_load_ps(a + (1 * step) + FLOAT_SIMD); \
    a_reg = _mm256_load_ps(c + (1 * step) + FLOAT_SIMD); \
    c3 = _mm256_fmadd_ps(c12, a_reg, c3);                 \
    b0 = _mm256_load_ps(a + (2 * step));                \
    b1 = _mm256_load_ps(c + (2 * step));                \
    c4 = _mm256_fmadd_ps(b0, b1, c4);                   \
    c12 = _mm256_load_ps(a + (2 * step) + FLOAT_SIMD); \
    a_reg = _mm256_load_ps(c + (2 * step) + FLOAT_SIMD); \
    c5 = _mm256_fmadd_ps(c12, a_reg, c5);                 \
    b0 = _mm256_load_ps(a + (3 * step));                \
    b1 = _mm256_load_ps(c + (3 * step));                \
    c6 = _mm256_fmadd_ps(b0, b1, c6);                   \
    c12 = _mm256_load_ps(a + (3 * step) + FLOAT_SIMD); \
    a_reg = _mm256_load_ps(c + (3 * step) + FLOAT_SIMD); \
    c7 = _mm256_fmadd_ps(c12, a_reg, c7);                 \
    b0 = _mm256_load_ps(a + (4 * step));                \
    b1 = _mm256_load_ps(c + (4 * step));                \
    c8 = _mm256_fmadd_ps(b0, b1, c8);                   \
    c12 = _mm256_load_ps(a + (4 * step) + FLOAT_SIMD); \
    a_reg = _mm256_load_ps(c + (4 * step) + FLOAT_SIMD); \
    c9 = _mm256_fmadd_ps(c12, a_reg, c9);                 \
    b0 = _mm256_load_ps(a + (5 * step));                \
    b1 = _mm256_load_ps(c + (5 * step));                \
    c10 = _mm256_fmadd_ps(b0, b1, c10);                 \
    c12 = _mm256_load_ps(a + (5 * step) + FLOAT_SIMD); \
    a_reg = _mm256_load_ps(c + (5 * step) + FLOAT_SIMD); \
    c11 = _mm256_fmadd_ps(c12, a_reg, c11);



#define FLOAT_MUL_END_C(step, a, c_cur, W_last, C_ob) \
    c_tile_t *c_pixel = c_cur;                                  \
    c_tile_t const *a_pixel = a;                                \
    for (uint32_t kk = 0; kk < W_last; kk++)                    \
    {                                                           \
        c_tile_t *c_channel = c_pixel;                          \
        c_tile_t const *a_channel = a_pixel;                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            *(c_channel) = *(a_channel) * *(c_channel);         \
            c_channel++;                                        \
            a_channel++;                                        \
        }                                                       \
        a_pixel += step;                                        \
        c_pixel += C_ob;                                        \
    }



#define FLOAT_FUSED_SLOPE_RELU_TILE_C(step, a_cur, c_cur) \
    FLOAT_MIN_SCALAR_TILE_C; \
    FLOAT_MUL_TILE_C(step, a_cur);  \
    FLOAT_STORE_TILE_C(c_tile); \
    FLOAT_LOAD_TILE_C(c_cur); \
    FLOAT_FUSED_RELU_TILE_C;  \
    FLOAT_ACCUM_TILE_C(step, c_tile);

//****************************************************************************
// CeLu activation
//****************************************************************************

#define FLOAT_CELU_TILE_C(b) \
    c12 = _mm256_broadcast_ss(b);   \
    a_reg = _mm256_setzero_ps();   \
    b0 = _mm256_set1_ps(1.0f);      \
    b1 = _mm256_div_ps(c0, c12);    \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c0 = _mm256_max_ps(c0, a_reg);  \
    c0 = _mm256_add_ps(c0, b1);     \
    b1 = _mm256_div_ps(c1, c12);    \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c1 = _mm256_max_ps(c1, a_reg);  \
    c1 = _mm256_add_ps(c1, b1);     \
    b1 = _mm256_div_ps(c2, c12);    \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c2 = _mm256_max_ps(c2, a_reg);  \
    c2 = _mm256_add_ps(c2, b1);     \
    b1 = _mm256_div_ps(c3, c12);    \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c3 = _mm256_max_ps(c3, a_reg);  \
    c3 = _mm256_add_ps(c3, b1);     \
    b1 = _mm256_div_ps(c4, c12);    \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c4 = _mm256_max_ps(c4, a_reg);  \
    c4 = _mm256_add_ps(c4, b1);     \
    b1 = _mm256_div_ps(c5, c12);    \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c5 = _mm256_max_ps(c5, a_reg);  \
    c5 = _mm256_add_ps(c5, b1);     \
    b1 = _mm256_div_ps(c6, c12);    \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c6 = _mm256_max_ps(c6, a_reg);  \
    c6 = _mm256_add_ps(c6, b1);     \
    b1 = _mm256_div_ps(c7, c12);    \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c7 = _mm256_max_ps(c7, a_reg);  \
    c7 = _mm256_add_ps(c7, b1);     \
    b1 = _mm256_div_ps(c8, c12);    \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c8 = _mm256_max_ps(c8, a_reg);  \
    c8 = _mm256_add_ps(c8, b1);     \
    b1 = _mm256_div_ps(c9, c12);    \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c9 = _mm256_max_ps(c9, a_reg);  \
    c9 = _mm256_add_ps(c9, b1);     \
    b1 = _mm256_div_ps(c10, c12);   \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c10 = _mm256_max_ps(c10, a_reg); \
    c10 = _mm256_add_ps(c10, b1);   \
    b1 = _mm256_div_ps(c11, c12);   \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD, b1); \
    for (uint32_t kk = 0; kk < FLOAT_SIMD; ++kk) \
    { \
        c_tile[kk] = std::exp(c_tile[kk]); \
    } \
    b1 = _mm256_loadu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD); \
    b1 = _mm256_sub_ps(b1, b0);     \
    b1 = _mm256_mul_ps(c12, b1);    \
    b1 = _mm256_min_ps(b1, a_reg);  \
    c11 = _mm256_max_ps(c11, a_reg); \
    c11 = _mm256_add_ps(c11, b1);  

#define FLOAT_CELU_END_C(b, c_cur, W_last, C_ob)                   \
    float *c_pixel = c_cur;                                           \
    float alpha = b[0];                                              \
    for (uint32_t kk = 0; kk < W_last; kk++)                          \
    {                                                                 \
        float *c_channel = c_pixel;                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                        \
        {                                                             \
            *(c_channel) = std::max(0.f, *(c_channel)) + std::min(0.f, alpha * (*(c_channel) / alpha - 1.0f)); \
            c_channel++;                                              \
        }                                                             \
        c_pixel += C_ob;                                              \
    }                                                           

#define FLOAT_FUSED_CELU_TILE_C(step, a, b, c) \
    FLOAT_DIV_SCALAR_TILE_C(b, W_ob, C_ob); \
    FLOAT_FUSED_EXP_TILE_C(W_ob, C_ob); \
    FLOAT_SUB_ONE_TILE_C(W_ob, C_ob); \
    FLOAT_MUL_SCALAR_TILE_C(b, W_ob, C_ob); \
    FLOAT_MIN_SCALAR_TILE_C(W_ob, C_ob); \
    FLOAT_STORE_TILE_C(c_tile, W_ob, C_ob); \
    FLOAT_LOAD_TILE_C(c, W_ob, C_ob); \
    FLOAT_FUSED_RELU_TILE_C(W_ob, C_ob); \
    FLOAT_ACCUM_TILE_C(step, c_tile, W_ob, C_ob);


#define FLOAT_DIV_SCALAR_TILE_C(b)  \
    c12 = _mm256_broadcast_ss(b);   \
    c0 = _mm256_div_ps(c0, c12);    \
    c1 = _mm256_div_ps(c1, c12);    \
    c2 = _mm256_div_ps(c2, c12);    \
    c3 = _mm256_div_ps(c3, c12);    \
    c4 = _mm256_div_ps(c4, c12);    \
    c5 = _mm256_div_ps(c5, c12);    \
    c6 = _mm256_div_ps(c6, c12);    \
    c7 = _mm256_div_ps(c7, c12);    \
    c8 = _mm256_div_ps(c8, c12);    \
    c9 = _mm256_div_ps(c9, c12);    \
    c10 = _mm256_div_ps(c10, c12);  \
    c11 = _mm256_div_ps(c11, c12);

#define FLOAT_SUB_ONE_TILE_C \
    b0 = _mm256_set1_ps(1.0f);      \
    c0 = _mm256_sub_ps(c0, b0);    \
    c1 = _mm256_sub_ps(c1, b0);    \
    c2 = _mm256_sub_ps(c2, b0);    \
    c3 = _mm256_sub_ps(c3, b0);    \
    c4 = _mm256_sub_ps(c4, b0);    \
    c5 = _mm256_sub_ps(c5, b0);    \
    c6 = _mm256_sub_ps(c6, b0);    \
    c7 = _mm256_sub_ps(c7, b0);    \
    c8 = _mm256_sub_ps(c8, b0);    \
    c9 = _mm256_sub_ps(c9, b0);    \
    c10 = _mm256_sub_ps(c10, b0);  \
    c11 = _mm256_sub_ps(c11, b0);

#define FLOAT_MUL_SCALAR_TILE_C(b) \
    c12 = _mm256_broadcast_ss(b);   \
    c0 = _mm256_mul_ps(c0, c12);    \
    c1 = _mm256_mul_ps(c1, c12);    \
    c2 = _mm256_mul_ps(c2, c12);    \
    c3 = _mm256_mul_ps(c3, c12);    \
    c4 = _mm256_mul_ps(c4, c12);    \
    c5 = _mm256_mul_ps(c5, c12);    \
    c6 = _mm256_mul_ps(c6, c12);    \
    c7 = _mm256_mul_ps(c7, c12);    \
    c8 = _mm256_mul_ps(c8, c12);    \
    c9 = _mm256_mul_ps(c9, c12);    \
    c10 = _mm256_mul_ps(c10, c12);  \
    c11 = _mm256_mul_ps(c11, c12);


#define FLOAT_SOFTMAX_TILE_C(step, a, c, d) \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD, c0);  \
    _mm256_storeu_ps(c_tile + 0 * C_ob + 1 * FLOAT_SIMD, c1);  \
    _mm256_storeu_ps(c_tile + 1 * C_ob + 0 * FLOAT_SIMD, c2);  \
    _mm256_storeu_ps(c_tile + 1 * C_ob + 1 * FLOAT_SIMD, c3);  \
    _mm256_storeu_ps(c_tile + 2 * C_ob + 0 * FLOAT_SIMD, c4);  \
    _mm256_storeu_ps(c_tile + 2 * C_ob + 1 * FLOAT_SIMD, c5);  \
    _mm256_storeu_ps(c_tile + 3 * C_ob + 0 * FLOAT_SIMD, c6);  \
    _mm256_storeu_ps(c_tile + 3 * C_ob + 1 * FLOAT_SIMD, c7);  \
    _mm256_storeu_ps(c_tile + 4 * C_ob + 0 * FLOAT_SIMD, c8);  \
    _mm256_storeu_ps(c_tile + 4 * C_ob + 1 * FLOAT_SIMD, c9);  \
    _mm256_storeu_ps(c_tile + 5 * C_ob + 0 * FLOAT_SIMD, c10); \
    _mm256_storeu_ps(c_tile + 5 * C_ob + 1 * FLOAT_SIMD, c11); \
    c_tile_t *c_pixel = c_tile;                                \
    c_tile_t *d_pixel = d;                                     \
    c_tile_t const *a_pixel = a;                               \
    for (uint32_t kk = 0; kk < W_ob; kk++)                     \
    {                                                          \
        c_tile_t *c_channel = c_pixel;                         \
        c_tile_t *d_channel = d_pixel;                         \
        c_tile_t const *a_channel = a_pixel;                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                 \
        {                                                      \
            *(d_channel) = std::exp(*a_channel);               \
            *(c_channel) += *(d_channel);                      \
            std::cout << "exp: " << *(d_channel) << " sum: " << *(c_channel) << std::endl; \
            c_channel++;                                       \
            d_channel++;                                       \
            a_channel++;                                       \
        }                                                      \
        a_pixel += step;                                       \
        c_pixel += C_ob;                                       \
        d_pixel += C_ob;                                       \
    }                                                          \
    c0 = _mm256_loadu_ps(c_tile + 0 * C_ob + 0 * FLOAT_SIMD);  \
    c1 = _mm256_loadu_ps(c_tile + 0 * C_ob + 1 * FLOAT_SIMD);  \
    c2 = _mm256_loadu_ps(c_tile + 1 * C_ob + 0 * FLOAT_SIMD);  \
    c3 = _mm256_loadu_ps(c_tile + 1 * C_ob + 1 * FLOAT_SIMD);  \
    c4 = _mm256_loadu_ps(c_tile + 2 * C_ob + 0 * FLOAT_SIMD);  \
    c5 = _mm256_loadu_ps(c_tile + 2 * C_ob + 1 * FLOAT_SIMD);  \
    c6 = _mm256_loadu_ps(c_tile + 3 * C_ob + 0 * FLOAT_SIMD);  \
    c7 = _mm256_loadu_ps(c_tile + 3 * C_ob + 1 * FLOAT_SIMD);  \
    c8 = _mm256_loadu_ps(c_tile + 4 * C_ob + 0 * FLOAT_SIMD);  \
    c9 = _mm256_loadu_ps(c_tile + 4 * C_ob + 1 * FLOAT_SIMD);  \
    c10 = _mm256_loadu_ps(c_tile + 5 * C_ob + 0 * FLOAT_SIMD); \
    c11 = _mm256_loadu_ps(c_tile + 5 * C_ob + 1 * FLOAT_SIMD);

#define FLOAT_FUSED_SOFTMAX_TILE_C(step, a, c) \
    FLOAT_STORE_TILE_C(c_tile, W_ob, C_ob); \
    FLOAT_LOAD_TILE_C(c, W_ob, C_ob);  \
    FLOAT_SOFTMAX_TILE_C(step, a, c, W_ob, C_ob); \
    FLOAT_DIV_TILE_C(step, a, c, W_ob, C_ob);

