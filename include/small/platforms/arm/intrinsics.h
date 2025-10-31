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

// https://developer.arm.com/architectures/instruction-sets/intrinsics/
#include <arm_neon.h>

#include <params.h>
#include <Buffer.hpp>

#include "arm_mathfun.h"

// scalar versions of all the float microkernels for platform portability
// Use the FLOAT_ prefix for all macros in this file.

/// @todo Replace this with FLOAT_SIMD? They are the same value.
///       The code blocks that are defined when this value
///       does not equal 1 actually use FLOAT_SIMD's value.
// Epilogue parameters
#define FLOAT_SIMD_EPILOGUE 4

namespace small
{
    namespace float_detail
    {
#if FLOAT_SIMD_EPILOGUE == 1
        typedef small::FloatBuffer::value_type  c_tile_t;
#else
        typedef float32x4_t c_tile_t;
#endif
    }
}


//****************************************************************************
// Definitions
//****************************************************************************

// float32x4_t vectorizes C_ob dim: [W_ob, C_ob] -> [W_ob, C_ob/FLOAT_SIMD, FLOAT_SIMD].
// assume FLOAT_SIMD == 4 and vec type is float.
// otherwise, FLOAT_SIMD = Neon bit width (128) / data type size.

#define FLOAT_DEF_TILE_C                                                \
    float32x4_t c_tile_v[FLOAT_W_ob * (FLOAT_C_ob / FLOAT_SIMD)];

/// @todo c_tile is different for different FLOAT_SIMD_EPILOGUE.  Does this need 2 variants?
/// @todo ZEN2 uses FLOAT_SIMD and not FLOAT_SIMD_EPILOGUE. What's the difference?
#define FLOAT_DEF_END_C(W_ob, C_ob)                                     \
    c_tile_t c_tile[W_ob * (FLOAT_C_ob / FLOAT_SIMD_EPILOGUE)];


//****************************************************************************
// Initializations
//****************************************************************************

#define FLOAT_ZERO_TILE_C                                               \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)       \
        {                                                               \
            c_tile_v[kk * (FLOAT_C_ob / FLOAT_SIMD) + jj] = vdupq_n_f32(0); \
        }                                                               \
    }


#if FLOAT_SIMD_EPILOGUE==1
#define FLOAT_ZERO_END_C(W_ob, C_ob)            \
    for (uint32_t kk = 0; kk < W_ob; kk++)      \
    {                                           \
        for (uint32_t jj = 0; jj < C_ob; jj++)  \
        {                                       \
            c_tile[kk * C_ob + jj] = {};        \
        }                                       \
    }

#else

#define FLOAT_ZERO_END_C(W_ob, C_ob)                                    \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)       \
        {                                                               \
            c_tile[kk * (FLOAT_C_ob / FLOAT_SIMD) + jj] = vdupq_n_f32(0); \
        }                                                               \
    }
#endif


//****************************************************************************
// Loads
//****************************************************************************

#define FLOAT_LOAD_TILE_C(I)                                            \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)       \
        {                                                               \
            c_tile_v[kk * (FLOAT_C_ob / FLOAT_SIMD) + jj] =             \
                vld1q_f32(I + kk * FLOAT_C_ob + jj * FLOAT_SIMD);       \
        }                                                               \
    }

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_LOAD_END_C(I, W_ob, C_ob)                 \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = I[kk * C_ob + jj]; \
        }                                               \
    }

#else

#define FLOAT_LOAD_END_C(I, W_ob, C_ob)                                 \
    if constexpr(C_ob == 1)                                             \
    {                                                                   \
        for(uint32_t kk = 0; kk < W_ob; kk++)                           \
        {                                                               \
            float c_channel_v[FLOAT_C_ob]={0};                          \
            c_channel_v[0] = I[kk * C_ob];                              \
            c_tile[kk * (FLOAT_C_ob/FLOAT_SIMD)]  = vld1q_f32(c_channel_v); \
        }                                                               \
    }                                                                   \
    else                                                                \
    {                                                                   \
        for (uint32_t kk = 0; kk < W_ob; kk++)                          \
        {                                                               \
            for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)         \
            {                                                           \
                c_tile[kk * (C_ob / FLOAT_SIMD) + jj] = vld1q_f32(I + kk * C_ob + jj * FLOAT_SIMD); \
            }                                                           \
        }                                                               \
    }
#endif


//****************************************************************************
// Pooling Loads
//****************************************************************************

/// @todo: merge FLOAT_LOAD_TILE_C and FLOAT_LOAD_TILE_C_strided? can use C_ob as step above.

//  strided loads
#define FLOAT_LOAD_TILE_C_strided(I, step)                              \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)       \
        {                                                               \
            c_tile_v[kk * (FLOAT_C_ob / FLOAT_SIMD) + jj] =             \
                vld1q_f32(I + kk * step + jj * FLOAT_SIMD);             \
        }                                                               \
    }

#if FLOAT_SIMD_EPILOGUE==1
#define FLOAT_LOAD_END_C_strided(I, step, W_ob, C_ob)   \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = I[kk * step + jj]; \
        }                                               \
    }

#else

#define FLOAT_LOAD_END_C_strided(O, step, W_ob, C_ob)                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            c_tile[kk * (C_ob / FLOAT_SIMD) + jj] =                     \
                vld1q_f32(O + kk * step + jj * FLOAT_SIMD);             \
        }                                                               \
    }
#endif


//****************************************************************************
// Upsampling loads (stride < 1, factor = 1/stride)
//****************************************************************************

#define FLOAT_LOAD_TILE_C_upsample(I, factor)                           \
    c_0_0 = vld1q_f32(I + (0/factor) * FLOAT_C_ob + 0 * FLOAT_SIMD);    \
    c_0_1 = vld1q_f32(I + (0/factor) * FLOAT_C_ob + 1 * FLOAT_SIMD);    \
    c_0_2 = vld1q_f32(I + (0/factor) * FLOAT_C_ob + 2 * FLOAT_SIMD);    \
    c_0_3 = vld1q_f32(I + (0/factor) * FLOAT_C_ob + 3 * FLOAT_SIMD);    \
    c_1_0 = vld1q_f32(I + (1/factor) * FLOAT_C_ob + 0 * FLOAT_SIMD);    \
    c_1_1 = vld1q_f32(I + (1/factor) * FLOAT_C_ob + 1 * FLOAT_SIMD);    \
    c_1_2 = vld1q_f32(I + (1/factor) * FLOAT_C_ob + 2 * FLOAT_SIMD);    \
    c_1_3 = vld1q_f32(I + (1/factor) * FLOAT_C_ob + 3 * FLOAT_SIMD);    \
    c_2_0 = vld1q_f32(I + (2/factor) * FLOAT_C_ob + 0 * FLOAT_SIMD);    \
    c_2_1 = vld1q_f32(I + (2/factor) * FLOAT_C_ob + 1 * FLOAT_SIMD);    \
    c_2_2 = vld1q_f32(I + (2/factor) * FLOAT_C_ob + 2 * FLOAT_SIMD);    \
    c_2_3 = vld1q_f32(I + (2/factor) * FLOAT_C_ob + 3 * FLOAT_SIMD);    \
    c_3_0 = vld1q_f32(I + (3/factor) * FLOAT_C_ob + 0 * FLOAT_SIMD);    \
    c_3_1 = vld1q_f32(I + (3/factor) * FLOAT_C_ob + 1 * FLOAT_SIMD);    \
    c_3_2 = vld1q_f32(I + (3/factor) * FLOAT_C_ob + 2 * FLOAT_SIMD);    \
    c_3_3 = vld1q_f32(I + (3/factor) * FLOAT_C_ob + 3 * FLOAT_SIMD);    \
    c_4_0 = vld1q_f32(I + (4/factor) * FLOAT_C_ob + 0 * FLOAT_SIMD);    \
    c_4_1 = vld1q_f32(I + (4/factor) * FLOAT_C_ob + 1 * FLOAT_SIMD);    \
    c_4_2 = vld1q_f32(I + (4/factor) * FLOAT_C_ob + 2 * FLOAT_SIMD);    \
    c_4_3 = vld1q_f32(I + (4/factor) * FLOAT_C_ob + 3 * FLOAT_SIMD);    \
    c_5_0 = vld1q_f32(I + (5/factor) * FLOAT_C_ob + 0 * FLOAT_SIMD);    \
    c_5_1 = vld1q_f32(I + (5/factor) * FLOAT_C_ob + 1 * FLOAT_SIMD);    \
    c_5_2 = vld1q_f32(I + (5/factor) * FLOAT_C_ob + 2 * FLOAT_SIMD);    \
    c_5_3 = vld1q_f32(I + (5/factor) * FLOAT_C_ob + 3 * FLOAT_SIMD);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_LOAD_END_C_upsample(I, factor, W_ob, C_ob)                \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            c_tile[kk * C_ob + jj] = I[(kk / factor) * (C_ob) + jj];    \
        }                                                               \
    }

#else

#define FLOAT_LOAD_END_C_upsample(I, factor, W_ob, C_ob)                \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            c_tile[kk * (C_ob / FLOAT_SIMD) + jj] =                     \
                vld1q_f32(I + (kk / factor) *(C_ob) + jj * FLOAT_SIMD); \
        }                                                               \
    }
#endif


//****************************************************************************
// Stores
//****************************************************************************

#define FLOAT_STORE_TILE_C(O)                                         \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                      \
    {                                                                 \
        for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)     \
        {                                                             \
            vst1q_f32(O + kk * FLOAT_C_ob + jj * FLOAT_SIMD,          \
                      c_tile_v[kk * (FLOAT_C_ob / FLOAT_SIMD) + jj]); \
        }                                                             \
    }

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_STORE_END_C(O, W_ob, C_ob)                \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
        }                                               \
    }

#else

#define FLOAT_STORE_END_C(O, W_ob, C_ob)                          \
    for (uint32_t kk = 0; kk < W_ob; kk++)                        \
    {                                                             \
        if constexpr (C_ob == 1)                                  \
        {                                                         \
            float c_pixel[FLOAT_SIMD];                            \
            vst1q_f32(c_pixel,                                    \
                      c_tile[kk * (FLOAT_C_ob / FLOAT_SIMD)]);    \
            O[kk] = c_pixel[0];                                   \
                                                                  \
        }                                                         \
        else                                                      \
        {                                                         \
            for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)   \
            {                                                     \
                vst1q_f32(O + kk * C_ob + jj * FLOAT_SIMD,        \
                          c_tile[kk * (C_ob / FLOAT_SIMD) + jj]); \
            }                                                     \
        }                                                         \
    }
#endif


//****************************************************************************
// Convolution Computation (Strided GEMM)
//****************************************************************************

// TODO: add unroll of C_ib dim.
// I: [W_ob, C_ob] -- unroll C_ob in vector register and broadcast from diff lane.
// W: [Hf, Wf, C_ob, C_ob] -- unroll k loop will stride by C_ob (indexes C_ob).
#define FLOAT_CONV_TILE_C(step, I, W)                                   \
    float32x4_t Wv[C_ob / FLOAT_SIMD];                                  \
    for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)           \
    {                                                                   \
        Wv[jj] = vld1q_f32(W + jj * FLOAT_SIMD);                        \
    }                                                                   \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        float32x4_t Iv = vld1q_dup_f32(I + kk * step);                  \
        for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)       \
        {                                                               \
            c_tile_v[kk * (FLOAT_C_ob / FLOAT_SIMD) + jj] =             \
                vfmaq_f32(c_tile_v[kk * (FLOAT_C_ob / FLOAT_SIMD) + jj], Iv, Wv[jj]); \
        }                                                               \
    }

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_CONV_END_C(step, I, W, c_cur, W_ob, C_ob)                 \
    for (uint32_t i = 0; i < _UNROLL; i++)                              \
    {                                                                   \
        for (uint32_t kk = 0; kk < W_ob; kk++)                          \
        {                                                               \
            for (uint32_t jj = 0; jj < C_ob; jj++)                      \
            {                                                           \
                c_cur[kk * C_ob + jj] += a[kk * step + i] * b[i * C_ob + jj]; \
            }                                                           \
        }                                                               \
    }

#else

#define FLOAT_CONV_END_C(step, I, W, c_cur, W_ob, C_ob)                 \
    float32x4_t Wv[C_ob / FLOAT_SIMD];                                  \
    float32x4_t Iv;                                                     \
    for (uint32_t ii_unroll = 0; ii_unroll < _UNROLL; ii_unroll++)      \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            Wv[jj] = vld1q_f32(W + (ii_unroll * C_ob) + jj * FLOAT_SIMD); \
        }                                                               \
        switch (W_ob)                                                   \
        {                                                               \
        case 5:                                                         \
            Iv = vld1q_dup_f32(I + ((4 * step) + ii_unroll));           \
            c_cur[(4 * (C_ob / FLOAT_SIMD)) + 0] =                      \
                vfmaq_f32(c_cur[(4 * (C_ob / FLOAT_SIMD)) + 0], Iv, Wv[0]); \
            c_cur[(4 * (C_ob / FLOAT_SIMD)) + 1] =                      \
                vfmaq_f32(c_cur[(4 * (C_ob / FLOAT_SIMD)) + 1], Iv, Wv[1]); \
            c_cur[(4 * (C_ob / FLOAT_SIMD)) + 2] =                      \
                vfmaq_f32(c_cur[(4 * (C_ob / FLOAT_SIMD)) + 2], Iv, Wv[2]); \
            c_cur[(4 * (C_ob / FLOAT_SIMD)) + 3] =                      \
                vfmaq_f32(c_cur[(4 * (C_ob / FLOAT_SIMD)) + 3], Iv, Wv[3]); \
        case 4:                                                         \
            Iv = vld1q_dup_f32(I + ((3 * step) + ii_unroll));           \
            c_cur[(3 * (C_ob / FLOAT_SIMD)) + 0] =                      \
                vfmaq_f32(c_cur[(3 * (C_ob / FLOAT_SIMD)) + 0], Iv, Wv[0]); \
            c_cur[(3 * (C_ob / FLOAT_SIMD)) + 1] =                      \
                vfmaq_f32(c_cur[(3 * (C_ob / FLOAT_SIMD)) + 1], Iv, Wv[1]); \
            c_cur[(3 * (C_ob / FLOAT_SIMD)) + 2] =                      \
                vfmaq_f32(c_cur[(3 * (C_ob / FLOAT_SIMD)) + 2], Iv, Wv[2]); \
            c_cur[(3 * (C_ob / FLOAT_SIMD)) + 3] =                      \
                vfmaq_f32(c_cur[(3 * (C_ob / FLOAT_SIMD)) + 3], Iv, Wv[3]); \
        case 3:                                                         \
            Iv = vld1q_dup_f32(I + ((2 * step) + ii_unroll));           \
            c_cur[(2 * (C_ob / FLOAT_SIMD)) + 0] =                      \
                vfmaq_f32(c_cur[(2 * (C_ob / FLOAT_SIMD)) + 0], Iv, Wv[0]); \
            c_cur[(2 * (C_ob / FLOAT_SIMD)) + 1] =                      \
                vfmaq_f32(c_cur[(2 * (C_ob / FLOAT_SIMD)) + 1], Iv, Wv[1]); \
            c_cur[(2 * (C_ob / FLOAT_SIMD)) + 2] =                      \
                vfmaq_f32(c_cur[(2 * (C_ob / FLOAT_SIMD)) + 2], Iv, Wv[2]); \
            c_cur[(2 * (C_ob / FLOAT_SIMD)) + 3] =                      \
                vfmaq_f32(c_cur[(2 * (C_ob / FLOAT_SIMD)) + 3], Iv, Wv[3]); \
        case 2:                                                         \
            Iv = vld1q_dup_f32(I + ((1 * step) + ii_unroll));           \
            c_cur[(1 * (C_ob / FLOAT_SIMD)) + 0] =                      \
                vfmaq_f32(c_cur[(1 * (C_ob / FLOAT_SIMD)) + 0], Iv, Wv[0]); \
            c_cur[(1 * (C_ob / FLOAT_SIMD)) + 1] =                      \
                vfmaq_f32(c_cur[(1 * (C_ob / FLOAT_SIMD)) + 1], Iv, Wv[1]); \
            c_cur[(1 * (C_ob / FLOAT_SIMD)) + 2] =                      \
                vfmaq_f32(c_cur[(1 * (C_ob / FLOAT_SIMD)) + 2], Iv, Wv[2]); \
            c_cur[(1 * (C_ob / FLOAT_SIMD)) + 3] =                      \
                vfmaq_f32(c_cur[(1 * (C_ob / FLOAT_SIMD)) + 3], Iv, Wv[3]); \
        case 1:                                                         \
            Iv = vld1q_dup_f32(I + ((0 * step) + ii_unroll));           \
            c_cur[(0 * (C_ob / FLOAT_SIMD)) + 0] =                      \
                vfmaq_f32(c_cur[(0 * (C_ob / FLOAT_SIMD)) + 0], Iv, Wv[0]); \
            c_cur[(0 * (C_ob / FLOAT_SIMD)) + 1] =                      \
                vfmaq_f32(c_cur[(0 * (C_ob / FLOAT_SIMD)) + 1], Iv, Wv[1]); \
            c_cur[(0 * (C_ob / FLOAT_SIMD)) + 2] =                      \
                vfmaq_f32(c_cur[(0 * (C_ob / FLOAT_SIMD)) + 2], Iv, Wv[2]); \
            c_cur[(0 * (C_ob / FLOAT_SIMD)) + 3] =                      \
                vfmaq_f32(c_cur[(0 * (C_ob / FLOAT_SIMD)) + 3], Iv, Wv[3]); \
                                                                        \
        }                                                               \
    }
#endif


//****************************************************************************
// Pooling
//   Max pooling
//****************************************************************************

#define FLOAT_MAX_TILE_C(step, I)                                        \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                         \
    {                                                                    \
        for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)        \
        {                                                                \
            float32x4_t Iv = vld1q_f32(I + kk * step + jj * FLOAT_SIMD); \
            c_tile_v[kk * (FLOAT_C_ob / FLOAT_SIMD) + jj] =              \
                vmaxq_f32(Iv, c_tile_v[kk * (FLOAT_C_ob / FLOAT_SIMD) + jj]); \
        }                                                                \
    }

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_MAX_END_C(step, I, c_cur, W_ob, C_ob)             \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            c_cur[kk * C_ob + jj] =                             \
                c_cur[kk * C_ob + jj] > a[kk * step + jj] ?     \
                c_cur[kk * C_ob + jj] : a[kk * step + jj];      \
        }                                                       \
    }

#else

#define FLOAT_MAX_END_C(step, I, c_cur, W_ob, C_ob)                      \
    for (uint32_t kk = 0; kk < W_ob; kk++)                               \
    {                                                                    \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)              \
        {                                                                \
            float32x4_t Iv = vld1q_f32(a + kk * step + jj * FLOAT_SIMD); \
            c_cur[(kk) * (C_ob / FLOAT_SIMD) + jj] =                     \
                vmaxq_f32(Iv, c_cur[(kk) * (C_ob / FLOAT_SIMD) + jj]);   \
        }                                                                \
    }
#endif


//****************************************************************************
//DW Convolution
//****************************************************************************

#define FLOAT_DW_TILE_C(step, I, W)                                      \
    float32x4_t Wv[FLOAT_C_ob / FLOAT_SIMD];                             \
    for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)            \
    {                                                                    \
        Wv[jj] = vld1q_f32(W + jj * FLOAT_SIMD);                         \
    }                                                                    \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                         \
    {                                                                    \
        for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)        \
        {                                                                \
            float32x4_t Iv = vld1q_f32(I + kk * step + jj * FLOAT_SIMD); \
            c_tile_v[kk * (FLOAT_C_ob / FLOAT_SIMD) + jj] =              \
                vfmaq_f32(c_tile_v[kk * (FLOAT_C_ob / FLOAT_SIMD) + jj], Iv, Wv[jj]); \
        }                                                                \
    }

// TODO: is this tested?
#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_DW_END_C(step, I, W, c_cur, W_ob, C_ob)           \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            c_cur[kk * C_ob + jj] += I[kk * step + jj] * W[jj]; \
        }                                                       \
    }

#else

#define FLOAT_DW_END_C(step, I, W, c_cur, W_ob, C_ob)                    \
    for (uint32_t kk = 0; kk < W_ob; kk++)                               \
    {                                                                    \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)              \
        {                                                                \
            float32x4_t Iv = vld1q_f32(I + kk * step + jj * FLOAT_SIMD); \
            float32x4_t Wv = vld1q_f32(W + jj * FLOAT_SIMD);             \
            c_cur[(kk) * (C_ob / FLOAT_SIMD) + jj] =                     \
                vfmaq_f32(c_cur[(kk) * (C_ob / FLOAT_SIMD) + jj], Iv, Wv); \
        }                                                                \
    }
#endif


//****************************************************************************
// ReLU Activation
//****************************************************************************

// Same kernel as Pooling, set to zero to start.
/// @todo in intrinsics_gen.h?

//#define FLOAT_FUSED_RELU_TILE_C

//#define FLOAT_FUSED_RELU_END_C

//****************************************************************************
// Leaky ReLU activation
//****************************************************************************

#define FLOAT_COND_SCALE_SIMD_C(c_x_x, mask, Wv, Iv, I, kk, jj)         \
    Iv = vld1q_f32(I + kk * step + jj * FLOAT_SIMD);                    \
    c_x_x = vmaxq_f32(Iv, c_x_x);                                       \
    mask = vcltq_f32(Iv, c_x_x);                                        \
    Iv = vmulq_f32(Iv, Wv);                                             \
    Iv = (float32x4_t) vandq_s32((int32x4_t)(Iv), (int32x4_t)(mask));   \
    c_x_x = vaddq_f32(Iv, c_x_x);

#define FLOAT_COND_SCALE_TILE_C(step, I, W)                             \
    float32x4_t Wv = vld1q_dup_f32(W);                                  \
    float32x4_t Iv;                                                     \
    uint32x4_t mask;                                                    \
    FLOAT_COND_SCALE_SIMD_C(c_0_0, mask, Wv, Iv, I, 0, 0);              \
    FLOAT_COND_SCALE_SIMD_C(c_0_1, mask, Wv, Iv, I, 0, 1);              \
    FLOAT_COND_SCALE_SIMD_C(c_0_2, mask, Wv, Iv, I, 0, 2);              \
    FLOAT_COND_SCALE_SIMD_C(c_0_3, mask, Wv, Iv, I, 0, 3);              \
    /**/                                                                \
    FLOAT_COND_SCALE_SIMD_C(c_1_0, mask, Wv, Iv, I, 1, 0);              \
    FLOAT_COND_SCALE_SIMD_C(c_1_1, mask, Wv, Iv, I, 1, 1);              \
    FLOAT_COND_SCALE_SIMD_C(c_1_2, mask, Wv, Iv, I, 1, 2);              \
    FLOAT_COND_SCALE_SIMD_C(c_1_3, mask, Wv, Iv, I, 1, 3);              \
    /**/                                                                \
    FLOAT_COND_SCALE_SIMD_C(c_2_0, mask, Wv, Iv, I, 2, 0);              \
    FLOAT_COND_SCALE_SIMD_C(c_2_1, mask, Wv, Iv, I, 2, 1);              \
    FLOAT_COND_SCALE_SIMD_C(c_2_2, mask, Wv, Iv, I, 2, 2);              \
    FLOAT_COND_SCALE_SIMD_C(c_2_3, mask, Wv, Iv, I, 2, 3);              \
    /**/                                                                \
    FLOAT_COND_SCALE_SIMD_C(c_3_0, mask, Wv, Iv, I, 3, 0);              \
    FLOAT_COND_SCALE_SIMD_C(c_3_1, mask, Wv, Iv, I, 3, 1);              \
    FLOAT_COND_SCALE_SIMD_C(c_3_2, mask, Wv, Iv, I, 3, 2);              \
    FLOAT_COND_SCALE_SIMD_C(c_3_3, mask, Wv, Iv, I, 3, 3);              \
    /**/                                                                \
    FLOAT_COND_SCALE_SIMD_C(c_4_0, mask, Wv, Iv, I, 4, 0);              \
    FLOAT_COND_SCALE_SIMD_C(c_4_1, mask, Wv, Iv, I, 4, 1);              \
    FLOAT_COND_SCALE_SIMD_C(c_4_2, mask, Wv, Iv, I, 4, 2);              \
    FLOAT_COND_SCALE_SIMD_C(c_4_3, mask, Wv, Iv, I, 4, 3);              \
    /**/                                                                \
    FLOAT_COND_SCALE_SIMD_C(c_5_0, mask, Wv, Iv, I, 5, 0);              \
    FLOAT_COND_SCALE_SIMD_C(c_5_1, mask, Wv, Iv, I, 5, 1);              \
    FLOAT_COND_SCALE_SIMD_C(c_5_2, mask, Wv, Iv, I, 5, 2);              \
    FLOAT_COND_SCALE_SIMD_C(c_5_3, mask, Wv, Iv, I, 5, 3);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_COND_SCALE_END_C(step, I, W, c_cur, W_ob, C_ob)           \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t const *I_pixel = I;                                        \
    c_tile_t scale = W[0];                                              \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *I_channel = I_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (*(I_channel) > *(c_channel)) ? *(I_channel) : (*(I_channel) * (scale)); \
            c_channel++;                                                \
            I_channel++;                                                \
        }                                                               \
        I_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#else

#define FLOAT_COND_SCALE_END_C(step, I, W, c_cur, W_ob, C_ob)           \
    float32x4_t Wv = vld1q_dup_f32(W);                                  \
    float32x4_t Iv;                                                     \
    uint32x4_t mask;                                                    \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            float32x4_t cv = c_cur[kk * (C_ob / FLOAT_SIMD) + jj];      \
            FLOAT_COND_SCALE_SIMD_C(cv, mask, Wv, Iv, I, kk, jj);       \
            c_cur[kk * (C_ob / FLOAT_SIMD) + jj] = cv;                  \
        }                                                               \
    }
#endif

#define FLOAT_FUSED_COND_SCALE_SIMD_C(c_x_x, mask, Wv, Iv)              \
    Iv = vmovq_n_f32(0.0f);                                             \
    mask = vcltq_f32(c_x_x, Iv);                                        \
    Iv = vmulq_f32(c_x_x, Wv);                                          \
    c_x_x = vmaxq_f32(vmovq_n_f32(0.0f), c_x_x);                        \
    Iv = (float32x4_t) vandq_s32((int32x4_t)(Iv), (int32x4_t)(mask));   \
    c_x_x = vaddq_f32(Iv, c_x_x);

#define FLOAT_FUSED_COND_SCALE_TILE_C(W)                                \
    float32x4_t Wv = vld1q_dup_f32(W);                                  \
    float32x4_t Iv;                                                     \
    uint32x4_t mask;                                                    \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_0_0, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_0_1, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_0_2, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_0_3, mask, Wv, Iv);                 \
    /**/                                                                \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_1_0, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_1_1, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_1_2, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_1_3, mask, Wv, Iv);                 \
    /**/                                                                \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_2_0, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_2_1, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_2_2, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_2_3, mask, Wv, Iv);                 \
    /**/                                                                \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_3_0, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_3_1, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_3_2, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_3_3, mask, Wv, Iv);                 \
    /**/                                                                \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_4_0, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_4_1, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_4_2, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_4_3, mask, Wv, Iv);                 \
    /**/                                                                \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_5_0, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_5_1, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_5_2, mask, Wv, Iv);                 \
    FLOAT_FUSED_COND_SCALE_SIMD_C(c_5_3, mask, Wv, Iv);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_FUSED_COND_SCALE_END_C(W, c_cur, W_ob, C_ob)              \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t scale = S[0];                                              \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (0.0 > *(c_channel)) ? (*(c_channel) * (scale)) : *(c_channel); \
            c_channel++;                                                \
        }                                                               \
        c_pixel += C_ob;                                                \
    }

#else

#define FLOAT_FUSED_COND_SCALE_END_C(W, c_cur, W_ob, C_ob)              \
    float32x4_t Wv = vld1q_dup_f32(W);                                  \
    float32x4_t Iv;                                                     \
    uint32x4_t mask;                                                    \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            float32x4_t cv = c_cur[kk * (C_ob / FLOAT_SIMD) + jj];      \
            FLOAT_FUSED_COND_SCALE_SIMD_C(cv, mask, Wv, Iv);            \
            c_cur[kk * (C_ob / FLOAT_SIMD) + jj] = cv;                  \
        }                                                               \
    }
#endif


//****************************************************************************
// Accumulation kernels
//****************************************************************************

#define FLOAT_ACCUM_TILE_C(step, I)                     \
    float32x4_t Iv;                                     \
    Iv = vld1q_f32(I + 0 * step + 0 * FLOAT_SIMD);      \
    c_0_0 = vaddq_f32(c_0_0, Iv);                       \
    Iv = vld1q_f32(I + 0 * step + 1 * FLOAT_SIMD);      \
    c_0_1 = vaddq_f32(c_0_1, Iv);                       \
    Iv = vld1q_f32(I + 0 * step + 2 * FLOAT_SIMD);      \
    c_0_2 = vaddq_f32(c_0_2, Iv);                       \
    Iv = vld1q_f32(I + 0 * step + 3 * FLOAT_SIMD);      \
    c_0_3 = vaddq_f32(c_0_3, Iv);                       \
    Iv = vld1q_f32(I + 1 * step + 0 * FLOAT_SIMD);      \
    c_1_0 = vaddq_f32(c_1_0, Iv);                       \
    Iv = vld1q_f32(I + 1 * step + 1 * FLOAT_SIMD);      \
    c_1_1 = vaddq_f32(c_1_1, Iv);                       \
    Iv = vld1q_f32(I + 1 * step + 2 * FLOAT_SIMD);      \
    c_1_2 = vaddq_f32(c_1_2, Iv);                       \
    Iv = vld1q_f32(I + 1 * step + 3 * FLOAT_SIMD);      \
    c_1_3 = vaddq_f32(c_1_3, Iv);                       \
    Iv = vld1q_f32(I + 2 * step + 0 * FLOAT_SIMD);      \
    c_2_0 = vaddq_f32(c_2_0, Iv);                       \
    Iv = vld1q_f32(I + 2 * step + 1 * FLOAT_SIMD);      \
    c_2_1 = vaddq_f32(c_2_1, Iv);                       \
    Iv = vld1q_f32(I + 2 * step + 2 * FLOAT_SIMD);      \
    c_2_2 = vaddq_f32(c_2_2, Iv);                       \
    Iv = vld1q_f32(I + 2 * step + 3 * FLOAT_SIMD);      \
    c_2_3 = vaddq_f32(c_2_3, Iv);                       \
    Iv = vld1q_f32(I + 3 * step + 0 * FLOAT_SIMD);      \
    c_3_0 = vaddq_f32(c_3_0, Iv);                       \
    Iv = vld1q_f32(I + 3 * step + 1 * FLOAT_SIMD);      \
    c_3_1 = vaddq_f32(c_3_1, Iv);                       \
    Iv = vld1q_f32(I + 3 * step + 2 * FLOAT_SIMD);      \
    c_3_2 = vaddq_f32(c_3_2, Iv);                       \
    Iv = vld1q_f32(I + 3 * step + 3 * FLOAT_SIMD);      \
    c_3_3 = vaddq_f32(c_3_3, Iv);                       \
    Iv = vld1q_f32(I + 4 * step + 0 * FLOAT_SIMD);      \
    c_4_0 = vaddq_f32(c_4_0, Iv);                       \
    Iv = vld1q_f32(I + 4 * step + 1 * FLOAT_SIMD);      \
    c_4_1 = vaddq_f32(c_4_1, Iv);                       \
    Iv = vld1q_f32(I + 4 * step + 2 * FLOAT_SIMD);      \
    c_4_2 = vaddq_f32(c_4_2, Iv);                       \
    Iv = vld1q_f32(I + 4 * step + 3 * FLOAT_SIMD);      \
    c_4_3 = vaddq_f32(c_4_3, Iv);                       \
    Iv = vld1q_f32(I + 5 * step + 0 * FLOAT_SIMD);      \
    c_5_0 = vaddq_f32(c_5_0, Iv);                       \
    Iv = vld1q_f32(I + 5 * step + 1 * FLOAT_SIMD);      \
    c_5_1 = vaddq_f32(c_5_1, Iv);                       \
    Iv = vld1q_f32(I + 5 * step + 2 * FLOAT_SIMD);      \
    c_5_2 = vaddq_f32(c_5_2, Iv);                       \
    Iv = vld1q_f32(I + 5 * step + 3 * FLOAT_SIMD);      \
    c_5_3 = vaddq_f32(c_5_3, Iv);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_ACCUM_END_C(step, I, c_cur, W_ob, C_ob)   \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_cur[kk * C_ob + jj] += I[kk * step + jj]; \
        }                                               \
    }

#else

#define FLOAT_ACCUM_END_C(step, I, c_cur, W_ob, C_ob)                    \
    for (uint32_t kk = 0; kk < W_ob; kk++)                               \
    {                                                                    \
        for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)        \
        {                                                                \
            float32x4_t Iv = vld1q_f32(I + kk * step + jj * FLOAT_SIMD); \
            c_cur[(kk) * (FLOAT_C_ob / FLOAT_SIMD) + jj] =               \
                vaddq_f32(Iv, c_cur[(kk) * (FLOAT_C_ob / FLOAT_SIMD) + jj]); \
        }                                                                \
    }
#endif


//****************************************************************************
// Broadcast multiplication kernels
//****************************************************************************

#define FLOAT_DIV_TILE_C(scale)                         \
    float32x4_t Iv;                                     \
    Iv = vld1q_dup_f32(&scale);                         \
    c_0_0 = vmulq_f32(c_0_0, Iv);                       \
    c_0_1 = vmulq_f32(c_0_1, Iv);                       \
    c_0_2 = vmulq_f32(c_0_2, Iv);                       \
    c_0_3 = vmulq_f32(c_0_3, Iv);                       \
    c_1_0 = vmulq_f32(c_1_0, Iv);                       \
    c_1_1 = vmulq_f32(c_1_1, Iv);                       \
    c_1_2 = vmulq_f32(c_1_2, Iv);                       \
    c_1_3 = vmulq_f32(c_1_3, Iv);                       \
    c_2_0 = vmulq_f32(c_2_0, Iv);                       \
    c_2_1 = vmulq_f32(c_2_1, Iv);                       \
    c_2_2 = vmulq_f32(c_2_2, Iv);                       \
    c_2_3 = vmulq_f32(c_2_3, Iv);                       \
    c_3_0 = vmulq_f32(c_3_0, Iv);                       \
    c_3_1 = vmulq_f32(c_3_1, Iv);                       \
    c_3_2 = vmulq_f32(c_3_2, Iv);                       \
    c_3_3 = vmulq_f32(c_3_3, Iv);                       \
    c_4_0 = vmulq_f32(c_4_0, Iv);                       \
    c_4_1 = vmulq_f32(c_4_1, Iv);                       \
    c_4_2 = vmulq_f32(c_4_2, Iv);                       \
    c_4_3 = vmulq_f32(c_4_3, Iv);                       \
    c_5_0 = vmulq_f32(c_5_0, Iv);                       \
    c_5_1 = vmulq_f32(c_5_1, Iv);                       \
    c_5_2 = vmulq_f32(c_5_2, Iv);                       \
    c_5_3 = vmulq_f32(c_5_3, Iv);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_DIV_END_C(c_cur, scale, W_ob, C_ob) \
    float *c_pixel = c_cur;                       \
    for (uint32_t kk = 0; kk < W_ob; kk++)        \
    {                                             \
        float *c_channel = c_pixel;               \
        for (uint32_t jj = 0; jj < C_ob; jj++)    \
        {                                         \
            *(c_channel) *= scale;                \
            c_channel++;                          \
        }                                         \
        c_pixel += C_ob;                          \
    }
#else
#define FLOAT_DIV_END_C(c_cur, scale, W_ob, C_ob)                  \
    float32x4_t Iv;                                                \
    Iv = vld1q_dup_f32(&scale);                                    \
    float32x4_t *c_pixel = c_cur;                                  \
    for (uint32_t kk = 0; kk < W_ob; kk++)                         \
    {                                                              \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)        \
        {                                                          \
            c_pixel[(kk) * (C_ob / FLOAT_SIMD) + jj] =             \
                vmulq_f32(c_pixel[(kk) * (C_ob / FLOAT_SIMD) + jj], Iv); \
        }                                                          \
    }
#endif

//****************************************************************************
// Broadcast addition kernels
//****************************************************************************

#define FLOAT_EWISE_ADD_SCALAR_TILE_C(scalar)           \
    float32x4_t Iv;                                     \
    Iv = vld1q_dup_f32(&scalar);                        \
    c_0_0 = vaddq_f32(c_0_0, Iv);                       \
    c_0_1 = vaddq_f32(c_0_1, Iv);                       \
    c_0_2 = vaddq_f32(c_0_2, Iv);                       \
    c_0_3 = vaddq_f32(c_0_3, Iv);                       \
    c_1_0 = vaddq_f32(c_1_0, Iv);                       \
    c_1_1 = vaddq_f32(c_1_1, Iv);                       \
    c_1_2 = vaddq_f32(c_1_2, Iv);                       \
    c_1_3 = vaddq_f32(c_1_3, Iv);                       \
    c_2_0 = vaddq_f32(c_2_0, Iv);                       \
    c_2_1 = vaddq_f32(c_2_1, Iv);                       \
    c_2_2 = vaddq_f32(c_2_2, Iv);                       \
    c_2_3 = vaddq_f32(c_2_3, Iv);                       \
    c_3_0 = vaddq_f32(c_3_0, Iv);                       \
    c_3_1 = vaddq_f32(c_3_1, Iv);                       \
    c_3_2 = vaddq_f32(c_3_2, Iv);                       \
    c_3_3 = vaddq_f32(c_3_3, Iv);                       \
    c_4_0 = vaddq_f32(c_4_0, Iv);                       \
    c_4_1 = vaddq_f32(c_4_1, Iv);                       \
    c_4_2 = vaddq_f32(c_4_2, Iv);                       \
    c_4_3 = vaddq_f32(c_4_3, Iv);                       \
    c_5_0 = vaddq_f32(c_5_0, Iv);                       \
    c_5_1 = vaddq_f32(c_5_1, Iv);                       \
    c_5_2 = vaddq_f32(c_5_2, Iv);                       \
    c_5_3 = vaddq_f32(c_5_3, Iv);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_EWISE_ADD_SCALAR_END_C(c_cur, scalar, W_ob, C_ob) \
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
#else
#define FLOAT_EWISE_ADD_SCALAR_END_C(c_cur, scalar, W_ob, C_ob)    \
    float32x4_t Iv;                                                \
    Iv = vld1q_dup_f32(&scalar);                                   \
    float32x4_t *c_pixel = c_cur;                                  \
    for (uint32_t kk = 0; kk < W_ob; kk++)                         \
    {                                                              \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)        \
        {                                                          \
            c_pixel[(kk) * (C_ob / FLOAT_SIMD) + jj] =             \
                vaddq_f32(c_pixel[(kk) * (C_ob / FLOAT_SIMD) + jj], Iv); \
        }                                                          \
    }
#endif

//****************************************************************************
// Accumulate upsampling
//****************************************************************************

//@todo build and test this on arm platform

#define FLOAT_ACCUM_TILE_C_upsample(I, factor)                   \
    float32x4_t a_0, a_1, a_2, a_3;                              \
    a_0 = vld1q_f32(I + (0 / factor) * C_ob + 0 * FLOAT_SIMD);   \
    a_1 = vld1q_f32(I + (0 / factor) * C_ob + 1 * FLOAT_SIMD);   \
    a_2 = vld1q_f32(I + (0 / factor) * C_ob + 2 * FLOAT_SIMD);   \
    a_3 = vld1q_f32(I + (0 / factor) * C_ob + 3 * FLOAT_SIMD);   \
    c_0_0 = vaddq_f32(c_0_0, a_0);                               \
    a_0 = vld1q_f32(I + (1 / factor) * C_ob + 0 * FLOAT_SIMD);   \
    c_0_1 = vaddq_f32(c_0_1, a_1);                               \
    a_1 = vld1q_f32(I + (1 / factor) * C_ob + 1 * FLOAT_SIMD);   \
    c_0_2 = vaddq_f32(c_0_2, a_2);                               \
    a_2 = vld1q_f32(I + (1 / factor) * C_ob + 2 * FLOAT_SIMD);   \
    c_0_3 = vaddq_f32(c_0_3, a_3);                               \
    a_3 = vld1q_f32(I + (1 / factor) * C_ob + 3 * FLOAT_SIMD);   \
    c_1_0 = vaddq_f32(c_1_0, a_0);                               \
    a_0 = vld1q_f32(I + (2 / factor) * C_ob + 0 * FLOAT_SIMD);   \
    c_1_1 = vaddq_f32(c_1_1, a_1);                               \
    a_1 = vld1q_f32(I + (2 / factor) * C_ob + 1 * FLOAT_SIMD);   \
    c_1_2 = vaddq_f32(c_1_2, a_2);                               \
    a_2 = vld1q_f32(I + (2 / factor) * C_ob + 2 * FLOAT_SIMD);   \
    c_1_3 = vaddq_f32(c_1_3, a_3);                               \
    a_3 = vld1q_f32(I + (2 / factor) * C_ob + 3 * FLOAT_SIMD);   \
    c_2_0 = vaddq_f32(c_2_0, a_0);                               \
    a_0 = vld1q_f32(I + (3 / factor) * C_ob + 0 * FLOAT_SIMD);   \
    c_2_1 = vaddq_f32(c_2_1, a_1);                               \
    a_1 = vld1q_f32(I + (3 / factor) * C_ob + 1 * FLOAT_SIMD);   \
    c_2_2 = vaddq_f32(c_2_2, a_2);                               \
    a_2 = vld1q_f32(I + (3 / factor) * C_ob + 2 * FLOAT_SIMD);   \
    c_2_3 = vaddq_f32(c_2_3, a_3);                               \
    a_3 = vld1q_f32(I + (3 / factor) * C_ob + 3 * FLOAT_SIMD);   \
    c_3_0 = vaddq_f32(c_3_0, a_0);                               \
    a_0 = vld1q_f32(I + (4 / factor) * C_ob + 0 * FLOAT_SIMD);   \
    c_3_1 = vaddq_f32(c_3_1, a_1);                               \
    a_1 = vld1q_f32(I + (4 / factor) * C_ob + 1 * FLOAT_SIMD);   \
    c_3_2 = vaddq_f32(c_3_2, a_2);                               \
    a_2 = vld1q_f32(I + (4 / factor) * C_ob + 2 * FLOAT_SIMD);   \
    c_3_3 = vaddq_f32(c_3_3, a_3);                               \
    a_3 = vld1q_f32(I + (4 / factor) * C_ob + 3 * FLOAT_SIMD);   \
    c_4_0 = vaddq_f32(c_4_0, a_0);                               \
    a_0 = vld1q_f32(I + (5 / factor) * C_ob + 0 * FLOAT_SIMD);   \
    c_4_1 = vaddq_f32(c_4_1, a_1);                               \
    a_1 = vld1q_f32(I + (5 / factor) * C_ob + 1 * FLOAT_SIMD);   \
    c_4_2 = vaddq_f32(c_4_2, a_2);                               \
    a_2 = vld1q_f32(I + (5 / factor) * C_ob + 2 * FLOAT_SIMD);   \
    c_4_3 = vaddq_f32(c_4_3, a_3);                               \
    a_3 = vld1q_f32(I + (5 / factor) * C_ob + 3 * FLOAT_SIMD);   \
    c_5_0 = vaddq_f32(c_5_0, a_0);                               \
    c_5_1 = vaddq_f32(c_5_1, a_1);                               \
    c_5_2 = vaddq_f32(c_5_2, a_2);                               \
    c_5_3 = vaddq_f32(c_5_3, a_3);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_ACCUM_END_C_upsample(I, factor, W_ob, C_ob)             \
    for (uint32_t kk = 0; kk < W_ob; kk++)                            \
    {                                                                 \
        for (uint32_t jj = 0; jj < C_ob; jj++)                        \
        {                                                             \
            c_tile[kk * C_ob + jj] += I[(kk / factor) * (C_ob) + jj]; \
        }                                                             \
    }

#else

#define FLOAT_ACCUM_END_C_upsample(I, factor, W_ob, C_ob)               \
    c_tile_t Iv;                                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            Iv = vld1q_f32(I + (kk / factor) * (C_ob) + jj * FLOAT_SIMD); \
            c_tile[kk * (C_ob / FLOAT_SIMD) + jj] =                     \
                vaddq_f32(c_tile[kk * (C_ob / FLOAT_SIMD) + jj], Iv);   \
        }                                                               \
    }
#endif

//

//****************************************************************************
// Accumulate channel dimension
//****************************************************************************

#define FLOAT_REDUCE_CHANNEL_END_C(O_w_left, C_ob)                      \
    if constexpr(C_ob == 1 && C_ob != FLOAT_SIMD_EPILOGUE)              \
    {                                                                   \
        float c_tile_array[FLOAT_C_ob];                                 \
        for (uint32_t kk = 0; kk < O_w_left; kk++)                      \
        {                                                               \
            float32x4_t *c_channel_v = c_tile + kk * (FLOAT_C_ob / FLOAT_SIMD); \
            c_channel_v[0] = vaddq_f32(c_channel_v[0], c_channel_v[1]); \
            c_channel_v[2] = vaddq_f32(c_channel_v[2], c_channel_v[3]); \
            c_channel_v[0] = vaddq_f32(c_channel_v[0], c_channel_v[2]); \
                                                                        \
            vst1q_f32(c_tile_array, c_channel_v[0]);                    \
            for (uint32_t jj = 1; jj < FLOAT_SIMD; jj++)                \
            {                                                           \
                c_tile_array[0] += c_tile_array[jj];                    \
                c_tile_array[jj] = 0;                                   \
            }                                                           \
                                                                        \
            c_channel_v[0] = vld1q_f32(c_tile_array);                   \
            c_channel_v[1] = vdupq_n_f32(0.0);                          \
            c_channel_v[2] = vdupq_n_f32(0.0);                          \
            c_channel_v[3] = vdupq_n_f32(0.0);                          \
        }                                                               \
    }

//****************************************************************************
// Ewise exponentiation (Softmax)
//****************************************************************************

/*#define FLOAT_EXP_TILE_C(step, a, W_ob, C_ob) \
    float32x4_t a_0, a_1, a_2, a_3;           \
    float const *a_pixel = a;                 \
    a_0 = vld1q_f32(a_pixel + 0 * FLOAT_SIMD);      \
    a_1 = vld1q_f32(a_pixel + 1 * FLOAT_SIMD);      \
    a_2 = vld1q_f32(a_pixel + 2 * FLOAT_SIMD);      \
    a_3 = vld1q_f32(a_pixel + 3 * FLOAT_SIMD);      \
    a_pixel += step;                          \
    c_0_0 = exp_ps(a_0);                      \
    c_0_1 = exp_ps(a_1);a_0 = vld1q_f32(a_pixel + 0 * FLOAT_SIMD);      \
    c_0_2 = exp_ps(a_2);a_1 = vld1q_f32(a_pixel + 1 * FLOAT_SIMD);      \
    c_0_3 = exp_ps(a_3);a_2 = vld1q_f32(a_pixel + 2 * FLOAT_SIMD);      \
    \
    c_1_0 = exp_ps(a_0);a_3 = vld1q_f32(a_pixel + 3 * FLOAT_SIMD); a_pixel += step;  \
    c_1_1 = exp_ps(a_1);a_0 = vld1q_f32(a_pixel + 0 * FLOAT_SIMD);      \
    c_1_2 = exp_ps(a_2);a_1 = vld1q_f32(a_pixel + 1 * FLOAT_SIMD);      \
    c_1_3 = exp_ps(a_3);a_2 = vld1q_f32(a_pixel + 2 * FLOAT_SIMD);      \
    \
    c_2_0 = exp_ps(a_0);a_3 = vld1q_f32(a_pixel + 3 * FLOAT_SIMD); a_pixel += step;  \
    c_2_1 = exp_ps(a_1);a_0 = vld1q_f32(a_pixel + 0 * FLOAT_SIMD);      \
    c_2_2 = exp_ps(a_2);a_1 = vld1q_f32(a_pixel + 1 * FLOAT_SIMD);      \
    c_2_3 = exp_ps(a_3);a_2 = vld1q_f32(a_pixel + 2 * FLOAT_SIMD);      \
    \
    c_3_0 = exp_ps(a_0);a_3 = vld1q_f32(a_pixel + 3 * FLOAT_SIMD); a_pixel += step;  \
    c_3_1 = exp_ps(a_1);a_0 = vld1q_f32(a_pixel + 0 * FLOAT_SIMD);      \
    c_3_2 = exp_ps(a_2);a_1 = vld1q_f32(a_pixel + 1 * FLOAT_SIMD);      \
    c_3_3 = exp_ps(a_3);a_2 = vld1q_f32(a_pixel + 2 * FLOAT_SIMD);      \
    \
    c_4_0 = exp_ps(a_0);a_3 = vld1q_f32(a_pixel + 3 * FLOAT_SIMD); a_pixel += step;  \
    c_4_1 = exp_ps(a_1);a_0 = vld1q_f32(a_pixel + 0 * FLOAT_SIMD);      \
    c_4_2 = exp_ps(a_2);a_1 = vld1q_f32(a_pixel + 1 * FLOAT_SIMD);      \
    c_4_3 = exp_ps(a_3);a_2 = vld1q_f32(a_pixel + 2 * FLOAT_SIMD);      \
    \
    c_5_0 = exp_ps(a_0);  \
    c_5_1 = exp_ps(a_1);      \
    c_5_2 = exp_ps(a_2);      \
    c_5_3 = exp_ps(a_3);      \
*/

#define FLOAT_EXP_TILE_C(step, I)                                    \
    float c_tile_scalar[FLOAT_W_ob * FLOAT_C_ob];                    \
    float *c_pixel = c_tile_scalar;                                  \
    float const *I_pixel = I;                                        \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                     \
    {                                                                \
        float *c_channel = c_pixel;                                  \
        float const *I_channel = I_pixel;                            \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                 \
        {                                                            \
            *(c_channel) = std::exp(*I_channel);                     \
            c_channel++;                                             \
            I_channel++;                                             \
        }                                                            \
        I_pixel += step;                                             \
        c_pixel += C_ob;                                             \
    }                                                                \
    c_0_0 = vld1q_f32(c_tile_scalar + 0 * C_ob + 0 * FLOAT_SIMD);    \
    c_0_1 = vld1q_f32(c_tile_scalar + 0 * C_ob + 1 * FLOAT_SIMD);    \
    c_0_2 = vld1q_f32(c_tile_scalar + 0 * C_ob + 2 * FLOAT_SIMD);    \
    c_0_3 = vld1q_f32(c_tile_scalar + 0 * C_ob + 3 * FLOAT_SIMD);    \
    c_1_0 = vld1q_f32(c_tile_scalar + 1 * C_ob + 0 * FLOAT_SIMD);    \
    c_1_1 = vld1q_f32(c_tile_scalar + 1 * C_ob + 1 * FLOAT_SIMD);    \
    c_1_2 = vld1q_f32(c_tile_scalar + 1 * C_ob + 2 * FLOAT_SIMD);    \
    c_1_3 = vld1q_f32(c_tile_scalar + 1 * C_ob + 3 * FLOAT_SIMD);    \
    c_2_0 = vld1q_f32(c_tile_scalar + 2 * C_ob + 0 * FLOAT_SIMD);    \
    c_2_1 = vld1q_f32(c_tile_scalar + 2 * C_ob + 1 * FLOAT_SIMD);    \
    c_2_2 = vld1q_f32(c_tile_scalar + 2 * C_ob + 2 * FLOAT_SIMD);    \
    c_2_3 = vld1q_f32(c_tile_scalar + 2 * C_ob + 3 * FLOAT_SIMD);    \
    c_3_0 = vld1q_f32(c_tile_scalar + 3 * C_ob + 0 * FLOAT_SIMD);    \
    c_3_1 = vld1q_f32(c_tile_scalar + 3 * C_ob + 1 * FLOAT_SIMD);    \
    c_3_2 = vld1q_f32(c_tile_scalar + 3 * C_ob + 2 * FLOAT_SIMD);    \
    c_3_3 = vld1q_f32(c_tile_scalar + 3 * C_ob + 3 * FLOAT_SIMD);    \
    c_4_0 = vld1q_f32(c_tile_scalar + 4 * C_ob + 0 * FLOAT_SIMD);    \
    c_4_1 = vld1q_f32(c_tile_scalar + 4 * C_ob + 1 * FLOAT_SIMD);    \
    c_4_2 = vld1q_f32(c_tile_scalar + 4 * C_ob + 2 * FLOAT_SIMD);    \
    c_4_3 = vld1q_f32(c_tile_scalar + 4 * C_ob + 3 * FLOAT_SIMD);    \
    c_5_0 = vld1q_f32(c_tile_scalar + 5 * C_ob + 0 * FLOAT_SIMD);    \
    c_5_1 = vld1q_f32(c_tile_scalar + 5 * C_ob + 1 * FLOAT_SIMD);    \
    c_5_2 = vld1q_f32(c_tile_scalar + 5 * C_ob + 2 * FLOAT_SIMD);    \
    c_5_3 = vld1q_f32(c_tile_scalar + 5 * C_ob + 3 * FLOAT_SIMD);


#if FLOAT_SIMD_EPILOGUE == 1
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
#else
/*#define FLOAT_EXP_END_C(step, I, c_cur, W_ob, C_ob)             \
  float32x4_t Iv;                                                 \
  float const * I_pixel = I;                                      \
  Iv = vld1q_f32(I_pixel);                                        \
  float32x4_t *c_pixel = c_cur;                                   \
  for (uint32_t kk = 0; kk < W_ob; kk++)                          \
  {                                                               \
     for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)          \
     {                                                            \
        c_pixel[(kk) * (C_ob / FLOAT_SIMD) + jj] = exp_ps(Iv);    \
        Iv = vld1q_f32(I_pixel + jj * FLOAT_SIMD);                \
     }                                                            \
     I_pixel += step;                                             \
  }
*/

#define FLOAT_EXP_END_C(step, I, c_cur, W_ob, C_ob)            \
    float c_tile_scalar[FLOAT_W_ob * FLOAT_C_ob];              \
    float *c_pixel = c_tile_scalar;                            \
    float const *I_pixel = I;                                  \
    for (uint32_t kk = 0; kk < W_ob; kk++)                     \
    {                                                          \
        float *c_channel = c_pixel;                            \
        float const *I_channel = I_pixel;                      \
        for (uint32_t jj = 0; jj < C_ob; jj++)                 \
        {                                                      \
            *(c_channel) = std::exp(*I_channel);               \
            c_channel++;                                       \
            I_channel++;                                       \
        }                                                      \
        c_cur[kk * (C_ob/FLOAT_SIMD) + 0] = vld1q_f32(c_pixel + 0 * FLOAT_SIMD); \
        c_cur[kk * (C_ob/FLOAT_SIMD) + 1] = vld1q_f32(c_pixel + 1 * FLOAT_SIMD); \
        c_cur[kk * (C_ob/FLOAT_SIMD) + 2] = vld1q_f32(c_pixel + 2 * FLOAT_SIMD); \
        c_cur[kk * (C_ob/FLOAT_SIMD) + 3] = vld1q_f32(c_pixel + 3 * FLOAT_SIMD); \
        I_pixel += step;                                       \
        c_pixel += C_ob;                                       \
    }                                                          \

#endif

/// @todo Missing FLOAT_FUSED_EXP_TILE_C
/// @todo Missing FLOAT_FUSED_EXP_END_C

//****************************************************************************
// Ewise logarithm
//****************************************************************************

/// @todo Missing FLOAT_LOG_TILE_C
/// @todo Missing FLOAT_LOG_END_C


//****************************************************************************
// Ewise Softsign
//****************************************************************************

#define FLOAT_SOFTSIGN_SIMD_C(c_x_x, Iv, I, kk, jj)   \
    Iv = vld1q_f32(I + kk * step + jj * FLOAT_SIMD);  \
    c_x_x = vabsq_f32(Iv);                            \
    c_x_x = vaddq_f32(c_x_x, vdupq_n_f32(1.0f));      \
    c_x_x = vdivq_f32(Iv, c_x_x);

#define FLOAT_SOFTSIGN_TILE_C(step, I)         \
    float32x4_t Iv;                            \
    FLOAT_SOFTSIGN_SIMD_C(c_0_0, Iv, I, 0, 0); \
    FLOAT_SOFTSIGN_SIMD_C(c_0_1, Iv, I, 0, 1); \
    FLOAT_SOFTSIGN_SIMD_C(c_0_2, Iv, I, 0, 2); \
    FLOAT_SOFTSIGN_SIMD_C(c_0_3, Iv, I, 0, 3); \
    /**/                                       \
    FLOAT_SOFTSIGN_SIMD_C(c_1_0, Iv, I, 1, 0); \
    FLOAT_SOFTSIGN_SIMD_C(c_1_1, Iv, I, 1, 1); \
    FLOAT_SOFTSIGN_SIMD_C(c_1_2, Iv, I, 1, 2); \
    FLOAT_SOFTSIGN_SIMD_C(c_1_3, Iv, I, 1, 3); \
    /**/                                       \
    FLOAT_SOFTSIGN_SIMD_C(c_2_0, Iv, I, 2, 0); \
    FLOAT_SOFTSIGN_SIMD_C(c_2_1, Iv, I, 2, 1); \
    FLOAT_SOFTSIGN_SIMD_C(c_2_2, Iv, I, 2, 2); \
    FLOAT_SOFTSIGN_SIMD_C(c_2_3, Iv, I, 2, 3); \
    /**/                                       \
    FLOAT_SOFTSIGN_SIMD_C(c_3_0, Iv, I, 3, 0); \
    FLOAT_SOFTSIGN_SIMD_C(c_3_1, Iv, I, 3, 1); \
    FLOAT_SOFTSIGN_SIMD_C(c_3_2, Iv, I, 3, 2); \
    FLOAT_SOFTSIGN_SIMD_C(c_3_3, Iv, I, 3, 3); \
    /**/                                       \
    FLOAT_SOFTSIGN_SIMD_C(c_4_0, Iv, I, 4, 0); \
    FLOAT_SOFTSIGN_SIMD_C(c_4_1, Iv, I, 4, 1); \
    FLOAT_SOFTSIGN_SIMD_C(c_4_2, Iv, I, 4, 2); \
    FLOAT_SOFTSIGN_SIMD_C(c_4_3, Iv, I, 4, 3); \
    /**/                                       \
    FLOAT_SOFTSIGN_SIMD_C(c_5_0, Iv, I, 5, 0); \
    FLOAT_SOFTSIGN_SIMD_C(c_5_1, Iv, I, 5, 1); \
    FLOAT_SOFTSIGN_SIMD_C(c_5_2, Iv, I, 5, 2); \
    FLOAT_SOFTSIGN_SIMD_C(c_5_3, Iv, I, 5, 3);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_SOFTSIGN_END_C(step, I, c_cur, W_ob, C_ob)                \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t const *I_pixel = I;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *I_channel = I_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = *(I_channel) / (1.0f + std::abs(*I_channel)); \
            c_channel++;                                                \
            I_channel++;                                                \
        }                                                               \
        I_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#else

#define FLOAT_SOFTSIGN_END_C(step, I, c_cur, W_ob, C_ob)                \
    float32x4_t Iv;                                                     \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            float32x4_t cv = c_cur[kk * (C_ob / FLOAT_SIMD) + jj];      \
            FLOAT_SOFTSIGN_SIMD_C(cv, Iv, I, kk, jj);                   \
            c_cur[kk * (C_ob / FLOAT_SIMD) + jj] = cv;                  \
        }                                                               \
    }
#endif

#define FLOAT_FUSED_SOFTSIGN_SIMD_C(c_x_x, Iv)    \
    Iv = c_x_x;                                   \
    c_x_x = vabsq_f32(Iv);                        \
    c_x_x = vaddq_f32(c_x_x, vdupq_n_f32(1.0f));  \
    c_x_x = vdivq_f32(Iv, c_x_x);

#define FLOAT_FUSED_SOFTSIGN_TILE_C               \
    float32x4_t Iv;                               \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_0_0, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_0_1, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_0_2, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_0_3, Iv);       \
    /**/                                          \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_1_0, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_1_1, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_1_2, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_1_3, Iv);       \
    /**/                                          \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_2_0, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_2_1, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_2_2, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_2_3, Iv);       \
    /**/                                          \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_3_0, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_3_1, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_3_2, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_3_3, Iv);       \
    /**/                                          \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_4_0, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_4_1, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_4_2, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_4_3, Iv);       \
    /**/                                          \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_5_0, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_5_1, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_5_2, Iv);       \
    FLOAT_FUSED_SOFTSIGN_SIMD_C(c_5_3, Iv);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_SOFTSIGN_END_C(c_cur, W_ob, C_ob)      \
    c_tile_t *c_pixel = c_cur;                       \
    for (uint32_t kk = 0; kk < W_ob; kk++)           \
    {                                                \
        c_tile_t *c_channel = c_pixel;               \
        for (uint32_t jj = 0; jj < C_ob; jj++)       \
        {                                            \
            *(c_channel) = *(c_channel) / (1.0f + std::abs(*c_channel)); \
            c_channel++;                             \
        }                                            \
        c_pixel += C_ob;                             \
    }

#else

#define FLOAT_FUSED_SOFTSIGN_END_C(c_cur, W_ob, C_ob)                 \
    float32x4_t Iv;                                                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)                            \
    {                                                                 \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)           \
        {                                                             \
            float32x4_t cv = c_cur[kk * (C_ob / FLOAT_SIMD) + jj];    \
            FLOAT_FUSED_SOFTSIGN_SIMD_C(cv, Iv, a, kk, jj);           \
            c_cur[kk * (C_ob / FLOAT_SIMD) + jj] = cv;                \
        }                                                             \
    }
#endif


//****************************************************************************
// Ewise abs
//****************************************************************************

#define FLOAT_ABS_TILE_C(step, I)                       \
    float32x4_t Iv;                                     \
    Iv = vld1q_f32(I + 0 * step + 0 * FLOAT_SIMD);      \
    c_0_0 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 0 * step + 1 * FLOAT_SIMD);      \
    c_0_1 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 0 * step + 2 * FLOAT_SIMD);      \
    c_0_2 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 0 * step + 3 * FLOAT_SIMD);      \
    c_0_3 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 1 * step + 0 * FLOAT_SIMD);      \
    c_1_0 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 1 * step + 1 * FLOAT_SIMD);      \
    c_1_1 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 1 * step + 2 * FLOAT_SIMD);      \
    c_1_2 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 1 * step + 3 * FLOAT_SIMD);      \
    c_1_3 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 2 * step + 0 * FLOAT_SIMD);      \
    c_2_0 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 2 * step + 1 * FLOAT_SIMD);      \
    c_2_1 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 2 * step + 2 * FLOAT_SIMD);      \
    c_2_2 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 2 * step + 3 * FLOAT_SIMD);      \
    c_2_3 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 3 * step + 0 * FLOAT_SIMD);      \
    c_3_0 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 3 * step + 1 * FLOAT_SIMD);      \
    c_3_1 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 3 * step + 2 * FLOAT_SIMD);      \
    c_3_2 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 3 * step + 3 * FLOAT_SIMD);      \
    c_3_3 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 4 * step + 0 * FLOAT_SIMD);      \
    c_4_0 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 4 * step + 1 * FLOAT_SIMD);      \
    c_4_1 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 4 * step + 2 * FLOAT_SIMD);      \
    c_4_2 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 4 * step + 3 * FLOAT_SIMD);      \
    c_4_3 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 5 * step + 0 * FLOAT_SIMD);      \
    c_5_0 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 5 * step + 1 * FLOAT_SIMD);      \
    c_5_1 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 5 * step + 2 * FLOAT_SIMD);      \
    c_5_2 = vabsq_f32(Iv);                              \
    Iv = vld1q_f32(I + 5 * step + 3 * FLOAT_SIMD);      \
    c_5_3 = vabsq_f32(Iv);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_ABS_END_C(step, I, c_cur, W_ob, C_ob)     \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_cur[kk * C_ob + jj] = std::abs(I[kk * step + jj]); \
        }                                               \
    }

#else

#define FLOAT_ABS_END_C(step, I, c_cur, W_ob, C_ob)                     \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)       \
        {                                                               \
            float32x4_t Iv = vld1q_f32(I + kk * step + jj * FLOAT_SIMD); \
            c_cur[(kk) * (FLOAT_C_ob / FLOAT_SIMD) + jj] = vabsq_f32(Iv);\
        }                                                               \
    }
#endif


//****************************************************************************
// Ewise div
//****************************************************************************

#define FLOAT_FUSED_DIV_TILE_C(step, I)                 \
    float32x4_t Iv;                                     \
    Iv = vld1q_f32(I + 0 * step + 0 * FLOAT_SIMD);      \
    c_0_0 = vdivq_f32(Iv, c_0_0);                       \
    Iv = vld1q_f32(I + 0 * step + 1 * FLOAT_SIMD);      \
    c_0_1 = vdivq_f32(Iv, c_0_1);                       \
    Iv = vld1q_f32(I + 0 * step + 2 * FLOAT_SIMD);      \
    c_0_2 = vdivq_f32(Iv, c_0_2);                       \
    Iv = vld1q_f32(I + 0 * step + 3 * FLOAT_SIMD);      \
    c_0_3 = vdivq_f32(Iv, c_0_3);                       \
    Iv = vld1q_f32(I + 1 * step + 0 * FLOAT_SIMD);      \
    c_1_0 = vdivq_f32(Iv, c_1_0);                       \
    Iv = vld1q_f32(I + 1 * step + 1 * FLOAT_SIMD);      \
    c_1_1 = vdivq_f32(Iv, c_1_1);                       \
    Iv = vld1q_f32(I + 1 * step + 2 * FLOAT_SIMD);      \
    c_1_2 = vdivq_f32(Iv, c_1_2);                       \
    Iv = vld1q_f32(I + 1 * step + 3 * FLOAT_SIMD);      \
    c_1_3 = vdivq_f32(Iv, c_1_3);                       \
    Iv = vld1q_f32(I + 2 * step + 0 * FLOAT_SIMD);      \
    c_2_0 = vdivq_f32(Iv, c_2_0);                       \
    Iv = vld1q_f32(I + 2 * step + 1 * FLOAT_SIMD);      \
    c_2_1 = vdivq_f32(Iv, c_2_1);                       \
    Iv = vld1q_f32(I + 2 * step + 2 * FLOAT_SIMD);      \
    c_2_2 = vdivq_f32(Iv, c_2_2);                       \
    Iv = vld1q_f32(I + 2 * step + 3 * FLOAT_SIMD);      \
    c_2_3 = vdivq_f32(Iv, c_2_3);                       \
    Iv = vld1q_f32(I + 3 * step + 0 * FLOAT_SIMD);      \
    c_3_0 = vdivq_f32(Iv, c_3_0);                       \
    Iv = vld1q_f32(I + 3 * step + 1 * FLOAT_SIMD);      \
    c_3_1 = vdivq_f32(Iv, c_3_1);                       \
    Iv = vld1q_f32(I + 3 * step + 2 * FLOAT_SIMD);      \
    c_3_2 = vdivq_f32(Iv, c_3_2);                       \
    Iv = vld1q_f32(I + 3 * step + 3 * FLOAT_SIMD);      \
    c_3_3 = vdivq_f32(Iv, c_3_3);                       \
    Iv = vld1q_f32(I + 4 * step + 0 * FLOAT_SIMD);      \
    c_4_0 = vdivq_f32(Iv, c_4_0);                       \
    Iv = vld1q_f32(I + 4 * step + 1 * FLOAT_SIMD);      \
    c_4_1 = vdivq_f32(Iv, c_4_1);                       \
    Iv = vld1q_f32(I + 4 * step + 2 * FLOAT_SIMD);      \
    c_4_2 = vdivq_f32(Iv, c_4_2);                       \
    Iv = vld1q_f32(I + 4 * step + 3 * FLOAT_SIMD);      \
    c_4_3 = vdivq_f32(Iv, c_4_3);                       \
    Iv = vld1q_f32(I + 5 * step + 0 * FLOAT_SIMD);      \
    c_5_0 = vdivq_f32(Iv, c_5_0);                       \
    Iv = vld1q_f32(I + 5 * step + 1 * FLOAT_SIMD);      \
    c_5_1 = vdivq_f32(Iv, c_5_1);                       \
    Iv = vld1q_f32(I + 5 * step + 2 * FLOAT_SIMD);      \
    c_5_2 = vdivq_f32(Iv, c_5_2);                       \
    Iv = vld1q_f32(I + 5 * step + 3 * FLOAT_SIMD);      \
    c_5_3 = vdivq_f32(Iv, c_5_3);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_FUSED_DIV_END_C(step, I, c_cur, W_ob, C_ob)  \
    for (uint32_t kk = 0; kk < W_ob; kk++)                 \
    {                                                      \
        for (uint32_t jj = 0; jj < C_ob; jj++)             \
        {                                                  \
            c_cur[kk * C_ob + jj] = I[kk * step + jj] / c; \
        }                                                  \
    }

#else

#define FLOAT_FUSED_DIV_END_C(step, I, c_cur, W_ob, C_ob)               \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)       \
        {                                                               \
            float32x4_t Iv = vld1q_f32(I + kk * step + jj * FLOAT_SIMD); \
            c_cur[(kk) * (FLOAT_C_ob / FLOAT_SIMD) + jj] =              \
                vdivq_f32(Iv, c_cur[(kk) * (FLOAT_C_ob / FLOAT_SIMD) + jj]); \
        }                                                               \
    }
#endif

//****************************************************************************
// OTHER MISC CODE.
//****************************************************************************

// TODO: is this tested?

#define FLOAT_ADD_TILE_C_G(I, W_ob_g, C_ob)                     \
    for (uint32_t mm = 0; mm < W_ob_g; mm++)                    \
    {                                                           \
        for (uint32_t kk = 0; kk < C_ob; kk++)                  \
        {                                                       \
            c_tile[mm * C_ob + kk] += I[mm * C_ob + kk];        \
        }                                                       \
    }



#define FLOAT_ADD_LAST_C_G(I, W_ob, C_ob)     \
    float *i_pixel = I;                         \
    float *c_pixel = c_tile;                    \
    for (uint32_t mm = 0; mm < W_ob; mm++)    \
    {                                           \
        float *c_channel = c_pixel;             \
        float *i_channel = i_pixel;             \
        for (uint32_t kk = 0; kk < C_ob; kk++)  \
        {                                       \
            *c_channel += *i_channel;           \
            c_channel++;                        \
            i_channel++;                        \
        }                                       \
        c_pixel += C_ob;                        \
        i_pixel += C_ob;                        \
    }

#define FLOAT_REDUCE_div_C(O, d, W_ob_g, C_ob)          \
    {                                                   \
        float *c_pixel = c_tile;                        \
        float *O_channel = O;                           \
        float *c_channel = c_pixel;                     \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)        \
        {                                               \
            float *O_channel = O;                       \
            float *c_channel = c_pixel;                 \
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

#define FLOAT_REDUCE_C(O, W_ob_g, C_ob)                 \
    {                                                   \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)        \
        {                                               \
            for (uint32_t kk = 0; kk < C_ob; kk++)      \
            {                                           \
                O[kk] += c_tile[mm * C_ob + kk];        \
            }                                           \
        }                                               \
    }

#define FLOAT_REDUCE_C_last(O, W_ob, C_ob)            \
    {                                                   \
        float *c_pixel = c_tile;                        \
        float *O_channel = O;                           \
        float *c_channel = c_pixel;                     \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)        \
        {                                               \
            float *O_channel = O;                       \
            float *c_channel = c_pixel;                 \
            for (uint32_t kk = 0; kk < C_ob; kk++)      \
            {                                           \
                *O_channel += *c_channel;               \
                O_channel++;                            \
                c_channel++;                            \
            }                                           \
            c_pixel += C_ob;                            \
        }                                               \
    }




#include "intrinsics-gen.h"
