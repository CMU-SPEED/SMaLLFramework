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

#include <arm_neon.h>
#include <params.h>
#include <Buffer.hpp>
#include "arm_mathfun.h"

// Scalar versions of all the float microkernels for platform portability
// Use the FLOAT_ prefix for all macros in this file.

/// @todo Replace this with FLOAT_SIMD? They are the same value.
///       The code blocks that are defined when this value
///       does not equal 1 actually use FLOAT_SIMD's value.

// Epilogue parameters
#define FLOAT_SIMD_EPILOGUE 4

// Define SIMD width for fp16 and float32
#define FLOAT_SIMD_FP16 8  // for float16x8_t
#define FLOAT_SIMD 4       // for float32x4_t

// Unroll parameter for convolution
//#define _UNROLL 4

namespace small {
namespace detail {

typedef small::FloatBuffer::value_type dtype;

#if FLOAT_SIMD_EPILOGUE == 1
typedef small::FloatBuffer::value_type c_tile_t;
#else
typedef float32x4_t c_tile_t;
#endif

}
}

//****************************************************************************
// Architecture-specific tiling params for float16 and float32
//****************************************************************************

// FP16 macros for loading, storing, and computing
typedef float16x8_t c_tile_t_fp16;

#define FLOAT_DEF_TILE_C_FP16(W_ob, C_ob) \
    float16x8_t c_tile_v[W_ob * (C_ob / FLOAT_SIMD_FP16)];

#define FLOAT_ZERO_TILE_C_FP16(W_ob, C_ob) \
    for (uint32_t kk = 0; kk < W_ob; kk++) { \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) { \
            c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj] = vdupq_n_f16(0); \
        } \
    }

#define FLOAT_LOAD_TILE_C_FP16(O, W_ob, C_ob) \
    for (uint32_t kk = 0; kk < W_ob; kk++) { \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) { \
            c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj] = vld1q_f16(O + kk * C_ob + jj * FLOAT_SIMD_FP16); \
        } \
    }

#define FLOAT_STORE_TILE_C_FP16(O, W_ob, C_ob) \
    for (uint32_t kk = 0; kk < W_ob; kk++) { \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) { \
            vst1q_f16(O + kk * C_ob + jj * FLOAT_SIMD_FP16, c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj]); \
        } \
    }

#define FLOAT_CONV_TILE_C_FP16(step, a, b, W_ob, C_ob) \
    float16x8_t bv[C_ob / FLOAT_SIMD_FP16]; \
    for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) { \
        bv[jj] = vld1q_f16(b + jj * FLOAT_SIMD_FP16); \
    } \
    for (uint32_t kk = 0; kk < W_ob; kk++) { \
        float16x8_t av = vdupq_n_f16(*(a + kk * step)); \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) { \
            c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj] = vfmaq_f16(c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj], av, bv[jj]); \
        } \
    }

// FP16 Depthwise Convolution Tile computation
#define FLOAT_DW_TILE_C_FP16(step, a, b, W_ob, C_ob)                   \
    float16x8_t bv[C_ob / FLOAT_SIMD_FP16];                            \
    for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)           \
    {                                                                   \
        bv[jj] = vld1q_f16(b + jj * FLOAT_SIMD_FP16);                  \
    }                                                                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)                             \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)       \
        {                                                               \
            float16x8_t av = vld1q_f16(a + kk * step + jj * FLOAT_SIMD_FP16); \
            c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj] =             \
                vfmaq_f16(c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj], av, bv[jj]); \
        }                                                               \
    }

// FP16 Depthwise Convolution End computation 
#define FLOAT_DW_END_C_FP16(step, a, b, c_cur, _W_ob, C_ob)           \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                           \
    {                                                                  \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)      \
        {                                                              \
            float16x8_t av = vld1q_f16(a + kk * step + jj * FLOAT_SIMD_FP16); \
            float16x8_t bv = vld1q_f16(b + jj * FLOAT_SIMD_FP16);     \
            c_cur[(kk) * (C_ob / FLOAT_SIMD_FP16) + jj] =             \
                vfmaq_f16(c_cur[(kk) * (C_ob / FLOAT_SIMD_FP16) + jj], av, bv); \
        }                                                              \
    }

//****************************************************************************
// Convolution computation for FP16 and FP32 with unrolling
//****************************************************************************
#define FLOAT_CONV_END_C_FP16(step, a, b, c_cur, W_ob, C_ob) \
    float16x8_t bv[C_ob / FLOAT_SIMD_FP16]; \
    for (uint32_t ii_unroll = 0; ii_unroll < _UNROLL; ii_unroll++) { \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) { \
            bv[jj] = vld1q_f16(b + (ii_unroll * C_ob) + jj * FLOAT_SIMD_FP16); \
        } \
        for (uint32_t kk = 0; kk < W_ob; kk++) { \
            float16x8_t av = vdupq_n_f16(*(a + (kk * step + ii_unroll))); \
            for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) { \
                c_cur[kk * (C_ob / FLOAT_SIMD_FP16) + jj] = vfmaq_f16(c_cur[kk * (C_ob / FLOAT_SIMD_FP16) + jj], av, bv[jj]); \
            } \
        } \
    }

#define FLOAT_CONV_END_C_FP32(step, a, b, c_cur, W_ob, C_ob) \
    float32x4_t bv[C_ob / FLOAT_SIMD]; \
    for (uint32_t ii_unroll = 0; ii_unroll < _UNROLL; ii_unroll++) { \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++) { \
            bv[jj] = vld1q_f32(b + (ii_unroll * C_ob) + jj * FLOAT_SIMD); \
        } \
        for (uint32_t kk = 0; kk < W_ob; kk++) { \
            float32x4_t av = vdupq_n_f32(*(a + (kk * step + ii_unroll))); \
            for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++) { \
                c_cur[kk * (C_ob / FLOAT_SIMD) + jj] = vfmaq_f32(c_cur[kk * (C_ob / FLOAT_SIMD) + jj], av, bv[jj]); \
            } \
        } \
    }

#define FLOAT16_CONV_END_C(step, a, b, c_cur, _W_ob, C_ob)              \
    float16x8_t bv[C_ob / FLOAT_SIMD_FP16];                             \
    float16x8_t av;                                                     \
    for (uint32_t ii_unroll = 0; ii_unroll < _UNROLL; ii_unroll++)      \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)        \
        {                                                               \
            bv[jj] = vld1q_f16(b + (ii_unroll * C_ob) + jj * FLOAT_SIMD_FP16); \
        }                                                               \
        switch (_W_ob)                                                  \
        {                                                               \
        case 5:                                                         \
            av = vdupq_n_f16(a + ((4 * step) + ii_unroll));             \
            c_cur[(4 * (C_ob / FLOAT_SIMD_FP16)) + 0] =                 \
                vfmaq_f16(c_cur[(4 * (C_ob / FLOAT_SIMD_FP16)) + 0], av, bv[0]); \
            c_cur[(4 * (C_ob / FLOAT_SIMD_FP16)) + 1] =                 \
                vfmaq_f16(c_cur[(4 * (C_ob / FLOAT_SIMD_FP16)) + 1], av, bv[1]); \
            c_cur[(4 * (C_ob / FLOAT_SIMD_FP16)) + 2] =                 \
                vfmaq_f16(c_cur[(4 * (C_ob / FLOAT_SIMD_FP16)) + 2], av, bv[2]); \
            c_cur[(4 * (C_ob / FLOAT_SIMD_FP16)) + 3] =                 \
                vfmaq_f16(c_cur[(4 * (C_ob / FLOAT_SIMD_FP16)) + 3], av, bv[3]); \
        case 4:                                                         \
            av = vdupq_n_f16(a + ((3 * step) + ii_unroll));             \
            c_cur[(3 * (C_ob / FLOAT_SIMD_FP16)) + 0] =                 \
                vfmaq_f16(c_cur[(3 * (C_ob / FLOAT_SIMD_FP16)) + 0], av, bv[0]); \
            c_cur[(3 * (C_ob / FLOAT_SIMD_FP16)) + 1] =                 \
                vfmaq_f16(c_cur[(3 * (C_ob / FLOAT_SIMD_FP16)) + 1], av, bv[1]); \
            c_cur[(3 * (C_ob / FLOAT_SIMD_FP16)) + 2] =                 \
                vfmaq_f16(c_cur[(3 * (C_ob / FLOAT_SIMD_FP16)) + 2], av, bv[2]); \
            c_cur[(3 * (C_ob / FLOAT_SIMD_FP16)) + 3] =                 \
                vfmaq_f16(c_cur[(3 * (C_ob / FLOAT_SIMD_FP16)) + 3], av, bv[3]); \
        case 3:                                                         \
            av = vdupq_n_f16(a + ((2 * step) + ii_unroll));             \
            c_cur[(2 * (C_ob / FLOAT_SIMD_FP16)) + 0] =                 \
                vfmaq_f16(c_cur[(2 * (C_ob / FLOAT_SIMD_FP16)) + 0], av, bv[0]); \
            c_cur[(2 * (C_ob / FLOAT_SIMD_FP16)) + 1] =                 \
                vfmaq_f16(c_cur[(2 * (C_ob / FLOAT_SIMD_FP16)) + 1], av, bv[1]); \
            c_cur[(2 * (C_ob / FLOAT_SIMD_FP16)) + 2] =                 \
                vfmaq_f16(c_cur[(2 * (C_ob / FLOAT_SIMD_FP16)) + 2], av, bv[2]); \
            c_cur[(2 * (C_ob / FLOAT_SIMD_FP16)) + 3] =                 \
                vfmaq_f16(c_cur[(2 * (C_ob / FLOAT_SIMD_FP16)) + 3], av, bv[3]); \
        case 2:                                                         \
            av = vdupq_n_f16(a + ((1 * step) + ii_unroll));             \
            c_cur[(1 * (C_ob / FLOAT_SIMD_FP16)) + 0] =                 \
                vfmaq_f16(c_cur[(1 * (C_ob / FLOAT_SIMD_FP16)) + 0], av, bv[0]); \
            c_cur[(1 * (C_ob / FLOAT_SIMD_FP16)) + 1] =                 \
                vfmaq_f16(c_cur[(1 * (C_ob / FLOAT_SIMD_FP16)) + 1], av, bv[1]); \
            c_cur[(1 * (C_ob / FLOAT_SIMD_FP16)) + 2] =                 \
                vfmaq_f16(c_cur[(1 * (C_ob / FLOAT_SIMD_FP16)) + 2], av, bv[2]); \
            c_cur[(1 * (C_ob / FLOAT_SIMD_FP16)) + 3] =                 \
                vfmaq_f16(c_cur[(1 * (C_ob / FLOAT_SIMD_FP16)) + 3], av, bv[3]); \
        case 1:                                                         \
            av = vdupq_n_f16(a + ((0 * step) + ii_unroll));             \
            c_cur[(0 * (C_ob / FLOAT_SIMD_FP16)) + 0] =                 \
                vfmaq_f16(c_cur[(0 * (C_ob / FLOAT_SIMD_FP16)) + 0], av, bv[0]); \
            c_cur[(0 * (C_ob / FLOAT_SIMD_FP16)) + 1] =                 \
                vfmaq_f16(c_cur[(0 * (C_ob / FLOAT_SIMD_FP16)) + 1], av, bv[1]); \
            c_cur[(0 * (C_ob / FLOAT_SIMD_FP16)) + 2] =                 \
                vfmaq_f16(c_cur[(0 * (C_ob / FLOAT_SIMD_FP16)) + 2], av, bv[2]); \
            c_cur[(0 * (C_ob / FLOAT_SIMD_FP16)) + 3] =                 \
                vfmaq_f16(c_cur[(0 * (C_ob / FLOAT_SIMD_FP16)) + 3], av, bv[3]); \
        }                                                               \
    }

#define FLOAT_ZERO_END_C_FP16(_W_ob, C_ob)                             \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                           \
    {                                                                  \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)      \
        {                                                              \
            c_tile[kk * (C_ob / FLOAT_SIMD_FP16) + jj] = vdupq_n_f16(0); \
        }                                                              \
    }

#define FLOAT_LOAD_END_C_FP16(O, _W_ob, C_ob)                         \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                           \
    {                                                                  \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)      \
        {                                                              \
            c_tile[kk * (C_ob / FLOAT_SIMD_FP16) + jj] =              \
                vld1q_f16(O + kk * C_ob + jj * FLOAT_SIMD_FP16);      \
        }                                                              \
    }

#define FLOAT_STORE_END_C_FP16(O, _W_ob, C_ob)                        \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                           \
    {                                                                  \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)      \
        {                                                              \
            vst1q_f16(O + kk * C_ob + jj * FLOAT_SIMD_FP16,           \
                      c_tile[kk * (C_ob / FLOAT_SIMD_FP16) + jj]);    \
        }                                                              \
    }

#define FLOAT_LOAD_TILE_C_strided_FP16(O, step, W_ob, C_ob)                  \
    for (uint32_t kk = 0; kk < W_ob; kk++)                                   \
    {                                                                        \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)            \
        {                                                                    \
            c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj] =                  \
                vld1q_f16(O + kk * step + jj * FLOAT_SIMD_FP16);            \
        }                                                                    \
    }

#define FLOAT_LOAD_TILE_C_upsample_FP16(O, stride, _C_ib, W_ob, C_ob)       \
    c_0_0 = vld1q_f16(O + (0/stride) * C_ob + 0 * FLOAT_SIMD_FP16);        \
    c_0_1 = vld1q_f16(O + (0/stride) * C_ob + 1 * FLOAT_SIMD_FP16);        \
    c_1_0 = vld1q_f16(O + (1/stride) * C_ob + 0 * FLOAT_SIMD_FP16);        \
    c_1_1 = vld1q_f16(O + (1/stride) * C_ob + 1 * FLOAT_SIMD_FP16);        \
    c_2_0 = vld1q_f16(O + (2/stride) * C_ob + 0 * FLOAT_SIMD_FP16);        \
    c_2_1 = vld1q_f16(O + (2/stride) * C_ob + 1 * FLOAT_SIMD_FP16);        \
    c_3_0 = vld1q_f16(O + (3/stride) * C_ob + 0 * FLOAT_SIMD_FP16);        \
    c_3_1 = vld1q_f16(O + (3/stride) * C_ob + 1 * FLOAT_SIMD_FP16);
    
#define FLOAT_LOAD_END_C_strided_FP16(O, step, _W_ob, C_ob)           \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                           \
    {                                                                  \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)      \
        {                                                              \
            c_tile[kk * (C_ob / FLOAT_SIMD_FP16) + jj] =              \
                vld1q_f16(O + kk * step + jj * FLOAT_SIMD_FP16);      \
        }                                                              \
    }

#define FLOAT_LOAD_END_C_upsample_FP16(O, stride, _C_ib, _W_ob, C_ob) \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                           \
    {                                                                  \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)      \
        {                                                              \
            c_tile[kk * (C_ob / FLOAT_SIMD_FP16) + jj] =              \
                vld1q_f16(O + (kk / stride) * (_C_ib) + jj * FLOAT_SIMD_FP16); \
        }                                                              \
    }

#define FLOAT_MAX_TILE_C_FP16(step, a, W_ob, C_ob)                           \
    for (uint32_t kk = 0; kk < W_ob; kk++)                                   \
    {                                                                        \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)            \
        {                                                                    \
            float16x8_t av = vld1q_f16(a + kk * step + jj * FLOAT_SIMD_FP16); \
            c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj] =                   \
                vmaxq_f16(av, c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj]); \
        }                                                                    \
    }

#define FLOAT_MAX_END_C_FP16(step, a, c_cur, W_last, C_ob)                   \
    for (uint32_t kk = 0; kk < W_last; kk++)                                 \
    {                                                                        \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)            \
        {                                                                    \
            float16x8_t av = vld1q_f16(a + kk * step + jj * FLOAT_SIMD_FP16); \
            c_cur[(kk) * (C_ob / FLOAT_SIMD_FP16) + jj] =                    \
                vmaxq_f16(av, c_cur[(kk) * (C_ob / FLOAT_SIMD_FP16) + jj]);  \
        }                                                                    \
    }


// Accumulation operations
#define FLOAT_ACCUM_TILE_C_FP16(step, a, W_ob, C_ob)         \
    float16x8_t av;                                     \
    av = vld1q_f16(a + 0 * step + 0 * FLOAT_SIMD_FP16);      \
    c_0_0 = vaddq_f16(c_0_0, av);                       \
    av = vld1q_f16(a + 0 * step + 1 * FLOAT_SIMD_FP16);      \
    c_0_1 = vaddq_f16(c_0_1, av);                       \
    av = vld1q_f16(a + 1 * step + 0 * FLOAT_SIMD_FP16);      \
    c_1_0 = vaddq_f16(c_1_0, av);                       \
    av = vld1q_f16(a + 1 * step + 1 * FLOAT_SIMD_FP16);      \
    c_1_1 = vaddq_f16(c_1_1, av);                       \
    av = vld1q_f16(a + 2 * step + 0 * FLOAT_SIMD_FP16);      \
    c_2_0 = vaddq_f16(c_2_0, av);                       \
    av = vld1q_f16(a + 2 * step + 1 * FLOAT_SIMD_FP16);      \
    c_2_1 = vaddq_f16(c_2_1, av);                       \
    av = vld1q_f16(a + 3 * step + 0 * FLOAT_SIMD_FP16);      \
    c_3_0 = vaddq_f16(c_3_0, av);                       \
    av = vld1q_f16(a + 3 * step + 1 * FLOAT_SIMD_FP16);      \
    c_3_1 = vaddq_f16(c_3_1, av);

#define FLOAT_ACCUM_END_C_FP16(step, a, c_cur, W_last, C_ob)                   \
    for (uint32_t kk = 0; kk < W_last; kk++)                                 \
    {                                                                        \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)            \
        {                                                                    \
            float16x8_t av = vld1q_f16(a + kk * step + jj * FLOAT_SIMD_FP16); \
            c_cur[(kk) * (C_ob / FLOAT_SIMD_FP16) + jj] =                    \
                vaddq_f16(av, c_cur[(kk) * (C_ob / FLOAT_SIMD_FP16) + jj]);  \
        }                                                                    \
    }

// Addition operations for groups
#define FLOAT_ADD_TILE_C_G_FP16(I, W_ob_g, C_ob)                     \
    for (uint32_t mm = 0; mm < W_ob_g; mm++)                         \
    {                                                                \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)     \
        {                                                            \
            float16x8_t iv = vld1q_f16(I + mm * C_ob + jj * FLOAT_SIMD_FP16); \
            c_tile_v[mm * (C_ob / FLOAT_SIMD_FP16) + jj] =           \
                vaddq_f16(c_tile_v[mm * (C_ob / FLOAT_SIMD_FP16) + jj], iv); \
        }                                                            \
    }

#define FLOAT_ADD_LAST_C_G_FP16(I, W_last, C_ob)     \
    for (uint32_t mm = 0; mm < W_last; mm++)         \
    {                                                \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) \
        {                                            \
            float16x8_t iv = vld1q_f16(I + mm * C_ob + jj * FLOAT_SIMD_FP16); \
            c_tile[mm * (C_ob / FLOAT_SIMD_FP16) + jj] = \
                vaddq_f16(c_tile[mm * (C_ob / FLOAT_SIMD_FP16) + jj], iv); \
        }                                            \
    }

// Conditional scale operations
#define FLOAT_COND_SCALE_TILE_C_FP16(step, a, b, W_ob, C_ob)                   \
    float16x8_t bv = vdupq_n_f16(*b);                                         \
    float16x8_t av;                                                           \
    uint16x8_t mask;                                                          \
    for (uint32_t kk = 0; kk < W_ob; kk++)                                    \
    {                                                                         \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)             \
        {                                                                     \
            av = vld1q_f16(a + kk * step + jj * FLOAT_SIMD_FP16);            \
            float16x8_t cv = c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj];    \
            cv = vmaxq_f16(av, cv);                                           \
            mask = vcltq_f16(av, cv);                                         \
            av = vmulq_f16(av, bv);                                          \
            av = vbslq_f16(mask, av, cv);                                     \
            c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj] = av;               \
        }                                                                     \
    }

#define FLOAT_COND_SCALE_END_C_FP16(step, a, b, c_cur, W_last, C_ob)          \
    float16x8_t bv = vdupq_n_f16(*b);                                         \
    float16x8_t av;                                                           \
    uint16x8_t mask;                                                          \
    for (uint32_t kk = 0; kk < W_last; kk++)                                  \
    {                                                                         \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)             \
        {                                                                     \
            av = vld1q_f16(a + kk * step + jj * FLOAT_SIMD_FP16);            \
            float16x8_t cv = c_cur[kk * (C_ob / FLOAT_SIMD_FP16) + jj];       \
            cv = vmaxq_f16(av, cv);                                           \
            mask = vcltq_f16(av, cv);                                         \
            av = vmulq_f16(av, bv);                                          \
            av = vbslq_f16(mask, av, cv);                                     \
            c_cur[kk * (C_ob / FLOAT_SIMD_FP16) + jj] = av;                  \
        }                                                                     \
    }

// Division operations
#define FLOAT_DIV_TILE_C_FP16(norm, W_ob, C_ob)        \
    float16x8_t av = vdupq_n_f16(norm);                \
    for (uint32_t kk = 0; kk < W_ob; kk++)             \
    {                                                  \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) \
        {                                              \
            c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj] = \
                vmulq_f16(c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj], av); \
        }                                              \
    }

#define FLOAT_DIV_END_C_FP16(c_cur, norm, W_last, C_ob)  \
    float16x8_t av = vdupq_n_f16(norm);                 \
    for (uint32_t kk = 0; kk < W_last; kk++)            \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) \
        {                                               \
            c_cur[kk * (C_ob / FLOAT_SIMD_FP16) + jj] = \
                vmulq_f16(c_cur[kk * (C_ob / FLOAT_SIMD_FP16) + jj], av); \
        }                                               \
    }

// Reduction operations
#define FLOAT_REDUCE_C_FP16(O, W_ob_g, C_ob)                 \
    for (uint32_t mm = 0; mm < W_ob_g; mm++)                \
    {                                                       \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) \
        {                                                   \
            float16x8_t acc = vld1q_f16(O + jj * FLOAT_SIMD_FP16); \
            acc = vaddq_f16(acc, c_tile_v[mm * (C_ob / FLOAT_SIMD_FP16) + jj]); \
            vst1q_f16(O + jj * FLOAT_SIMD_FP16, acc);      \
        }                                                   \
    }

#define FLOAT_REDUCE_C_last_FP16(O, W_last, C_ob)           \
    for (uint32_t mm = 0; mm < W_last; mm++)               \
    {                                                      \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) \
        {                                                  \
            float16x8_t acc = vld1q_f16(O + jj * FLOAT_SIMD_FP16); \
            acc = vaddq_f16(acc, c_tile[mm * (C_ob / FLOAT_SIMD_FP16) + jj]); \
            vst1q_f16(O + jj * FLOAT_SIMD_FP16, acc);     \
        }                                                  \
    }

#define FLOAT_REDUCE_div_C_FP16(O, d, W_ob_g, C_ob)         \
    {                                                       \
        FLOAT_REDUCE_C_FP16(O, W_ob_g, C_ob);              \
        float16x8_t dv = vdupq_n_f16(d);                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) \
        {                                                   \
            float16x8_t acc = vld1q_f16(O + jj * FLOAT_SIMD_FP16); \
            acc = vmulq_f16(acc, dv);                       \
            vst1q_f16(O + jj * FLOAT_SIMD_FP16, acc);      \
        }                                                   \
    }

#define FLOAT_REDUCE_CHANNEL_END_C_FP16(O_w_left, _C_ob)                      \
    if constexpr(_C_ob == 1 && _C_ob != FLOAT_SIMD_EPILOGUE)                 \
    {                                                                         \
        float16_t c_tile_array[FLOAT_SIMD_FP16];                             \
        for (uint32_t kk = 0; kk < O_w_left; kk++)                           \
        {                                                                     \
            float16x8_t *c_channel_v = c_tile + kk * (_C_ob / FLOAT_SIMD_FP16); \
            float16x8_t sum = c_channel_v[0];                                \
            for (uint32_t jj = 1; jj < _C_ob / FLOAT_SIMD_FP16; jj++)       \
            {                                                                \
                sum = vaddq_f16(sum, c_channel_v[jj]);                      \
            }                                                                \
            vst1q_f16(c_tile_array, sum);                                   \
            float16_t total = 0;                                            \
            for (uint32_t i = 0; i < FLOAT_SIMD_FP16; i++)                 \
            {                                                               \
                total += c_tile_array[i];                                   \
            }                                                               \
            c_tile_array[0] = total;                                        \
            for (uint32_t i = 1; i < FLOAT_SIMD_FP16; i++)                 \
            {                                                               \
                c_tile_array[i] = 0;                                        \
            }                                                               \
            c_channel_v[0] = vld1q_f16(c_tile_array);                      \
        }                                                                   \
    }


#define FLOAT_LOAD_END_C_strided_FP16(O, step, _W_ob, C_ob)           \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                           \
    {                                                                  \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++)      \
        {                                                              \
            c_tile[kk * (C_ob / FLOAT_SIMD_FP16) + jj] =              \
                vld1q_f16(O + kk * step + jj * FLOAT_SIMD_FP16);      \
        }                                                              \
    }

#define FLOAT_EXP_TILE_C_FP16(step, a, W_ob, C_ob)                  \
    float16_t c_tile_scalar[FLOAT_W_ob * FLOAT_C_ob];               \
    float16_t *c_pixel = c_tile_scalar;                             \
    float16_t const *a_pixel = a;                                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)                         \
    {                                                              \
        float16_t *c_channel = c_pixel;                            \
        float16_t const *a_channel = a_pixel;                      \
        for (uint32_t jj = 0; jj < C_ob; jj++)                     \
        {                                                          \
            *(c_channel) = expf16(*a_channel);                     \
            c_channel++;                                           \
            a_channel++;                                           \
        }                                                          \
        a_pixel += step;                                           \
        c_pixel += C_ob;                                           \
    }                                                              \
    c_0_0 = vld1q_f16(c_tile_scalar + 0 * C_ob + 0 * FLOAT_SIMD_FP16);    \
    c_0_1 = vld1q_f16(c_tile_scalar + 0 * C_ob + 1 * FLOAT_SIMD_FP16);    \
    c_1_0 = vld1q_f16(c_tile_scalar + 1 * C_ob + 0 * FLOAT_SIMD_FP16);    \
    c_1_1 = vld1q_f16(c_tile_scalar + 1 * C_ob + 1 * FLOAT_SIMD_FP16);    \
    c_2_0 = vld1q_f16(c_tile_scalar + 2 * C_ob + 0 * FLOAT_SIMD_FP16);    \
    c_2_1 = vld1q_f16(c_tile_scalar + 2 * C_ob + 1 * FLOAT_SIMD_FP16);    \
    c_3_0 = vld1q_f16(c_tile_scalar + 3 * C_ob + 0 * FLOAT_SIMD_FP16);    \
    c_3_1 = vld1q_f16(c_tile_scalar + 3 * C_ob + 1 * FLOAT_SIMD_FP16);

#define FLOAT_FUSED_EXP_END_C_FP16(step, a, c_cur, W_last, C_ob) \
    float16_t c_tile_scalar[FLOAT_W_ob * FLOAT_C_ob];            \
    float16_t *c_pixel = c_tile_scalar;                          \
    float16_t const *a_pixel = a;                                \
    for (uint32_t kk = 0; kk < W_last; kk++)                     \
    {                                                            \
        float16_t *c_channel = c_pixel;                          \
        float16_t const *a_channel = a_pixel;                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)                   \
        {                                                        \
            *(c_channel) = expf16(*a_channel);                   \
            c_channel++;                                         \
            a_channel++;                                         \
        }                                                        \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) \
        {                                                        \
            c_cur[kk * (C_ob/FLOAT_SIMD_FP16) + jj] =           \
                vld1q_f16(c_pixel + jj * FLOAT_SIMD_FP16);      \
        }                                                        \
        a_pixel += step;                                         \
        c_pixel += C_ob;                                         \
    }

#define FLOAT_FUSED_EXP_TILE_C_FP16(step, a, W_ob, C_ob)        \
    float16x8_t zero = vdupq_n_f16(0.0f);                       \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) \
        {                                                       \
            float16x8_t av = vld1q_f16(a + kk * step + jj * FLOAT_SIMD_FP16); \
            av = exp_ps_f16(av);                               \
            c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj] =     \
                vmaxq_f16(av, zero);                           \
        }                                                      \
    }

#define FLOAT_FUSED_RELU_END_C_FP16(step, a, c_cur, W_last, C_ob) \
    float16x8_t zero = vdupq_n_f16(0.0f);                        \
    for (uint32_t kk = 0; kk < W_last; kk++)                     \
    {                                                            \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) \
        {                                                        \
            float16x8_t av = vld1q_f16(a + kk * step + jj * FLOAT_SIMD_FP16); \
            c_cur[kk * (C_ob / FLOAT_SIMD_FP16) + jj] =         \
                vmaxq_f16(av, zero);                            \
        }                                                       \
    }

#define FLOAT_FUSED_RELU_TILE_C_FP16(step, a, W_ob, C_ob)       \
    float16x8_t zero = vdupq_n_f16(0.0f);                       \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD_FP16; jj++) \
        {                                                       \
            float16x8_t av = vld1q_f16(a + kk * step + jj * FLOAT_SIMD_FP16); \
            c_tile_v[kk * (C_ob / FLOAT_SIMD_FP16) + jj] =     \
                vmaxq_f16(av, zero);                           \
        }                                                      \
    }