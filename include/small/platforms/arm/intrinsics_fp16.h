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
