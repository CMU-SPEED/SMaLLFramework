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

// scalar versions of all the float microkernels for platform portability
// Use the FLOAT_ prefix for all macros in this file.

/// @todo Replace this with FLOAT_SIMD? They are the same value.
///       The code blocks that are defined when this value
///       does not equal 1 actually use FLOAT_SIMD's value.
// Epilogue parameters
#define FLOAT_SIMD_EPILOGUE 4

namespace small {
namespace detail {

/// @todo both pairs of typedefs should not be needed.
typedef small::FloatBuffer::value_type  dtype;

/// @todo Any way to move this typedef to Buffer class?
#if FLOAT_SIMD_EPILOGUE == 1
//typedef float c_tile_t;
typedef small::FloatBuffer::value_type  c_tile_t;

#else
typedef float32x4_t c_tile_t;
#endif

}
}

// https://developer.arm.com/architectures/instruction-sets/intrinsics/

// Architecture specific tiling params

//****************************************************************************
// Initializations
//****************************************************************************

// float32x4_t vectorizes C_ob dim: [W_ob, C_ob] -> [W_ob, C_ob/FLOAT_SIMD, FLOAT_SIMD].
// assume FLOAT_SIMD == 4 and vec type is float.
// otherwise, FLOAT_SIMD = Neon bit width (128) / data type size.

#define FLOAT_DEF_TILE_C(W_ob, C_ob)                    \
    float32x4_t c_tile_v[W_ob * (C_ob / FLOAT_SIMD)];

#define FLOAT_DEF_END_C(W_ob, C_ob)                             \
    c_tile_t c_tile[W_ob * (FLOAT_C_ob / FLOAT_SIMD_EPILOGUE)];


#define FLOAT_ZERO_TILE_C(W_ob, C_ob)                                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            c_tile_v[kk * (C_ob / FLOAT_SIMD) + jj] = vdupq_n_f32(0);   \
        }                                                               \
    }


#if FLOAT_SIMD_EPILOGUE==1
#define FLOAT_ZERO_END_C(_W_ob, C_ob)           \
    for (uint32_t kk = 0; kk < _W_ob; kk++)     \
    {                                           \
        for (uint32_t jj = 0; jj < C_ob; jj++)  \
        {                                       \
            c_tile[kk * C_ob + jj] = {};        \
        }                                       \
    }

#else

#define FLOAT_ZERO_END_C(_W_ob, C_ob)                                     \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                               \
    {                                                                     \
        for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)         \
        {                                                                 \
            c_tile[kk * (FLOAT_C_ob / FLOAT_SIMD) + jj] = vdupq_n_f32(0); \
        }                                                                 \
    }
#endif


//****************************************************************************
// Loads
//****************************************************************************

#define FLOAT_LOAD_TILE_C(O, W_ob, C_ob)                                \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            c_tile_v[kk * (C_ob / FLOAT_SIMD) + jj] = vld1q_f32(O + kk * C_ob + jj * FLOAT_SIMD); \
        }                                                               \
    }

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_LOAD_END_C(O, _W_ob, C_ob)                \
    for (uint32_t kk = 0; kk < _W_ob; kk++)             \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
        }                                               \
    }

#else

#define FLOAT_LOAD_END_C(O, _W_ob, C_ob)                                \
if constexpr(C_ob == 1)\
{\
    for(uint32_t kk = 0; kk < _W_ob; kk++)\
    {\
      float c_channel_v[FLOAT_C_ob]={0};\
      c_channel_v[0] = O[kk * C_ob];\
      c_tile[kk * (FLOAT_C_ob/FLOAT_SIMD)]  = vld1q_f32(c_channel_v);\
    }\
}\
else\
{\
    for (uint32_t kk = 0; kk < _W_ob; kk++)                             \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            c_tile[kk * (C_ob / FLOAT_SIMD) + jj] = vld1q_f32(O + kk * C_ob + jj * FLOAT_SIMD); \
        }                                                               \
    }\
}
#endif


//****************************************************************************
// Pooling Loads
//****************************************************************************
// TODO: merge FLOAT_LOAD_TILE_C and FLOAT_LOAD_TILE_C_strided? can use C_ob as step above.

//  strided loads
#define FLOAT_LOAD_TILE_C_strided(O, step, W_ob, C_ob)                  \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            c_tile_v[kk * (C_ob / FLOAT_SIMD) + jj] = vld1q_f32(O + kk * step + jj * FLOAT_SIMD); \
        }                                                               \
    }

#if FLOAT_SIMD_EPILOGUE==1
#define FLOAT_LOAD_END_C_strided(O, step, _W_ob, C_ob)  \
    for (uint32_t kk = 0; kk < _W_ob; kk++)             \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = O[kk * step + jj]; \
        }                                               \
    }

#else

#define FLOAT_LOAD_END_C_strided(O, step, _W_ob, C_ob)                  \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                             \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            c_tile[kk * (C_ob / FLOAT_SIMD) + jj] = vld1q_f32(O + kk * step + jj * FLOAT_SIMD); \
        }                                                               \
    }
#endif


//****************************************************************************
// Upsampling loads (stride < 1)
//****************************************************************************

//@todo build and test this on arm platform

#define FLOAT_LOAD_TILE_C_upsample(O, stride, _C_ib, _W_ob, C_ob)       \
    c_0_0 = vld1q_f32(O + (0/stride) * C_ob + 0 * FLOAT_SIMD);          \
    c_0_1 = vld1q_f32(O + (0/stride) * C_ob + 1 * FLOAT_SIMD);          \
    c_0_2 = vld1q_f32(O + (0/stride) * C_ob + 2 * FLOAT_SIMD);          \
    c_0_3 = vld1q_f32(O + (0/stride) * C_ob + 3 * FLOAT_SIMD);          \
    c_1_0 = vld1q_f32(O + (1/stride) * C_ob + 0 * FLOAT_SIMD);          \
    c_1_1 = vld1q_f32(O + (1/stride) * C_ob + 1 * FLOAT_SIMD);          \
    c_1_2 = vld1q_f32(O + (1/stride) * C_ob + 2 * FLOAT_SIMD);          \
    c_1_3 = vld1q_f32(O + (1/stride) * C_ob + 3 * FLOAT_SIMD);          \
    c_2_0 = vld1q_f32(O + (2/stride) * C_ob + 0 * FLOAT_SIMD);          \
    c_2_1 = vld1q_f32(O + (2/stride) * C_ob + 1 * FLOAT_SIMD);          \
    c_2_2 = vld1q_f32(O + (2/stride) * C_ob + 2 * FLOAT_SIMD);          \
    c_2_3 = vld1q_f32(O + (2/stride) * C_ob + 3 * FLOAT_SIMD);          \
    c_3_0 = vld1q_f32(O + (3/stride) * C_ob + 0 * FLOAT_SIMD);          \
    c_3_1 = vld1q_f32(O + (3/stride) * C_ob + 1 * FLOAT_SIMD);          \
    c_3_2 = vld1q_f32(O + (3/stride) * C_ob + 2 * FLOAT_SIMD);          \
    c_3_3 = vld1q_f32(O + (3/stride) * C_ob + 3 * FLOAT_SIMD);          \
    c_4_0 = vld1q_f32(O + (4/stride) * C_ob + 0 * FLOAT_SIMD);          \
    c_4_1 = vld1q_f32(O + (4/stride) * C_ob + 1 * FLOAT_SIMD);          \
    c_4_2 = vld1q_f32(O + (4/stride) * C_ob + 2 * FLOAT_SIMD);          \
    c_4_3 = vld1q_f32(O + (4/stride) * C_ob + 3 * FLOAT_SIMD);          \
    c_5_0 = vld1q_f32(O + (5/stride) * C_ob + 0 * FLOAT_SIMD);          \
    c_5_1 = vld1q_f32(O + (5/stride) * C_ob + 1 * FLOAT_SIMD);          \
    c_5_2 = vld1q_f32(O + (5/stride) * C_ob + 2 * FLOAT_SIMD);          \
    c_5_3 = vld1q_f32(O + (5/stride) * C_ob + 3 * FLOAT_SIMD);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_LOAD_END_C_upsample(O, stride, _C_ib, _W_ob, C_ob)        \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                             \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            c_tile[kk * C_ob + jj] = O[(kk / stride) * (_C_ib) + jj];   \
        }                                                               \
    }

#else

#define FLOAT_LOAD_END_C_upsample(O, stride, _C_ib, _W_ob, C_ob)        \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                             \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            c_tile[kk * (C_ob / FLOAT_SIMD) + jj] =                     \
                vld1q_f32(O + (kk / stride) *(_C_ib) + jj * FLOAT_SIMD); \
        }                                                               \
    }
#endif


//****************************************************************************
// Stores
//****************************************************************************

#define FLOAT_STORE_TILE_C(O, W_ob, C_ob)                       \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)     \
        {                                                       \
            vst1q_f32(O + kk * C_ob + jj * FLOAT_SIMD,          \
                      c_tile_v[kk * (C_ob / FLOAT_SIMD) + jj]); \
        }                                                       \
    }

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_STORE_END_C(O, _W_ob, C_ob)               \
    for (uint32_t kk = 0; kk < _W_ob; kk++)             \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
        }                                               \
    }

#else

#define FLOAT_STORE_END_C(O, _W_ob, C_ob)                         \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                       \
    {                                                             \
        if constexpr (C_ob == 1)                                  \
        {                                                         \
                float c_pixel[FLOAT_SIMD];                                    \
                vst1q_f32(c_pixel,        \
                          c_tile[kk * (FLOAT_C_ob / FLOAT_SIMD)]); \
                O[kk] = c_pixel[0];\
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
// Convolution Computation
//****************************************************************************

// TODO: add unroll of C_ib dim.
// a: [W_ob, C_ib] -- unroll C_ib in vector register and broadcast from diff lane.
// b: [Hf, Wf, C_ib, C_ob] -- unroll k loop will stride by C_ob (indexes C_ib).
#define FLOAT_CONV_TILE_C(step, a, b, W_ob, C_ob)                       \
    float32x4_t bv[C_ob / FLOAT_SIMD];                                  \
    for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)                 \
    {                                                                   \
        bv[jj] = vld1q_f32(b + jj * FLOAT_SIMD);                        \
    }                                                                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        float32x4_t av = vld1q_dup_f32(a + kk * step);                  \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            c_tile_v[kk * (C_ob / FLOAT_SIMD) + jj] =                   \
                vfmaq_f32(c_tile_v[kk * (C_ob / FLOAT_SIMD) + jj], av, bv[jj]); \
        }                                                               \
    }

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_CONV_END_C(step, a, b, c_cur, _W_ob, C_ob)                \
    for (uint32_t i = 0; i < _UNROLL; i++)                              \
    {                                                                   \
        for (uint32_t kk = 0; kk < _W_ob; kk++)                         \
        {                                                               \
            for (uint32_t jj = 0; jj < C_ob; jj++)                      \
            {                                                           \
                c_cur[kk * C_ob + jj] += a[kk * step + i] * b[i * C_ob + jj]; \
            }                                                           \
        }                                                               \
    }

#else
//todo: make this use the fmla as well
#define FLOAT_CONV_END_C(step, a, b, c_cur, _W_ob, C_ob)                \
    float32x4_t bv[C_ob / FLOAT_SIMD];                                  \
    float32x4_t av;                                                     \
    for (uint32_t ii_unroll = 0; ii_unroll < _UNROLL; ii_unroll++)      \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            bv[jj] = vld1q_f32(b + (ii_unroll * C_ob) + jj * FLOAT_SIMD); \
        }                                                               \
        switch (_W_ob)                                                  \
        {                                                               \
        case 5:                                                         \
            av = vld1q_dup_f32(a + ((4 * step) + ii_unroll));           \
            c_cur[(4 * (C_ob / FLOAT_SIMD)) + 0] =                      \
                vfmaq_f32(c_cur[(4 * (C_ob / FLOAT_SIMD)) + 0], av, bv[0]); \
            c_cur[(4 * (C_ob / FLOAT_SIMD)) + 1] =                      \
                vfmaq_f32(c_cur[(4 * (C_ob / FLOAT_SIMD)) + 1], av, bv[1]); \
            c_cur[(4 * (C_ob / FLOAT_SIMD)) + 2] =                      \
                vfmaq_f32(c_cur[(4 * (C_ob / FLOAT_SIMD)) + 2], av, bv[2]); \
            c_cur[(4 * (C_ob / FLOAT_SIMD)) + 3] =                      \
                vfmaq_f32(c_cur[(4 * (C_ob / FLOAT_SIMD)) + 3], av, bv[3]); \
        case 4:                                                         \
            av = vld1q_dup_f32(a + ((3 * step) + ii_unroll));           \
            c_cur[(3 * (C_ob / FLOAT_SIMD)) + 0] =                      \
                vfmaq_f32(c_cur[(3 * (C_ob / FLOAT_SIMD)) + 0], av, bv[0]); \
            c_cur[(3 * (C_ob / FLOAT_SIMD)) + 1] =                      \
                vfmaq_f32(c_cur[(3 * (C_ob / FLOAT_SIMD)) + 1], av, bv[1]); \
            c_cur[(3 * (C_ob / FLOAT_SIMD)) + 2] =                      \
                vfmaq_f32(c_cur[(3 * (C_ob / FLOAT_SIMD)) + 2], av, bv[2]); \
            c_cur[(3 * (C_ob / FLOAT_SIMD)) + 3] =                      \
                vfmaq_f32(c_cur[(3 * (C_ob / FLOAT_SIMD)) + 3], av, bv[3]); \
        case 3:                                                         \
            av = vld1q_dup_f32(a + ((2 * step) + ii_unroll));           \
            c_cur[(2 * (C_ob / FLOAT_SIMD)) + 0] =                      \
                vfmaq_f32(c_cur[(2 * (C_ob / FLOAT_SIMD)) + 0], av, bv[0]); \
            c_cur[(2 * (C_ob / FLOAT_SIMD)) + 1] =                      \
                vfmaq_f32(c_cur[(2 * (C_ob / FLOAT_SIMD)) + 1], av, bv[1]); \
            c_cur[(2 * (C_ob / FLOAT_SIMD)) + 2] =                      \
                vfmaq_f32(c_cur[(2 * (C_ob / FLOAT_SIMD)) + 2], av, bv[2]); \
            c_cur[(2 * (C_ob / FLOAT_SIMD)) + 3] =                      \
                vfmaq_f32(c_cur[(2 * (C_ob / FLOAT_SIMD)) + 3], av, bv[3]); \
        case 2:                                                         \
            av = vld1q_dup_f32(a + ((1 * step) + ii_unroll));           \
            c_cur[(1 * (C_ob / FLOAT_SIMD)) + 0] =                      \
                vfmaq_f32(c_cur[(1 * (C_ob / FLOAT_SIMD)) + 0], av, bv[0]); \
            c_cur[(1 * (C_ob / FLOAT_SIMD)) + 1] =                      \
                vfmaq_f32(c_cur[(1 * (C_ob / FLOAT_SIMD)) + 1], av, bv[1]); \
            c_cur[(1 * (C_ob / FLOAT_SIMD)) + 2] =                      \
                vfmaq_f32(c_cur[(1 * (C_ob / FLOAT_SIMD)) + 2], av, bv[2]); \
            c_cur[(1 * (C_ob / FLOAT_SIMD)) + 3] =                      \
                vfmaq_f32(c_cur[(1 * (C_ob / FLOAT_SIMD)) + 3], av, bv[3]); \
        case 1:                                                         \
            av = vld1q_dup_f32(a + ((0 * step) + ii_unroll));           \
            c_cur[(0 * (C_ob / FLOAT_SIMD)) + 0] =                      \
                vfmaq_f32(c_cur[(0 * (C_ob / FLOAT_SIMD)) + 0], av, bv[0]); \
            c_cur[(0 * (C_ob / FLOAT_SIMD)) + 1] =                      \
                vfmaq_f32(c_cur[(0 * (C_ob / FLOAT_SIMD)) + 1], av, bv[1]); \
            c_cur[(0 * (C_ob / FLOAT_SIMD)) + 2] =                      \
                vfmaq_f32(c_cur[(0 * (C_ob / FLOAT_SIMD)) + 2], av, bv[2]); \
            c_cur[(0 * (C_ob / FLOAT_SIMD)) + 3] =                      \
                vfmaq_f32(c_cur[(0 * (C_ob / FLOAT_SIMD)) + 3], av, bv[3]); \
                                                                        \
        }                                                               \
    }
#endif


//****************************************************************************
//Pooling
//  Max pooling / ReLU
//****************************************************************************

#define FLOAT_MAX_TILE_C(step, a, W_ob, C_ob)                           \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            float32x4_t av = vld1q_f32(a + kk * step + jj * FLOAT_SIMD); \
            c_tile_v[kk * (C_ob / FLOAT_SIMD) + jj] =                   \
                vmaxq_f32(av, c_tile_v[kk * (C_ob / FLOAT_SIMD) + jj]); \
        }                                                               \
    }

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_MAX_END_C(step, a, c_cur, W_last, C_ob)           \
    for (uint32_t kk = 0; kk < W_last; kk++)                    \
    {                                                           \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            c_cur[kk * C_ob + jj] =                             \
                c_cur[kk * C_ob + jj] > a[kk * step + jj] ?     \
                c_cur[kk * C_ob + jj] : a[kk * step + jj];      \
        }                                                       \
    }

#else

#define FLOAT_MAX_END_C(step, a, c_cur, W_last, C_ob)                   \
    for (uint32_t kk = 0; kk < W_last; kk++)                            \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            float32x4_t av = vld1q_f32(a + kk * step + jj * FLOAT_SIMD); \
            c_cur[(kk) * (C_ob / FLOAT_SIMD) + jj] =                    \
                vmaxq_f32(av, c_cur[(kk) * (C_ob / FLOAT_SIMD) + jj]);  \
        }                                                               \
    }
#endif


//****************************************************************************
//DW Convolution
//****************************************************************************

#define FLOAT_DW_TILE_C(step, a, b, W_ob, C_ob)                         \
    float32x4_t bv[C_ob / FLOAT_SIMD];                                  \
    for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)                 \
    {                                                                   \
        bv[jj] = vld1q_f32(b + jj * FLOAT_SIMD);                        \
    }                                                                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            float32x4_t av = vld1q_f32(a + kk * step + jj * FLOAT_SIMD); \
            c_tile_v[kk * (C_ob / FLOAT_SIMD) + jj] =                   \
                vfmaq_f32(c_tile_v[kk * (C_ob / FLOAT_SIMD) + jj], av, bv[jj]); \
        }                                                               \
    }

// TODO: is this tested?
#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_DW_END_C(step, a, b, c_cur, _W_ob, C_ob)          \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                     \
    {                                                           \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            c_cur[kk * C_ob + jj] += a[kk * step + jj] * b[jj]; \
        }                                                       \
    }

#else

#define FLOAT_DW_END_C(step, a, b, c_cur, _W_ob, C_ob)                  \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                             \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            float32x4_t av = vld1q_f32(a + kk * step + jj * FLOAT_SIMD); \
            float32x4_t bv = vld1q_f32(b + jj * FLOAT_SIMD);            \
            c_cur[(kk) * (C_ob / FLOAT_SIMD) + jj] =                    \
                vfmaq_f32(c_cur[(kk) * (C_ob / FLOAT_SIMD) + jj], av, bv); \
        }                                                               \
    }
#endif


//****************************************************************************
// ReLU Activation
//****************************************************************************

// Same kernel as Pooling, set to zero to start.


//****************************************************************************
// Leaky ReLU activation
//****************************************************************************

//@todo implement neon intrinsics version
#define FLOAT_COND_SCALE_SIMD_C(c_x_x, mask, bv, av, a, kk, jj, _W_ob, _C_ob) \
    av = vld1q_f32(a + kk * step + jj * FLOAT_SIMD);                    \
    c_x_x = vmaxq_f32(av, c_x_x);                                       \
    mask = vcltq_f32(av, c_x_x);                                        \
    av = vmulq_f32(av, bv);                                             \
    av = (float32x4_t) vandq_s32((int32x4_t)(av), (int32x4_t)(mask));   \
    c_x_x = vaddq_f32(av, c_x_x);

#define FLOAT_COND_SCALE_TILE_C(step, a, b, _W_ob, _C_ob)               \
    float32x4_t bv = vld1q_dup_f32(b);                                  \
    float32x4_t av;                                                     \
    uint32x4_t mask;                                                    \
    FLOAT_COND_SCALE_SIMD_C(c_0_0, mask, bv, av, a, 0, 0, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_0_1, mask, bv, av, a, 0, 1, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_0_2, mask, bv, av, a, 0, 2, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_0_3, mask, bv, av, a, 0, 3, _W_ob, _C_ob); \
    /**/                                                                \
    FLOAT_COND_SCALE_SIMD_C(c_1_0, mask, bv, av, a, 1, 0, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_1_1, mask, bv, av, a, 1, 1, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_1_2, mask, bv, av, a, 1, 2, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_1_3, mask, bv, av, a, 1, 3, _W_ob, _C_ob); \
    /**/                                                                \
    FLOAT_COND_SCALE_SIMD_C(c_2_0, mask, bv, av, a, 2, 0, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_2_1, mask, bv, av, a, 2, 1, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_2_2, mask, bv, av, a, 2, 2, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_2_3, mask, bv, av, a, 2, 3, _W_ob, _C_ob); \
    /**/                                                                \
    FLOAT_COND_SCALE_SIMD_C(c_3_0, mask, bv, av, a, 3, 0, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_3_1, mask, bv, av, a, 3, 1, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_3_2, mask, bv, av, a, 3, 2, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_3_3, mask, bv, av, a, 3, 3, _W_ob, _C_ob); \
    /**/                                                                \
    FLOAT_COND_SCALE_SIMD_C(c_4_0, mask, bv, av, a, 4, 0, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_4_1, mask, bv, av, a, 4, 1, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_4_2, mask, bv, av, a, 4, 2, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_4_3, mask, bv, av, a, 4, 3, _W_ob, _C_ob); \
    /**/                                                                \
    FLOAT_COND_SCALE_SIMD_C(c_5_0, mask, bv, av, a, 5, 0, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_5_1, mask, bv, av, a, 5, 1, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_5_2, mask, bv, av, a, 5, 2, _W_ob, _C_ob); \
    FLOAT_COND_SCALE_SIMD_C(c_5_3, mask, bv, av, a, 5, 3, _W_ob, _C_ob);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_COND_SCALE_END_C(step, a, b, c_cur, W_last, C_ob)         \
    dtype *c_pixel = c_cur;                                             \
    dtype const *a_pixel = a;                                           \
    dtype scale = b[0];                                                 \
    for (uint32_t kk = 0; kk < W_last; kk++)                            \
    {                                                                   \
        dtype *c_channel = c_pixel;                                     \
        dtype const *a_channel = a_pixel;                               \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (*(a_channel) > *(c_channel)) ? *(a_channel) : (*(a_channel) * (scale)); \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#else

#define FLOAT_COND_SCALE_END_C(step, a, b, c_cur, W_last, C_ob)         \
    float32x4_t bv = vld1q_dup_f32(b);                                  \
    float32x4_t av;                                                     \
    uint32x4_t mask;                                                    \
    for (uint32_t kk = 0; kk < W_last; kk++)                            \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)             \
        {                                                               \
            float32x4_t cv = c_cur[kk * (C_ob / FLOAT_SIMD) + jj];      \
            FLOAT_COND_SCALE_SIMD_C(cv, mask, bv, av, a, kk, jj, W_last, C_ob); \
            c_cur[kk * (C_ob / FLOAT_SIMD) + jj] = cv;                  \
        }                                                               \
    }
#endif


//****************************************************************************
// Accumulation kernels
//****************************************************************************

#define FLOAT_ACCUM_TILE_C(step, a, W_ob, C_ob)         \
    float32x4_t av;                                     \
    av = vld1q_f32(a + 0 * step + 0 * FLOAT_SIMD);      \
    c_0_0 = vaddq_f32(c_0_0, av);                       \
    av = vld1q_f32(a + 0 * step + 1 * FLOAT_SIMD);      \
    c_0_1 = vaddq_f32(c_0_1, av);                       \
    av = vld1q_f32(a + 0 * step + 2 * FLOAT_SIMD);      \
    c_0_2 = vaddq_f32(c_0_2, av);                       \
    av = vld1q_f32(a + 0 * step + 3 * FLOAT_SIMD);      \
    c_0_3 = vaddq_f32(c_0_3, av);                       \
    av = vld1q_f32(a + 1 * step + 0 * FLOAT_SIMD);      \
    c_1_0 = vaddq_f32(c_1_0, av);                       \
    av = vld1q_f32(a + 1 * step + 1 * FLOAT_SIMD);      \
    c_1_1 = vaddq_f32(c_1_1, av);                       \
    av = vld1q_f32(a + 1 * step + 2 * FLOAT_SIMD);      \
    c_1_2 = vaddq_f32(c_1_2, av);                       \
    av = vld1q_f32(a + 1 * step + 3 * FLOAT_SIMD);      \
    c_1_3 = vaddq_f32(c_1_3, av);                       \
    av = vld1q_f32(a + 2 * step + 0 * FLOAT_SIMD);      \
    c_2_0 = vaddq_f32(c_2_0, av);                       \
    av = vld1q_f32(a + 2 * step + 1 * FLOAT_SIMD);      \
    c_2_1 = vaddq_f32(c_2_1, av);                       \
    av = vld1q_f32(a + 2 * step + 2 * FLOAT_SIMD);      \
    c_2_2 = vaddq_f32(c_2_2, av);                       \
    av = vld1q_f32(a + 2 * step + 3 * FLOAT_SIMD);      \
    c_2_3 = vaddq_f32(c_2_3, av);                       \
    av = vld1q_f32(a + 3 * step + 0 * FLOAT_SIMD);      \
    c_3_0 = vaddq_f32(c_3_0, av);                       \
    av = vld1q_f32(a + 3 * step + 1 * FLOAT_SIMD);      \
    c_3_1 = vaddq_f32(c_3_1, av);                       \
    av = vld1q_f32(a + 3 * step + 2 * FLOAT_SIMD);      \
    c_3_2 = vaddq_f32(c_3_2, av);                       \
    av = vld1q_f32(a + 3 * step + 3 * FLOAT_SIMD);      \
    c_3_3 = vaddq_f32(c_3_3, av);                       \
    av = vld1q_f32(a + 4 * step + 0 * FLOAT_SIMD);      \
    c_4_0 = vaddq_f32(c_4_0, av);                       \
    av = vld1q_f32(a + 4 * step + 1 * FLOAT_SIMD);      \
    c_4_1 = vaddq_f32(c_4_1, av);                       \
    av = vld1q_f32(a + 4 * step + 2 * FLOAT_SIMD);      \
    c_4_2 = vaddq_f32(c_4_2, av);                       \
    av = vld1q_f32(a + 4 * step + 3 * FLOAT_SIMD);      \
    c_4_3 = vaddq_f32(c_4_3, av);                       \
    av = vld1q_f32(a + 5 * step + 0 * FLOAT_SIMD);      \
    c_5_0 = vaddq_f32(c_5_0, av);                       \
    av = vld1q_f32(a + 5 * step + 1 * FLOAT_SIMD);      \
    c_5_1 = vaddq_f32(c_5_1, av);                       \
    av = vld1q_f32(a + 5 * step + 2 * FLOAT_SIMD);      \
    c_5_2 = vaddq_f32(c_5_2, av);                       \
    av = vld1q_f32(a + 5 * step + 3 * FLOAT_SIMD);      \
    c_5_3 = vaddq_f32(c_5_3, av);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_ACCUM_END_C(step, a, c_cur, W_last, C_ob) \
    for (uint32_t kk = 0; kk < W_last; kk++)            \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_cur[kk * C_ob + jj] += a[kk * step + jj]; \
        }                                               \
    }

#else

#define FLOAT_ACCUM_END_C(step, a, c_cur, W_last, C_ob)                              \
    for (uint32_t kk = 0; kk < W_last; kk++)                                         \
    {                                                                                \
        for (uint32_t jj = 0; jj < FLOAT_C_ob / FLOAT_SIMD; jj++)                          \
        {                                                                            \
            float32x4_t av = vld1q_f32(a + kk * step + jj * FLOAT_SIMD);             \
            c_cur[(kk) * (FLOAT_C_ob / FLOAT_SIMD) + jj] =                                 \
                vaddq_f32(av, c_cur[(kk) * (FLOAT_C_ob / FLOAT_SIMD) + jj]);               \
        }                                                                            \
    }
#endif


//****************************************************************************
// Broadcast multiplication kernels
//****************************************************************************

#define FLOAT_DIV_TILE_C(norm, W_ob, C_ob)        \
     float32x4_t av;                                     \
     av = vld1q_dup_f32(&norm);\
    c_0_0 = vmulq_f32(c_0_0, av);                       \
    c_0_1 = vmulq_f32(c_0_1, av);                       \
    c_0_2 = vmulq_f32(c_0_2, av);                       \
    c_0_3 = vmulq_f32(c_0_3, av);                       \
    c_1_0 = vmulq_f32(c_1_0, av);                       \
    c_1_1 = vmulq_f32(c_1_1, av);                       \
    c_1_2 = vmulq_f32(c_1_2, av);                       \
    c_1_3 = vmulq_f32(c_1_3, av);                       \
    c_2_0 = vmulq_f32(c_2_0, av);                       \
    c_2_1 = vmulq_f32(c_2_1, av);                       \
    c_2_2 = vmulq_f32(c_2_2, av);                       \
    c_2_3 = vmulq_f32(c_2_3, av);                       \
    c_3_0 = vmulq_f32(c_3_0, av);                       \
    c_3_1 = vmulq_f32(c_3_1, av);                       \
    c_3_2 = vmulq_f32(c_3_2, av);                       \
    c_3_3 = vmulq_f32(c_3_3, av);                       \
    c_4_0 = vmulq_f32(c_4_0, av);                       \
    c_4_1 = vmulq_f32(c_4_1, av);                       \
    c_4_2 = vmulq_f32(c_4_2, av);                       \
    c_4_3 = vmulq_f32(c_4_3, av);                       \
    c_5_0 = vmulq_f32(c_5_0, av);                       \
    c_5_1 = vmulq_f32(c_5_1, av);                       \
    c_5_2 = vmulq_f32(c_5_2, av);                       \
    c_5_3 = vmulq_f32(c_5_3, av);

#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_DIV_END_C(c_cur, norm, W_last, C_ob) \
  float *c_pixel = c_cur;                  \
  for (uint32_t kk = 0; kk < W_last; kk++)  \
  {                                         \
    float *c_channel = c_pixel;             \
    for (uint32_t jj = 0; jj < C_ob; jj++)  \
    {                                       \
      *(c_channel) *= norm;                 \
      c_channel++;                          \
    }                                       \
    c_pixel += C_ob;                        \
    }
#else
#define FLOAT_DIV_END_C(c_cur, norm, W_last, C_ob)                      \
  float32x4_t av;                                                \
  av = vld1q_dup_f32(&norm);                                     \
  float32x4_t *c_pixel = c_cur;                                   \
  for (uint32_t kk = 0; kk < W_last; kk++)                       \
  {                                                              \
    for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)          \
    {                                                            \
      c_pixel[(kk) * (C_ob / FLOAT_SIMD) + jj] =                   \
          vmulq_f32(c_pixel[(kk) * (C_ob / FLOAT_SIMD) + jj], av); \
    }                                                            \
    }
#endif

//****************************************************************************
// Softmax  (Ewise exponentiation)
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

#define FLOAT_EXP_TILE_C(step, a, W_ob, C_ob)                  \
    float c_tile_scalar[FLOAT_W_ob * FLOAT_C_ob];                  \
    float *c_pixel = c_tile_scalar;                                \
    float const *a_pixel = a;                               \
    for (uint32_t kk = 0; kk < W_ob; kk++)                     \
    {                                                          \
        float *c_channel = c_pixel;                         \
        float const *a_channel = a_pixel;                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                 \
        {                                                      \
            *(c_channel) = std::exp(*a_channel);               \
            c_channel++;                                       \
            a_channel++;                                       \
        }                                                      \
        a_pixel += step;                                       \
        c_pixel += C_ob;                                       \
    }                                                          \
    c_0_0 = vld1q_f32(c_tile_scalar + 0 * C_ob + 0 * FLOAT_SIMD);    c_0_1 = vld1q_f32(c_tile_scalar + 0 * C_ob + 1 * FLOAT_SIMD);     c_0_2 = vld1q_f32(c_tile_scalar + 0 * C_ob + 2 * FLOAT_SIMD);    c_0_3 = vld1q_f32(c_tile_scalar + 0 * C_ob + 3 * FLOAT_SIMD);   \
    c_1_0 = vld1q_f32(c_tile_scalar + 1 * C_ob + 0 * FLOAT_SIMD);    c_1_1 = vld1q_f32(c_tile_scalar + 1 * C_ob + 1 * FLOAT_SIMD);     c_1_2 = vld1q_f32(c_tile_scalar + 1 * C_ob + 2 * FLOAT_SIMD);    c_1_3 = vld1q_f32(c_tile_scalar + 1 * C_ob + 3 * FLOAT_SIMD);   \
    c_2_0 = vld1q_f32(c_tile_scalar + 2 * C_ob + 0 * FLOAT_SIMD);    c_2_1 = vld1q_f32(c_tile_scalar + 2 * C_ob + 1 * FLOAT_SIMD);     c_2_2 = vld1q_f32(c_tile_scalar + 2 * C_ob + 2 * FLOAT_SIMD);    c_2_3 = vld1q_f32(c_tile_scalar + 2 * C_ob + 3 * FLOAT_SIMD);   \
    c_3_0 = vld1q_f32(c_tile_scalar + 3 * C_ob + 0 * FLOAT_SIMD);    c_3_1 = vld1q_f32(c_tile_scalar + 3 * C_ob + 1 * FLOAT_SIMD);     c_3_2 = vld1q_f32(c_tile_scalar + 3 * C_ob + 2 * FLOAT_SIMD);    c_3_3 = vld1q_f32(c_tile_scalar + 3 * C_ob + 3 * FLOAT_SIMD);   \
    c_4_0 = vld1q_f32(c_tile_scalar + 4 * C_ob + 0 * FLOAT_SIMD);    c_4_1 = vld1q_f32(c_tile_scalar + 4 * C_ob + 1 * FLOAT_SIMD);     c_4_2 = vld1q_f32(c_tile_scalar + 4 * C_ob + 2 * FLOAT_SIMD);    c_4_3 = vld1q_f32(c_tile_scalar + 4 * C_ob + 3 * FLOAT_SIMD);   \
    c_5_0 = vld1q_f32(c_tile_scalar + 5 * C_ob + 0 * FLOAT_SIMD);    c_5_1 = vld1q_f32(c_tile_scalar + 5 * C_ob + 1 * FLOAT_SIMD);     c_5_2 = vld1q_f32(c_tile_scalar + 5 * C_ob + 2 * FLOAT_SIMD);    c_5_3 = vld1q_f32(c_tile_scalar + 5 * C_ob + 3 * FLOAT_SIMD);   


#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_EXP_END_C(step, a, c_cur, W_last, C_ob) \
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
#else
#define FLOAT_EXP_END_C(step, a, c_cur, W_last, C_ob) \
    float c_tile_scalar[FLOAT_W_ob * FLOAT_C_ob];                  \
    float *c_pixel = c_tile_scalar;                                \
    float const *a_pixel = a;                               \
    for (uint32_t kk = 0; kk < W_last; kk++)                     \
    {                                                          \
        float *c_channel = c_pixel;                         \
        float const *a_channel = a_pixel;                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                 \
        {                                                      \
            *(c_channel) = std::exp(*a_channel);               \
            c_channel++;                                       \
            a_channel++;                                       \
        }                                                      \
        c_cur[kk * (C_ob/FLOAT_SIMD) + 0] = vld1q_f32(c_pixel + 0 * FLOAT_SIMD);    c_cur[kk * (C_ob/FLOAT_SIMD) + 1] = vld1q_f32(c_pixel + 1 * FLOAT_SIMD);     c_cur[kk * (C_ob/FLOAT_SIMD) + 2] = vld1q_f32(c_pixel + 2 * FLOAT_SIMD);    c_cur[kk * (C_ob/FLOAT_SIMD) + 3] = vld1q_f32(c_pixel + 3 * FLOAT_SIMD); \
        a_pixel += step;                                       \
        c_pixel += C_ob;                                       \
    }                                                          \

#endif

//****************************************************************************
// Accumulate upsampling
//****************************************************************************

//@todo build and test this on arm platform

#define FLOAT_ACCUM_TILE_C_upsample(I, stride, _C_ib, _W_ob, C_ob) \
    float32x4_t a_0, a_1, a_2, a_3;\
    a_0 = vld1q_f32(I + (0 / stride) * C_ob + 0 * FLOAT_SIMD);  \
    a_1 = vld1q_f32(I + (0 / stride) * C_ob + 1 * FLOAT_SIMD);  \
    a_2 = vld1q_f32(I + (0 / stride) * C_ob + 2 * FLOAT_SIMD);  \
    a_3 = vld1q_f32(I + (0 / stride) * C_ob + 3 * FLOAT_SIMD);  \
    c_0_0 = vaddq_f32(c_0_0, a_0);a_0 = vld1q_f32(I + (1 / stride) * C_ob + 0 * FLOAT_SIMD);  \
    c_0_1 = vaddq_f32(c_0_1, a_1);a_1 = vld1q_f32(I + (1 / stride) * C_ob + 1 * FLOAT_SIMD);  \
    c_0_2 = vaddq_f32(c_0_2, a_2);a_2 = vld1q_f32(I + (1 / stride) * C_ob + 2 * FLOAT_SIMD);  \
    c_0_3 = vaddq_f32(c_0_3, a_3);a_3 = vld1q_f32(I + (1 / stride) * C_ob + 3 * FLOAT_SIMD);  \
    c_1_0 = vaddq_f32(c_1_0, a_0);a_0 = vld1q_f32(I + (2 / stride) * C_ob + 0 * FLOAT_SIMD);  \
    c_1_1 = vaddq_f32(c_1_1, a_1);a_1 = vld1q_f32(I + (2 / stride) * C_ob + 1 * FLOAT_SIMD);  \
    c_1_2 = vaddq_f32(c_1_2, a_2);a_2 = vld1q_f32(I + (2 / stride) * C_ob + 2 * FLOAT_SIMD);  \
    c_1_3 = vaddq_f32(c_1_3, a_3);a_3 = vld1q_f32(I + (2 / stride) * C_ob + 3 * FLOAT_SIMD);  \
    c_2_0 = vaddq_f32(c_2_0, a_0);a_0  = vld1q_f32(I + (3 / stride) * C_ob + 0 * FLOAT_SIMD);  \
    c_2_1 = vaddq_f32(c_2_1, a_1);a_1  = vld1q_f32(I + (3 / stride) * C_ob + 1 * FLOAT_SIMD);  \
    c_2_2 = vaddq_f32(c_2_2, a_2);a_2  = vld1q_f32(I + (3 / stride) * C_ob + 2 * FLOAT_SIMD);  \
    c_2_3 = vaddq_f32(c_2_3, a_3);a_3  = vld1q_f32(I + (3 / stride) * C_ob + 3 * FLOAT_SIMD);  \
    c_3_0 = vaddq_f32(c_3_0, a_0);a_0  = vld1q_f32(I + (4 / stride) * C_ob + 0 * FLOAT_SIMD);  \
    c_3_1 = vaddq_f32(c_3_1, a_1);a_1  = vld1q_f32(I + (4 / stride) * C_ob + 1 * FLOAT_SIMD);  \
    c_3_2 = vaddq_f32(c_3_2, a_2);a_2  = vld1q_f32(I + (4 / stride) * C_ob + 2 * FLOAT_SIMD);  \
    c_3_3 = vaddq_f32(c_3_3, a_3);a_3  = vld1q_f32(I + (4 / stride) * C_ob + 3 * FLOAT_SIMD);  \
    c_4_0 = vaddq_f32(c_4_0, a_0);a_0 = vld1q_f32(I + (5 / stride) * C_ob + 0 * FLOAT_SIMD);  \
    c_4_1 = vaddq_f32(c_4_1, a_1);a_1 = vld1q_f32(I + (5 / stride) * C_ob + 1 * FLOAT_SIMD);  \
    c_4_2 = vaddq_f32(c_4_2, a_2);a_2 = vld1q_f32(I + (5 / stride) * C_ob + 2 * FLOAT_SIMD);  \
    c_4_3 = vaddq_f32(c_4_3, a_3);a_3 = vld1q_f32(I + (5 / stride) * C_ob + 3 * FLOAT_SIMD);  \
    c_5_0 = vaddq_f32(c_5_0, a_0); \
    c_5_1 = vaddq_f32(c_5_1, a_1); \
    c_5_2 = vaddq_f32(c_5_2, a_2); \
    c_5_3 = vaddq_f32(c_5_3, a_3);
    
#if FLOAT_SIMD_EPILOGUE == 1
#define FLOAT_ACCUM_END_C_upsample(I, stride, _C_ib, _W_ob, C_ob)      \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                           \
    {                                                                 \
        for (uint32_t jj = 0; jj < C_ob; jj++)                        \
        {                                                             \
            c_tile[kk * C_ob + jj] += I[(kk / stride) * (_C_ib) + jj]; \
        }                                                             \
    }

#else

#define FLOAT_ACCUM_END_C_upsample(I, stride, _C_ib, _W_ob, C_ob)                                     \
    c_tile_t av;                                                                                      \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                                                           \
    {                                                                                                 \
        for (uint32_t jj = 0; jj < C_ob / FLOAT_SIMD; jj++)                                           \
        {                                                                                             \
            av =                                                                                      \
                vld1q_f32(I + (kk / stride) * (_C_ib) + jj * FLOAT_SIMD);                             \
            c_tile[kk * (C_ob / FLOAT_SIMD) + jj] = vaddq_f32(c_tile[kk * (C_ob / FLOAT_SIMD) + jj], av); \
        }                                                                                             \
    }
#endif

//

//****************************************************************************
// Accumulate channel dimension
//****************************************************************************

#define FLOAT_REDUCE_CHANNEL_END_C(O_w_left, _C_ob)                                \
if constexpr(_C_ob == 1 && _C_ob != FLOAT_SIMD_EPILOGUE)\
{\
    float c_tile_array[FLOAT_C_ob];                                         \
    for (uint32_t kk = 0; kk < O_w_left; kk++)                              \
    {                                                                       \
        float32x4_t *c_channel_v = c_tile + kk * (FLOAT_C_ob / FLOAT_SIMD); \
        c_channel_v[0] = vaddq_f32(c_channel_v[0], c_channel_v[1]);         \
        c_channel_v[2] = vaddq_f32(c_channel_v[2], c_channel_v[3]);         \
        c_channel_v[0] = vaddq_f32(c_channel_v[0], c_channel_v[2]);         \
                                                                            \
        vst1q_f32(c_tile_array, c_channel_v[0]);                            \
        for (uint32_t jj = 1; jj < FLOAT_SIMD; jj++)                        \
        {                                                                   \
            c_tile_array[0] += c_tile_array[jj];                            \
            c_tile_array[jj] = 0;                                           \
        }                                                                   \
                                                                            \
        c_channel_v[0] = vld1q_f32(c_tile_array);                           \
        c_channel_v[1] = vdupq_n_f32(0.0);                                  \
        c_channel_v[2] = vdupq_n_f32(0.0);                                  \
        c_channel_v[3] = vdupq_n_f32(0.0);                                  \
    }\
}

//****************************************************************************
// AVG Pooling
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



#define FLOAT_ADD_LAST_C_G(I, W_last, C_ob)     \
    float *i_pixel = I;                         \
    float *c_pixel = c_tile;                    \
    for (uint32_t mm = 0; mm < W_last; mm++)    \
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

#define FLOAT_REDUCE_C_last(O, W_last, C_ob)            \
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
