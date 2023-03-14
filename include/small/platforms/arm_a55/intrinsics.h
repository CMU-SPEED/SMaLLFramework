// SMaLL, Software for Machine Learning Libraries
// Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126

#include <arm_neon.h>

// Epilogue parameters
#define SIMD_EPILOGUE 4

#if SIMD_EPILOGUE == 1
typedef float c_tile_t;
#else
typedef float32x4_t c_tile_t;
#endif
// https://developer.arm.com/architectures/instruction-sets/intrinsics/

// Architecture specific tiling params

// __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

// Initializations

// float32x4_t vectorizes C_ob dim: [W_ob, C_ob] -> [W_ob, C_ob/SIMD, SIMD].
// assume SIMD == 4 and vec type is float.
// otherwise, SIMD = Neon bit width (128) / data type size.

#define DEF_TILE_C(W_ob, C_ob) \
  float c_tile[W_ob * C_ob];   \
  float32x4_t c_tile_v[W_ob * (C_ob / SIMD)];

#define DEF_END_C(W_ob, C_ob) \
c_tile_t c_tile[W_ob * (C_ob / SIMD_EPILOGUE)];

// === Initialize C tile to zero ==============================================
#define ZERO_TILE_C(W_ob, C_ob)                           \
  for (uint32_t kk = 0; kk < W_ob; kk++)                  \
  {                                                       \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)         \
    {                                                     \
      c_tile_v[kk * (C_ob / SIMD) + jj] = vdupq_n_f32(0); \
    }                                                     \
  }


#if SIMD_EPILOGUE==1
 #define ZERO_END_C(_W_ob, C_ob)                                               \
for (uint32_t kk = 0; kk < _W_ob; kk++)					\
  {									\
    for (uint32_t jj = 0; jj < C_ob; jj++)				\
      {									\
	c_tile[kk * C_ob + jj] = {};					\
      }									\
  }
#else
#define ZERO_END_C(_W_ob, C_ob)                           \
  for (uint32_t kk = 0; kk < _W_ob; kk++)                 \
  {                                                       \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)         \
    {                                                     \
      c_tile[kk * (C_ob / SIMD) + jj] = vdupq_n_f32(0); \
    }                                                     \
  }
#endif
// === End Initialize C tile to zero ==========================================

// === Loads ==================================================================
#define LOAD_TILE_C(O, W_ob, C_ob)                                              \
  for (uint32_t kk = 0; kk < W_ob; kk++)                                        \
  {                                                                             \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                               \
    {                                                                           \
      c_tile_v[kk * (C_ob / SIMD) + jj] = vld1q_f32(O + kk * C_ob + jj * SIMD); \
    }                                                                           \
  }
#if SIMD_EPILOGUE == 1

#define LOAD_END_C(O, _W_ob, C_ob)                                            \
  for (uint32_t kk = 0; kk < _W_ob; kk++)                                     \
  {                                                                           \
    for (uint32_t jj = 0; jj < C_ob; jj++)                                    \
    {                                                                         \
      c_tile[kk * C_ob + jj] = O[kk * C_ob + jj];                             \
    }                                                                         \
  }
#else
#define LOAD_END_C(O, _W_ob, C_ob)                                              \
  for (uint32_t kk = 0; kk < _W_ob; kk++)                                       \
  {                                                                             \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                               \
    {                                                                           \
      c_tile[kk * (C_ob / SIMD) + jj] = vld1q_f32(O + kk * C_ob + jj * SIMD); \
    }                                                                           \
  }
#endif
// TODO: merge LOAD_TILE_C and LOAD_TILE_C_stride? can use C_ob as step above.

#define LOAD_TILE_C_strided(O, step, W_ob, C_ob)                                \
  for (uint32_t kk = 0; kk < W_ob; kk++)                                        \
  {                                                                             \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                               \
    {                                                                           \
      c_tile_v[kk * (C_ob / SIMD) + jj] = vld1q_f32(O + kk * step + jj * SIMD); \
    }                                                                           \
  }

#if SIMD_EPILOGUE==1
#define LOAD_END_C_strided(O, step, _W_ob, C_ob)                              \
  for (uint32_t kk = 0; kk < _W_ob; kk++)                                     \
  {                                                                           \
    for (uint32_t jj = 0; jj < C_ob; jj++)                                    \
    {                                                                         \
      c_tile[kk * C_ob + jj] = O[kk * step + jj];                             \
    }                                                                         \
  }
#else
#define LOAD_END_C_strided(O, step, _W_ob, C_ob)                                \
  for (uint32_t kk = 0; kk < _W_ob; kk++)                                       \
  {                                                                             \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                               \
    {                                                                           \
      c_tile[kk * (C_ob / SIMD) + jj] = vld1q_f32(O + kk * step + jj * SIMD); \
    }                                                                           \
  }
#endif
// === End Loads ==============================================================

// === Stores =================================================================
#define STORE_TILE_C(O, W_ob, C_ob)                                            \
  for (uint32_t kk = 0; kk < W_ob; kk++)                                       \
  {                                                                            \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                              \
    {                                                                          \
      vst1q_f32(O + kk * C_ob + jj * SIMD, c_tile_v[kk * (C_ob / SIMD) + jj]); \
    }                                                                          \
  }
#if SIMD_EPILOGUE == 1
#define STORE_END_C(O, _W_ob, C_ob)                                           \
  for (uint32_t kk = 0; kk < _W_ob; kk++)                                     \
  {                                                                           \
    for (uint32_t jj = 0; jj < C_ob; jj++)                                    \
    {                                                                         \
      O[kk * C_ob + jj] = c_tile[kk * C_ob + jj];                             \
    }                                                                         \
  }
#else
#define STORE_END_C(O, _W_ob, C_ob)                                            \
  for (uint32_t kk = 0; kk < _W_ob; kk++)                                      \
  {                                                                            \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                              \
    {                                                                          \
      vst1q_f32(O + kk * C_ob + jj * SIMD, c_tile[kk * (C_ob / SIMD) + jj]); \
    }                                                                          \
  }
#endif
// === End Stores =============================================================

// === Convolution ============================================================
// TODO: add unroll of C_ib dim.
// a: [W_ob, C_ib] -- unroll C_ib in vector register and broadcast from diff lane.
// b: [Hf, Wf, C_ib, C_ob] -- unroll k loop will stride by C_ob (indexes C_ib).
#define CONV_TILE_C(step, a, b, W_ob, C_ob)                         \
  float32x4_t bv[C_ob / SIMD];                                      \
  for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                     \
  {                                                                 \
    bv[jj] = vld1q_f32(b + jj * SIMD);                              \
  }                                                                 \
  for (uint32_t kk = 0; kk < W_ob; kk++)                            \
  {                                                                 \
    float32x4_t av = vld1q_dup_f32(a + kk * step);                  \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                   \
    {                                                               \
      c_tile_v[kk * (C_ob / SIMD) + jj] =                           \
          vfmaq_f32(c_tile_v[kk * (C_ob / SIMD) + jj], av, bv[jj]); \
    }                                                               \
  }

#if SIMD_EPILOGUE == 1
#define CONV_END_C(step, a, b, c_cur, _W_ob, C_ob)                            \
  for (uint32_t i = 0; i < _UNROLL; i++)                                       \
  {                                                                           \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                                   \
    {                                                                         \
      for (uint32_t jj = 0; jj < C_ob; jj++)                                  \
      {                                                                       \
        c_cur[kk * C_ob + jj] += a[kk * step + i] * b[i * C_ob + jj];         \
      }                                                                       \
    }                                                                         \
  }
#else

#define CONV_END_C(step, a, b, c_cur, _W_ob, C_ob)	    \
  float32x4_t bv[C_ob / SIMD];				    \
  float32x4_t av;\
  for (uint32_t ii_unroll = 0; ii_unroll < _UNROLL; ii_unroll++)    \
  {                                                                 \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                   \
    {                                                               \
      bv[jj] = vld1q_f32(b + (ii_unroll * C_ob) + jj * SIMD);       \
    }                                                               \
    switch (_W_ob)                                                     \
    {                                                               \
    case 5:                                                         \
      av = vld1q_dup_f32(a + ((4 * step) + ii_unroll)); \
       c_cur[(4 * (C_ob / SIMD)) + 0] =                            \
          vfmaq_f32(c_cur[(4 * (C_ob / SIMD)) + 0], av, bv[0]);     \
      c_cur[(4 * (C_ob / SIMD)) + 1] =                              \
          vfmaq_f32(c_cur[(4 * (C_ob / SIMD)) + 1], av, bv[1]);     \
      c_cur[(4 * (C_ob / SIMD)) + 2] =                              \
          vfmaq_f32(c_cur[(4 * (C_ob / SIMD)) + 2], av, bv[2]);     \
      c_cur[(4 * (C_ob / SIMD)) + 3] =                              \
          vfmaq_f32(c_cur[(4 * (C_ob / SIMD)) + 3], av, bv[3]);     \
    case 4:                                                         \
      av = vld1q_dup_f32(a + ((3 * step) + ii_unroll)); \
       c_cur[(3 * (C_ob / SIMD)) + 0] =                            \
          vfmaq_f32(c_cur[(3 * (C_ob / SIMD)) + 0], av, bv[0]);     \
      c_cur[(3 * (C_ob / SIMD)) + 1] =                              \
          vfmaq_f32(c_cur[(3 * (C_ob / SIMD)) + 1], av, bv[1]);     \
      c_cur[(3 * (C_ob / SIMD)) + 2] =                              \
          vfmaq_f32(c_cur[(3 * (C_ob / SIMD)) + 2], av, bv[2]);     \
      c_cur[(3 * (C_ob / SIMD)) + 3] =                              \
          vfmaq_f32(c_cur[(3 * (C_ob / SIMD)) + 3], av, bv[3]);     \
    case 3:                                                         \
      av = vld1q_dup_f32(a + ((2 * step) + ii_unroll)); \
       c_cur[(2 * (C_ob / SIMD)) + 0] =                            \
          vfmaq_f32(c_cur[(2 * (C_ob / SIMD)) + 0], av, bv[0]);     \
      c_cur[(2 * (C_ob / SIMD)) + 1] =                              \
          vfmaq_f32(c_cur[(2 * (C_ob / SIMD)) + 1], av, bv[1]);     \
      c_cur[(2 * (C_ob / SIMD)) + 2] =                              \
          vfmaq_f32(c_cur[(2 * (C_ob / SIMD)) + 2], av, bv[2]);     \
      c_cur[(2 * (C_ob / SIMD)) + 3] =                              \
          vfmaq_f32(c_cur[(2 * (C_ob / SIMD)) + 3], av, bv[3]);     \
    case 2:                                                         \
      av = vld1q_dup_f32(a + ((1 * step) + ii_unroll)); \
       c_cur[(1 * (C_ob / SIMD)) + 0] =                            \
          vfmaq_f32(c_cur[(1 * (C_ob / SIMD)) + 0], av, bv[0]);     \
      c_cur[(1 * (C_ob / SIMD)) + 1] =                              \
          vfmaq_f32(c_cur[(1 * (C_ob / SIMD)) + 1], av, bv[1]);     \
      c_cur[(1 * (C_ob / SIMD)) + 2] =                              \
          vfmaq_f32(c_cur[(1 * (C_ob / SIMD)) + 2], av, bv[2]);     \
      c_cur[(1 * (C_ob / SIMD)) + 3] =                              \
          vfmaq_f32(c_cur[(1 * (C_ob / SIMD)) + 3], av, bv[3]);     \
    case 1:                                                         \
      av = vld1q_dup_f32(a + ((0 * step) + ii_unroll)); \
       c_cur[(0 * (C_ob / SIMD)) + 0] =                            \
          vfmaq_f32(c_cur[(0 * (C_ob / SIMD)) + 0], av, bv[0]);     \
      c_cur[(0 * (C_ob / SIMD)) + 1] =                              \
          vfmaq_f32(c_cur[(0 * (C_ob / SIMD)) + 1], av, bv[1]);     \
      c_cur[(0 * (C_ob / SIMD)) + 2] =                              \
          vfmaq_f32(c_cur[(0 * (C_ob / SIMD)) + 2], av, bv[2]);     \
      c_cur[(0 * (C_ob / SIMD)) + 3] =                              \
          vfmaq_f32(c_cur[(0 * (C_ob / SIMD)) + 3], av, bv[3]);     \
                                                                                                                  \
    }                                                               \
  }

//#define CONV_END_C(step, a, b, c_cur, _W_ob, C_ob)		     \
float32x4_t bv[C_ob / SIMD];					     \
  for (uint32_t ii_unroll = 0; ii_unroll < _UNROLL; ii_unroll++)     \
  {                                                                  \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                    \
    {                                                                \
      bv[jj] = vld1q_f32(b + (ii_unroll * C_ob) + jj * SIMD);        \
    }                                                                \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                          \
    {                                                                \
      float32x4_t av = vld1q_dup_f32(a + ((kk * step) + ii_unroll)); \
      for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                  \
      {                                                              \
        c_cur[(kk * (C_ob / SIMD)) + jj] =                           \
            vfmaq_f32(c_cur[(kk * (C_ob / SIMD)) + jj], av, bv[jj]); \
      }                                                              \
    }                                                                \
  }
#endif
// === End Convolution ========================================================

// === Max Pooling/ReLU =======================================================
#define MAX_TILE_C(step, a, W_ob, C_ob)                      \
  for (uint32_t kk = 0; kk < W_ob; kk++)                     \
  {                                                          \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)            \
    {                                                        \
      float32x4_t av = vld1q_f32(a + kk * step + jj * SIMD); \
      c_tile_v[kk * (C_ob / SIMD) + jj] =                    \
          vmaxq_f32(av, c_tile_v[kk * (C_ob / SIMD) + jj]);  \
    }                                                        \
  }

#if SIMD_EPILOGUE == 1
#define MAX_END_C(step, a, b, c_cur, W_last, C_ob)                            \
  for (uint32_t kk = 0; kk < W_last; kk++)                                    \
  {                                                                           \
    for (uint32_t jj = 0; jj < C_ob; jj++)                                    \
    {                                                                         \
      c_cur[kk * C_ob + jj] =                                                 \
        c_cur[kk * C_ob + jj] > a[kk * step + jj] ?                           \
        c_cur[kk * C_ob + jj] : a[kk * step + jj];                            \
    }                                                                         \
  }
#else
#define MAX_END_C(step, a, b, c_cur, W_last, C_ob)                          \
  for (uint32_t kk = 0; kk < W_last; kk++)                                  \
  {                                                                         \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                           \
    {                                                                       \
      float32x4_t av = vld1q_f32(a + kk * step + jj * SIMD);                \
      c_cur[(kk) * (C_ob / SIMD) + jj] =                   \
          vmaxq_f32(av, c_cur[(kk) * (C_ob / SIMD) + jj]); \
    }                                                                       \
  }
#endif
// === End Max Pooling/ReLU ===================================================

// === Depthwise Convolution ==================================================
#define DW_TILE_C(step, a, b, W_ob, C_ob)                           \
  float32x4_t bv[C_ob / SIMD];                                      \
  for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                     \
  {                                                                 \
    bv[jj] = vld1q_f32(b + jj * SIMD);                              \
  }                                                                 \
  for (uint32_t kk = 0; kk < W_ob; kk++)                            \
  {                                                                 \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                   \
    {                                                               \
      float32x4_t av = vld1q_f32(a + kk * step + jj * SIMD);        \
      c_tile_v[kk * (C_ob / SIMD) + jj] =                           \
          vfmaq_f32(c_tile_v[kk * (C_ob / SIMD) + jj], av, bv[jj]); \
    }                                                               \
  }

// TODO: is this tested?
#if SIMD_EPILOGUE == 1
#define DW_END_C(step, a, b, c_cur, _W_ob, C_ob)                              \
  for (uint32_t kk = 0; kk < _W_ob; kk++)                                     \
  {                                                                           \
    for (uint32_t jj = 0; jj < C_ob; jj++)                                    \
    {                                                                         \
      c_cur[kk * C_ob + jj] += a[kk * step + jj] * b[jj];                     \
    }                                                                         \
  }
#else
#define DW_END_C(step, a, b, c_cur, _W_ob, C_ob)                                \
  for (uint32_t kk = 0; kk < _W_ob; kk++)                                       \
  {                                                                             \
    for (uint32_t jj = 0; jj < C_ob / SIMD; jj++)                               \
    {                                                                           \
      float32x4_t av = vld1q_f32(a + kk * step + jj * SIMD);                    \
      float32x4_t bv = vld1q_f32(b + jj * SIMD);                                \
      c_cur[(kk) * (C_ob / SIMD) + jj] =                       \
          vfmaq_f32(c_cur[(kk) * (C_ob / SIMD) + jj], av, bv); \
    }                                                                           \
  }
#endif
// === End Depthwise Convolution ==============================================

// TODO: is this tested?
// AVG Pooling
#define ADD_TILE_C_G(I, W_ob_g, C_ob)              \
  for (uint32_t mm = 0; mm < W_ob_g; mm++)         \
  {                                                \
    for (uint32_t kk = 0; kk < C_ob; kk++)         \
    {                                              \
      c_tile[mm * C_ob + kk] += I[mm * C_ob + kk]; \
    }                                              \
  }

#define ADD_LAST_C_G(I, W_last, C_ob)      \
  float *i_pixel = I;                      \
  float *c_pixel = c_tile;                 \
  for (uint32_t mm = 0; mm < W_last; mm++) \
  {                                        \
    float *c_channel = c_pixel;            \
    float *i_channel = i_pixel;            \
    for (uint32_t kk = 0; kk < C_ob; kk++) \
    {                                      \
      *c_channel += *i_channel;            \
      c_channel++;                         \
      i_channel++;                         \
    }                                      \
    c_pixel += C_ob;                       \
    i_pixel += C_ob;                       \
  }

#define REDUCE_div_C(O, d, W_ob_g, C_ob)     \
  {                                          \
    float *c_pixel = c_tile;                 \
    float *O_channel = O;                    \
    float *c_channel = c_pixel;              \
    for (uint32_t mm = 0; mm < W_ob_g; mm++) \
    {                                        \
      float *O_channel = O;                  \
      float *c_channel = c_pixel;            \
      for (uint32_t kk = 0; kk < C_ob; kk++) \
      {                                      \
        *O_channel += *c_channel;            \
        O_channel++;                         \
        c_channel++;                         \
      }                                      \
      c_pixel += C_ob;                       \
    }                                        \
    O_channel = O;                           \
    for (uint32_t kk = 0; kk < C_ob; kk++)   \
    {                                        \
      *O_channel *= d;                       \
      O_channel++;                           \
    }                                        \
  }

#define REDUCE_C(O, W_ob_g, C_ob)            \
  {                                          \
    for (uint32_t mm = 0; mm < W_ob_g; mm++) \
    {                                        \
      for (uint32_t kk = 0; kk < C_ob; kk++) \
      {                                      \
        O[kk] += c_tile[mm * C_ob + kk];     \
      }                                      \
    }                                        \
  }

#define REDUCE_C_last(O, W_last, C_ob)       \
  {                                          \
    float *c_pixel = c_tile;                 \
    float *O_channel = O;                    \
    float *c_channel = c_pixel;              \
    for (uint32_t mm = 0; mm < W_ob_g; mm++) \
    {                                        \
      float *O_channel = O;                  \
      float *c_channel = c_pixel;            \
      for (uint32_t kk = 0; kk < C_ob; kk++) \
      {                                      \
        *O_channel += *c_channel;            \
        O_channel++;                         \
        c_channel++;                         \
      }                                      \
      c_pixel += C_ob;                       \
    }                                        \
  }



#include "intrinsics-gen.h"
