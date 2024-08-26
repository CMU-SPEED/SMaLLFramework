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

#include <arm_neon.h>
#ifdef FLOAT_DEF_TILE_C
#undef FLOAT_DEF_TILE_C
#endif

#define FLOAT_DEF_TILE_C(W_ob, C_ob)\
  /*float c_tile[W_ob * C_ob];*/\
  float32x4_t c_0_0;\
  float32x4_t c_0_1;\
  float32x4_t c_0_2;\
  float32x4_t c_0_3;\
  float32x4_t c_1_0;\
  float32x4_t c_1_1;\
  float32x4_t c_1_2;\
  float32x4_t c_1_3;\
  float32x4_t c_2_0;\
  float32x4_t c_2_1;\
  float32x4_t c_2_2;\
  float32x4_t c_2_3;\
  float32x4_t c_3_0;\
  float32x4_t c_3_1;\
  float32x4_t c_3_2;\
  float32x4_t c_3_3;\
  float32x4_t c_4_0;\
  float32x4_t c_4_1;\
  float32x4_t c_4_2;\
  float32x4_t c_4_3;\
  float32x4_t c_5_0;\
  float32x4_t c_5_1;\
  float32x4_t c_5_2;\
  float32x4_t c_5_3;

#ifdef FLOAT_ZERO_TILE_C
#undef FLOAT_ZERO_TILE_C
#endif

#define FLOAT_ZERO_TILE_C(W_ob, C_ob) \
  c_0_0 = vdupq_n_f32(0);             \
  c_0_1 = vdupq_n_f32(0);             \
  c_0_2 = vdupq_n_f32(0);             \
  c_0_3 = vdupq_n_f32(0);             \
  c_1_0 = vdupq_n_f32(0);             \
  c_1_1 = vdupq_n_f32(0);             \
  c_1_2 = vdupq_n_f32(0);             \
  c_1_3 = vdupq_n_f32(0);             \
  c_2_0 = vdupq_n_f32(0);             \
  c_2_1 = vdupq_n_f32(0);             \
  c_2_2 = vdupq_n_f32(0);             \
  c_2_3 = vdupq_n_f32(0);             \
  c_3_0 = vdupq_n_f32(0);             \
  c_3_1 = vdupq_n_f32(0);             \
  c_3_2 = vdupq_n_f32(0);             \
  c_3_3 = vdupq_n_f32(0);             \
  c_4_0 = vdupq_n_f32(0);             \
  c_4_1 = vdupq_n_f32(0);             \
  c_4_2 = vdupq_n_f32(0);             \
  c_4_3 = vdupq_n_f32(0);             \
  c_5_0 = vdupq_n_f32(0);             \
  c_5_1 = vdupq_n_f32(0);             \
  c_5_2 = vdupq_n_f32(0);             \
  c_5_3 = vdupq_n_f32(0);\
  

#ifdef FLOAT_LOAD_TILE_C
#undef FLOAT_LOAD_TILE_C
#endif

#define FLOAT_LOAD_TILE_C(O, W_ob, C_ob)                  \
  if constexpr (W_ob == FLOAT_W_ob && C_ob == FLOAT_C_ob) \
  {                                                       \
    c_0_0 = vld1q_f32(O + 0 * C_ob + 0 * FLOAT_SIMD);     \
    c_0_1 = vld1q_f32(O + 0 * C_ob + 1 * FLOAT_SIMD);     \
    c_0_2 = vld1q_f32(O + 0 * C_ob + 2 * FLOAT_SIMD);     \
    c_0_3 = vld1q_f32(O + 0 * C_ob + 3 * FLOAT_SIMD);     \
    c_1_0 = vld1q_f32(O + 1 * C_ob + 0 * FLOAT_SIMD);     \
    c_1_1 = vld1q_f32(O + 1 * C_ob + 1 * FLOAT_SIMD);     \
    c_1_2 = vld1q_f32(O + 1 * C_ob + 2 * FLOAT_SIMD);     \
    c_1_3 = vld1q_f32(O + 1 * C_ob + 3 * FLOAT_SIMD);     \
    c_2_0 = vld1q_f32(O + 2 * C_ob + 0 * FLOAT_SIMD);     \
    c_2_1 = vld1q_f32(O + 2 * C_ob + 1 * FLOAT_SIMD);     \
    c_2_2 = vld1q_f32(O + 2 * C_ob + 2 * FLOAT_SIMD);     \
    c_2_3 = vld1q_f32(O + 2 * C_ob + 3 * FLOAT_SIMD);     \
    c_3_0 = vld1q_f32(O + 3 * C_ob + 0 * FLOAT_SIMD);     \
    c_3_1 = vld1q_f32(O + 3 * C_ob + 1 * FLOAT_SIMD);     \
    c_3_2 = vld1q_f32(O + 3 * C_ob + 2 * FLOAT_SIMD);     \
    c_3_3 = vld1q_f32(O + 3 * C_ob + 3 * FLOAT_SIMD);     \
    c_4_0 = vld1q_f32(O + 4 * C_ob + 0 * FLOAT_SIMD);     \
    c_4_1 = vld1q_f32(O + 4 * C_ob + 1 * FLOAT_SIMD);     \
    c_4_2 = vld1q_f32(O + 4 * C_ob + 2 * FLOAT_SIMD);     \
    c_4_3 = vld1q_f32(O + 4 * C_ob + 3 * FLOAT_SIMD);     \
    c_5_0 = vld1q_f32(O + 5 * C_ob + 0 * FLOAT_SIMD);     \
    c_5_1 = vld1q_f32(O + 5 * C_ob + 1 * FLOAT_SIMD);     \
    c_5_2 = vld1q_f32(O + 5 * C_ob + 2 * FLOAT_SIMD);     \
    c_5_3 = vld1q_f32(O + 5 * C_ob + 3 * FLOAT_SIMD);     \
  }                                                       \
  else if constexpr (W_ob == 5 && C_ob == FLOAT_C_ob)     \
  {                                                       \
    c_0_0 = vld1q_f32(O + 0 * C_ob + 0 * FLOAT_SIMD);     \
    c_0_1 = vld1q_f32(O + 0 * C_ob + 1 * FLOAT_SIMD);     \
    c_0_2 = vld1q_f32(O + 0 * C_ob + 2 * FLOAT_SIMD);     \
    c_0_3 = vld1q_f32(O + 0 * C_ob + 3 * FLOAT_SIMD);     \
    c_1_0 = vld1q_f32(O + 1 * C_ob + 0 * FLOAT_SIMD);     \
    c_1_1 = vld1q_f32(O + 1 * C_ob + 1 * FLOAT_SIMD);     \
    c_1_2 = vld1q_f32(O + 1 * C_ob + 2 * FLOAT_SIMD);     \
    c_1_3 = vld1q_f32(O + 1 * C_ob + 3 * FLOAT_SIMD);     \
    c_2_0 = vld1q_f32(O + 2 * C_ob + 0 * FLOAT_SIMD);     \
    c_2_1 = vld1q_f32(O + 2 * C_ob + 1 * FLOAT_SIMD);     \
    c_2_2 = vld1q_f32(O + 2 * C_ob + 2 * FLOAT_SIMD);     \
    c_2_3 = vld1q_f32(O + 2 * C_ob + 3 * FLOAT_SIMD);     \
    c_3_0 = vld1q_f32(O + 3 * C_ob + 0 * FLOAT_SIMD);     \
    c_3_1 = vld1q_f32(O + 3 * C_ob + 1 * FLOAT_SIMD);     \
    c_3_2 = vld1q_f32(O + 3 * C_ob + 2 * FLOAT_SIMD);     \
    c_3_3 = vld1q_f32(O + 3 * C_ob + 3 * FLOAT_SIMD);     \
    c_4_0 = vld1q_f32(O + 4 * C_ob + 0 * FLOAT_SIMD);     \
    c_4_1 = vld1q_f32(O + 4 * C_ob + 1 * FLOAT_SIMD);     \
    c_4_2 = vld1q_f32(O + 4 * C_ob + 2 * FLOAT_SIMD);     \
    c_4_3 = vld1q_f32(O + 4 * C_ob + 3 * FLOAT_SIMD);     \
  }                                                       \
  if constexpr (W_ob == 4 && C_ob == FLOAT_C_ob)          \
  {                                                       \
    c_0_0 = vld1q_f32(O + 0 * C_ob + 0 * FLOAT_SIMD);     \
    c_0_1 = vld1q_f32(O + 0 * C_ob + 1 * FLOAT_SIMD);     \
    c_0_2 = vld1q_f32(O + 0 * C_ob + 2 * FLOAT_SIMD);     \
    c_0_3 = vld1q_f32(O + 0 * C_ob + 3 * FLOAT_SIMD);     \
    c_1_0 = vld1q_f32(O + 1 * C_ob + 0 * FLOAT_SIMD);     \
    c_1_1 = vld1q_f32(O + 1 * C_ob + 1 * FLOAT_SIMD);     \
    c_1_2 = vld1q_f32(O + 1 * C_ob + 2 * FLOAT_SIMD);     \
    c_1_3 = vld1q_f32(O + 1 * C_ob + 3 * FLOAT_SIMD);     \
    c_2_0 = vld1q_f32(O + 2 * C_ob + 0 * FLOAT_SIMD);     \
    c_2_1 = vld1q_f32(O + 2 * C_ob + 1 * FLOAT_SIMD);     \
    c_2_2 = vld1q_f32(O + 2 * C_ob + 2 * FLOAT_SIMD);     \
    c_2_3 = vld1q_f32(O + 2 * C_ob + 3 * FLOAT_SIMD);     \
    c_3_0 = vld1q_f32(O + 3 * C_ob + 0 * FLOAT_SIMD);     \
    c_3_1 = vld1q_f32(O + 3 * C_ob + 1 * FLOAT_SIMD);     \
    c_3_2 = vld1q_f32(O + 3 * C_ob + 2 * FLOAT_SIMD);     \
    c_3_3 = vld1q_f32(O + 3 * C_ob + 3 * FLOAT_SIMD);     \
  }

#ifdef FLOAT_LOAD_TILE_C_strided
#undef FLOAT_LOAD_TILE_C_strided
#endif

#define FLOAT_LOAD_TILE_C_strided(O, step, W_ob, C_ob)    \
  if constexpr (W_ob == FLOAT_W_ob && C_ob == FLOAT_C_ob) \
  {                                                       \
    c_0_0 = vld1q_f32(O + 0 * step + 0 * FLOAT_SIMD);     \
    c_0_1 = vld1q_f32(O + 0 * step + 1 * FLOAT_SIMD);     \
    c_0_2 = vld1q_f32(O + 0 * step + 2 * FLOAT_SIMD);     \
    c_0_3 = vld1q_f32(O + 0 * step + 3 * FLOAT_SIMD);     \
    c_1_0 = vld1q_f32(O + 1 * step + 0 * FLOAT_SIMD);     \
    c_1_1 = vld1q_f32(O + 1 * step + 1 * FLOAT_SIMD);     \
    c_1_2 = vld1q_f32(O + 1 * step + 2 * FLOAT_SIMD);     \
    c_1_3 = vld1q_f32(O + 1 * step + 3 * FLOAT_SIMD);     \
    c_2_0 = vld1q_f32(O + 2 * step + 0 * FLOAT_SIMD);     \
    c_2_1 = vld1q_f32(O + 2 * step + 1 * FLOAT_SIMD);     \
    c_2_2 = vld1q_f32(O + 2 * step + 2 * FLOAT_SIMD);     \
    c_2_3 = vld1q_f32(O + 2 * step + 3 * FLOAT_SIMD);     \
    c_3_0 = vld1q_f32(O + 3 * step + 0 * FLOAT_SIMD);     \
    c_3_1 = vld1q_f32(O + 3 * step + 1 * FLOAT_SIMD);     \
    c_3_2 = vld1q_f32(O + 3 * step + 2 * FLOAT_SIMD);     \
    c_3_3 = vld1q_f32(O + 3 * step + 3 * FLOAT_SIMD);     \
    c_4_0 = vld1q_f32(O + 4 * step + 0 * FLOAT_SIMD);     \
    c_4_1 = vld1q_f32(O + 4 * step + 1 * FLOAT_SIMD);     \
    c_4_2 = vld1q_f32(O + 4 * step + 2 * FLOAT_SIMD);     \
    c_4_3 = vld1q_f32(O + 4 * step + 3 * FLOAT_SIMD);     \
    c_5_0 = vld1q_f32(O + 5 * step + 0 * FLOAT_SIMD);     \
    c_5_1 = vld1q_f32(O + 5 * step + 1 * FLOAT_SIMD);     \
    c_5_2 = vld1q_f32(O + 5 * step + 2 * FLOAT_SIMD);     \
    c_5_3 = vld1q_f32(O + 5 * step + 3 * FLOAT_SIMD);     \
  }                                                       \
  if constexpr (W_ob == 5 && C_ob == FLOAT_C_ob)          \
  {                                                       \
    c_0_0 = vld1q_f32(O + 0 * step + 0 * FLOAT_SIMD);     \
    c_0_1 = vld1q_f32(O + 0 * step + 1 * FLOAT_SIMD);     \
    c_0_2 = vld1q_f32(O + 0 * step + 2 * FLOAT_SIMD);     \
    c_0_3 = vld1q_f32(O + 0 * step + 3 * FLOAT_SIMD);     \
    c_1_0 = vld1q_f32(O + 1 * step + 0 * FLOAT_SIMD);     \
    c_1_1 = vld1q_f32(O + 1 * step + 1 * FLOAT_SIMD);     \
    c_1_2 = vld1q_f32(O + 1 * step + 2 * FLOAT_SIMD);     \
    c_1_3 = vld1q_f32(O + 1 * step + 3 * FLOAT_SIMD);     \
    c_2_0 = vld1q_f32(O + 2 * step + 0 * FLOAT_SIMD);     \
    c_2_1 = vld1q_f32(O + 2 * step + 1 * FLOAT_SIMD);     \
    c_2_2 = vld1q_f32(O + 2 * step + 2 * FLOAT_SIMD);     \
    c_2_3 = vld1q_f32(O + 2 * step + 3 * FLOAT_SIMD);     \
    c_3_0 = vld1q_f32(O + 3 * step + 0 * FLOAT_SIMD);     \
    c_3_1 = vld1q_f32(O + 3 * step + 1 * FLOAT_SIMD);     \
    c_3_2 = vld1q_f32(O + 3 * step + 2 * FLOAT_SIMD);     \
    c_3_3 = vld1q_f32(O + 3 * step + 3 * FLOAT_SIMD);     \
    c_4_0 = vld1q_f32(O + 4 * step + 0 * FLOAT_SIMD);     \
    c_4_1 = vld1q_f32(O + 4 * step + 1 * FLOAT_SIMD);     \
    c_4_2 = vld1q_f32(O + 4 * step + 2 * FLOAT_SIMD);     \
    c_4_3 = vld1q_f32(O + 4 * step + 3 * FLOAT_SIMD);     \
  }                                                       \
  if constexpr (W_ob == 4 && C_ob == FLOAT_C_ob)          \
  {                                                       \
    c_0_0 = vld1q_f32(O + 0 * step + 0 * FLOAT_SIMD);     \
    c_0_1 = vld1q_f32(O + 0 * step + 1 * FLOAT_SIMD);     \
    c_0_2 = vld1q_f32(O + 0 * step + 2 * FLOAT_SIMD);     \
    c_0_3 = vld1q_f32(O + 0 * step + 3 * FLOAT_SIMD);     \
    c_1_0 = vld1q_f32(O + 1 * step + 0 * FLOAT_SIMD);     \
    c_1_1 = vld1q_f32(O + 1 * step + 1 * FLOAT_SIMD);     \
    c_1_2 = vld1q_f32(O + 1 * step + 2 * FLOAT_SIMD);     \
    c_1_3 = vld1q_f32(O + 1 * step + 3 * FLOAT_SIMD);     \
    c_2_0 = vld1q_f32(O + 2 * step + 0 * FLOAT_SIMD);     \
    c_2_1 = vld1q_f32(O + 2 * step + 1 * FLOAT_SIMD);     \
    c_2_2 = vld1q_f32(O + 2 * step + 2 * FLOAT_SIMD);     \
    c_2_3 = vld1q_f32(O + 2 * step + 3 * FLOAT_SIMD);     \
    c_3_0 = vld1q_f32(O + 3 * step + 0 * FLOAT_SIMD);     \
    c_3_1 = vld1q_f32(O + 3 * step + 1 * FLOAT_SIMD);     \
    c_3_2 = vld1q_f32(O + 3 * step + 2 * FLOAT_SIMD);     \
    c_3_3 = vld1q_f32(O + 3 * step + 3 * FLOAT_SIMD);     \
  }

#ifdef FLOAT_STORE_TILE_C
#undef FLOAT_STORE_TILE_C
#endif

#define FLOAT_STORE_TILE_C(O, W_ob, C_ob)                 \
  if constexpr (W_ob == FLOAT_W_ob && C_ob == FLOAT_C_ob) \
  {                                                       \
    vst1q_f32(O + 0 * C_ob + 0 * FLOAT_SIMD, c_0_0);      \
    vst1q_f32(O + 0 * C_ob + 1 * FLOAT_SIMD, c_0_1);      \
    vst1q_f32(O + 0 * C_ob + 2 * FLOAT_SIMD, c_0_2);      \
    vst1q_f32(O + 0 * C_ob + 3 * FLOAT_SIMD, c_0_3);      \
    vst1q_f32(O + 1 * C_ob + 0 * FLOAT_SIMD, c_1_0);      \
    vst1q_f32(O + 1 * C_ob + 1 * FLOAT_SIMD, c_1_1);      \
    vst1q_f32(O + 1 * C_ob + 2 * FLOAT_SIMD, c_1_2);      \
    vst1q_f32(O + 1 * C_ob + 3 * FLOAT_SIMD, c_1_3);      \
    vst1q_f32(O + 2 * C_ob + 0 * FLOAT_SIMD, c_2_0);      \
    vst1q_f32(O + 2 * C_ob + 1 * FLOAT_SIMD, c_2_1);      \
    vst1q_f32(O + 2 * C_ob + 2 * FLOAT_SIMD, c_2_2);      \
    vst1q_f32(O + 2 * C_ob + 3 * FLOAT_SIMD, c_2_3);      \
    vst1q_f32(O + 3 * C_ob + 0 * FLOAT_SIMD, c_3_0);      \
    vst1q_f32(O + 3 * C_ob + 1 * FLOAT_SIMD, c_3_1);      \
    vst1q_f32(O + 3 * C_ob + 2 * FLOAT_SIMD, c_3_2);      \
    vst1q_f32(O + 3 * C_ob + 3 * FLOAT_SIMD, c_3_3);      \
    vst1q_f32(O + 4 * C_ob + 0 * FLOAT_SIMD, c_4_0);      \
    vst1q_f32(O + 4 * C_ob + 1 * FLOAT_SIMD, c_4_1);      \
    vst1q_f32(O + 4 * C_ob + 2 * FLOAT_SIMD, c_4_2);      \
    vst1q_f32(O + 4 * C_ob + 3 * FLOAT_SIMD, c_4_3);      \
    vst1q_f32(O + 5 * C_ob + 0 * FLOAT_SIMD, c_5_0);      \
    vst1q_f32(O + 5 * C_ob + 1 * FLOAT_SIMD, c_5_1);      \
    vst1q_f32(O + 5 * C_ob + 2 * FLOAT_SIMD, c_5_2);      \
    vst1q_f32(O + 5 * C_ob + 3 * FLOAT_SIMD, c_5_3);      \
  }                                                       \
  else if constexpr (W_ob == 5 && C_ob == FLOAT_C_ob)     \
  {                                                       \
    vst1q_f32(O + 0 * C_ob + 0 * FLOAT_SIMD, c_0_0);      \
    vst1q_f32(O + 0 * C_ob + 1 * FLOAT_SIMD, c_0_1);      \
    vst1q_f32(O + 0 * C_ob + 2 * FLOAT_SIMD, c_0_2);      \
    vst1q_f32(O + 0 * C_ob + 3 * FLOAT_SIMD, c_0_3);      \
    vst1q_f32(O + 1 * C_ob + 0 * FLOAT_SIMD, c_1_0);      \
    vst1q_f32(O + 1 * C_ob + 1 * FLOAT_SIMD, c_1_1);      \
    vst1q_f32(O + 1 * C_ob + 2 * FLOAT_SIMD, c_1_2);      \
    vst1q_f32(O + 1 * C_ob + 3 * FLOAT_SIMD, c_1_3);      \
    vst1q_f32(O + 2 * C_ob + 0 * FLOAT_SIMD, c_2_0);      \
    vst1q_f32(O + 2 * C_ob + 1 * FLOAT_SIMD, c_2_1);      \
    vst1q_f32(O + 2 * C_ob + 2 * FLOAT_SIMD, c_2_2);      \
    vst1q_f32(O + 2 * C_ob + 3 * FLOAT_SIMD, c_2_3);      \
    vst1q_f32(O + 3 * C_ob + 0 * FLOAT_SIMD, c_3_0);      \
    vst1q_f32(O + 3 * C_ob + 1 * FLOAT_SIMD, c_3_1);      \
    vst1q_f32(O + 3 * C_ob + 2 * FLOAT_SIMD, c_3_2);      \
    vst1q_f32(O + 3 * C_ob + 3 * FLOAT_SIMD, c_3_3);      \
    vst1q_f32(O + 4 * C_ob + 0 * FLOAT_SIMD, c_4_0);      \
    vst1q_f32(O + 4 * C_ob + 1 * FLOAT_SIMD, c_4_1);      \
    vst1q_f32(O + 4 * C_ob + 2 * FLOAT_SIMD, c_4_2);      \
    vst1q_f32(O + 4 * C_ob + 3 * FLOAT_SIMD, c_4_3);      \
  }                                                       \
  if constexpr (W_ob == 4 && C_ob == FLOAT_C_ob)     \
  {                                                       \
    vst1q_f32(O + 0 * C_ob + 0 * FLOAT_SIMD, c_0_0);      \
    vst1q_f32(O + 0 * C_ob + 1 * FLOAT_SIMD, c_0_1);      \
    vst1q_f32(O + 0 * C_ob + 2 * FLOAT_SIMD, c_0_2);      \
    vst1q_f32(O + 0 * C_ob + 3 * FLOAT_SIMD, c_0_3);      \
    vst1q_f32(O + 1 * C_ob + 0 * FLOAT_SIMD, c_1_0);      \
    vst1q_f32(O + 1 * C_ob + 1 * FLOAT_SIMD, c_1_1);      \
    vst1q_f32(O + 1 * C_ob + 2 * FLOAT_SIMD, c_1_2);      \
    vst1q_f32(O + 1 * C_ob + 3 * FLOAT_SIMD, c_1_3);      \
    vst1q_f32(O + 2 * C_ob + 0 * FLOAT_SIMD, c_2_0);      \
    vst1q_f32(O + 2 * C_ob + 1 * FLOAT_SIMD, c_2_1);      \
    vst1q_f32(O + 2 * C_ob + 2 * FLOAT_SIMD, c_2_2);      \
    vst1q_f32(O + 2 * C_ob + 3 * FLOAT_SIMD, c_2_3);      \
    vst1q_f32(O + 3 * C_ob + 0 * FLOAT_SIMD, c_3_0);      \
    vst1q_f32(O + 3 * C_ob + 1 * FLOAT_SIMD, c_3_1);      \
    vst1q_f32(O + 3 * C_ob + 2 * FLOAT_SIMD, c_3_2);      \
    vst1q_f32(O + 3 * C_ob + 3 * FLOAT_SIMD, c_3_3);      \
  }

#ifdef FLOAT_CONV_TILE_C
#undef FLOAT_CONV_TILE_C
#endif

#define FLOAT_CONV_TILE_C(step, a, b, W_ob, C_ob)                                      \
  if constexpr (_UNROLL == 1 && W_ob == FLOAT_W_ob && C_ob == FLOAT_C_ob)              \
  {                                                                                    \
    /*float const *aa = a;*/                                                           \
    float const *bb = b;                                                               \
    float32x4_t a_0;                                                                   \
    float32x4_t a_1;                                                                   \
    float32x4_t a_2;                                                                   \
    float32x4_t a_3;                                                                   \
    float32x4_t a_4;                                                                   \
    float32x4_t a_5;                                                                   \
    float32x4_t b_0;                                                                   \
    float32x4_t b_1;                                                                   \
    float32x4_t b_2;                                                                   \
    float32x4_t b_3;                                                                   \
    a_0 = vld1q_dup_f32(a + 0 * step + 0 * FLOAT_SIMD);                                \
    b_0 = vld1q_f32(bb + 0 * C_ob + (0 * 4 + 0) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    b_1 = vld1q_f32(bb + 0 * C_ob + (0 * 4 + 1) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    b_2 = vld1q_f32(bb + 0 * C_ob + (0 * 4 + 2) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_2) : "w"(b_2), "w"(a_0)); \
    b_3 = vld1q_f32(bb + 0 * C_ob + (0 * 4 + 3) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_3) : "w"(b_3), "w"(a_0)); \
    a_1 = vld1q_dup_f32(a + 1 * step + 0 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_2) : "w"(b_2), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_3) : "w"(b_3), "w"(a_1)); \
    a_2 = vld1q_dup_f32(a + 2 * step + 0 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_2) : "w"(b_2), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_3) : "w"(b_3), "w"(a_2)); \
    a_3 = vld1q_dup_f32(a + 3 * step + 0 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_2) : "w"(b_2), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_3) : "w"(b_3), "w"(a_3)); \
    a_4 = vld1q_dup_f32(a + 4 * step + 0 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_2) : "w"(b_2), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_3) : "w"(b_3), "w"(a_4)); \
    a_5 = vld1q_dup_f32(a + 5 * step + 0 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_5_0) : "w"(b_0), "w"(a_5)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_5_1) : "w"(b_1), "w"(a_5)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_5_2) : "w"(b_2), "w"(a_5)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_5_3) : "w"(b_3), "w"(a_5)); \
  }                                                                                    \
  else if constexpr (_UNROLL == 4 && W_ob == FLOAT_W_ob && C_ob == FLOAT_C_ob)         \
  {                                                                                    \
    /*float const *aa = a;*/                                                           \
    float const *bb = b;                                                               \
    float32x4_t a_0;                                                                   \
    float32x4_t a_1;                                                                   \
    float32x4_t a_2;                                                                   \
    float32x4_t a_3;                                                                   \
    float32x4_t a_4;                                                                   \
    float32x4_t a_5;                                                                   \
    float32x4_t b_0;                                                                   \
    float32x4_t b_1;                                                                   \
    a_0 = vld1q_f32(a + 0 * step + 0 * FLOAT_SIMD);                                    \
    b_0 = vld1q_f32(bb + 0 * C_ob + (0 * 4 + 0) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    b_1 = vld1q_f32(bb + 0 * C_ob + (0 * 4 + 1) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    a_1 = vld1q_f32(a + 1 * step + 0 * FLOAT_SIMD);                                    \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    a_2 = vld1q_f32(a + 2 * step + 0 * FLOAT_SIMD);                                    \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    a_3 = vld1q_f32(a + 3 * step + 0 * FLOAT_SIMD);                                    \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    a_4 = vld1q_f32(a + 4 * step + 0 * FLOAT_SIMD);                                    \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4)); \
    a_5 = vld1q_f32(a + 5 * step + 0 * FLOAT_SIMD);                                    \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_5_0) : "w"(b_0), "w"(a_5)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_5_1) : "w"(b_1), "w"(a_5)); \
    b_0 = vld1q_f32(bb + 0 * C_ob + (0 * 4 + 2) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    b_1 = vld1q_f32(bb + 0 * C_ob + (0 * 4 + 3) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_5_2) : "w"(b_0), "w"(a_5)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_5_3) : "w"(b_1), "w"(a_5)); \
                                                                                       \
    b_0 = vld1q_f32(bb + 1 * C_ob + (0 * 4 + 0) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    b_1 = vld1q_f32(bb + 1 * C_ob + (0 * 4 + 1) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_5_0) : "w"(b_0), "w"(a_5)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_5_1) : "w"(b_1), "w"(a_5)); \
    b_0 = vld1q_f32(bb + 1 * C_ob + (0 * 4 + 2) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    b_1 = vld1q_f32(bb + 1 * C_ob + (0 * 4 + 3) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_5_2) : "w"(b_0), "w"(a_5)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_5_3) : "w"(b_1), "w"(a_5)); \
                                                                                       \
    b_0 = vld1q_f32(bb + 2 * C_ob + (0 * 4 + 0) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    b_1 = vld1q_f32(bb + 2 * C_ob + (0 * 4 + 1) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_5_0) : "w"(b_0), "w"(a_5)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_5_1) : "w"(b_1), "w"(a_5)); \
    b_0 = vld1q_f32(bb + 2 * C_ob + (0 * 4 + 2) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    b_1 = vld1q_f32(bb + 2 * C_ob + (0 * 4 + 3) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_5_2) : "w"(b_0), "w"(a_5)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_5_3) : "w"(b_1), "w"(a_5)); \
                                                                                       \
    b_0 = vld1q_f32(bb + 3 * C_ob + (0 * 4 + 0) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    b_1 = vld1q_f32(bb + 3 * C_ob + (0 * 4 + 1) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_5_0) : "w"(b_0), "w"(a_5)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_5_1) : "w"(b_1), "w"(a_5)); \
    b_0 = vld1q_f32(bb + 3 * C_ob + (0 * 4 + 2) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    b_1 = vld1q_f32(bb + 3 * C_ob + (0 * 4 + 3) * FLOAT_SIMD);                         \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_5_2) : "w"(b_0), "w"(a_5)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_5_3) : "w"(b_1), "w"(a_5)); \
  }                                                                                    \
  else if constexpr (_UNROLL == 4 && W_ob == 5 && C_ob == FLOAT_C_ob)                  \
  {                                                                                    \
    float32x4_t a_0;                                                                   \
    float32x4_t a_1;                                                                   \
    float32x4_t a_2;                                                                   \
    float32x4_t a_3;                                                                   \
    float32x4_t a_4;                                                                   \
    float32x4_t b_0;                                                                   \
    float32x4_t b_1;                                                                   \
    a_0 = vld1q_f32(a + 0 * step + 0);                                                 \
    a_1 = vld1q_f32(a + 1 * step + 0);                                                 \
    a_2 = vld1q_f32(a + 2 * step + 0);                                                 \
    a_3 = vld1q_f32(a + 3 * step + 0);                                                 \
    a_4 = vld1q_f32(a + 4 * step + 0);                                                 \
    b_0 = vld1q_f32(b + 0 * C_ob + 0 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 0 * C_ob + 0 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 0 * C_ob + 8 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 0 * C_ob + 8 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 1 * C_ob + 0 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 1 * C_ob + 0 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 1 * C_ob + 8 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 1 * C_ob + 8 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 2 * C_ob + 0 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 2 * C_ob + 0 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 2 * C_ob + 8 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 2 * C_ob + 8 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 3 * C_ob + 0 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 3 * C_ob + 0 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 3 * C_ob + 8 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 3 * C_ob + 8 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4)); \
  }                                                                                    \
  else if constexpr (_UNROLL == 4 && W_ob == 5 && C_ob == 16)                          \
  {                                                                                    \
    float32x4_t a_0;                                                                   \
    float32x4_t a_1;                                                                   \
    float32x4_t a_2;                                                                   \
    float32x4_t a_3;                                                                   \
    float32x4_t a_4;                                                                   \
    float32x4_t b_0;                                                                   \
    float32x4_t b_1;                                                                   \
    a_0 = vld1q_f32(a + 0 * step + 0);                                                 \
    a_1 = vld1q_f32(a + 1 * step + 0);                                                 \
    a_2 = vld1q_f32(a + 2 * step + 0);                                                 \
    a_3 = vld1q_f32(a + 3 * step + 0);                                                 \
    a_4 = vld1q_f32(a + 4 * step + 0);                                                 \
    b_0 = vld1q_f32(b + 0 * C_ob + 0 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 0 * C_ob + 0 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 0 * C_ob + 8 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 0 * C_ob + 8 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 1 * C_ob + 0 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 1 * C_ob + 0 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 1 * C_ob + 8 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 1 * C_ob + 8 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 2 * C_ob + 0 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 2 * C_ob + 0 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 2 * C_ob + 8 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 2 * C_ob + 8 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 3 * C_ob + 0 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 3 * C_ob + 0 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4)); \
    b_0 = vld1q_f32(b + 3 * C_ob + 8 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 3 * C_ob + 8 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4)); \
  }                                                                                    \
  if constexpr (_UNROLL == 4 && W_ob == 4 && C_ob == FLOAT_C_ob)                       \
  {                                                                                    \
    float32x4_t a_0;                                                                   \
    float32x4_t a_1;                                                                   \
    float32x4_t a_2;                                                                   \
    float32x4_t a_3;                                                                   \
    float32x4_t b_0;                                                                   \
    float32x4_t b_1;                                                                   \
    a_0 = vld1q_f32(a + 0 * step + 0);                                             \
    a_1 = vld1q_f32(a + 1 * step + 0);                                             \
    a_2 = vld1q_f32(a + 2 * step + 0);                                             \
    a_3 = vld1q_f32(a + 3 * step + 0);                                             \
    b_0 = vld1q_f32(b + 0 * C_ob + 0 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 0 * C_ob + 0 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    b_0 = vld1q_f32(b + 0 * C_ob + 8 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 0 * C_ob + 8 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    b_0 = vld1q_f32(b + 1 * C_ob + 0 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 1 * C_ob + 0 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    b_0 = vld1q_f32(b + 1 * C_ob + 8 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 1 * C_ob + 8 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    b_0 = vld1q_f32(b + 2 * C_ob + 0 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 2 * C_ob + 0 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    b_0 = vld1q_f32(b + 2 * C_ob + 8 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 2 * C_ob + 8 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
    b_0 = vld1q_f32(b + 3 * C_ob + 0 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 3 * C_ob + 0 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3)); \
    b_0 = vld1q_f32(b + 3 * C_ob + 8 + 0 * FLOAT_SIMD);                                \
    b_1 = vld1q_f32(b + 3 * C_ob + 8 + 1 * FLOAT_SIMD);                                \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3)); \
    __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3)); \
  }

#ifdef FLOAT_MAX_TILE_C
#undef FLOAT_MAX_TILE_C
#endif

#define FLOAT_MAX_TILE_C(step, a, W_ob, C_ob)             \
  if constexpr (W_ob == FLOAT_W_ob && C_ob == FLOAT_C_ob) \
  {                                                       \
    float32x4_t av;                                       \
    av = vld1q_f32(a + 0 * step + 0 * FLOAT_SIMD);        \
    c_0_0 = vmaxq_f32(c_0_0, av);                         \
    av = vld1q_f32(a + 0 * step + 1 * FLOAT_SIMD);        \
    c_0_1 = vmaxq_f32(c_0_1, av);                         \
    av = vld1q_f32(a + 0 * step + 2 * FLOAT_SIMD);        \
    c_0_2 = vmaxq_f32(c_0_2, av);                         \
    av = vld1q_f32(a + 0 * step + 3 * FLOAT_SIMD);        \
    c_0_3 = vmaxq_f32(c_0_3, av);                         \
    av = vld1q_f32(a + 1 * step + 0 * FLOAT_SIMD);        \
    c_1_0 = vmaxq_f32(c_1_0, av);                         \
    av = vld1q_f32(a + 1 * step + 1 * FLOAT_SIMD);        \
    c_1_1 = vmaxq_f32(c_1_1, av);                         \
    av = vld1q_f32(a + 1 * step + 2 * FLOAT_SIMD);        \
    c_1_2 = vmaxq_f32(c_1_2, av);                         \
    av = vld1q_f32(a + 1 * step + 3 * FLOAT_SIMD);        \
    c_1_3 = vmaxq_f32(c_1_3, av);                         \
    av = vld1q_f32(a + 2 * step + 0 * FLOAT_SIMD);        \
    c_2_0 = vmaxq_f32(c_2_0, av);                         \
    av = vld1q_f32(a + 2 * step + 1 * FLOAT_SIMD);        \
    c_2_1 = vmaxq_f32(c_2_1, av);                         \
    av = vld1q_f32(a + 2 * step + 2 * FLOAT_SIMD);        \
    c_2_2 = vmaxq_f32(c_2_2, av);                         \
    av = vld1q_f32(a + 2 * step + 3 * FLOAT_SIMD);        \
    c_2_3 = vmaxq_f32(c_2_3, av);                         \
    av = vld1q_f32(a + 3 * step + 0 * FLOAT_SIMD);        \
    c_3_0 = vmaxq_f32(c_3_0, av);                         \
    av = vld1q_f32(a + 3 * step + 1 * FLOAT_SIMD);        \
    c_3_1 = vmaxq_f32(c_3_1, av);                         \
    av = vld1q_f32(a + 3 * step + 2 * FLOAT_SIMD);        \
    c_3_2 = vmaxq_f32(c_3_2, av);                         \
    av = vld1q_f32(a + 3 * step + 3 * FLOAT_SIMD);        \
    c_3_3 = vmaxq_f32(c_3_3, av);                         \
    av = vld1q_f32(a + 4 * step + 0 * FLOAT_SIMD);        \
    c_4_0 = vmaxq_f32(c_4_0, av);                         \
    av = vld1q_f32(a + 4 * step + 1 * FLOAT_SIMD);        \
    c_4_1 = vmaxq_f32(c_4_1, av);                         \
    av = vld1q_f32(a + 4 * step + 2 * FLOAT_SIMD);        \
    c_4_2 = vmaxq_f32(c_4_2, av);                         \
    av = vld1q_f32(a + 4 * step + 3 * FLOAT_SIMD);        \
    c_4_3 = vmaxq_f32(c_4_3, av);                         \
    av = vld1q_f32(a + 5 * step + 0 * FLOAT_SIMD);        \
    c_5_0 = vmaxq_f32(c_5_0, av);                         \
    av = vld1q_f32(a + 5 * step + 1 * FLOAT_SIMD);        \
    c_5_1 = vmaxq_f32(c_5_1, av);                         \
    av = vld1q_f32(a + 5 * step + 2 * FLOAT_SIMD);        \
    c_5_2 = vmaxq_f32(c_5_2, av);                         \
    av = vld1q_f32(a + 5 * step + 3 * FLOAT_SIMD);        \
    c_5_3 = vmaxq_f32(c_5_3, av);                         \
  }                                                       \
  if constexpr (W_ob == 5 && C_ob == FLOAT_C_ob)          \
  {                                                       \
    float32x4_t av;                                       \
    av = vld1q_f32(a + 0 * step + 0 * FLOAT_SIMD);        \
    c_0_0 = vmaxq_f32(c_0_0, av);                         \
    av = vld1q_f32(a + 0 * step + 1 * FLOAT_SIMD);        \
    c_0_1 = vmaxq_f32(c_0_1, av);                         \
    av = vld1q_f32(a + 0 * step + 2 * FLOAT_SIMD);        \
    c_0_2 = vmaxq_f32(c_0_2, av);                         \
    av = vld1q_f32(a + 0 * step + 3 * FLOAT_SIMD);        \
    c_0_3 = vmaxq_f32(c_0_3, av);                         \
    av = vld1q_f32(a + 1 * step + 0 * FLOAT_SIMD);        \
    c_1_0 = vmaxq_f32(c_1_0, av);                         \
    av = vld1q_f32(a + 1 * step + 1 * FLOAT_SIMD);        \
    c_1_1 = vmaxq_f32(c_1_1, av);                         \
    av = vld1q_f32(a + 1 * step + 2 * FLOAT_SIMD);        \
    c_1_2 = vmaxq_f32(c_1_2, av);                         \
    av = vld1q_f32(a + 1 * step + 3 * FLOAT_SIMD);        \
    c_1_3 = vmaxq_f32(c_1_3, av);                         \
    av = vld1q_f32(a + 2 * step + 0 * FLOAT_SIMD);        \
    c_2_0 = vmaxq_f32(c_2_0, av);                         \
    av = vld1q_f32(a + 2 * step + 1 * FLOAT_SIMD);        \
    c_2_1 = vmaxq_f32(c_2_1, av);                         \
    av = vld1q_f32(a + 2 * step + 2 * FLOAT_SIMD);        \
    c_2_2 = vmaxq_f32(c_2_2, av);                         \
    av = vld1q_f32(a + 2 * step + 3 * FLOAT_SIMD);        \
    c_2_3 = vmaxq_f32(c_2_3, av);                         \
    av = vld1q_f32(a + 3 * step + 0 * FLOAT_SIMD);        \
    c_3_0 = vmaxq_f32(c_3_0, av);                         \
    av = vld1q_f32(a + 3 * step + 1 * FLOAT_SIMD);        \
    c_3_1 = vmaxq_f32(c_3_1, av);                         \
    av = vld1q_f32(a + 3 * step + 2 * FLOAT_SIMD);        \
    c_3_2 = vmaxq_f32(c_3_2, av);                         \
    av = vld1q_f32(a + 3 * step + 3 * FLOAT_SIMD);        \
    c_3_3 = vmaxq_f32(c_3_3, av);                         \
    av = vld1q_f32(a + 4 * step + 0 * FLOAT_SIMD);        \
    c_4_0 = vmaxq_f32(c_4_0, av);                         \
    av = vld1q_f32(a + 4 * step + 1 * FLOAT_SIMD);        \
    c_4_1 = vmaxq_f32(c_4_1, av);                         \
    av = vld1q_f32(a + 4 * step + 2 * FLOAT_SIMD);        \
    c_4_2 = vmaxq_f32(c_4_2, av);                         \
    av = vld1q_f32(a + 4 * step + 3 * FLOAT_SIMD);        \
    c_4_3 = vmaxq_f32(c_4_3, av);                         \
  }                                                       \
  if constexpr (W_ob == 4 && C_ob == FLOAT_C_ob)          \
  {                                                       \
    float32x4_t av;                                       \
    av = vld1q_f32(a + 0 * step + 0 * FLOAT_SIMD);        \
    c_0_0 = vmaxq_f32(c_0_0, av);                         \
    av = vld1q_f32(a + 0 * step + 1 * FLOAT_SIMD);        \
    c_0_1 = vmaxq_f32(c_0_1, av);                         \
    av = vld1q_f32(a + 0 * step + 2 * FLOAT_SIMD);        \
    c_0_2 = vmaxq_f32(c_0_2, av);                         \
    av = vld1q_f32(a + 0 * step + 3 * FLOAT_SIMD);        \
    c_0_3 = vmaxq_f32(c_0_3, av);                         \
    av = vld1q_f32(a + 1 * step + 0 * FLOAT_SIMD);        \
    c_1_0 = vmaxq_f32(c_1_0, av);                         \
    av = vld1q_f32(a + 1 * step + 1 * FLOAT_SIMD);        \
    c_1_1 = vmaxq_f32(c_1_1, av);                         \
    av = vld1q_f32(a + 1 * step + 2 * FLOAT_SIMD);        \
    c_1_2 = vmaxq_f32(c_1_2, av);                         \
    av = vld1q_f32(a + 1 * step + 3 * FLOAT_SIMD);        \
    c_1_3 = vmaxq_f32(c_1_3, av);                         \
    av = vld1q_f32(a + 2 * step + 0 * FLOAT_SIMD);        \
    c_2_0 = vmaxq_f32(c_2_0, av);                         \
    av = vld1q_f32(a + 2 * step + 1 * FLOAT_SIMD);        \
    c_2_1 = vmaxq_f32(c_2_1, av);                         \
    av = vld1q_f32(a + 2 * step + 2 * FLOAT_SIMD);        \
    c_2_2 = vmaxq_f32(c_2_2, av);                         \
    av = vld1q_f32(a + 2 * step + 3 * FLOAT_SIMD);        \
    c_2_3 = vmaxq_f32(c_2_3, av);                         \
    av = vld1q_f32(a + 3 * step + 0 * FLOAT_SIMD);        \
    c_3_0 = vmaxq_f32(c_3_0, av);                         \
    av = vld1q_f32(a + 3 * step + 1 * FLOAT_SIMD);        \
    c_3_1 = vmaxq_f32(c_3_1, av);                         \
    av = vld1q_f32(a + 3 * step + 2 * FLOAT_SIMD);        \
    c_3_2 = vmaxq_f32(c_3_2, av);                         \
    av = vld1q_f32(a + 3 * step + 3 * FLOAT_SIMD);        \
    c_3_3 = vmaxq_f32(c_3_3, av);                         \
  }

#ifdef FLOAT_DW_TILE_C
#undef FLOAT_DW_TILE_C
#endif

#define FLOAT_DW_TILE_C(step, a, b, W_ob, C_ob)           \
  if constexpr (W_ob == FLOAT_W_ob && C_ob == FLOAT_C_ob) \
  {                                                       \
    float32x4_t av;                                       \
    float32x4_t b_0 = vld1q_f32(b + 0 * FLOAT_SIMD);      \
    float32x4_t b_1 = vld1q_f32(b + 1 * FLOAT_SIMD);      \
    float32x4_t b_2 = vld1q_f32(b + 2 * FLOAT_SIMD);      \
    float32x4_t b_3 = vld1q_f32(b + 3 * FLOAT_SIMD);      \
    av = vld1q_f32(a + 0 * step + 0 * FLOAT_SIMD);        \
    c_0_0 = vfmaq_f32(c_0_0, av, b_0);                    \
    av = vld1q_f32(a + 0 * step + 1 * FLOAT_SIMD);        \
    c_0_1 = vfmaq_f32(c_0_1, av, b_1);                    \
    av = vld1q_f32(a + 0 * step + 2 * FLOAT_SIMD);        \
    c_0_2 = vfmaq_f32(c_0_2, av, b_2);                    \
    av = vld1q_f32(a + 0 * step + 3 * FLOAT_SIMD);        \
    c_0_3 = vfmaq_f32(c_0_3, av, b_3);                    \
    av = vld1q_f32(a + 1 * step + 0 * FLOAT_SIMD);        \
    c_1_0 = vfmaq_f32(c_1_0, av, b_0);                    \
    av = vld1q_f32(a + 1 * step + 1 * FLOAT_SIMD);        \
    c_1_1 = vfmaq_f32(c_1_1, av, b_1);                    \
    av = vld1q_f32(a + 1 * step + 2 * FLOAT_SIMD);        \
    c_1_2 = vfmaq_f32(c_1_2, av, b_2);                    \
    av = vld1q_f32(a + 1 * step + 3 * FLOAT_SIMD);        \
    c_1_3 = vfmaq_f32(c_1_3, av, b_3);                    \
    av = vld1q_f32(a + 2 * step + 0 * FLOAT_SIMD);        \
    c_2_0 = vfmaq_f32(c_2_0, av, b_0);                    \
    av = vld1q_f32(a + 2 * step + 1 * FLOAT_SIMD);        \
    c_2_1 = vfmaq_f32(c_2_1, av, b_1);                    \
    av = vld1q_f32(a + 2 * step + 2 * FLOAT_SIMD);        \
    c_2_2 = vfmaq_f32(c_2_2, av, b_2);                    \
    av = vld1q_f32(a + 2 * step + 3 * FLOAT_SIMD);        \
    c_2_3 = vfmaq_f32(c_2_3, av, b_3);                    \
    av = vld1q_f32(a + 3 * step + 0 * FLOAT_SIMD);        \
    c_3_0 = vfmaq_f32(c_3_0, av, b_0);                    \
    av = vld1q_f32(a + 3 * step + 1 * FLOAT_SIMD);        \
    c_3_1 = vfmaq_f32(c_3_1, av, b_1);                    \
    av = vld1q_f32(a + 3 * step + 2 * FLOAT_SIMD);        \
    c_3_2 = vfmaq_f32(c_3_2, av, b_2);                    \
    av = vld1q_f32(a + 3 * step + 3 * FLOAT_SIMD);        \
    c_3_3 = vfmaq_f32(c_3_3, av, b_3);                    \
    av = vld1q_f32(a + 4 * step + 0 * FLOAT_SIMD);        \
    c_4_0 = vfmaq_f32(c_4_0, av, b_0);                    \
    av = vld1q_f32(a + 4 * step + 1 * FLOAT_SIMD);        \
    c_4_1 = vfmaq_f32(c_4_1, av, b_1);                    \
    av = vld1q_f32(a + 4 * step + 2 * FLOAT_SIMD);        \
    c_4_2 = vfmaq_f32(c_4_2, av, b_2);                    \
    av = vld1q_f32(a + 4 * step + 3 * FLOAT_SIMD);        \
    c_4_3 = vfmaq_f32(c_4_3, av, b_3);                    \
    av = vld1q_f32(a + 5 * step + 0 * FLOAT_SIMD);        \
    c_5_0 = vfmaq_f32(c_5_0, av, b_0);                    \
    av = vld1q_f32(a + 5 * step + 1 * FLOAT_SIMD);        \
    c_5_1 = vfmaq_f32(c_5_1, av, b_1);                    \
    av = vld1q_f32(a + 5 * step + 2 * FLOAT_SIMD);        \
    c_5_2 = vfmaq_f32(c_5_2, av, b_2);                    \
    av = vld1q_f32(a + 5 * step + 3 * FLOAT_SIMD);        \
    c_5_3 = vfmaq_f32(c_5_3, av, b_3);                    \
  }                                                       \
  if constexpr (W_ob == 5 && C_ob == FLOAT_C_ob)          \
  {                                                       \
    float32x4_t av;                                       \
    float32x4_t b_0 = vld1q_f32(b + 0 * FLOAT_SIMD);      \
    float32x4_t b_1 = vld1q_f32(b + 1 * FLOAT_SIMD);      \
    float32x4_t b_2 = vld1q_f32(b + 2 * FLOAT_SIMD);      \
    float32x4_t b_3 = vld1q_f32(b + 3 * FLOAT_SIMD);      \
    av = vld1q_f32(a + 0 * step + 0 * FLOAT_SIMD);        \
    c_0_0 = vfmaq_f32(c_0_0, av, b_0);                    \
    av = vld1q_f32(a + 0 * step + 1 * FLOAT_SIMD);        \
    c_0_1 = vfmaq_f32(c_0_1, av, b_1);                    \
    av = vld1q_f32(a + 0 * step + 2 * FLOAT_SIMD);        \
    c_0_2 = vfmaq_f32(c_0_2, av, b_2);                    \
    av = vld1q_f32(a + 0 * step + 3 * FLOAT_SIMD);        \
    c_0_3 = vfmaq_f32(c_0_3, av, b_3);                    \
    av = vld1q_f32(a + 1 * step + 0 * FLOAT_SIMD);        \
    c_1_0 = vfmaq_f32(c_1_0, av, b_0);                    \
    av = vld1q_f32(a + 1 * step + 1 * FLOAT_SIMD);        \
    c_1_1 = vfmaq_f32(c_1_1, av, b_1);                    \
    av = vld1q_f32(a + 1 * step + 2 * FLOAT_SIMD);        \
    c_1_2 = vfmaq_f32(c_1_2, av, b_2);                    \
    av = vld1q_f32(a + 1 * step + 3 * FLOAT_SIMD);        \
    c_1_3 = vfmaq_f32(c_1_3, av, b_3);                    \
    av = vld1q_f32(a + 2 * step + 0 * FLOAT_SIMD);        \
    c_2_0 = vfmaq_f32(c_2_0, av, b_0);                    \
    av = vld1q_f32(a + 2 * step + 1 * FLOAT_SIMD);        \
    c_2_1 = vfmaq_f32(c_2_1, av, b_1);                    \
    av = vld1q_f32(a + 2 * step + 2 * FLOAT_SIMD);        \
    c_2_2 = vfmaq_f32(c_2_2, av, b_2);                    \
    av = vld1q_f32(a + 2 * step + 3 * FLOAT_SIMD);        \
    c_2_3 = vfmaq_f32(c_2_3, av, b_3);                    \
    av = vld1q_f32(a + 3 * step + 0 * FLOAT_SIMD);        \
    c_3_0 = vfmaq_f32(c_3_0, av, b_0);                    \
    av = vld1q_f32(a + 3 * step + 1 * FLOAT_SIMD);        \
    c_3_1 = vfmaq_f32(c_3_1, av, b_1);                    \
    av = vld1q_f32(a + 3 * step + 2 * FLOAT_SIMD);        \
    c_3_2 = vfmaq_f32(c_3_2, av, b_2);                    \
    av = vld1q_f32(a + 3 * step + 3 * FLOAT_SIMD);        \
    c_3_3 = vfmaq_f32(c_3_3, av, b_3);                    \
    av = vld1q_f32(a + 4 * step + 0 * FLOAT_SIMD);        \
    c_4_0 = vfmaq_f32(c_4_0, av, b_0);                    \
    av = vld1q_f32(a + 4 * step + 1 * FLOAT_SIMD);        \
    c_4_1 = vfmaq_f32(c_4_1, av, b_1);                    \
    av = vld1q_f32(a + 4 * step + 2 * FLOAT_SIMD);        \
    c_4_2 = vfmaq_f32(c_4_2, av, b_2);                    \
    av = vld1q_f32(a + 4 * step + 3 * FLOAT_SIMD);        \
    c_4_3 = vfmaq_f32(c_4_3, av, b_3);                    \
  }                                                       \
  if constexpr (W_ob == 4 && C_ob == FLOAT_C_ob)          \
  {                                                       \
    float32x4_t av;                                       \
    float32x4_t b_0 = vld1q_f32(b + 0 * FLOAT_SIMD);      \
    float32x4_t b_1 = vld1q_f32(b + 1 * FLOAT_SIMD);      \
    float32x4_t b_2 = vld1q_f32(b + 2 * FLOAT_SIMD);      \
    float32x4_t b_3 = vld1q_f32(b + 3 * FLOAT_SIMD);      \
    av = vld1q_f32(a + 0 * step + 0 * FLOAT_SIMD);        \
    c_0_0 = vfmaq_f32(c_0_0, av, b_0);                    \
    av = vld1q_f32(a + 0 * step + 1 * FLOAT_SIMD);        \
    c_0_1 = vfmaq_f32(c_0_1, av, b_1);                    \
    av = vld1q_f32(a + 0 * step + 2 * FLOAT_SIMD);        \
    c_0_2 = vfmaq_f32(c_0_2, av, b_2);                    \
    av = vld1q_f32(a + 0 * step + 3 * FLOAT_SIMD);        \
    c_0_3 = vfmaq_f32(c_0_3, av, b_3);                    \
    av = vld1q_f32(a + 1 * step + 0 * FLOAT_SIMD);        \
    c_1_0 = vfmaq_f32(c_1_0, av, b_0);                    \
    av = vld1q_f32(a + 1 * step + 1 * FLOAT_SIMD);        \
    c_1_1 = vfmaq_f32(c_1_1, av, b_1);                    \
    av = vld1q_f32(a + 1 * step + 2 * FLOAT_SIMD);        \
    c_1_2 = vfmaq_f32(c_1_2, av, b_2);                    \
    av = vld1q_f32(a + 1 * step + 3 * FLOAT_SIMD);        \
    c_1_3 = vfmaq_f32(c_1_3, av, b_3);                    \
    av = vld1q_f32(a + 2 * step + 0 * FLOAT_SIMD);        \
    c_2_0 = vfmaq_f32(c_2_0, av, b_0);                    \
    av = vld1q_f32(a + 2 * step + 1 * FLOAT_SIMD);        \
    c_2_1 = vfmaq_f32(c_2_1, av, b_1);                    \
    av = vld1q_f32(a + 2 * step + 2 * FLOAT_SIMD);        \
    c_2_2 = vfmaq_f32(c_2_2, av, b_2);                    \
    av = vld1q_f32(a + 2 * step + 3 * FLOAT_SIMD);        \
    c_2_3 = vfmaq_f32(c_2_3, av, b_3);                    \
    av = vld1q_f32(a + 3 * step + 0 * FLOAT_SIMD);        \
    c_3_0 = vfmaq_f32(c_3_0, av, b_0);                    \
    av = vld1q_f32(a + 3 * step + 1 * FLOAT_SIMD);        \
    c_3_1 = vfmaq_f32(c_3_1, av, b_1);                    \
    av = vld1q_f32(a + 3 * step + 2 * FLOAT_SIMD);        \
    c_3_2 = vfmaq_f32(c_3_2, av, b_2);                    \
    av = vld1q_f32(a + 3 * step + 3 * FLOAT_SIMD);        \
    c_3_3 = vfmaq_f32(c_3_3, av, b_3);                    \
  }