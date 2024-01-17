
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

#define SMALL_HAS_FLOAT_SUPPORT  1

#include <arm_neon.h>
#ifdef FLOAT_DEF_TILE_C
#undef FLOAT_DEF_TILE_C
#endif

#define FLOAT_DEF_TILE_C(W_ob, C_ob)\
float c_tile[W_ob * C_ob];\
float32x4_t c_0_0;\
float32x4_t c_0_1;\
float32x4_t c_1_0;\
float32x4_t c_1_1;\
float32x4_t c_2_0;\
float32x4_t c_2_1;\
float32x4_t c_3_0;\
float32x4_t c_3_1;\
float32x4_t c_4_0;\
float32x4_t c_4_1;\
float32x4_t c_5_0;\
float32x4_t c_5_1;\

#ifdef FLOAT_ZERO_TILE_C
#undef FLOAT_ZERO_TILE_C
#endif

#define FLOAT_ZERO_TILE_C(W_ob, C_ob)\
c_0_0 = vdupq_n_f32(0);\
c_0_1 = vdupq_n_f32(0);\
c_1_0 = vdupq_n_f32(0);\
c_1_1 = vdupq_n_f32(0);\
c_2_0 = vdupq_n_f32(0);\
c_2_1 = vdupq_n_f32(0);\
c_3_0 = vdupq_n_f32(0);\
c_3_1 = vdupq_n_f32(0);\
c_4_0 = vdupq_n_f32(0);\
c_4_1 = vdupq_n_f32(0);\
c_5_0 = vdupq_n_f32(0);\
c_5_1 = vdupq_n_f32(0);\

#ifdef FLOAT_LOAD_TILE_C
#undef FLOAT_LOAD_TILE_C
#endif

#define FLOAT_LOAD_TILE_C(O, W_ob, C_ob)\
c_0_0 = vld1q_f32(O + 0 * C_ob + 0 * SIMD);\
c_0_1 = vld1q_f32(O + 0 * C_ob + 1 * SIMD);\
c_1_0 = vld1q_f32(O + 1 * C_ob + 0 * SIMD);\
c_1_1 = vld1q_f32(O + 1 * C_ob + 1 * SIMD);\
c_2_0 = vld1q_f32(O + 2 * C_ob + 0 * SIMD);\
c_2_1 = vld1q_f32(O + 2 * C_ob + 1 * SIMD);\
c_3_0 = vld1q_f32(O + 3 * C_ob + 0 * SIMD);\
c_3_1 = vld1q_f32(O + 3 * C_ob + 1 * SIMD);\
c_4_0 = vld1q_f32(O + 4 * C_ob + 0 * SIMD);\
c_4_1 = vld1q_f32(O + 4 * C_ob + 1 * SIMD);\
c_5_0 = vld1q_f32(O + 5 * C_ob + 0 * SIMD);\
c_5_1 = vld1q_f32(O + 5 * C_ob + 1 * SIMD);\

#ifdef FLOAT_LOAD_TILE_C_strided
#undef FLOAT_LOAD_TILE_C_strided
#endif

#define FLOAT_LOAD_TILE_C_strided(O, step, W_ob, C_ob)\
c_0_0 = vld1q_f32(O + 0 * step + 0 * SIMD);\
c_0_1 = vld1q_f32(O + 0 * step + 1 * SIMD);\
c_1_0 = vld1q_f32(O + 1 * step + 0 * SIMD);\
c_1_1 = vld1q_f32(O + 1 * step + 1 * SIMD);\
c_2_0 = vld1q_f32(O + 2 * step + 0 * SIMD);\
c_2_1 = vld1q_f32(O + 2 * step + 1 * SIMD);\
c_3_0 = vld1q_f32(O + 3 * step + 0 * SIMD);\
c_3_1 = vld1q_f32(O + 3 * step + 1 * SIMD);\
c_4_0 = vld1q_f32(O + 4 * step + 0 * SIMD);\
c_4_1 = vld1q_f32(O + 4 * step + 1 * SIMD);\
c_5_0 = vld1q_f32(O + 5 * step + 0 * SIMD);\
c_5_1 = vld1q_f32(O + 5 * step + 1 * SIMD);\

#ifdef STORE_TILE_C
#undef STORE_TILE_C
#endif

#define STORE_TILE_C(O, W_ob, C_ob)\
vst1q_f32(O + 0 * C_ob + 0 * SIMD, c_0_0);\
vst1q_f32(O + 0 * C_ob + 1 * SIMD, c_0_1);\
vst1q_f32(O + 1 * C_ob + 0 * SIMD, c_1_0);\
vst1q_f32(O + 1 * C_ob + 1 * SIMD, c_1_1);\
vst1q_f32(O + 2 * C_ob + 0 * SIMD, c_2_0);\
vst1q_f32(O + 2 * C_ob + 1 * SIMD, c_2_1);\
vst1q_f32(O + 3 * C_ob + 0 * SIMD, c_3_0);\
vst1q_f32(O + 3 * C_ob + 1 * SIMD, c_3_1);\
vst1q_f32(O + 4 * C_ob + 0 * SIMD, c_4_0);\
vst1q_f32(O + 4 * C_ob + 1 * SIMD, c_4_1);\
vst1q_f32(O + 5 * C_ob + 0 * SIMD, c_5_0);\
vst1q_f32(O + 5 * C_ob + 1 * SIMD, c_5_1);\

#ifdef CONV_TILE_C
#undef CONV_TILE_C
#endif

#define CONV_TILE_C(step, a, b, W_ob, C_ob)\
float *aa = a;\
float *bb = b;\
float32x4_t a_0;\
float32x4_t a_1;\
float32x4_t a_2;\
float32x4_t a_3;\
float32x4_t a_4;\
float32x4_t a_5;\
float32x4_t b_0;\
float32x4_t b_1;\

#ifdef MAX_TILE_C
#undef MAX_TILE_C
#endif

#define MAX_TILE_C(step, a, W_ob, C_ob)\
float32x4_t av; \
av = vld1q_f32(a + 0 * step + 0 * SIMD);\
c_0_0 = vmaxq_f32(c_0_0, av);\
av = vld1q_f32(a + 0 * step + 1 * SIMD);\
c_0_1 = vmaxq_f32(c_0_1, av);\
av = vld1q_f32(a + 1 * step + 0 * SIMD);\
c_1_0 = vmaxq_f32(c_1_0, av);\
av = vld1q_f32(a + 1 * step + 1 * SIMD);\
c_1_1 = vmaxq_f32(c_1_1, av);\
av = vld1q_f32(a + 2 * step + 0 * SIMD);\
c_2_0 = vmaxq_f32(c_2_0, av);\
av = vld1q_f32(a + 2 * step + 1 * SIMD);\
c_2_1 = vmaxq_f32(c_2_1, av);\
av = vld1q_f32(a + 3 * step + 0 * SIMD);\
c_3_0 = vmaxq_f32(c_3_0, av);\
av = vld1q_f32(a + 3 * step + 1 * SIMD);\
c_3_1 = vmaxq_f32(c_3_1, av);\
av = vld1q_f32(a + 4 * step + 0 * SIMD);\
c_4_0 = vmaxq_f32(c_4_0, av);\
av = vld1q_f32(a + 4 * step + 1 * SIMD);\
c_4_1 = vmaxq_f32(c_4_1, av);\
av = vld1q_f32(a + 5 * step + 0 * SIMD);\
c_5_0 = vmaxq_f32(c_5_0, av);\
av = vld1q_f32(a + 5 * step + 1 * SIMD);\
c_5_1 = vmaxq_f32(c_5_1, av);\

#ifdef DW_TILE_C
#undef DW_TILE_C
#endif

#define DW_TILE_C(step, a, b, W_ob, C_ob)\
float32x4_t av; \
float32x4_t b_0 = vld1q_f32(b + 0*SIMD);\
float32x4_t b_1 = vld1q_f32(b + 1*SIMD);\
av = vld1q_f32(a + 0 * step + 0 * SIMD);\
c_0_0 = vfmaq_f32(c_0_0, av, b_0);\
av = vld1q_f32(a + 0 * step + 1 * SIMD);\
c_0_1 = vfmaq_f32(c_0_1, av, b_1);\
av = vld1q_f32(a + 1 * step + 0 * SIMD);\
c_1_0 = vfmaq_f32(c_1_0, av, b_0);\
av = vld1q_f32(a + 1 * step + 1 * SIMD);\
c_1_1 = vfmaq_f32(c_1_1, av, b_1);\
av = vld1q_f32(a + 2 * step + 0 * SIMD);\
c_2_0 = vfmaq_f32(c_2_0, av, b_0);\
av = vld1q_f32(a + 2 * step + 1 * SIMD);\
c_2_1 = vfmaq_f32(c_2_1, av, b_1);\
av = vld1q_f32(a + 3 * step + 0 * SIMD);\
c_3_0 = vfmaq_f32(c_3_0, av, b_0);\
av = vld1q_f32(a + 3 * step + 1 * SIMD);\
c_3_1 = vfmaq_f32(c_3_1, av, b_1);\
av = vld1q_f32(a + 4 * step + 0 * SIMD);\
c_4_0 = vfmaq_f32(c_4_0, av, b_0);\
av = vld1q_f32(a + 4 * step + 1 * SIMD);\
c_4_1 = vfmaq_f32(c_4_1, av, b_1);\
av = vld1q_f32(a + 5 * step + 0 * SIMD);\
c_5_0 = vfmaq_f32(c_5_0, av, b_0);\
av = vld1q_f32(a + 5 * step + 1 * SIMD);\
c_5_1 = vfmaq_f32(c_5_1, av, b_1);\

