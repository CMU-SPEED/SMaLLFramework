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
#endif

#define FLOAT_CONV_TILE_C(W_stride, a, b, W_ob, C_ob)

if (_UNROLL == 1 && W_ob == FLOAT_W_ob && C_ob == FLOAT_C_ob)

{

  float const *bb = b;
  float32x4_t a_0;
  float32x4_t a_1;

  /* ... */
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_5_3) : "w"(b_3), "w"(a_5));
}
else if (_UNROLL == 4 && W_ob == FLOAT_W_ob && C_ob == FLOAT_C_ob)
{ /* ... */
  b_1 = vld1q_f32(bb + 3 * C_ob + (0 * 4 + 3) * FLOAT_SIMD);
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_5_3) : "w"(b_1), "w"(a_5));
}
else if (_UNROLL == 4 && W_ob == 5 && C_ob == 16)
{
  float32x4_t a_0;\
  float32x4_t a_1;\
  float32x4_t a_2;\
  float32x4_t a_3;\
  float32x4_t a_4;\
  float32x4_t b_0;\
  float32x4_t b_1;\
  a_0 = vld1q_f32(a + 0 * W_stride + 0);\
  a_1 = vld1q_f32(a + 1 * W_stride + 0);\
  a_2 = vld1q_f32(a + 2 * W_stride + 0);\
  a_3 = vld1q_f32(a + 3 * W_stride + 0);\
  a_4 = vld1q_f32(a + 4 * W_stride + 0);\
  b_0 = vld1q_f32(b + 0 * C_ob + 0 + 0 * FLOAT_SIMD);\
  b_1 = vld1q_f32(b + 0 * C_ob + 0 + 1 * FLOAT_SIMD);\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4));\
  b_0 = vld1q_f32(b + 0 * C_ob + 8 + 0 * FLOAT_SIMD);\
  b_1 = vld1q_f32(b + 0 * C_ob + 8 + 1 * FLOAT_SIMD);\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[0]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4));\
  b_0 = vld1q_f32(b + 1 * C_ob + 0 + 0 * FLOAT_SIMD);\
  b_1 = vld1q_f32(b + 1 * C_ob + 0 + 1 * FLOAT_SIMD);\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4));\
  b_0 = vld1q_f32(b + 1 * C_ob + 8 + 0 * FLOAT_SIMD);\
  b_1 = vld1q_f32(b + 1 * C_ob + 8 + 1 * FLOAT_SIMD);\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[1]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4));\
  b_0 = vld1q_f32(b + 2 * C_ob + 0 + 0 * FLOAT_SIMD);\
  b_1 = vld1q_f32(b + 2 * C_ob + 0 + 1 * FLOAT_SIMD);\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4));\
  b_0 = vld1q_f32(b + 2 * C_ob + 8 + 0 * FLOAT_SIMD);\
  b_1 = vld1q_f32(b + 2 * C_ob + 8 + 1 * FLOAT_SIMD);\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[2]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4));\
  b_0 = vld1q_f32(b + 3 * C_ob + 0 + 0 * FLOAT_SIMD);\
  b_1 = vld1q_f32(b + 3 * C_ob + 0 + 1 * FLOAT_SIMD);\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_0) : "w"(b_0), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_1) : "w"(b_1), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_0) : "w"(b_0), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_1) : "w"(b_1), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_0) : "w"(b_0), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_1) : "w"(b_1), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_0) : "w"(b_0), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_1) : "w"(b_1), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_0) : "w"(b_0), "w"(a_4));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_1) : "w"(b_1), "w"(a_4));\
  b_0 = vld1q_f32(b + 3 * C_ob + 8 + 0 * FLOAT_SIMD);\
  b_1 = vld1q_f32(b + 3 * C_ob + 8 + 1 * FLOAT_SIMD);\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_2) : "w"(b_0), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_0_3) : "w"(b_1), "w"(a_0));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_2) : "w"(b_0), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_1_3) : "w"(b_1), "w"(a_1));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_2) : "w"(b_0), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_2_3) : "w"(b_1), "w"(a_2));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_2) : "w"(b_0), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_3_3) : "w"(b_1), "w"(a_3));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_2) : "w"(b_0), "w"(a_4));\
  __asm__ volatile("fmla %0.4s, %1.4s, %2.s[3]" : "+w"(c_4_3) : "w"(b_1), "w"(a_4));\
}
\