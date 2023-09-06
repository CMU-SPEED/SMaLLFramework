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

#include <Buffer.hpp>

// Scalar versions of all the quint8 microkernels for platform portability.
// Use the QUINT8_ prefix for all macros in this file.

#define QUINT8_SIMD_EPILOGUE 1

namespace small {
namespace quint8_detail {

/// @todo both pairs of typedefs should not be needed.
typedef small::QUInt8Buffer::value_type dtype;
typedef small::QUInt8Buffer::accum_type atype;

typedef small::QUInt8Buffer::value_type c_tile_out_t;
typedef small::QUInt8Buffer::accum_type c_tile_t;

}
}


//****************************************************************************
// Initializations
//****************************************************************************

#define QUINT8_DEF_TILE_C(W_ob, C_ob)           \
    c_tile_t c_tile[W_ob * C_ob];


#define QUINT8_DEF_END_C(W_ob, C_ob)            \
    c_tile_t c_tile[W_ob * C_ob];


// USED
#define QUINT8_ZERO_TILE_C(W_ob, C_ob, zero)    \
    for (uint32_t kk = 0; kk < W_ob; kk++)      \
    {                                           \
        for (uint32_t jj = 0; jj < C_ob; jj++)  \
        {                                       \
            c_tile[kk * C_ob + jj] = zero;      \
        }                                       \
    }

#define QUINT8_ZERO_END_C(_W_ob, C_ob, zero)    \
    for (uint32_t kk = 0; kk < _W_ob; kk++)     \
    {                                           \
        for (uint32_t jj = 0; jj < C_ob; jj++)  \
        {                                       \
            c_tile[kk * C_ob + jj] = zero;      \
        }                                       \
    }


//****************************************************************************
// Loads
//****************************************************************************

// USED
#define QUINT8_LOAD_TILE_C(O, W_ob, C_ob)                               \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            c_tile[kk * C_ob + jj] = static_cast<c_tile_t>(O[kk * C_ob + jj]); \
        }                                                               \
    }

//  c_tile_out_t c_tile[W_ob * C_ob];
#define QUINT8_LOAD_END_C(O, _W_ob, C_ob)                               \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                             \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            c_tile[kk * C_ob + jj] = static_cast<c_tile_t>(O[kk * C_ob + jj]); \
        }                                                               \
    }


//****************************************************************************
// Pooling Loads
//****************************************************************************

// strided loads
#define QUINT8_LOAD_TILE_C_strided(O, step, _W_ob, _C_ob)               \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                             \
    {                                                                   \
        for (uint32_t jj = 0; jj < _C_ob; jj++)                         \
        {                                                               \
            c_tile[kk * _C_ob + jj] = static_cast<c_tile_t>(O[kk * step + jj]); \
        }                                                               \
    }

//  c_tile_out_t c_tile[W_ob * C_ob];
#define QUINT8_LOAD_END_C_strided(O, step, _W_ob, C_ob)                 \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                             \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            c_tile[kk * C_ob + jj] = static_cast<c_tile_t>(O[kk * step + jj]); \
        }                                                               \
    }


//****************************************************************************
// Upsampling loads (stride < 1)
//****************************************************************************

// MISSING
/*
  #define QUINT8_LOAD_TILE_C_upsample(O, stride, _C_ib, _W_ob, C_ob)             \
  for (uint32_t kk = 0; kk < _W_ob; kk++)                             \
  {                                                                   \
  for (uint32_t jj = 0; jj < C_ob; jj++)                          \
  {                                                               \
  c_tile[kk * C_ob + jj] = O[(kk / stride) * (_C_ib) + jj];   \
  }                                                               \
  }

  #define QUINT8_LOAD_END_C_upsample(O, stride, _C_ib, _W_ob, C_ob)              \
  for (uint32_t kk = 0; kk < _W_ob; kk++)                             \
  {                                                                   \
  for (uint32_t jj = 0; jj < C_ob; jj++)                          \
  {                                                               \
  c_tile[kk * C_ob + jj] = O[(kk / stride) * (_C_ib) + jj];   \
  }                                                               \
  }
*/


//****************************************************************************
// Stores
//****************************************************************************

// USED
#define QUINT8_STORE_TILE_C(O, W_ob, C_ob)              \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
        }                                               \
    }

#define QUINT8_STORE_END_C(O, _W_ob, C_ob)              \
    for (uint32_t kk = 0; kk < _W_ob; kk++)             \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
        }                                               \
    }

// USED
#define QUINT8_STORE_Q_TILE_C(O, W_ob, C_ob)                            \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            O[kk * C_ob + jj] = static_cast<ScalarT>(c_tile[kk * C_ob + jj]); \
        }                                                               \
    }

#define QUINT8_STORE_Q_END_C(O, _W_ob, C_ob)                            \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                             \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            O[kk * C_ob + jj] = static_cast<ScalarT>(c_tile[kk * C_ob + jj]); \
        }                                                               \
    }


//****************************************************************************
// Convolution
//****************************************************************************

// USED, main
#if QUINT8_C_ob == 1 && QUINT8_W_ob == 1
#define QUINT8_CONV_TILE_C(step, a, b, W_ob, C_ob, a_offset, b_offset)  \
    c_tile_t a_val0 = *(a + 0 * step) + a_offset;                       \
    c_tile_t b_val0 = *(b + 0) + b_offset;                              \
    c_tile[0 * C_ob + 0] += (a_val0) * (b_val0);
#elif QUINT8_C_ob == 2 && QUINT8_W_ob == 1
#define QUINT8_CONV_TILE_C(step, a, b, W_ob, C_ob, a_offset, b_offset)  \
    c_tile_t a_val0 = *(a + 0 * step) + a_offset;                       \
    c_tile_t b_val0 = *(b + 0) + b_offset;                              \
    c_tile_t b_val1 = *(b + 1) + b_offset;                              \
    c_tile[0 * C_ob + 0] += (a_val0) * (b_val0);                        \
    c_tile[0 * C_ob + 1] += (a_val0) * (b_val1);
#elif QUINT8_C_ob == 4 && QUINT8_W_ob == 1
#define QUINT8_CONV_TILE_C(step, a, b, W_ob, C_ob, a_offset, b_offset)  \
    c_tile_t a_val0 = *(a + 0 * step) + a_offset;                       \
    c_tile_t b_val0 = *(b + 0) + b_offset;                              \
    c_tile_t b_val1 = *(b + 1) + b_offset;                              \
    c_tile_t b_val2 = *(b + 2) + b_offset;                              \
    c_tile_t b_val3 = *(b + 3) + b_offset;                              \
    c_tile[0 * C_ob + 0] += (a_val0) * (b_val0);                        \
    c_tile[0 * C_ob + 1] += (a_val0) * (b_val1);                        \
    c_tile[0 * C_ob + 2] += (a_val0) * (b_val2);                        \
    c_tile[0 * C_ob + 3] += (a_val0) * (b_val3);
#elif QUINT8_C_ob == 8 && QUINT8_W_ob == 1
#define QUINT8_CONV_TILE_C(step, a, b, W_ob, C_ob, a_offset, b_offset)  \
    c_tile_t a_val0 = *(a + 0 * step) + a_offset;                       \
    c_tile_t b_val0 = *(b + 0) + b_offset;                              \
    c_tile_t b_val1 = *(b + 1) + b_offset;                              \
    c_tile_t b_val2 = *(b + 2) + b_offset;                              \
    c_tile_t b_val3 = *(b + 3) + b_offset;                              \
    c_tile_t b_val4 = *(b + 4) + b_offset;                              \
    c_tile_t b_val5 = *(b + 5) + b_offset;                              \
    c_tile_t b_val6 = *(b + 6) + b_offset;                              \
    c_tile_t b_val7 = *(b + 7) + b_offset;                              \
    c_tile[0 * C_ob + 0] += (a_val0) * (b_val0);                        \
    c_tile[0 * C_ob + 1] += (a_val0) * (b_val1);                        \
    c_tile[0 * C_ob + 2] += (a_val0) * (b_val2);                        \
    c_tile[0 * C_ob + 3] += (a_val0) * (b_val3);                        \
    c_tile[0 * C_ob + 4] += (a_val0) * (b_val4);                        \
    c_tile[0 * C_ob + 5] += (a_val0) * (b_val5);                        \
    c_tile[0 * C_ob + 6] += (a_val0) * (b_val6);                        \
    c_tile[0 * C_ob + 7] += (a_val0) * (b_val7);
#elif QUINT8_C_ob == 16 && QUINT8_W_ob == 1
#define QUINT8_CONV_TILE_C(step, a, b, W_ob, C_ob, a_offset, b_offset)  \
    c_tile_t a_val0 = *(a + 0 * step) + a_offset;                       \
    c_tile_t b_val0 = *(b + 0x0) + b_offset;                            \
    c_tile_t b_val1 = *(b + 0x1) + b_offset;                            \
    c_tile_t b_val2 = *(b + 0x2) + b_offset;                            \
    c_tile_t b_val3 = *(b + 0x3) + b_offset;                            \
    c_tile_t b_val4 = *(b + 0x4) + b_offset;                            \
    c_tile_t b_val5 = *(b + 0x5) + b_offset;                            \
    c_tile_t b_val6 = *(b + 0x6) + b_offset;                            \
    c_tile_t b_val7 = *(b + 0x7) + b_offset;                            \
    c_tile_t b_val8 = *(b + 0x8) + b_offset;                            \
    c_tile_t b_val9 = *(b + 0x9) + b_offset;                            \
    c_tile_t b_valA = *(b + 0xA) + b_offset;                            \
    c_tile_t b_valB = *(b + 0xB) + b_offset;                            \
    c_tile_t b_valC = *(b + 0xC) + b_offset;                            \
    c_tile_t b_valD = *(b + 0xD) + b_offset;                            \
    c_tile_t b_valE = *(b + 0xE) + b_offset;                            \
    c_tile_t b_valF = *(b + 0xF) + b_offset;                            \
    c_tile[0 * C_ob + 0x0] += (a_val0) * (b_val0);                      \
    c_tile[0 * C_ob + 0x1] += (a_val0) * (b_val1);                      \
    c_tile[0 * C_ob + 0x2] += (a_val0) * (b_val2);                      \
    c_tile[0 * C_ob + 0x3] += (a_val0) * (b_val3);                      \
    c_tile[0 * C_ob + 0x4] += (a_val0) * (b_val4);                      \
    c_tile[0 * C_ob + 0x5] += (a_val0) * (b_val5);                      \
    c_tile[0 * C_ob + 0x6] += (a_val0) * (b_val6);                      \
    c_tile[0 * C_ob + 0x7] += (a_val0) * (b_val7);                      \
    c_tile[0 * C_ob + 0x8] += (a_val0) * (b_val8);                      \
    c_tile[0 * C_ob + 0x9] += (a_val0) * (b_val9);                      \
    c_tile[0 * C_ob + 0xA] += (a_val0) * (b_valA);                      \
    c_tile[0 * C_ob + 0xB] += (a_val0) * (b_valB);                      \
    c_tile[0 * C_ob + 0xC] += (a_val0) * (b_valC);                      \
    c_tile[0 * C_ob + 0xD] += (a_val0) * (b_valD);                      \
    c_tile[0 * C_ob + 0xE] += (a_val0) * (b_valE);                      \
    c_tile[0 * C_ob + 0xF] += (a_val0) * (b_valF);
#elif QUINT8_C_ob == 32 && QUINT8_W_ob == 1
#define QUINT8_CONV_TILE_C(step, a, b, W_ob, C_ob, a_offset, b_offset)  \
    c_tile_t a_val0 = *(a + 0 * step) + a_offset;                       \
    c_tile_t b_val00 = *(b + 0x00) + b_offset;                          \
    c_tile_t b_val01 = *(b + 0x01) + b_offset;                          \
    c_tile_t b_val02 = *(b + 0x02) + b_offset;                          \
    c_tile_t b_val03 = *(b + 0x03) + b_offset;                          \
    c_tile_t b_val04 = *(b + 0x04) + b_offset;                          \
    c_tile_t b_val05 = *(b + 0x05) + b_offset;                          \
    c_tile_t b_val06 = *(b + 0x06) + b_offset;                          \
    c_tile_t b_val07 = *(b + 0x07) + b_offset;                          \
    c_tile_t b_val08 = *(b + 0x08) + b_offset;                          \
    c_tile_t b_val09 = *(b + 0x09) + b_offset;                          \
    c_tile_t b_val0A = *(b + 0x0A) + b_offset;                          \
    c_tile_t b_val0B = *(b + 0x0B) + b_offset;                          \
    c_tile_t b_val0C = *(b + 0x0C) + b_offset;                          \
    c_tile_t b_val0D = *(b + 0x0D) + b_offset;                          \
    c_tile_t b_val0E = *(b + 0x0E) + b_offset;                          \
    c_tile_t b_val0F = *(b + 0x0F) + b_offset;                          \
    c_tile_t b_val10 = *(b + 0x10) + b_offset;                          \
    c_tile_t b_val11 = *(b + 0x11) + b_offset;                          \
    c_tile_t b_val12 = *(b + 0x12) + b_offset;                          \
    c_tile_t b_val13 = *(b + 0x13) + b_offset;                          \
    c_tile_t b_val14 = *(b + 0x14) + b_offset;                          \
    c_tile_t b_val15 = *(b + 0x15) + b_offset;                          \
    c_tile_t b_val16 = *(b + 0x16) + b_offset;                          \
    c_tile_t b_val17 = *(b + 0x17) + b_offset;                          \
    c_tile_t b_val18 = *(b + 0x18) + b_offset;                          \
    c_tile_t b_val19 = *(b + 0x19) + b_offset;                          \
    c_tile_t b_val1A = *(b + 0x1A) + b_offset;                          \
    c_tile_t b_val1B = *(b + 0x1B) + b_offset;                          \
    c_tile_t b_val1C = *(b + 0x1C) + b_offset;                          \
    c_tile_t b_val1D = *(b + 0x1D) + b_offset;                          \
    c_tile_t b_val1E = *(b + 0x1E) + b_offset;                          \
    c_tile_t b_val1F = *(b + 0x1F) + b_offset;                          \
    c_tile[0 * C_ob + 0x00] += (a_val0) * (b_val00);                    \
    c_tile[0 * C_ob + 0x01] += (a_val0) * (b_val01);                    \
    c_tile[0 * C_ob + 0x02] += (a_val0) * (b_val02);                    \
    c_tile[0 * C_ob + 0x03] += (a_val0) * (b_val03);                    \
    c_tile[0 * C_ob + 0x04] += (a_val0) * (b_val04);                    \
    c_tile[0 * C_ob + 0x05] += (a_val0) * (b_val05);                    \
    c_tile[0 * C_ob + 0x06] += (a_val0) * (b_val06);                    \
    c_tile[0 * C_ob + 0x07] += (a_val0) * (b_val07);                    \
    c_tile[0 * C_ob + 0x08] += (a_val0) * (b_val08);                    \
    c_tile[0 * C_ob + 0x09] += (a_val0) * (b_val09);                    \
    c_tile[0 * C_ob + 0x0A] += (a_val0) * (b_val0A);                    \
    c_tile[0 * C_ob + 0x0B] += (a_val0) * (b_val0B);                    \
    c_tile[0 * C_ob + 0x0C] += (a_val0) * (b_val0C);                    \
    c_tile[0 * C_ob + 0x0D] += (a_val0) * (b_val0D);                    \
    c_tile[0 * C_ob + 0x0E] += (a_val0) * (b_val0E);                    \
    c_tile[0 * C_ob + 0x0F] += (a_val0) * (b_val0F);                    \
    c_tile[0 * C_ob + 0x10] += (a_val0) * (b_val10);                    \
    c_tile[0 * C_ob + 0x11] += (a_val0) * (b_val11);                    \
    c_tile[0 * C_ob + 0x12] += (a_val0) * (b_val12);                    \
    c_tile[0 * C_ob + 0x13] += (a_val0) * (b_val13);                    \
    c_tile[0 * C_ob + 0x14] += (a_val0) * (b_val14);                    \
    c_tile[0 * C_ob + 0x15] += (a_val0) * (b_val15);                    \
    c_tile[0 * C_ob + 0x16] += (a_val0) * (b_val16);                    \
    c_tile[0 * C_ob + 0x17] += (a_val0) * (b_val17);                    \
    c_tile[0 * C_ob + 0x18] += (a_val0) * (b_val18);                    \
    c_tile[0 * C_ob + 0x19] += (a_val0) * (b_val19);                    \
    c_tile[0 * C_ob + 0x1A] += (a_val0) * (b_val1A);                    \
    c_tile[0 * C_ob + 0x1B] += (a_val0) * (b_val1B);                    \
    c_tile[0 * C_ob + 0x1C] += (a_val0) * (b_val1C);                    \
    c_tile[0 * C_ob + 0x1D] += (a_val0) * (b_val1D);                    \
    c_tile[0 * C_ob + 0x1E] += (a_val0) * (b_val1E);                    \
    c_tile[0 * C_ob + 0x1F] += (a_val0) * (b_val1F);
#elif QUINT8_C_ob == 2 && QUINT8_W_ob == 2
#define QUINT8_CONV_TILE_C(step, a, b, W_ob, C_ob, a_offset, b_offset)  \
    c_tile_t a_val0 = *(a + 0 * step) + a_offset;                       \
    c_tile_t a_val1 = *(a + 1 * step) + a_offset;                       \
    c_tile_t b_val0 = *(b + 0 * 1) + b_offset;                          \
    c_tile_t b_val1 = *(b + 1 * 1) + b_offset;                          \
    c_tile[0 * C_ob + 0] += (a_val0) * (b_val0);                        \
    c_tile[0 * C_ob + 1] += (a_val0) * (b_val1);                        \
    c_tile[1 * C_ob + 0] += (a_val1) * (b_val0);                        \
    c_tile[1 * C_ob + 1] += (a_val1) * (b_val1);
#else
#define QUINT8_CONV_TILE_C(step, a, b, W_ob, C_ob, a_offset, b_offset)  \
    c_tile_out_t const *a_channel = a;                                  \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t a_val = *(a_channel) + a_offset;                       \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            c_tile_t b_val = *(b + jj) + b_offset;                      \
            c_tile[kk * C_ob + jj] += (a_val) * (b_val);                \
        }                                                               \
        a_channel += step;                                              \
    }
#endif

/// @todo the #else clause above is quite different from quantized_reference impl.

#define QUINT8_CONV_END_C(step, a, b, c_cur, _W_ob, C_ob, a_offset, b_offset) \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_out_t const *a_channel = a;                                  \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                             \
    {                                                                   \
        c_tile_out_t a_val = *(a_channel);                              \
        c_tile_t *c_channel = c_pixel;                                  \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            c_tile_out_t b_val = *(b + jj);                             \
            *(c_channel) += (a_val + a_offset) * (b_val + b_offset);    \
            c_channel++;                                                \
        }                                                               \
        a_channel += step;                                              \
        c_pixel += C_ob;                                                \
    }


//****************************************************************************
// Pooling
//   Max pooling
//****************************************************************************

// USED
#if QUINT8_C_ob == 16 && QUINT8_W_ob == 1
#define QUINT8_MAX_TILE_C(step, a, W_ob, C_ob, a_offset)                \
  c_tile_t a_val = *(a) + a_offset;                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0x0];                          \
      c_tile[0 * C_ob + 0x0] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0x1];                          \
      c_tile[0 * C_ob + 0x1] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0x2];                          \
      c_tile[0 * C_ob + 0x2] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0x3];                          \
      c_tile[0 * C_ob + 0x3] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0x4];                          \
      c_tile[0 * C_ob + 0x4] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0x5];                          \
      c_tile[0 * C_ob + 0x5] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0x6];                          \
      c_tile[0 * C_ob + 0x6] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0x7];                          \
      c_tile[0 * C_ob + 0x7] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0x8];                          \
      c_tile[0 * C_ob + 0x8] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0x9];                          \
      c_tile[0 * C_ob + 0x9] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0xA];                          \
      c_tile[0 * C_ob + 0xA] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0xB];                          \
      c_tile[0 * C_ob + 0xB] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0xC];                          \
      c_tile[0 * C_ob + 0xC] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0xD];                          \
      c_tile[0 * C_ob + 0xD] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0xE];                          \
      c_tile[0 * C_ob + 0xE] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }                                                                     \
  {                                                                     \
      c_tile_t c_val = c_tile[0 * C_ob + 0xF];                          \
      c_tile[0 * C_ob + 0xF] = ((a_val) > c_val) ? (a_val) : c_val;     \
  }
#else
#define QUINT8_MAX_TILE_C(step, a, W_ob, C_ob, a_offset)                \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_out_t const *a_pixel = a;                                    \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_out_t const *a_channel = a_pixel;                        \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = ((*(a_channel) + a_offset) > *(c_channel)) ? (*(a_channel) + a_offset) : *(c_channel); \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }
#endif

#define QUINT8_MAX_END_C(step, a, c_cur, W_last, C_ob, a_offset)        \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_out_t const *a_pixel = a;                                    \
    for (uint32_t kk = 0; kk < W_last; kk++)                            \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_out_t const *a_channel = a_pixel;                        \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = ((*(a_channel) + a_offset) > *(c_channel)) ? (*(a_channel) + a_offset) : *(c_channel); \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }


//****************************************************************************
// DW Convolution
//****************************************************************************

#define QUINT8_DW_TILE_C(step, a, b, W_ob, C_ob, a_offset, b_offset)    \
    {                                                                   \
        c_tile_t *c_pixel = c_tile;                                     \
        c_tile_out_t const *a_pixel = a;                                \
        for (uint32_t kk = 0; kk < W_ob; kk++)                          \
        {                                                               \
            c_tile_t *c_channel = c_pixel;                              \
            c_tile_out_t const *a_channel = a_pixel;                    \
            c_tile_out_t const *b_channel = b;                          \
            for (uint32_t jj = 0; jj < C_ob; jj++)                      \
            {                                                           \
                *(c_channel) += ((*(a_channel) + a_offset) * (*(b_channel) + b_offset)); \
                c_channel++;                                            \
                b_channel++;                                            \
                a_channel++;                                            \
            }                                                           \
            a_pixel += step;                                            \
            c_pixel += C_ob;                                            \
        }                                                               \
    }

#define QUINT8_DW_END_C(step, a, b, c_cur, _W_ob, C_ob, a_offset, b_offset) \
    {                                                                   \
        c_tile_t *c_pixel = c_cur;                                      \
        c_tile_out_t const *a_pixel = a;                                \
        for (uint32_t kk = 0; kk < _W_ob; kk++)                         \
        {                                                               \
            c_tile_t *c_channel = c_pixel;                              \
            c_tile_out_t const *a_channel = a_pixel;                    \
            c_tile_out_t const *b_channel = b;                          \
            for (uint32_t jj = 0; jj < C_ob; jj++)                      \
            {                                                           \
                *(c_channel) += (((*(a_channel) + a_offset) * (*(b_channel) + b_offset))); \
                c_channel++;                                            \
                b_channel++;                                            \
                a_channel++;                                            \
            }                                                           \
            a_pixel += step;                                            \
            c_pixel += C_ob;                                            \
        }                                                               \
    }


//****************************************************************************
// Leaky ReLU activation
//****************************************************************************

// MISSING
/*
  #define QUINT8_COND_SCALE_TILE_C(step, a, b, W_ob, C_ob)                       \
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

  #define QUINT8_COND_SCALE_END_C(step, a, b, c_cur, W_last, C_ob)               \
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
*/

//****************************************************************************
// Accumulation kernels
//****************************************************************************

// MISSING


//****************************************************************************
// AVG Pooling
//****************************************************************************

#define QUINT8_ADD_TILE_C_G(I, W_ob_g, C_ob)            \
    c_tile_out_t const *i_pixel = I;                    \
    c_tile_t *c_pixel = c_tile;                         \
    for (uint32_t mm = 0; mm < W_ob_g; mm++)            \
    {                                                   \
        c_tile_t *c_channel = c_pixel;                  \
        c_tile_out_t const *i_channel = i_pixel;        \
        for (uint32_t kk = 0; kk < C_ob; kk++)          \
        {                                               \
            *c_channel += *i_channel;                   \
            c_channel++;                                \
            i_channel++;                                \
        }                                               \
        c_pixel += C_ob;                                \
        i_pixel += C_ob;                                \
    }

#define QUINT8_ADD_LAST_C_G(I, W_last, C_ob)            \
    c_tile_out_t const *i_pixel = I;                    \
    c_tile_t *c_pixel = c_tile;                         \
    for (uint32_t mm = 0; mm < W_last; mm++)            \
    {                                                   \
        c_tile_t *c_channel = c_pixel;                  \
        c_tile_out_t const *i_channel = i_pixel;        \
        for (uint32_t kk = 0; kk < C_ob; kk++)          \
        {                                               \
            *c_channel += *i_channel;                   \
            c_channel++;                                \
            i_channel++;                                \
        }                                               \
        c_pixel += C_ob;                                \
        i_pixel += C_ob;                                \
    }


#define QUINT8_REDUCE_div_C(O, d, W_ob_g, C_ob)         \
    {                                                   \
        c_tile_t *c_pixel = c_tile;                     \
        c_tile_out_t *O_channel = O;                    \
        c_tile_t *c_channel = c_pixel;                  \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)        \
        {                                               \
            c_tile_out_t *O_channel = O;                \
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

#define QUINT8_REDUCE_C(O, W_ob_g, C_ob)                \
    {                                                   \
        c_tile_t *c_pixel = c_tile;                     \
        c_tile_out_t *O_channel = O;                    \
        c_tile_t *c_channel = c_pixel;                  \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)        \
        {                                               \
            c_tile_out_t *O_channel = O;                \
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

#define QUINT8_REDUCE_C_last(O, W_last, C_ob)           \
    {                                                   \
        c_tile_t *c_pixel = c_tile;                     \
        c_tile_out_t *O_channel = O;                    \
        c_tile_t *c_channel = c_pixel;                  \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)        \
        {                                               \
            c_tile_out_t *O_channel = O;                \
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
// Multiply by quantized multiplier
// Left shift: same computation as the ARMv7 NEON VQRDMULH
//
// Add offset
// Clip to range of output type
//
// @todo Add prefix to internal macros
// @todo INT32MAX and INT32MIN not used
//****************************************************************************

//#define INT32MAX  2147483647
//#define INT32MIN -2147483648
//#define INT32MAX 1 << 31 - 1
//#define INT32MIN -1 << 31

// USED
#define VQRDMULH(a, b, out)                                     \
    bool overflow = a == b && a == -2147483648;                 \
    int64_t prod = a * b;                                       \
    int32_t nudge = prod >= 0 ? (1 << 30) : (1 - (1 << 30));    \
    int32_t ab_x2_high32 =                                      \
        static_cast<int32_t>((prod + nudge) / (1ll << 31));     \
    out = overflow ? 2147483647 : ab_x2_high32;

// USED
#define RNDRSHIFT(a, b, out)                                            \
    int32_t mask = (1ll << b) - 1;                                      \
    int32_t mod = (a & mask);                                           \
    int32_t one = ~0;                                                   \
    int32_t threshold = (mask >> 1) + (one & ((a < 0) ? ~0 : 0));       \
    out = (a >> b) + (one & ((mod > threshold) ? ~0 : 0));

// USED
/// @todo _C_ob not used
#define QUINT8_QUANTIZE_TILE_C(_W_ob, _C_ob, left_shift, right_shift, q_mul, offset, max_val, min_val) \
    {                                                                   \
        for (uint32_t kk = 0; kk < _W_ob; kk++)                         \
        {                                                               \
            for (uint32_t jj = 0; jj < QUINT8_C_ob; jj++)               \
            {                                                           \
                auto val = c_tile[kk * QUINT8_C_ob + jj];               \
                /*Multiply by q_mul and left shift*/                    \
                int64_t left_shift_val = static_cast<int64_t>(val * (1 << left_shift)); \
                VQRDMULH(left_shift_val, static_cast<int64_t>(q_mul), val); \
                RNDRSHIFT(val, right_shift, val);                       \
                c_tile[kk * QUINT8_C_ob + jj] = (val > min_val) ? ((val < max_val) ? val \
                                                            : max_val)  \
                    : min_val;                                          \
            }                                                           \
        }                                                               \
    }

/// @todo _C_ob not used
#define QUINT8_QUANTIZE_END_C(_W_ob, _C_ob, left_shift, right_shift, q_mul, offset, max_val, min_val) \
    {                                                                   \
        for (uint32_t kk = 0; kk < _W_ob; kk++)                         \
        {                                                               \
            for (uint32_t jj = 0; jj < QUINT8_C_ob; jj++)               \
            {                                                           \
                auto val = c_tile[kk * QUINT8_C_ob + jj];               \
                int64_t left_shift_val = static_cast<int64_t>(val * (1 << left_shift)); \
                VQRDMULH(left_shift_val, static_cast<int64_t>(q_mul), val); \
                RNDRSHIFT(val, right_shift, val);                       \
                c_tile[kk * QUINT8_C_ob + jj] = (val > min_val) ? ((val < max_val) ? val \
                                                            : max_val)  \
                    : min_val;                                          \
            }                                                           \
        }                                                               \
    }
