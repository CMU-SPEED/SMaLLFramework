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

// #include <QUInt8Buffer.hpp>

// Scalar versions of all the quint8 microkernels for platform portability.
// Use the QUINT8_ prefix for all macros in this file.

#define QUINT8_SIMD_EPILOGUE 1

typedef int32_t c_tile_t;
typedef uint8_t c_tile_out_t;
typedef int8_t dtype;
typedef int32_t atype;

// Architecture specific tiling params

// __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

//****************************************************************************
// Initializations
//****************************************************************************

#define QUINT8_DEF_TILE_C(W_ob, C_ob) \
    c_tile_t c_tile[W_ob * C_ob];

#define QUINT8_DEF_END_C(W_ob, C_ob) \
    c_tile_t c_tile[W_ob * C_ob];

#define QUINT8_ZERO_TILE_C(W_ob, C_ob, zero)   \
    for (uint32_t kk = 0; kk < W_ob; kk++)     \
    {                                          \
        for (uint32_t jj = 0; jj < C_ob; jj++) \
        {                                      \
            c_tile[kk * C_ob + jj] = zero;     \
        }                                      \
    }

#define QUINT8_ZERO_END_C(_W_ob, C_ob, zero)   \
    for (uint32_t kk = 0; kk < _W_ob; kk++)    \
    {                                          \
        for (uint32_t jj = 0; jj < C_ob; jj++) \
        {                                      \
            c_tile[kk * C_ob + jj] = zero;     \
        }                                      \
    }

//****************************************************************************
// Loads
//****************************************************************************

#define QUINT8_LOAD_TILE_C(O, W_ob, C_ob)                                      \
    for (uint32_t kk = 0; kk < W_ob; kk++)                                     \
    {                                                                          \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                 \
        {                                                                      \
            c_tile[kk * C_ob + jj] = static_cast<c_tile_t>(O[kk * C_ob + jj]); \
        }                                                                      \
    }

//  c_tile_out_t c_tile[W_ob * C_ob];
#define QUINT8_LOAD_END_C(O, _W_ob, C_ob)                                      \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                                    \
    {                                                                          \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                 \
        {                                                                      \
            c_tile[kk * C_ob + jj] = static_cast<c_tile_t>(O[kk * C_ob + jj]); \
        }                                                                      \
    }

//****************************************************************************
// Pooling Loads
//****************************************************************************

// strided loads
#define QUINT8_LOAD_TILE_C_strided(O, step, _W_ob, _C_ob)                       \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                                     \
    {                                                                           \
        for (uint32_t jj = 0; jj < _C_ob; jj++)                                 \
        {                                                                       \
            c_tile[kk * _C_ob + jj] = static_cast<c_tile_t>(O[kk * step + jj]); \
        }                                                                       \
    }

//  c_tile_out_t c_tile[W_ob * C_ob];
#define QUINT8_LOAD_END_C_strided(O, step, _W_ob, C_ob)                        \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                                    \
    {                                                                          \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                 \
        {                                                                      \
            c_tile[kk * C_ob + jj] = static_cast<c_tile_t>(O[kk * step + jj]); \
        }                                                                      \
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

#define QUINT8_STORE_Q_TILE_C(O, W_ob, C_ob)                                  \
    for (uint32_t kk = 0; kk < W_ob; kk++)                                    \
    {                                                                         \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                \
        {                                                                     \
            O[kk * C_ob + jj] = static_cast<ScalarT>(c_tile[kk * C_ob + jj]); \
        }                                                                     \
    }

#define QUINT8_STORE_Q_END_C(O, _W_ob, C_ob)                                  \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                                   \
    {                                                                         \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                \
        {                                                                     \
            O[kk * C_ob + jj] = static_cast<ScalarT>(c_tile[kk * C_ob + jj]); \
        }                                                                     \
    }

//****************************************************************************
// Convolution
//****************************************************************************

#define QUINT8_CONV_TILE_C(step, a, b, W_ob, C_ob, a_offset, b_offset) \
    c_tile_t *c_pixel = c_tile;                                        \
    c_tile_out_t const *a_channel = a;                                 \
    for (uint32_t kk = 0; kk < W_ob; kk++)                             \
    {                                                                  \
        c_tile_out_t a_val = *(a_channel);                             \
        c_tile_t *c_channel = c_pixel;                                 \
        for (uint32_t jj = 0; jj < C_ob; jj++)                         \
        {                                                              \
            c_tile_out_t b_val = *(b + jj);                            \
            *(c_channel) += (a_val + a_offset) * (b_val + b_offset);   \
            c_channel++;                                               \
        }                                                              \
        a_channel += step;                                             \
        c_pixel += C_ob;                                               \
    }

#define QUINT8_CONV_END_C(step, a, b, c_cur, _W_ob, C_ob, a_offset, b_offset) \
    c_tile_t *c_pixel = c_cur;                                                \
    c_tile_out_t const *a_channel = a;                                        \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                                   \
    {                                                                         \
        c_tile_out_t a_val = *(a_channel);                                    \
        c_tile_t *c_channel = c_pixel;                                        \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                \
        {                                                                     \
            c_tile_out_t b_val = *(b + jj);                                   \
            *(c_channel) += (a_val + a_offset) * (b_val + b_offset);          \
            c_channel++;                                                      \
        }                                                                     \
        a_channel += step;                                                    \
        c_pixel += C_ob;                                                      \
    }

//****************************************************************************
// Pooling
//   Max pooling
//****************************************************************************

#define QUINT8_MAX_TILE_C(step, a, W_ob, C_ob, a_offset)                                                          \
    c_tile_t *c_pixel = c_tile;                                                                                   \
    c_tile_out_t const *a_pixel = a;                                                                              \
    for (uint32_t kk = 0; kk < W_ob; kk++)                                                                        \
    {                                                                                                             \
        c_tile_t *c_channel = c_pixel;                                                                            \
        c_tile_out_t const *a_channel = a_pixel;                                                                  \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                                                    \
        {                                                                                                         \
            *(c_channel) = ((*(a_channel) + a_offset) > *(c_channel)) ? (*(a_channel) + a_offset) : *(c_channel); \
            c_channel++;                                                                                          \
            a_channel++;                                                                                          \
        }                                                                                                         \
        a_pixel += step;                                                                                          \
        c_pixel += C_ob;                                                                                          \
    }

#define QUINT8_MAX_END_C(step, a, c_cur, W_last, C_ob, a_offset)                                                  \
    c_tile_t *c_pixel = c_cur;                                                                                    \
    c_tile_out_t const *a_pixel = a;                                                                              \
    for (uint32_t kk = 0; kk < W_last; kk++)                                                                      \
    {                                                                                                             \
        c_tile_t *c_channel = c_pixel;                                                                            \
        c_tile_out_t const *a_channel = a_pixel;                                                                  \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                                                    \
        {                                                                                                         \
            *(c_channel) = ((*(a_channel) + a_offset) > *(c_channel)) ? (*(a_channel) + a_offset) : *(c_channel); \
            c_channel++;                                                                                          \
            a_channel++;                                                                                          \
        }                                                                                                         \
        a_pixel += step;                                                                                          \
        c_pixel += C_ob;                                                                                          \
    }

//****************************************************************************
// DW Convolution
//****************************************************************************

#define QUINT8_DW_TILE_C(step, a, b, W_ob, C_ob, a_offset, b_offset)                     \
    {                                                                                    \
        c_tile_t *c_pixel = c_tile;                                                      \
        c_tile_out_t const *a_pixel = a;                                                 \
        for (uint32_t kk = 0; kk < W_ob; kk++)                                           \
        {                                                                                \
            c_tile_t *c_channel = c_pixel;                                               \
            c_tile_out_t const *a_channel = a_pixel;                                     \
            c_tile_out_t const *b_channel = b;                                           \
            for (uint32_t jj = 0; jj < C_ob; jj++)                                       \
            {                                                                            \
                *(c_channel) += ((*(a_channel) + a_offset) * (*(b_channel) + b_offset)); \
                c_channel++;                                                             \
                b_channel++;                                                             \
                a_channel++;                                                             \
            }                                                                            \
            a_pixel += step;                                                             \
            c_pixel += C_ob;                                                             \
        }                                                                                \
    }

#define QUINT8_DW_END_C(step, a, b, c_cur, _W_ob, C_ob, a_offset, b_offset)                \
    {                                                                                      \
        c_tile_t *c_pixel = c_cur;                                                         \
        c_tile_out_t const *a_pixel = a;                                                   \
        for (uint32_t kk = 0; kk < _W_ob; kk++)                                            \
        {                                                                                  \
            c_tile_t *c_channel = c_pixel;                                                 \
            c_tile_out_t const *a_channel = a_pixel;                                       \
            c_tile_out_t const *b_channel = b;                                             \
            for (uint32_t jj = 0; jj < C_ob; jj++)                                         \
            {                                                                              \
                *(c_channel) += (((*(a_channel) + a_offset) * (*(b_channel) + b_offset))); \
                c_channel++;                                                               \
                b_channel++;                                                               \
                a_channel++;                                                               \
            }                                                                              \
            a_pixel += step;                                                               \
            c_pixel += C_ob;                                                               \
        }                                                                                  \
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

#define OLD_QUINT8_ADD_TILE_C_G(I, W_ob_g, C_ob) \
    c_tile_out_t const *i_pixel = I;             \
    c_tile_t *c_pixel = c_tile;                  \
    for (uint32_t mm = 0; mm < W_ob_g; mm++)     \
    {                                            \
        c_tile_t *c_channel = c_pixel;           \
        c_tile_out_t const *i_channel = i_pixel; \
        for (uint32_t kk = 0; kk < C_ob; kk++)   \
        {                                        \
            *c_channel += *i_channel;            \
            c_channel++;                         \
            i_channel++;                         \
        }                                        \
        c_pixel += C_ob;                         \
        i_pixel += C_ob;                         \
    }

#define QUINT8_ADD_TILE_C_G(I, W_ob_g, C_ob)                \
    c_tile_t *c_channel = c_tile;                           \
    c_tile_out_t const *i_channel = I;                      \
    const c_tile_t *end_addr = c_channel + (W_ob_g * C_ob); \
    /*address of the last element in c_tile*/               \
                                                            \
    while (c_channel < end_addr)                            \
    {                                                       \
        *c_channel++ += *i_channel++;                       \
    }

#define QUINT8_ADD_LAST_C_G(I, W_last, C_ob)     \
    c_tile_out_t const *i_pixel = I;             \
    c_tile_t *c_pixel = c_tile;                  \
    for (uint32_t mm = 0; mm < W_last; mm++)     \
    {                                            \
        c_tile_t *c_channel = c_pixel;           \
        c_tile_out_t const *i_channel = i_pixel; \
        for (uint32_t kk = 0; kk < C_ob; kk++)   \
        {                                        \
            *c_channel += *i_channel;            \
            c_channel++;                         \
            i_channel++;                         \
        }                                        \
        c_pixel += C_ob;                         \
        i_pixel += C_ob;                         \
    }

#define QUINT8_REDUCE_div_C(O, d, W_ob_g, C_ob)    \
    {                                              \
        c_tile_t *c_pixel = c_tile;                \
        c_tile_out_t *O_channel = O;               \
        c_tile_t *c_channel = c_pixel;             \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)   \
        {                                          \
            c_tile_out_t *O_channel = O;           \
            c_tile_t *c_channel = c_pixel;         \
            for (uint32_t kk = 0; kk < C_ob; kk++) \
            {                                      \
                *O_channel += *c_channel;          \
                O_channel++;                       \
                c_channel++;                       \
            }                                      \
            c_pixel += C_ob;                       \
        }                                          \
        O_channel = O;                             \
        for (uint32_t kk = 0; kk < C_ob; kk++)     \
        {                                          \
            *O_channel *= d;                       \
            O_channel++;                           \
        }                                          \
    }

#define QUINT8_REDUCE_C(O, W_ob_g, C_ob)           \
    {                                              \
        c_tile_t *c_pixel = c_tile;                \
        c_tile_out_t *O_channel = O;               \
        c_tile_t *c_channel = c_pixel;             \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)   \
        {                                          \
            c_tile_out_t *O_channel = O;           \
            c_tile_t *c_channel = c_pixel;         \
            for (uint32_t kk = 0; kk < C_ob; kk++) \
            {                                      \
                *O_channel += *c_channel;          \
                O_channel++;                       \
                c_channel++;                       \
            }                                      \
            c_pixel += C_ob;                       \
        }                                          \
    }

#define QUINT8_REDUCE_C_last(O, W_last, C_ob)      \
    {                                              \
        c_tile_t *c_pixel = c_tile;                \
        c_tile_out_t *O_channel = O;               \
        c_tile_t *c_channel = c_pixel;             \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)   \
        {                                          \
            c_tile_out_t *O_channel = O;           \
            c_tile_t *c_channel = c_pixel;         \
            for (uint32_t kk = 0; kk < C_ob; kk++) \
            {                                      \
                *O_channel += *c_channel;          \
                O_channel++;                       \
                c_channel++;                       \
            }                                      \
            c_pixel += C_ob;                       \
        }                                          \
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

// #define INT32MAX  2147483647
// #define INT32MIN -2147483648
// #define INT32MAX 1 << 31 - 1
// #define INT32MIN -1 << 31

#define VQRDMULH(a, b, out)                                  \
    bool overflow = a == b && a == -2147483648;              \
    int64_t prod = a * b;                                    \
    int32_t nudge = prod >= 0 ? (1 << 30) : (1 - (1 << 30)); \
    int32_t ab_x2_high32 =                                   \
        static_cast<int32_t>((prod + nudge) / (1ll << 31));  \
    out = overflow ? 2147483647 : ab_x2_high32;

#define RNDRSHIFT(a, b, out)                                      \
    int32_t mask = (1ll << b) - 1;                                \
    int32_t mod = (a & mask);                                     \
    int32_t one = ~0;                                             \
    int32_t threshold = (mask >> 1) + (one & ((a < 0) ? ~0 : 0)); \
    out = (a >> b) + (one & ((mod > threshold) ? ~0 : 0));

/// @todo _C_ob not used
#define QUINT8_QUANTIZE_TILE_C(_W_ob, _C_ob, left_shift, right_shift, q_mul, offset, max_val, min_val) \
    {                                                                                                  \
        for (uint32_t kk = 0; kk < _W_ob; kk++)                                                        \
        {                                                                                              \
            for (uint32_t jj = 0; jj < QUINT8_C_ob; jj++)                                              \
            {                                                                                          \
                auto val = c_tile[kk * QUINT8_C_ob + jj];                                              \
                /*Multiply by q_mul and left shift*/                                                   \
                int64_t left_shift_val = static_cast<int64_t>(val * (1 << left_shift));                \
                VQRDMULH(left_shift_val, static_cast<int64_t>(q_mul), val);                            \
                RNDRSHIFT(val, right_shift, val);                                                      \
                c_tile[kk * QUINT8_C_ob + jj] = (val > min_val) ? ((val < max_val) ? val               \
                                                                                   : max_val)          \
                                                                : min_val;                             \
            }                                                                                          \
        }                                                                                              \
    }

/// @todo _C_ob not used
#define QUINT8_QUANTIZE_END_C(_W_ob, _C_ob, left_shift, right_shift, q_mul, offset, max_val, min_val) \
    {                                                                                                 \
        for (uint32_t kk = 0; kk < _W_ob; kk++)                                                       \
        {                                                                                             \
            for (uint32_t jj = 0; jj < QUINT8_C_ob; jj++)                                             \
            {                                                                                         \
                auto val = c_tile[kk * QUINT8_C_ob + jj];                                             \
                int64_t left_shift_val = static_cast<int64_t>(val * (1 << left_shift));               \
                VQRDMULH(left_shift_val, static_cast<int64_t>(q_mul), val);                           \
                RNDRSHIFT(val, right_shift, val);                                                     \
                c_tile[kk * QUINT8_C_ob + jj] = (val > min_val) ? ((val < max_val) ? val              \
                                                                                   : max_val)         \
                                                                : min_val;                            \
            }                                                                                         \
        }                                                                                             \
    }
