// SMaLL, Software for Machine Learning Libraries
// Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126

//scalar versions of all the microkernels for platform portability

#define SIMD_EPILOGUE 1
typedef dtype c_tile_t;

//Architecture specific tiling params

// __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;


// Initializations

#define DEF_TILE_C(W_ob, C_ob)\
  dtype c_tile[W_ob * C_ob];


#define DEF_END_C(W_ob, C_ob) \
  dtype c_tile[W_ob * C_ob];


#define ZERO_TILE_C(W_ob, C_ob)            \
  for (uint32_t kk = 0; kk < W_ob; kk++)   \
  {                                        \
    for (uint32_t jj = 0; jj < C_ob; jj++) \
    {                                      \
      c_tile[kk * C_ob + jj] = 0.0;        \
    }                                             \
  }


#define ZERO_END_C(_W_ob, C_ob)                    \
  for (uint32_t kk = 0; kk < _W_ob; kk++)          \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = 0.0; \
    }                                             \
  }


// Loads

#define LOAD_TILE_C(O, W_ob, C_ob)                \
  for (uint32_t kk = 0; kk < W_ob; kk++)          \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
    }                                             \
  }

//  dtype c_tile[W_ob * C_ob];
#define LOAD_END_C(O, _W_ob, C_ob)           \
  for (uint32_t kk = 0; kk < _W_ob; kk++)        \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
    }                                             \
  }


//Pooling Loads

// strided loads
#define LOAD_TILE_C_strided(O, step, _W_ob, _C_ob) \
  for (uint32_t kk = 0; kk < _W_ob; kk++)          \
  {                                                \
    for (uint32_t jj = 0; jj < _C_ob; jj++)        \
    {                                              \
      c_tile[kk * _C_ob + jj] = O[kk * step + jj]; \
    }                                              \
  }

//  dtype c_tile[W_ob * C_ob];
#define LOAD_END_C_strided(O, step, _W_ob, C_ob) \
  for (uint32_t kk = 0; kk < _W_ob; kk++)               \
  {                                                      \
    for (uint32_t jj = 0; jj < C_ob; jj++)               \
    {                                                    \
      c_tile[kk * C_ob + jj] = O[kk * step + jj];        \
    }                                                    \
  }


// Stores

#define STORE_TILE_C(O, W_ob, C_ob)               \
  for (uint32_t kk = 0; kk < W_ob; kk++)          \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
    }                                             \
  }

#define STORE_END_C(O, _W_ob, C_ob)       \
  for (uint32_t kk = 0; kk < _W_ob; kk++)        \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
    }                                             \
  }


#if 0
// Convolution Computation
//(Strided GEMM)
// Pouint32_ter to C defined in the outer scope
#define FMA_TILE_C(step, a, b, p_cur, W_ob, C_ob) \
  dtype *c_pixel;                                 \
  for (uint32_t kk = 0; kk < W_ob; kk++)          \
  {                                               \
    dtype a_val = *(a + p_cur + kk * step);       \
    c_pixel = c_tile + kk * C_ob;                 \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      dtype b_val = *(b + p_cur * C_ob + jj);     \
      *(c_pixel + jj) += a_val * b_val;           \
    }                                             \
  }

#define FMA_END_C(step, a, b, p_cur, W_ob, C_ob, W_last) \
  dtype *c_pixel;                                        \
  dtype const *a_channel = a + p_cur;                          \
  for (uint32_t kk = 0; kk < W_last; kk++)               \
  {                                                      \
    dtype a_val = *(a_channel);                          \
    c_pixel = c_tile + kk * C_ob;                        \
    for (uint32_t jj = 0; jj < C_ob; jj++)               \
    {                                                    \
      dtype b_val = *(b + p_cur * C_ob + jj);            \
      *(c_pixel + jj) += a_val * b_val;                  \
    }                                                    \
    a_channel += step;                                   \
  }
#endif

#define CONV_TILE_C(step, a, b, W_ob, C_ob) \
  dtype *c_pixel = c_tile;                           \
  dtype const *a_channel = a;                     \
  for (uint32_t kk = 0; kk < W_ob; kk++)   \
  {                                         \
    dtype a_val = *(a_channel);             \
    dtype * c_channel = c_pixel;            \
    for (uint32_t jj = 0; jj < C_ob; jj++)  \
    {                                       \
      dtype b_val = *(b + jj);              \
      *(c_channel) += a_val * b_val;     \
      c_channel++;\
    }                                       \
    a_channel += step;                      \
    c_pixel += C_ob;\
  }

#define CONV_END_C(step, a, b, c_cur, _W_ob, C_ob) \
  dtype *c_pixel = c_cur;                                        \
  dtype const *a_channel = a;                          \
  for (uint32_t kk = 0; kk < _W_ob; kk++)               \
  {                                                      \
    dtype a_val = *(a_channel);                          \
    dtype * c_channel = c_pixel;                        \
    for (uint32_t jj = 0; jj < C_ob; jj++)               \
    {                                                    \
      dtype b_val = *(b + jj);            \
      *(c_channel) += a_val * b_val;\
      c_channel++;                  \
    }                                                    \
    a_channel += step;\
    c_pixel += C_ob;\
  }

//Pooling
//  Max pooling

#define MAX_TILE_C(step, a, W_ob, C_ob)                                           \
  dtype *c_pixel = c_tile;                                                        \
  dtype const *a_pixel = a;                                                             \
  for (uint32_t kk = 0; kk < W_ob; kk++)                                          \
  {                                                                               \
    dtype *c_channel = c_pixel;                                                   \
    dtype const *a_channel = a_pixel;                                                   \
    for (uint32_t jj = 0; jj < C_ob; jj++)                                        \
    {                                                                             \
      *(c_channel) = (*(a_channel) > *(c_channel)) ? *(a_channel) : *(c_channel); \
      c_channel++;                                                                \
      a_channel++;                                                                \
    }                                                                             \
    a_pixel += step;                                                              \
    c_pixel += C_ob;                                                              \
  }


#define MAX_END_C(step, a, b, c_cur, W_last, C_ob)                                          \
  dtype *c_pixel = c_cur;                                                        \
  dtype const *a_pixel = a;                                                             \
  for (uint32_t kk = 0; kk < W_last; kk++)                                        \
  {                                                                               \
    dtype *c_channel = c_pixel;                                                   \
    dtype const *a_channel = a_pixel;                                                   \
    for (uint32_t jj = 0; jj < C_ob; jj++)                                        \
    {                                                                             \
      *(c_channel) = (*(a_channel) > *(c_channel)) ? *(a_channel) : *(c_channel); \
      c_channel++;                                                                \
      a_channel++;                                                                \
    }                                                                             \
    a_pixel += step;                                                              \
    c_pixel += C_ob;                                                              \
  }

//DW Convolution
#define DW_TILE_C(step, a, b, W_ob, C_ob)              \
  {                                                    \
    dtype *c_pixel = c_tile;                           \
    dtype const *a_pixel = a;                                \
    for (uint32_t kk = 0; kk < W_ob; kk++)             \
    {                                                  \
      dtype *c_channel = c_pixel;                      \
      dtype const *a_channel = a_pixel;                      \
      dtype const *b_channel = b;                            \
      for (uint32_t jj = 0; jj < C_ob; jj++)           \
      {                                                \
        *(c_channel) += (*(a_channel) * *(b_channel)); \
        c_channel++;                                   \
        b_channel++;                                   \
        a_channel++;                                   \
      }                                                \
      a_pixel += step;                                 \
      c_pixel += C_ob;                                 \
    }                                                  \
  }

#define DW_END_C(step, a, b, c_cur, _W_ob, C_ob)               \
  {                                                    \
    dtype *c_pixel = c_cur;                           \
    dtype const *a_pixel = a;                                \
    for (uint32_t kk = 0; kk < _W_ob; kk++)             \
    {                                                  \
      dtype *c_channel = c_pixel;                      \
      dtype const *a_channel = a_pixel;                      \
      dtype const *b_channel = b;                            \
      for (uint32_t jj = 0; jj < C_ob; jj++)           \
      {                                                \
        *(c_channel) += (*(a_channel) * *(b_channel)); \
  c_channel++;\
  b_channel++;\
  a_channel++;                                   \
      }                                                \
      a_pixel += step;                                 \
      c_pixel += C_ob;                                 \
    }                                                  \
  }



// AVG Pooling
#define ADD_TILE_C_G(I, W_ob_g, C_ob)      \
  dtype const *i_pixel = I;                      \
  dtype *c_pixel = c_tile;                 \
  for (uint32_t mm = 0; mm < W_ob_g; mm++) \
  {                                        \
    dtype *c_channel = c_pixel;            \
    dtype const *i_channel = i_pixel;           \
    for (uint32_t kk = 0; kk < C_ob; kk++) \
    {                                      \
      *c_channel += *i_channel;              \
      c_channel++;                         \
      i_channel++;                         \
    }                                      \
    c_pixel += C_ob;                       \
    i_pixel += C_ob;                       \
  }

#define ADD_LAST_C_G(I, W_last, C_ob)      \
  dtype const *i_pixel = I;                      \
  dtype *c_pixel = c_tile;                 \
  for (uint32_t mm = 0; mm < W_last; mm++) \
  {                                        \
    dtype *c_channel = c_pixel;            \
    dtype const *i_channel = i_pixel;           \
    for (uint32_t kk = 0; kk < C_ob; kk++) \
    {                                      \
      *c_channel += *i_channel;              \
      c_channel++;                         \
      i_channel++;                         \
    }                                      \
    c_pixel += C_ob;                       \
    i_pixel += C_ob;                       \
  }

#define REDUCE_div_C(O, d, W_ob_g, C_ob)       \
{ dtype *c_pixel = c_tile;                 \
  dtype *O_channel = O;                    \
  dtype *c_channel = c_pixel;              \
  for (uint32_t mm = 0; mm < W_ob_g; mm++) \
  {                                        \
    dtype *O_channel = O;                  \
    dtype *c_channel = c_pixel;            \
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
  }\
}

#define REDUCE_C(O, W_ob_g, C_ob)         \
  {                                          \
    dtype *c_pixel = c_tile;                 \
    dtype *O_channel = O;                    \
    dtype *c_channel = c_pixel;              \
    for (uint32_t mm = 0; mm < W_ob_g; mm++) \
    {                                        \
      dtype *O_channel = O;                  \
      dtype *c_channel = c_pixel;            \
      for (uint32_t kk = 0; kk < C_ob; kk++) \
      {                                      \
        *O_channel += *c_channel;            \
        O_channel++;                         \
        c_channel++;                         \
      }                                      \
      c_pixel += C_ob;                       \
    }                                        \
  }

#define REDUCE_C_last(O, W_last, C_ob)            \
  {                                          \
    dtype *c_pixel = c_tile;                 \
    dtype *O_channel = O;                    \
    dtype *c_channel = c_pixel;              \
    for (uint32_t mm = 0; mm < W_ob_g; mm++) \
    {                                        \
      dtype *O_channel = O;                  \
      dtype *c_channel = c_pixel;            \
      for (uint32_t kk = 0; kk < C_ob; kk++) \
      {                                      \
        *O_channel += *c_channel;            \
        O_channel++;                         \
        c_channel++;                         \
      }                                      \
      c_pixel += C_ob;                       \
    }                                        \
  }
