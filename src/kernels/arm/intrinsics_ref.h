#include <arm_neon.h>

// https://developer.arm.com/architectures/instruction-sets/intrinsics/

//Architecture specific tiling params


// __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

// Initializations

// float32x4_t vectorizes C_ob dim: [W_ob, C_ob] -> [W_ob, C_ob/SIMD, SIMD].
// assume SIMD == 4 and vec type is float.
// otherwise, SIMD = Neon bit width (128) / data type size.
#define DEF_TILE_C(W_ob, C_ob)\
  float c_tile[W_ob * C_ob];

#define DEF_END_C(W_ob, C_ob) \
  float c_tile[W_ob * C_ob];
  

//  float c_tile[W_ob * C_ob];
#define ZERO_TILE_C(W_ob, C_ob)            \
  for (uint32_t kk = 0; kk < W_ob; kk++)   \
  {                                        \
    for (uint32_t jj = 0; jj < C_ob; jj++) \
    {                                      \
      c_tile[kk * C_ob + jj] = 0.0;        \
    }                                      \
  }

//  float c_tile[W_ob * C_ob];
#define ZERO_END_C(_W_ob, C_ob)                   \
  for (uint32_t kk = 0; kk < _W_ob; kk++)         \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = 0.0;               \
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

//  float c_tile[W_ob * C_ob]; 
#define LOAD_END_C(O, _W_ob, C_ob)                \
  for (uint32_t kk = 0; kk < _W_ob; kk++)         \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
    }                                             \
  }


//Pooling Loads
// TODO: vectorize pooling loads
// strided loads
#define LOAD_TILE_C_strided(O, step, W_ob, C_ob) \
  for (uint32_t kk = 0; kk < W_ob; kk++)          \
  {                                                \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                              \
      c_tile[kk * C_ob + jj] = O[kk * step + jj]; \
    }                                              \
  }

//  float c_tile[W_ob * C_ob];
#define LOAD_END_C_strided(O, step, _W_ob, C_ob)   \
  for (uint32_t kk = 0; kk < _W_ob; kk++)          \
  {                                                \
    for (uint32_t jj = 0; jj < C_ob; jj++)         \
    {                                              \
      c_tile[kk * C_ob + jj] = O[kk * step + jj];  \
    }                                              \
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

#define STORE_END_C(O, _W_ob, C_ob)               \
  for (uint32_t kk = 0; kk < _W_ob; kk++)         \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
    }                                             \
  }


// Convolution Computation
//(Strided GEMM)
// // Pouint32_ter to C defined in the outer scope
// #define FMA_TILE_C(step, a, b, p_cur, W_ob, C_ob) \
//   float *c_pixel;                                 \
//   for (uint32_t kk = 0; kk < W_ob; kk++)          \
//   {                                               \
//     float a_val = *(a + p_cur + kk * step);       \
//     c_pixel = c_tile + kk * C_ob;                 \
//     for (uint32_t jj = 0; jj < C_ob; jj++)        \
//     {                                             \
//       float b_val = *(b + p_cur * C_ob + jj);     \
//       *(c_pixel + jj) += a_val * b_val;           \
//     }                                             \
//   }

// #define FMA_END_C(step, a, b, p_cur, W_ob, C_ob, W_last) \
//   float *c_pixel;                                        \
//   float *a_channel = a + p_cur;                          \
//   for (uint32_t kk = 0; kk < W_last; kk++)               \
//   {                                                      \
//     float a_val = *(a_channel);                          \
//     c_pixel = c_tile + kk * C_ob;                        \
//     for (uint32_t jj = 0; jj < C_ob; jj++)               \
//     {                                                    \
//       float b_val = *(b + p_cur * C_ob + jj);            \
//       *(c_pixel + jj) += a_val * b_val;                  \
//     }                                                    \
//     a_channel += step;                                   \
//   }

#define CONV_TILE_C(step, a, b, W_ob, C_ob)           \
  for (uint32_t kk = 0; kk < W_ob; kk++)              \
  {                                                   \
    for (uint32_t jj = 0; jj < C_ob; jj++)            \
    {                                                 \
      c_tile[kk * C_ob + jj] += a[kk * step] * b[jj]; \
    }                                                 \
  }

#define CONV_END_C(step, a, b, c_cur, _W_ob, C_ob)   \
  for (uint32_t kk = 0; kk < _W_ob; kk++)            \
  {                                                  \
    for (uint32_t jj = 0; jj < C_ob; jj++)           \
    {                                                \
      c_cur[kk * C_ob + jj] += a[kk * step] * b[jj]; \
    }                                                \
  }

//Pooling
//  Max pooling
// ReLU

#define MAX_TILE_C(step, a, W_ob, C_ob)               \
  for (uint32_t kk = 0; kk < W_ob; kk++)              \
  {                                                   \
    for (uint32_t jj = 0; jj < C_ob; jj++)            \
    {                                                 \
      c_tile[kk * C_ob + jj] =                        \
        a[kk * step + jj] > c_tile[kk * C_ob + jj] ?  \
        a[kk * step + jj] : c_tile[kk * C_ob + jj];   \
    }                                                 \
  }

#define MAX_END_C(step, a, b, c_cur, W_last, C_ob)                                \
  float *c_pixel = c_cur;                                                         \
  float *a_pixel = a;                                                             \
  for (uint32_t kk = 0; kk < W_last; kk++)                                        \
  {                                                                               \
    float *c_channel = c_pixel;                                                   \
    float *a_channel = a_pixel;                                                   \
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
#define DW_TILE_C(step, a, b, W_ob, C_ob)                  \
  for (uint32_t kk = 0; kk < W_ob; kk++)                   \
  {                                                        \
    for (uint32_t jj = 0; jj < C_ob; jj++)                 \
    {                                                      \
      c_tile[kk * C_ob + jj] += a[kk * step + jj] * b[jj]; \
    }                                                      \
  }

#define DW_END_C(step, a, b, c_cur, _W_ob, C_ob)       \
  {                                                    \
    float *c_pixel = c_cur;                            \
    float *a_pixel = a;                                \
    for (uint32_t kk = 0; kk < _W_ob; kk++)            \
    {                                                  \
      float *c_channel = c_pixel;                      \
      float *a_channel = a_pixel;                      \
      float *b_channel = b;                            \
      for (uint32_t jj = 0; jj < C_ob; jj++)           \
      {                                                \
        *(c_channel) += (*(a_channel) * *(b_channel)); \
  c_channel++;\
  b_channel++;\
  a_channel++;                                         \
      }                                                \
      a_pixel += step;                                 \
      c_pixel += C_ob;                                 \
    }                                                  \
  }





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

#define REDUCE_div_C(O, d, W_ob_g, C_ob)   \
{ float *c_pixel = c_tile;                 \
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
  }\
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

  