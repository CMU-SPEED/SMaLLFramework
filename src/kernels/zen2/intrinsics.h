//scalar versions of all the microkernels for platform portability

#include<immintrin.h>
//Architecture specific tiling params
// #define W_ob_dw 6
// #define W_ob_pool 3
// #define W_ob 6
// #define C_ob 16
// #define C_ib 16




// Initializations
#define DEF_TILE_C(W_ob, C_ob)\
__m256 a_reg, b0, b1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13;\

#define DEF_TILE_C_END(W_ob, C_ob) \
  float c_tile[W_ob * C_ob];

#define ZERO_TILE_C(W_ob, C_ob) \
  c0 = _mm256_setzero_ps();     \
  c1 = _mm256_setzero_ps();     \
  c2 = _mm256_setzero_ps();     \
  c3 = _mm256_setzero_ps();     \
  c4 = _mm256_setzero_ps();     \
  c5 = _mm256_setzero_ps();     \
  c6 = _mm256_setzero_ps();     \
  c7 = _mm256_setzero_ps();     \
  c8 = _mm256_setzero_ps();     \
  c9 = _mm256_setzero_ps();     \
  c10 = _mm256_setzero_ps();    \
  c11 = _mm256_setzero_ps();

#define ZERO_END_C(W_ob, C_ob)                    \
  for (uint32_t kk = 0; kk < W_ob; kk++)          \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = 0.0; \
    }                                             \
  }

// Loads

#define LOAD_TILE_C(O, W_ob, C_ob)                                                    \
    c0 = _mm256_load_ps(O + (0 * C_ob));                                              \
    c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);                                       \
    c2 = _mm256_load_ps(O + (1 * C_ob));                                              \
    c3 = _mm256_load_ps(O + (1 * C_ob) + SIMD);                                       \
    c4 = _mm256_load_ps(O + (2 * C_ob));                                              \
    c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);                                       \
    c6 = _mm256_load_ps(O + (3 * C_ob));                                              \
    c7 = _mm256_load_ps(O + (3 * C_ob) + SIMD);                                       \
    c8 = _mm256_load_ps(O + (4 * C_ob));                                              \
    c9 = _mm256_load_ps(O + (4 * C_ob) + SIMD);                                       \
    c10 = _mm256_load_ps(O + (5 * C_ob));                                             \
    c11 = _mm256_load_ps(O + (5 * C_ob) + SIMD);

#define LOAD_LAST_C(O, W_ob, C_ob, W_last)        \
  for (uint32_t kk = 0; kk < W_last; kk++)        \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
    }                                             \
  }


//Pooling Loads

#define LOAD_TILE_C_POOL(O, W_ob, C_ob)                                               \
    __m256 a_reg, b0, b1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13; \
    c0 = _mm256_load_ps(O + (0 * C_ob));                                              \
    c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);                                       \
    c2 = _mm256_load_ps(O + (1 * C_ob));                                              \
    c3 = _mm256_load_ps(O + (1 * C_ob) + SIMD);                                       \
    c4 = _mm256_load_ps(O + (2 * C_ob));                                              \
    c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);


#define LOAD_LAST_C_POOL(O, W_ob, C_ob, W_last)   \
  float c_tile[W_ob * C_ob];                      \
  for (uint32_t kk = 0; kk < W_last; kk++)        \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
    }                                             \
  }

#define LOAD_TILE_C_DW(O, W_ob, C_ob)                                                 \
    __m256 a_reg, b0, b1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13; \
    c0 = _mm256_load_ps(O + (0 * C_ob));                                              \
    c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);                                       \
    c2 = _mm256_load_ps(O + (1 * C_ob));                                              \
    c3 = _mm256_load_ps(O + (1 * C_ob) + SIMD);                                       \
    c4 = _mm256_load_ps(O + (2 * C_ob));                                              \
    c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);                                       \
    c6 = _mm256_load_ps(O + (3 * C_ob));                                              \
    c7 = _mm256_load_ps(O + (3 * C_ob) + SIMD);                                       \
    c8 = _mm256_load_ps(O + (4 * C_ob));                                              \
    c9 = _mm256_load_ps(O + (4 * C_ob) + SIMD);                                       \
    c10 = _mm256_load_ps(O + (5 * C_ob));                                             \
    c11 = _mm256_load_ps(O + (5 * C_ob) + SIMD);
// strided loads
#define LOAD_TILE_C_strided(O, step, _W_ob, _C_ob)                                  \
  c0 = _mm256_load_ps(O + (0 * step));                                              \
  c1 = _mm256_load_ps(O + (0 * step) + SIMD);                                       \
  c2 = _mm256_load_ps(O + (1 * step));                                              \
  c3 = _mm256_load_ps(O + (1 * step) + SIMD);                                       \
  c4 = _mm256_load_ps(O + (2 * step));                                              \
  c5 = _mm256_load_ps(O + (2 * step) + SIMD);                                       \
  c6 = _mm256_load_ps(O + (3 * step));                                              \
  c7 = _mm256_load_ps(O + (3 * step) + SIMD);                                       \
  c8 = _mm256_load_ps(O + (4 * step));                                              \
  c9 = _mm256_load_ps(O + (4 * step) + SIMD);                                       \
  c10 = _mm256_load_ps(O + (5 * step));                                             \
  c11 = _mm256_load_ps(O + (5 * step) + SIMD);

#define LOAD_TILE_C_strided_DW(O, step, _W_ob, _C_ob)                                    \
    __m256 a_reg, b0, b1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13; \
    c0 = _mm256_load_ps(O + (0 * step));                                              \
    c1 = _mm256_load_ps(O + (0 * step) + SIMD);                                       \
    c2 = _mm256_load_ps(O + (1 * step));                                              \
    c3 = _mm256_load_ps(O + (1 * step) + SIMD);                                       \
    c4 = _mm256_load_ps(O + (2 * step));                                              \
    c5 = _mm256_load_ps(O + (2 * step) + SIMD);                                       \
    c6 = _mm256_load_ps(O + (3 * step));                                              \
    c7 = _mm256_load_ps(O + (3 * step) + SIMD);                                       \
    c8 = _mm256_load_ps(O + (4 * step));                                              \
    c9 = _mm256_load_ps(O + (4 * step) + SIMD);                                       \
    c10 = _mm256_load_ps(O + (5 * step));                                             \
    c11 = _mm256_load_ps(O + (5 * step) + SIMD);

#define LOAD_LAST_C_strided(O, step, W_ob, C_ob, W_last) \
  for (uint32_t kk = 0; kk < W_last; kk++)               \
  {                                                      \
    for (uint32_t jj = 0; jj < C_ob; jj++)               \
    {                                                    \
      c_tile[kk * C_ob + jj] = O[kk * step + jj];        \
    }                                                    \
  }

//Stores

#define STORE_TILE_C(O, W_ob, C_ob)             \
    _mm256_store_ps(O + (0 * C_ob), c0);        \
    _mm256_store_ps(O + (0 * C_ob) + SIMD, c1); \
    _mm256_store_ps(O + (1 * C_ob), c2);        \
    _mm256_store_ps(O + (1 * C_ob + SIMD), c3); \
    _mm256_store_ps(O + (2 * C_ob), c4);        \
    _mm256_store_ps(O + (2 * C_ob + SIMD), c5); \
    _mm256_store_ps(O + (3 * C_ob), c6);        \
    _mm256_store_ps(O + (3 * C_ob + SIMD), c7); \
    _mm256_store_ps(O + (4 * C_ob), c8);        \
    _mm256_store_ps(O + (4 * C_ob + SIMD), c9); \
    _mm256_store_ps(O + (5 * C_ob), c10);       \
    _mm256_store_ps(O + (5 * C_ob + SIMD), c11);

#define STORE_END_C(O, W_ob, C_ob, W_last)        \
  for (uint32_t kk = 0; kk < W_last; kk++)        \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
    }                                             \
  }

#define STORE_TILE_C_POOL(O, W_ob_pool, C_ob)   \
    _mm256_store_ps(O + (0 * C_ob), c0);        \
    _mm256_store_ps(O + (0 * C_ob) + SIMD, c1); \
    _mm256_store_ps(O + (1 * C_ob), c2);        \
    _mm256_store_ps(O + (1 * C_ob + SIMD), c3); \
    _mm256_store_ps(O + (2 * C_ob), c4);        \
    _mm256_store_ps(O + (2 * C_ob + SIMD), c5); 


#define STORE_END_C_POOL(O, W_ob_pool, C_ob, W_last)        \
  for (uint32_t kk = 0; kk < W_last; kk++)        \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
    }                                             \
  }

#define STORE_TILE_INTER(W_ob, C_ob)             \
    float c_tile[W_ob*C_ob];\
    _mm256_store_ps(c_tile + (0 * C_ob), c0);        \
    _mm256_store_ps(c_tile + (0 * C_ob) + SIMD, c1); \
    _mm256_store_ps(c_tile + (1 * C_ob), c2);        \
    _mm256_store_ps(c_tile + (1 * C_ob + SIMD), c3); \
    _mm256_store_ps(c_tile + (2 * C_ob), c4);        \
    _mm256_store_ps(c_tile + (2 * C_ob + SIMD), c5); \
    _mm256_store_ps(c_tile + (3 * C_ob), c6);        \
    _mm256_store_ps(c_tile + (3 * C_ob + SIMD), c7); \
    _mm256_store_ps(c_tile + (4 * C_ob), c8);        \
    _mm256_store_ps(c_tile + (4 * C_ob + SIMD), c9); \
    _mm256_store_ps(c_tile + (5 * C_ob), c10);       \
    _mm256_store_ps(c_tile + (5 * C_ob + SIMD), c11);

//Convolution Computation
//(Strided GEMM)
// Pouint32_ter to C defined in the outer scope
#define FMA_TILE_C(step, a, b, p_cur, W_ob, C_ob)   \
    b0 = _mm256_load_ps(b + (p_cur * C_ob));        \
    b1 = _mm256_load_ps(b + (p_cur * C_ob + SIMD)); \
    a_reg = _mm256_broadcast_ss(a + (p_cur));       \
    p_cur += step;                                  \
    c0 = _mm256_fmadd_ps(a_reg, b0, c0);            \
    c1 = _mm256_fmadd_ps(a_reg, b1, c1);            \
    a_reg = _mm256_broadcast_ss(a + (p_cur));       \
    p_cur += step;                                  \
    c2 = _mm256_fmadd_ps(a_reg, b0, c2);            \
    c3 = _mm256_fmadd_ps(a_reg, b1, c3);            \
    a_reg = _mm256_broadcast_ss(a + (p_cur));       \
    p_cur += step;                                  \
    c4 = _mm256_fmadd_ps(a_reg, b0, c4);            \
    c5 = _mm256_fmadd_ps(a_reg, b1, c5);            \
    a_reg = _mm256_broadcast_ss(a + (p_cur));       \
    p_cur += step;                                  \
    c6 = _mm256_fmadd_ps(a_reg, b0, c6);            \
    c7 = _mm256_fmadd_ps(a_reg, b1, c7);            \
    a_reg = _mm256_broadcast_ss(a + (p_cur));       \
    p_cur += step;                                  \
    c8 = _mm256_fmadd_ps(a_reg, b0, c8);            \
    c9 = _mm256_fmadd_ps(a_reg, b1, c9);            \
    a_reg = _mm256_broadcast_ss(a + (p_cur));       \
    p_cur += step;                                  \
    c10 = _mm256_fmadd_ps(a_reg, b0, c10);          \
    c11 = _mm256_fmadd_ps(a_reg, b1, c11);

#define FMA_END_C(step, a, b, p_cur, W_ob, C_ob, W_last) \
  float *c_pixel;                                        \
  float *a_channel = a + p_cur;                          \
  for (uint32_t kk = 0; kk < W_last; kk++)               \
  {                                                      \
    float a_val = *(a_channel);                          \
    c_pixel = c_tile + kk * C_ob;                        \
    for (uint32_t jj = 0; jj < C_ob; jj++)               \
    {                                                    \
      float b_val = *(b + p_cur * C_ob + jj);            \
      *(c_pixel + jj) += a_val * b_val;                  \
    }                                                    \
    a_channel += step;                                   \
  }

//Pooling
//  Max pooling

#define MAX_TILE_C(step, a, W_ob, C_ob)        \
  b0 = _mm256_load_ps(a + (0 * step));         \
  b1 = _mm256_load_ps(a + (0 * step) + SIMD);  \
  c0 = _mm256_max_ps(b0, c0);                  \
  c1 = _mm256_max_ps(b1, c1);                  \
  a_reg = _mm256_load_ps(a + (1 * step));      \
  c12 = _mm256_load_ps(a + (1 * step) + SIMD); \
  c2 = _mm256_max_ps(a_reg, c2);               \
  c3 = _mm256_max_ps(c12, c3);                 \
  b0 = _mm256_load_ps(a + (2 * step));         \
  b1 = _mm256_load_ps(a + (2 * step) + SIMD);  \
  c4 = _mm256_max_ps(b0, c4);                  \
  c5 = _mm256_max_ps(b1, c5);                  \
  a_reg = _mm256_load_ps(a + (3 * step));      \
  c12 = _mm256_load_ps(a + (3 * step) + SIMD); \
  c6 = _mm256_max_ps(a_reg, c6);               \
  c7 = _mm256_max_ps(c12, c7);                 \
  b0 = _mm256_load_ps(a + (4 * step));         \
  b1 = _mm256_load_ps(a + (4 * step) + SIMD);  \
  c8 = _mm256_max_ps(b0, c8);                  \
  c9 = _mm256_max_ps(b1, c9);                  \
  a_reg = _mm256_load_ps(a + (5 * step));      \
  c12 = _mm256_load_ps(a + (5 * step) + SIMD); \
  c10 = _mm256_max_ps(a_reg, c10);             \
  c11 = _mm256_max_ps(c12, c11);

#define MAX_END_C(step, a, W_last, C_ob)                                          \
  float *c_pixel = c_tile;                                                        \
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

#define MAX_TILE_IP(pool_col_stride, W_ob, C_ob, pool_stride, pool_H_f, pool_W_f, O_row, O_col, O_pool, H_o, W_o_full)  \
  float *c_pixel = c_tile;                                                                                              \
  for (uint32_t kk = 0; kk < W_ob; kk++)                                                                                \
  {                                                                                                                     \
    if (O_row % pool_stride == 0 && (O_row + pool_H_f - 1) < H_o)                                                             \
    {                                                                                                                   \
      float *p_row = O_pool + ((O_row) / pool_stride) * pool_col_stride;                                                \
      if (O_col % pool_stride == 0 && (O_col + pool_W_f - 1) < W_o_full)                                                \
      {                                                                                                                 \
        float *p_pixel = p_row + ((O_col) / pool_stride) * C_ob;                                                        \
        float *p_channel = p_pixel;                                                                                     \
        float *c_channel = c_pixel;                                                                                     \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                                                          \
        {                                                                                                               \
          *(p_channel) = *(c_channel);                                                                                  \
          p_channel++;                                                                                                  \
          c_channel++;                                                                                                  \
        }                                                                                                               \
      }                                                                                                                 \
      for (uint32_t m_p = 1; m_p < pool_W_f; m_p++)                                                                     \
      {                                                                                                                 \
        if ((O_col - m_p) % pool_stride == 0 && (int)(O_col - m_p) >= 0 && (O_col + pool_W_f - (m_p + 1)) < W_o_full)   \
        {                                                                                                               \
          float *p_pixel = p_row + ((O_col - m_p) / pool_stride) * C_ob;                                                \
          float *p_channel = p_pixel;                                                                                   \
          float *c_channel = c_pixel;                                                                                   \
          for (uint32_t jj = 0; jj < C_ob; jj++)                                                                        \
          {                                                                                                             \
            *(p_channel) = (*(c_channel) > *(p_channel)) ? *(c_channel) : *(p_channel);                                 \
            p_channel++;                                                                                                \
            c_channel++;                                                                                                \
          }                                                                                                             \
        }                                                                                                               \
      }                                                                                                                 \
    }                                                                                                                   \
    for (uint32_t n_p = 1; n_p < pool_H_f; n_p++)                                                                       \
    {                                                                                                                   \
      if ((O_row - n_p) % pool_stride == 0 && (int)(O_row - n_p) >= 0 && (O_row + pool_H_f - (n_p + 1)) < H_o)          \
      {                                                                                                                 \
        float *p_row = O_pool + ((O_row - n_p) / pool_stride) * pool_col_stride;                                        \
        for (uint32_t m_p = 0; m_p < pool_W_f; m_p++)                                                                   \
        {                                                                                                               \
          if ((O_col - m_p) % pool_stride == 0 && (int)(O_col - m_p) >= 0 && (O_col + pool_W_f - (m_p + 1)) < W_o_full) \
          {                                                                                                             \
            float *p_pixel = p_row + ((O_col - m_p) / pool_stride) * C_ob;                                              \
            float *p_channel = p_pixel;                                                                                 \
            float *c_channel = c_pixel;                                                                                 \
            for (uint32_t jj = 0; jj < C_ob; jj++)                                                                      \
            {                                                                                                           \
              *(p_channel) = (*(c_channel) > *(p_channel)) ? *(c_channel) : *(p_channel);                               \
              p_channel++;                                                                                              \
              c_channel++;                                                                                              \
            }                                                                                                           \
          }                                                                                                             \
        }                                                                                                               \
      }                                                                                                                 \
    }                                                                                                                   \
    c_pixel += C_ob;                                                                                                    \
    O_col++;                                                                                                            \
  }

#define MAX_END_IP(pool_col_stride, W_last, C_ob, pool_stride, pool_H_f, pool_W_f, O_row, O_col, O_pool, H_o, W_o_full)             \
  float *c_pixel = c_tile;                                                                                                          \
  uint32_t O_col_cur = O_col;                                                                                                       \
  for (uint32_t kk = 0; kk < W_last; kk++)                                                                                          \
  {                                                                                                                                 \
    c_pixel = c_tile + kk * C_ob;                                                                                                   \
    if (O_row % pool_stride == 0 && (O_row + pool_H_f - 1)< H_o)                                                                         \
    {                                                                                                                               \
      float *p_row = O_pool + ((O_row) / pool_stride) * pool_col_stride;                                                            \
      if (O_col_cur % pool_stride == 0 && (O_col_cur + pool_W_f - 1) < W_o_full)                                                    \
      {                                                                                                                             \
        float *p_pixel = p_row + ((O_col_cur) / pool_stride) * C_ob;                                                                \
        float *p_channel = p_pixel;                                                                                                 \
        float *c_channel = c_pixel;                                                                                                 \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                                                                      \
        {                                                                                                                           \
          *(p_channel) = *(c_channel);                                                                                              \
                                                                                                                                    \
          p_channel++;                                                                                                              \
          c_channel++;                                                                                                              \
        }                                                                                                                           \
      }                                                                                                                             \
      for (uint32_t m_p = 1; m_p < pool_W_f; m_p++)                                                                                 \
      {                                                                                                                             \
        if ((O_col_cur - m_p) % pool_stride == 0 && (int)(O_col_cur - m_p) >= 0 && (O_col_cur + pool_W_f - (m_p + 1)) < W_o_full)   \
        {                                                                                                                           \
          float *p_pixel = p_row + ((O_col_cur - m_p) / pool_stride) * C_ob;                                                        \
          float *p_channel = p_pixel;                                                                                               \
          float *c_channel = c_pixel;                                                                                               \
          for (uint32_t jj = 0; jj < C_ob; jj++)                                                                                    \
          {                                                                                                                         \
            *(p_channel) = (*(c_channel) > *(p_channel)) ? *(c_channel) : *(p_channel);                                             \
            p_channel++;                                                                                                            \
            c_channel++;                                                                                                            \
          }                                                                                                                         \
        }                                                                                                                           \
      }                                                                                                                             \
    }                                                                                                                               \
    for (uint32_t n_p = 1; n_p < pool_H_f; n_p++)                                                                                   \
    {                                                                                                                               \
      if ((O_row - n_p) % pool_stride == 0 && (int)(O_row - n_p) >= 0 && (O_row + pool_H_f - (n_p + 1)) < H_o)                      \
      {                                                                                                                             \
        float *p_row = O_pool + ((O_row - n_p) / pool_stride) * pool_col_stride;                                                    \
        for (uint32_t m_p = 0; m_p < pool_W_f; m_p++)                                                                               \
        {                                                                                                                           \
          if ((O_col_cur - m_p) % pool_stride == 0 && (int)(O_col_cur - m_p) >= 0 && (O_col_cur + pool_W_f - (m_p + 1)) < W_o_full) \
          {                                                                                                                         \
            float *p_pixel = p_row + ((O_col_cur - m_p) / pool_stride) * C_ob;                                                      \
            float *p_channel = p_pixel;                                                                                             \
            float *c_channel = c_pixel;                                                                                             \
            for (uint32_t jj = 0; jj < C_ob; jj++)                                                                                  \
            {                                                                                                                       \
              *(p_channel) = (*(c_channel) > *(p_channel)) ? *(c_channel) : *(p_channel);                                           \
              p_channel++;                                                                                                          \
              c_channel++;                                                                                                          \
            }                                                                                                                       \
          }                                                                                                                         \
        }                                                                                                                           \
      }                                                                                                                             \
    }                                                                                                                               \
    O_col_cur++;                                                                                                                    \
  }

//DW Convolution
#define MUL_TILE_C(b, W_ob, C_ob)  \
    b0 = _mm256_load_ps(b);        \
    b1 = _mm256_load_ps(b + SIMD); \
    c0 = _mm256_mul_ps(b0, c0);    \
    c1 = _mm256_mul_ps(b1, c1);    \
    c2 = _mm256_mul_ps(b0, c2);    \
    c3 = _mm256_mul_ps(b1, c3);    \
    c4 = _mm256_mul_ps(b0, c4);    \
    c5 = _mm256_mul_ps(b1, c5);    \
    c6 = _mm256_mul_ps(b0, c6);    \
    c7 = _mm256_mul_ps(b1, c7);    \
    c8 = _mm256_mul_ps(b0, c8);    \
    c9 = _mm256_mul_ps(b1, c9);    \
    c10 = _mm256_mul_ps(b0, c10);    \
    c11 = _mm256_mul_ps(b1, c11);

#define MUL_END_C(b, W_ob, C_ob)           \
  float *c_pixel = c_tile;                 \
  for (uint32_t kk = 0; kk < W_ob; kk++)   \
  {                                        \
    float *c_channel = c_pixel;            \
    float *b_channel = b;                  \
    for (uint32_t jj = 0; jj < C_ob; jj++) \
    {                                      \
      *(c_channel) *= *(b_channel);        \
      c_channel++;                         \
      b_channel++;                         \
    }                                      \
    c_pixel += C_ob;                       \
  }

#define DW_TILE_C(step, a, b, W_ob, C_ob)        \
    b0 = _mm256_load_ps(b);                      \
    b1 = _mm256_load_ps(b + SIMD);               \
    c12 = _mm256_load_ps(a + (0 * step));        \
    c13 = _mm256_load_ps(a + (0 * step) + SIMD); \
    c0 = _mm256_fmadd_ps(b0, c12, c0);           \
    c1 = _mm256_fmadd_ps(b1, c13, c1);           \
    c12 = _mm256_load_ps(a + (1 * step));        \
    c13 = _mm256_load_ps(a + (1 * step) + SIMD); \
    c2 = _mm256_fmadd_ps(b0, c12, c2);                  \
    c3 = _mm256_fmadd_ps(b1, c13, c3);                  \
    c12 = _mm256_load_ps(a + (2 * step));        \
    c13 = _mm256_load_ps(a + (2 * step) + SIMD); \
    c4 = _mm256_fmadd_ps(b0, c12, c4);                  \
    c5 = _mm256_fmadd_ps(b1, c13, c5);                  \
    c12 = _mm256_load_ps(a + (3 * step));        \
    c13 = _mm256_load_ps(a + (3 * step) + SIMD); \
    c6 = _mm256_fmadd_ps(b0, c12, c6);                  \
    c7 = _mm256_fmadd_ps(b1, c13, c7);                  \
    c12 = _mm256_load_ps(a + (4 * step));        \
    c13 = _mm256_load_ps(a + (4 * step) + SIMD); \
    c8 = _mm256_fmadd_ps(b0, c12, c8);                  \
    c9 = _mm256_fmadd_ps(b1, c13, c9);                  \
    c12 = _mm256_load_ps(a + (5 * step));        \
    c13 = _mm256_load_ps(a + (5* step) + SIMD); \
    c10 = _mm256_fmadd_ps(b0, c12, c10);                \
    c11 = _mm256_fmadd_ps(b1, c13, c11);

#define DW_END_C(step, a, b, W_ob, C_ob)               \
  {                                                    \
    float *c_pixel = c_tile;                           \
    float *a_pixel = a;                                \
    for (uint32_t kk = 0; kk < W_ob; kk++)             \
    {                                                  \
      float *c_channel = c_pixel;                      \
      float *a_channel = a_pixel;                      \
      float *b_channel = b;                            \
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

#define DW_TILE_IP(pool_col_stride, W_ob, C_ob, pool_stride, pool_H_f, pool_W_f, F, O_row, O_col, O_pool, H_o, W_o_full) \
  float *c_pixel = c_tile;                                                                                               \
  for (uint32_t kk = 0; kk < W_ob; kk++)                                                                                 \
  {                                                                                                                      \
    if (O_row % pool_stride == 0 && (O_row + pool_H_f - 1) < H_o)                                                        \
    {                                                                                                                    \
      float *p_row = O_pool + ((O_row) / pool_stride) * pool_col_stride;                                                 \
      if (O_col % pool_stride == 0 && (O_col + pool_W_f - 1) < W_o_full)                                                 \
      {                                                                                                                  \
        float *p_pixel = p_row + ((O_col) / pool_stride) * C_ob;                                                         \
        float *p_channel = p_pixel;                                                                                      \
        float *c_channel = c_pixel;                                                                                      \
        float *b = F;                                                                                                    \
        float *b_channel = b;                                                                                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                                                           \
        {                                                                                                                \
          *(p_channel) = *(c_channel) * *(b_channel);                                                                    \
          p_channel++;                                                                                                   \
          c_channel++;                                                                                                   \
          b_channel++;                                                                                                   \
        }                                                                                                                \
      }                                                                                                                  \
      for (uint32_t m_p = 1; m_p < pool_W_f; m_p++)                                                                      \
      {                                                                                                                  \
        if ((O_col - m_p) % pool_stride == 0 && (int)(O_col - m_p) >= 0 && (O_col + pool_W_f - (m_p + 1)) < W_o_full)    \
        {                                                                                                                \
          float *p_pixel = p_row + ((O_col - m_p) / pool_stride) * C_ob;                                                 \
          float *p_channel = p_pixel;                                                                                    \
          float *c_channel = c_pixel;                                                                                    \
          float *b = F + m_p * C_ob;                                                                                     \
          float *b_channel = b;                                                                                          \
          for (uint32_t jj = 0; jj < C_ob; jj++)                                                                         \
          {                                                                                                              \
            *(p_channel) += *(c_channel) * *(b_channel);                                                                 \
            p_channel++;                                                                                                 \
            c_channel++;                                                                                                 \
            b_channel++;                                                                                                 \
          }                                                                                                              \
        }                                                                                                                \
      }                                                                                                                  \
    }                                                                                                                    \
    for (uint32_t n_p = 1; n_p < pool_H_f; n_p++)                                                                        \
    {                                                                                                                    \
      if ((O_row - n_p) % pool_stride == 0 && (int)(O_row - n_p) >= 0 && (O_row + pool_H_f - (n_p + 1)) < H_o)           \
      {                                                                                                                  \
        float *p_row = O_pool + ((O_row - n_p) / pool_stride) * pool_col_stride;                                         \
        for (uint32_t m_p = 0; m_p < pool_W_f; m_p++)                                                                    \
        {                                                                                                                \
          if ((O_col - m_p) % pool_stride == 0 && (int)(O_col - m_p) >= 0 && (O_col + pool_W_f - (m_p + 1)) < W_o_full)  \
          {                                                                                                              \
            float *p_pixel = p_row + ((O_col - m_p) / pool_stride) * C_ob;                                               \
            float *p_channel = p_pixel;                                                                                  \
            float *c_channel = c_pixel;                                                                                  \
            float *b = F + n_p * pool_W_f * C_ob + m_p * C_ob;                                                           \
            float *b_channel = b;                                                                                        \
            for (uint32_t jj = 0; jj < C_ob; jj++)                                                                       \
            {                                                                                                            \
              *(p_channel) += *(c_channel) * *(b_channel);                                                               \
              p_channel++;                                                                                               \
              c_channel++;                                                                                               \
              b_channel++;                                                                                               \
            }                                                                                                            \
          }                                                                                                              \
        }                                                                                                                \
      }                                                                                                                  \
    }                                                                                                                    \
    c_pixel += C_ob;                                                                                                     \
    O_col++;                                                                                                             \
  }

#define DW_END_IP(pool_col_stride, W_last, C_ob, pool_stride, pool_H_f, pool_W_f, F, O_row, O_col, O_pool, H_o, W_o_full) \
  float *c_pixel = c_tile;                                                                                                \
  uint32_t O_col_cur = O_col;                                                                                             \
  for (uint32_t kk = 0; kk < W_last; kk++)                                                                                \
  {                                                                                                                       \
    {                                                                                                                     \
      if (O_row % pool_stride == 0 && (O_row + pool_H_f - 1) < H_o)                                                       \
      {                                                                                                                   \
        float *p_row = O_pool + ((O_row) / pool_stride) * pool_col_stride;                                                \
        if (O_col % pool_stride == 0 && (O_col + pool_W_f - 1) < W_o_full)                                                \
        {                                                                                                                 \
          float *p_pixel = p_row + ((O_col) / pool_stride) * C_ob;                                                        \
          float *p_channel = p_pixel;                                                                                     \
          float *c_channel = c_pixel;                                                                                     \
          float *b = F;                                                                                                   \
          float *b_channel = b;                                                                                           \
          for (uint32_t jj = 0; jj < C_ob; jj++)                                                                          \
          {                                                                                                               \
            *(p_channel) = *(c_channel) * *(b_channel);                                                                   \
            p_channel++;                                                                                                  \
            c_channel++;                                                                                                  \
            b_channel++;                                                                                                  \
          }                                                                                                               \
        }                                                                                                                 \
        for (uint32_t m_p = 1; m_p < pool_W_f; m_p++)                                                                     \
        {                                                                                                                 \
          if ((O_col - m_p) % pool_stride == 0 && (int)(O_col - m_p) >= 0 && (O_col + pool_W_f - (m_p + 1)) < W_o_full)   \
          {                                                                                                               \
            float *p_pixel = p_row + ((O_col - m_p) / pool_stride) * C_ob;                                                \
            float *p_channel = p_pixel;                                                                                   \
            float *c_channel = c_pixel;                                                                                   \
            float *b = F + m_p * C_ob;                                                                                    \
            float *b_channel = b;                                                                                         \
            for (uint32_t jj = 0; jj < C_ob; jj++)                                                                        \
            {                                                                                                             \
              *(p_channel) += *(c_channel) * *(b_channel);                                                                \
              p_channel++;                                                                                                \
              c_channel++;                                                                                                \
              b_channel++;                                                                                                \
            }                                                                                                             \
          }                                                                                                               \
        }                                                                                                                 \
      }                                                                                                                   \
      for (uint32_t n_p = 1; n_p < pool_H_f; n_p++)                                                                       \
      {                                                                                                                   \
        if ((O_row - n_p) % pool_stride == 0 && (int)(O_row - n_p) >= 0 && (O_row + pool_H_f - (n_p + 1)) < H_o)          \
        {                                                                                                                 \
          float *p_row = O_pool + ((O_row - n_p) / pool_stride) * pool_col_stride;                                        \
          for (uint32_t m_p = 0; m_p < pool_W_f; m_p++)                                                                   \
          {                                                                                                               \
            if ((O_col - m_p) % pool_stride == 0 && (int)(O_col - m_p) >= 0 && (O_col + pool_W_f - (m_p + 1)) < W_o_full) \
            {                                                                                                             \
              float *p_pixel = p_row + ((O_col - m_p) / pool_stride) * C_ob;                                              \
              float *p_channel = p_pixel;                                                                                 \
              float *c_channel = c_pixel;                                                                                 \
              float *b = F + n_p * pool_W_f * C_ob + m_p * C_ob;                                                          \
              float *b_channel = b;                                                                                       \
              for (uint32_t jj = 0; jj < C_ob; jj++)                                                                      \
              {                                                                                                           \
                *(p_channel) += *(c_channel) * *(b_channel);                                                              \
                p_channel++;                                                                                              \
                c_channel++;                                                                                              \
                b_channel++;                                                                                              \
              }                                                                                                           \
            }                                                                                                             \
          }                                                                                                               \
        }                                                                                                                 \
      }                                                                                                                   \
      c_pixel += C_ob;                                                                                                    \
      O_col++;                                                                                                            \
    }                                                                                                                     \
  }


//ReLU Activation
// Same kernel as Pooling, set to zero to start.
