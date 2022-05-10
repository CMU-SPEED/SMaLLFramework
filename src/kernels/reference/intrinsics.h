//scalar versions of all the microkernels for platform portability

//Architecture specific tiling params


// __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

// Initializations
#define DEF_TILE_C(W_ob, C_ob)
  float c_tile[W_ob * C_ob];

#define DEF_TILE_C_END(W_ob, C_ob) \
  float c_tile[W_ob * C_ob];       \
//  float c_tile[W_ob * C_ob];
#define ZERO_TILE_C(W_ob, C_ob)            \
  for (uint32_t kk = 0; kk < W_ob; kk++)   \
  {                                        \
    for (uint32_t jj = 0; jj < C_ob; jj++) \
    {                                      \
      c_tile[kk * C_ob + jj] = 0.0;        \
    }                                             \
  }

//  float c_tile[W_ob * C_ob];

#define ZERO_END_C(W_ob, C_ob)                    \
  for (uint32_t kk = 0; kk < W_ob; kk++)          \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = 0.0; \
    }                                             \
  }

// Loads
//  float c_tile[W_ob * C_ob]; 
#define LOAD_TILE_C(O, W_ob, C_ob)                \
  for (uint32_t kk = 0; kk < W_ob; kk++)          \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
    }                                             \
  }

//  float c_tile[W_ob * C_ob]; 
#define LOAD_LAST_C(O, W_ob, C_ob, W_last)        \
  for (uint32_t kk = 0; kk < W_last; kk++)        \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
    }                                             \
  }


//Pooling Loads

#define LOAD_TILE_C_POOL(O, W_ob, C_ob)      \
  float c_tile[W_ob * C_ob];                      \
  for (uint32_t kk = 0; kk < W_ob; kk++)          \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
    }                                             \
  }

#define LOAD_LAST_C_POOL(O, W_ob, C_ob, W_last)   \
  float c_tile[W_ob * C_ob];                      \
  for (uint32_t kk = 0; kk < W_last; kk++)        \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
    }                                             \
  }

#define LOAD_TILE_C_DW(O, W_ob, C_ob)             \
  float c_tile[W_ob * C_ob];                      \
  for (uint32_t kk = 0; kk < W_ob; kk++)        \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
    }                                             \
  }

// strided loads
//  float c_tile[_W_ob * _C_ob];
#define LOAD_TILE_C_strided(O, step, _W_ob, _C_ob) \
  for (uint32_t kk = 0; kk < _W_ob; kk++)          \
  {                                                \
    for (uint32_t jj = 0; jj < _C_ob; jj++)        \
    {                                              \
      c_tile[kk * _C_ob + jj] = O[kk * step + jj]; \
    }                                              \
  }

//  float c_tile[W_ob * C_ob];
#define LOAD_LAST_C_strided(O, step, W_ob, C_ob, W_last) \
  for (uint32_t kk = 0; kk < W_last; kk++)               \
  {                                                      \
    for (uint32_t jj = 0; jj < C_ob; jj++)               \
    {                                                    \
      c_tile[kk * C_ob + jj] = O[kk * step + jj];        \
    }                                                    \
  }

#define LOAD_TILE_C_strided_DW(O, step, _W_ob, _C_ob) \
  float c_tile[_W_ob * _C_ob];                        \
  for (uint32_t kk = 0; kk < _W_ob; kk++)             \
  {                                                   \
    for (uint32_t jj = 0; jj < _C_ob; jj++)           \
    {                                                 \
      c_tile[kk * _C_ob + jj] = O[kk * step + jj];    \
    }                                                 \
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

#define STORE_END_C(O, W_ob, C_ob, W_last)        \
  for (uint32_t kk = 0; kk < W_last; kk++)        \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
    }                                             \
  }

#define STORE_TILE_C_POOL(O, W_ob_pool, C_ob)               \
  for (uint32_t kk = 0; kk < W_ob_pool; kk++)          \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
    }                                             \
  }

#define STORE_END_C_POOL(O, W_ob_pool, C_ob, W_last)        \
  for (uint32_t kk = 0; kk < W_last; kk++)        \
  {                                               \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
    }                                             \
  }

#define STORE_TILE_INTER(W_ob, C_ob)               \
  void * do_nothing = NULL;
                             // Convolution Computation
//(Strided GEMM)
// Pouint32_ter to C defined in the outer scope
#define FMA_TILE_C(step, a, b, p_cur, W_ob, C_ob) \
  float *c_pixel;                                 \
  for (uint32_t kk = 0; kk < W_ob; kk++)          \
  {                                               \
    float a_val = *(a + p_cur + kk * step);       \
    c_pixel = c_tile + kk * C_ob;                 \
    for (uint32_t jj = 0; jj < C_ob; jj++)        \
    {                                             \
      float b_val = *(b + p_cur * C_ob + jj);     \
      *(c_pixel + jj) += a_val * b_val;           \
    }                                             \
  }

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

#define MAX_TILE_C(step, a, W_ob, C_ob)                                           \
  float *c_pixel = c_tile;                                                        \
  float *a_pixel = a;                                                             \
  for (uint32_t kk = 0; kk < W_ob; kk++)                                          \
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
#define MUL_TILE_C(b, W_ob, C_ob)          \
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

#define DW_TILE_C(step, a, b, W_ob, C_ob)              \
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
        c_channel++;                                   \
        b_channel++;                                   \
        a_channel++;                                   \
      }                                                \
      a_pixel += step;                                 \
      c_pixel += C_ob;                                 \
    }                                                  \
  }

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
        c_channel++;                                   \
        b_channel++;                                   \
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

#define ADD_TILE_C_G(I, W_ob_g, C_ob)      \
  float *i_pixel = I;                      \
  float *c_pixel = c_tile;                 \
  for (uint32_t mm = 0; mm < W_ob_g; mm++) \
  {                                        \
    float *c_channel = c_pixel;            \
    float *i_channel = i_pixel;           \
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
  float *i_pixel = I;                      \
  float *c_pixel = c_tile;                 \
  for (uint32_t mm = 0; mm < W_last; mm++) \
  {                                        \
    float *c_channel = c_pixel;            \
    float *i_channel = i_pixel;           \
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

#define REDUCE_C(O, W_ob_g, C_ob)         \
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

#define REDUCE_C_last(O, W_last, C_ob)            \
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

  