// Header File For different Versions of Fusing Pooling with a Convolution
#define POOL_UNROLL 8


#define LOAD_C(){\
  c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));\
   c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);\
   c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));\
   c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);\
   c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));\
   c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);\
   c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));\
   c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);\
   c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));\
   c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);\
   c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));\
   c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);\
}

#define STORE_C(){\
  _mm256_store_ps(O_buffer + (0 * C_ob), c0);\
  _mm256_store_ps(O_buffer + (0 * C_ob) + SIMD, c1);\
  _mm256_store_ps(O_buffer + (1 * C_ob), c2);\
  _mm256_store_ps(O_buffer + (1 * C_ob + SIMD), c3);\
  _mm256_store_ps(O_buffer + (2 * C_ob), c4);\
  _mm256_store_ps(O_buffer + (2 * C_ob + SIMD), c5);\
  _mm256_store_ps(O_buffer + (3 * C_ob), c6);\
  _mm256_store_ps(O_buffer + (3 * C_ob + SIMD), c7);\
  _mm256_store_ps(O_buffer + (4 * C_ob), c8);\
  _mm256_store_ps(O_buffer + (4 * C_ob + SIMD), c9);\
  _mm256_store_ps(O_buffer + (5 * C_ob), c10);\
  _mm256_store_ps(O_buffer + (5 * C_ob + SIMD), c11);\
}

// kernels
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_pool_first_row_start(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }


  // horizontal pooling
  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);



  // store to output of pooling layer

  _mm256_store_ps(O + (0 * C_ob), c2);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);

  _mm256_store_ps(O + (1 * C_ob), c6);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c7);

  _mm256_store_ps(O + (2 * C_ob), c10);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c11);


}
// load partial output from previous tile
// compute partial update for next tile
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_pool_first_row(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }

  // horizontal pooling

  // load partial update from previous
  b0 = _mm256_load_ps(O);
  b1 = _mm256_load_ps(O + SIMD);

  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);

  //update previoud tile
  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1,b1);

  // store to output of pooling layer
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O + (2 * C_ob), c6);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);

  _mm256_store_ps(O + (3 * C_ob), c10);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c11);


}

template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_pool_first_row_end(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
      }
    }
  }

  // horizontal pooling
  // load the partial update from the previous tile
  b0 = _mm256_load_ps(O);
  b1 = _mm256_load_ps(O + SIMD);

  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  // c10 = _mm256_max_ps(c10,c8);
  // c11 = _mm256_max_ps(c11,c9);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);

  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1,b1);

  // store to output of pooling layer
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O + (2 * C_ob), c6);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);




}


template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_pool_accum_start(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }

  // horizontal pooling
  // load partial updates from previous rows (only ones being finalized)
  b0 = _mm256_load_ps(O + (0 * C_ob));
  b1 = _mm256_load_ps(O + (1 * C_ob));
  a_reg = _mm256_load_ps(O + (2 * C_ob));
  __m256 temp = _mm256_load_ps(O + (0 * C_ob) + SIMD);

  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

  c0 = _mm256_load_ps(O + (1 * C_ob) + SIMD);
  c1 = _mm256_load_ps(O + (2 * C_ob) + SIMD);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);

  //accumulate with previous rows
  c2 = _mm256_max_ps(c2, b0);
  c3 = _mm256_max_ps(c3, temp);

  c6 = _mm256_max_ps(c6, b1);
  c7 = _mm256_max_ps(c7, c0);

  c10 = _mm256_max_ps(c10, a_reg);
  c11 = _mm256_max_ps(c11, c1);

  // store to output of pooling layer

  _mm256_store_ps(O + (0 * C_ob), c2);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);

  _mm256_store_ps(O + (1 * C_ob), c6);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c7);

  _mm256_store_ps(O + (2 * C_ob), c10);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c11);


}
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_pool_accum(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }

  // horizontal pooling previous tile
  b0 = _mm256_load_ps(O);
  b1 = _mm256_load_ps(O + SIMD);


  a_reg = _mm256_load_ps(O + (1 * C_ob));
  __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);

  //
  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);


  c4 = _mm256_load_ps(O + (2 * C_ob));
  c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);

  c8 = _mm256_load_ps(O + (3 * C_ob));
  c9 = _mm256_load_ps(O + (3 * C_ob) + SIMD);


  // accumulate with previous tile
  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1,b1);
  //accumulate with previous row
  c2 = _mm256_max_ps(c2, a_reg);
  c3 = _mm256_max_ps(c3, temp);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);


  // store to output of pooling layer
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O + (2 * C_ob), c6);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);

  _mm256_store_ps(O + (3 * C_ob), c10);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c11);


}
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_pool_accum_end(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        // p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
      }
    }
  }

  // horizontal pooling
  b0 = _mm256_load_ps(O);
  b1 = _mm256_load_ps(O + SIMD);

  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  // c10 = _mm256_max_ps(c10,c8);
  // c11 = _mm256_max_ps(c11,c9);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);

  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1,b1);

  // accumulate with previous row
  b0 = _mm256_load_ps(O + (1 * C_ob));
  b1 = _mm256_load_ps(O + (2 * C_ob));
  // a_reg = _mm256_load_ps(O + (3 * C_ob));
  a_reg = _mm256_load_ps(O + (1 * C_ob) + SIMD);
  __m256 temp = _mm256_load_ps(O + (2 * C_ob) + SIMD);

  c2 = _mm256_max_ps(c2, b0);
  c3 = _mm256_max_ps(c3, a_reg);

  c6 = _mm256_max_ps(c6, b1);
  c7 = _mm256_max_ps(c7, temp);

  // c10 = _mm256_max_ps(c10, c0);
  // c11 = _mm256_max_ps(c11, c1);


  // store to output of pooling layer
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O + (2 * C_ob), c6);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);




}

// accumulate to the previous pool output row
//TODO: do these stores non temporally
// write to new new output row
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_pool_start(
                            uint32_t input_col_stride,
                            uint32_t pool_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }

  // horizontal pooling
  // load partial updates from previous rows
  b0 = _mm256_load_ps(O + (0 * C_ob));
  b1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);
  a_reg = _mm256_load_ps(O + (1 * C_ob));
  __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);

  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);
  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);

  // store partial updates to next row

  _mm256_store_ps(O_next_row + (0 * C_ob), c2);
  _mm256_store_ps(O_next_row + (0 * C_ob) + SIMD, c3);

  _mm256_store_ps(O_next_row + (1 * C_ob), c6);
  _mm256_store_ps(O_next_row + (1 * C_ob + SIMD), c7);

  _mm256_store_ps(O_next_row + (2 * C_ob), c10);
  _mm256_store_ps(O_next_row + (2 * C_ob + SIMD), c11);

  //accumulate with previous rows
  c2 = _mm256_max_ps(c2, b0);
  c3 = _mm256_max_ps(c3, b1);

  c6 = _mm256_max_ps(c6, a_reg);
  c7 = _mm256_max_ps(c7, temp);

  // c10 = _mm256_max_ps(c10, a_reg);
  // c11 = _mm256_max_ps(c11, c1);

  // store to output of pooling layer

  _mm256_store_ps(O + (0 * C_ob), c2);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);

  _mm256_store_ps(O + (1 * C_ob), c6);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c7);

  //
  // _mm256_store_ps(O + (2 * C_ob), c10);
  // _mm256_store_ps(O + (2 * C_ob + SIMD), c11);


}
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_pool(
                            uint32_t input_col_stride,
                            uint32_t pool_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }

  // horizontal pooling previous tile
  b0 = _mm256_load_ps(O_next_row);
  b1 = _mm256_load_ps(O_next_row + SIMD);


  a_reg = _mm256_load_ps(O + (1 * C_ob));
  __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);

  //
  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);


  // accumulate with previous tile
  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1,b1);

  //Store Partial Outputs to next row
  _mm256_store_ps(O_next_row + (0 * C_ob), c0);
  _mm256_store_ps(O_next_row + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O_next_row + (1 * C_ob), c2);
  _mm256_store_ps(O_next_row + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O_next_row + (2 * C_ob), c6);
  _mm256_store_ps(O_next_row + (2 * C_ob + SIMD), c7);

  _mm256_store_ps(O_next_row + (3 * C_ob), c10);
  _mm256_store_ps(O_next_row + (3 * C_ob + SIMD), c11);

  //Load partial outputs from previous row
  c4 = _mm256_load_ps(O + (2 * C_ob));
  c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);

  c8 = _mm256_load_ps(O + (3 * C_ob));
  c9 = _mm256_load_ps(O + (3 * C_ob) + SIMD);

  b0 = _mm256_load_ps(O);
  b1 = _mm256_load_ps(O + SIMD);


  //accumulate with previous row
  c2 = _mm256_max_ps(c2, a_reg);
  c3 = _mm256_max_ps(c3, temp);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);
  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1, b1);

  // store to output of pooling layer
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O + (2 * C_ob), c6);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);

  _mm256_store_ps(O + (3 * C_ob), c10);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c11);


}
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_pool_end(
                            uint32_t input_col_stride,
                            uint32_t pool_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float * O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        // p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
      }
    }
  }

  // horizontal pooling
  b0 = _mm256_load_ps(O_next_row);
  b1 = _mm256_load_ps(O_next_row + SIMD);


  a_reg = _mm256_load_ps(O);
  __m256 temp = _mm256_load_ps(O + SIMD);
  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);



  // accumulate with previous tile
  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1,b1);



  // store partial updates to next row
  _mm256_store_ps(O_next_row + (0 * C_ob), c0);
  _mm256_store_ps(O_next_row + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O_next_row + (1 * C_ob), c2);
  _mm256_store_ps(O_next_row + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O_next_row + (2 * C_ob), c6);
  _mm256_store_ps(O_next_row + (2 * C_ob + SIMD), c7);

  // load partial updates from previous row
  c4 = _mm256_load_ps(O + (1 * C_ob));
  c8 = _mm256_load_ps(O + (2 * C_ob));

  c5 = _mm256_load_ps(O + (1 * C_ob) + SIMD);
  c9 = _mm256_load_ps(O + (2 * C_ob) + SIMD);

  c2 = _mm256_max_ps(c2, c4);
  c3 = _mm256_max_ps(c3, c5);

  c6 = _mm256_max_ps(c6, c8);
  c7 = _mm256_max_ps(c7, c9);

  c0 = _mm256_max_ps(c0,a_reg);
  c1 = _mm256_max_ps(c1, temp);

  // c10 = _mm256_max_ps(c10, c0);
  // c11 = _mm256_max_ps(c11, c1);


  // store to output of pooling layer
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O + (2 * C_ob), c6);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);




}




























//start fully computed kernels
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void complete_conv_microkernel_pool_first_row_start(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }


  STORE_C();
  // horizontal pooling
  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);



  // store to output of pooling layer

  _mm256_store_ps(O + (0 * C_ob), c2);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);

  _mm256_store_ps(O + (1 * C_ob), c6);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c7);

  _mm256_store_ps(O + (2 * C_ob), c10);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c11);


}
// load partial output from previous tile
// compute partial update for next tile
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void complete_conv_microkernel_pool_first_row(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }

  STORE_C();
  // horizontal pooling

  // load partial update from previous
  b0 = _mm256_load_ps(O);
  b1 = _mm256_load_ps(O + SIMD);

  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);

  //update previoud tile
  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1,b1);

  // store to output of pooling layer
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O + (2 * C_ob), c6);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);

  _mm256_store_ps(O + (3 * C_ob), c10);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c11);


}

template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void complete_conv_microkernel_pool_first_row_end(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }

  STORE_C();
  // horizontal pooling
  // load the partial update from the previous tile
  b0 = _mm256_load_ps(O);
  b1 = _mm256_load_ps(O + SIMD);

  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  // c10 = _mm256_max_ps(c10,c8);
  // c11 = _mm256_max_ps(c11,c9);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);

  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1,b1);

  // store to output of pooling layer
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O + (2 * C_ob), c6);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);




}


template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void complete_conv_microkernel_pool_accum_start(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }

  STORE_C();
  // horizontal pooling
  // load partial updates from previous rows (only ones being finalized)
  b0 = _mm256_load_ps(O + (0 * C_ob));
  b1 = _mm256_load_ps(O + (1 * C_ob));
  a_reg = _mm256_load_ps(O + (2 * C_ob));
  __m256 temp = _mm256_load_ps(O + (0 * C_ob) + SIMD);

  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

  c0 = _mm256_load_ps(O + (1 * C_ob) + SIMD);
  c1 = _mm256_load_ps(O + (2 * C_ob) + SIMD);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);

  //accumulate with previous rows
  c2 = _mm256_max_ps(c2, b0);
  c3 = _mm256_max_ps(c3, temp);

  c6 = _mm256_max_ps(c6, b1);
  c7 = _mm256_max_ps(c7, c0);

  c10 = _mm256_max_ps(c10, a_reg);
  c11 = _mm256_max_ps(c11, c1);

  // store to output of pooling layer

  _mm256_store_ps(O + (0 * C_ob), c2);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);

  _mm256_store_ps(O + (1 * C_ob), c6);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c7);

  _mm256_store_ps(O + (2 * C_ob), c10);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c11);


}
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void complete_conv_microkernel_pool_accum(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }

  STORE_C();
  // horizontal pooling previous tile
  b0 = _mm256_load_ps(O);
  b1 = _mm256_load_ps(O + SIMD);


  a_reg = _mm256_load_ps(O + (1 * C_ob));
  __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);

  //
  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);


  c4 = _mm256_load_ps(O + (2 * C_ob));
  c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);

  c8 = _mm256_load_ps(O + (3 * C_ob));
  c9 = _mm256_load_ps(O + (3 * C_ob) + SIMD);


  // accumulate with previous tile
  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1,b1);
  //accumulate with previous row
  c2 = _mm256_max_ps(c2, a_reg);
  c3 = _mm256_max_ps(c3, temp);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);


  // store to output of pooling layer
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O + (2 * C_ob), c6);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);

  _mm256_store_ps(O + (3 * C_ob), c10);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c11);


}
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void complete_conv_microkernel_pool_accum_end(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }

  STORE_C();
  // horizontal pooling
  b0 = _mm256_load_ps(O);
  b1 = _mm256_load_ps(O + SIMD);

  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  // c10 = _mm256_max_ps(c10,c8);
  // c11 = _mm256_max_ps(c11,c9);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);

  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1,b1);

  // accumulate with previous row
  b0 = _mm256_load_ps(O + (1 * C_ob));
  b1 = _mm256_load_ps(O + (2 * C_ob));
  // a_reg = _mm256_load_ps(O + (3 * C_ob));
  a_reg = _mm256_load_ps(O + (1 * C_ob) + SIMD);
  __m256 temp = _mm256_load_ps(O + (2 * C_ob) + SIMD);

  c2 = _mm256_max_ps(c2, b0);
  c3 = _mm256_max_ps(c3, a_reg);

  c6 = _mm256_max_ps(c6, b1);
  c7 = _mm256_max_ps(c7, temp);

  // c10 = _mm256_max_ps(c10, c0);
  // c11 = _mm256_max_ps(c11, c1);


  // store to output of pooling layer
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O + (2 * C_ob), c6);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);




}

// accumulate to the previous pool output row
//TODO: do these stores non temporally
// write to new new output row
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void complete_conv_microkernel_pool_start(
                            uint32_t input_col_stride,
                            uint32_t pool_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }

  STORE_C();
  // horizontal pooling
  // load partial updates from previous rows
  b0 = _mm256_load_ps(O + (0 * C_ob));
  b1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);
  a_reg = _mm256_load_ps(O + (1 * C_ob));
  __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);

  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);
  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);

  // store partial updates to next row

  _mm256_store_ps(O_next_row + (0 * C_ob), c2);
  _mm256_store_ps(O_next_row + (0 * C_ob) + SIMD, c3);

  _mm256_store_ps(O_next_row + (1 * C_ob), c6);
  _mm256_store_ps(O_next_row + (1 * C_ob + SIMD), c7);

  _mm256_store_ps(O_next_row + (2 * C_ob), c10);
  _mm256_store_ps(O_next_row + (2 * C_ob + SIMD), c11);

  //accumulate with previous rows
  c2 = _mm256_max_ps(c2, b0);
  c3 = _mm256_max_ps(c3, b1);

  c6 = _mm256_max_ps(c6, a_reg);
  c7 = _mm256_max_ps(c7, temp);

  // c10 = _mm256_max_ps(c10, a_reg);
  // c11 = _mm256_max_ps(c11, c1);

  // store to output of pooling layer

  _mm256_store_ps(O + (0 * C_ob), c2);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);

  _mm256_store_ps(O + (1 * C_ob), c6);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c7);

  //
  // _mm256_store_ps(O + (2 * C_ob), c10);
  // _mm256_store_ps(O + (2 * C_ob + SIMD), c11);


}
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void complete_conv_microkernel_pool(
                            uint32_t input_col_stride,
                            uint32_t pool_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }

  STORE_C();
  // horizontal pooling previous tile
  b0 = _mm256_load_ps(O_next_row);
  b1 = _mm256_load_ps(O_next_row + SIMD);


  a_reg = _mm256_load_ps(O + (1 * C_ob));
  __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);

  //
  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);


  // accumulate with previous tile
  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1,b1);

  //Store Partial Outputs to next row
  _mm256_store_ps(O_next_row + (0 * C_ob), c0);
  _mm256_store_ps(O_next_row + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O_next_row + (1 * C_ob), c2);
  _mm256_store_ps(O_next_row + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O_next_row + (2 * C_ob), c6);
  _mm256_store_ps(O_next_row + (2 * C_ob + SIMD), c7);

  _mm256_store_ps(O_next_row + (3 * C_ob), c10);
  _mm256_store_ps(O_next_row + (3 * C_ob + SIMD), c11);

  //Load partial outputs from previous row
  c4 = _mm256_load_ps(O + (2 * C_ob));
  c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);

  c8 = _mm256_load_ps(O + (3 * C_ob));
  c9 = _mm256_load_ps(O + (3 * C_ob) + SIMD);

  b0 = _mm256_load_ps(O);
  b1 = _mm256_load_ps(O + SIMD);


  //accumulate with previous row
  c2 = _mm256_max_ps(c2, a_reg);
  c3 = _mm256_max_ps(c3, temp);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);
  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1, b1);

  // store to output of pooling layer
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O + (2 * C_ob), c6);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);

  _mm256_store_ps(O + (3 * C_ob), c10);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c11);


}
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void complete_conv_microkernel_pool_end(
                            uint32_t input_col_stride,
                            uint32_t pool_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float * O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_C();
  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        int p_cur = ii;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }

  STORE_C();
  // horizontal pooling
  b0 = _mm256_load_ps(O_next_row);
  b1 = _mm256_load_ps(O_next_row + SIMD);


  a_reg = _mm256_load_ps(O);
  __m256 temp = _mm256_load_ps(O + SIMD);
  c2 = _mm256_max_ps(c2,c0);
  c3 = _mm256_max_ps(c3,c1);
  c6 = _mm256_max_ps(c6,c4);
  c7 = _mm256_max_ps(c7,c5);

  c2 = _mm256_max_ps(c2,c4);
  c3 = _mm256_max_ps(c3,c5);
  c6 = _mm256_max_ps(c6,c8);
  c7 = _mm256_max_ps(c7,c9);



  // accumulate with previous tile
  c0 = _mm256_max_ps(c0, b0);
  c1 = _mm256_max_ps(c1,b1);



  // store partial updates to next row
  _mm256_store_ps(O_next_row + (0 * C_ob), c0);
  _mm256_store_ps(O_next_row + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O_next_row + (1 * C_ob), c2);
  _mm256_store_ps(O_next_row + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O_next_row + (2 * C_ob), c6);
  _mm256_store_ps(O_next_row + (2 * C_ob + SIMD), c7);

  // load partial updates from previous row
  c4 = _mm256_load_ps(O + (1 * C_ob));
  c8 = _mm256_load_ps(O + (2 * C_ob));

  c5 = _mm256_load_ps(O + (1 * C_ob) + SIMD);
  c9 = _mm256_load_ps(O + (2 * C_ob) + SIMD);

  c2 = _mm256_max_ps(c2, c4);
  c3 = _mm256_max_ps(c3, c5);

  c6 = _mm256_max_ps(c6, c8);
  c7 = _mm256_max_ps(c7, c9);

  c0 = _mm256_max_ps(c0,a_reg);
  c1 = _mm256_max_ps(c1, temp);

  // c10 = _mm256_max_ps(c10, c0);
  // c11 = _mm256_max_ps(c11, c1);


  // store to output of pooling layer
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);

  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);

  _mm256_store_ps(O + (2 * C_ob), c6);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);




}

// end fully computed kernels

// asssumes C_i > 32
template <uint32_t stride, uint32_t H_f, uint32_t W_f>
inline void fused_pooling_direct_convolution(
  uint32_t C_i,
  uint32_t C_o,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
  float * I,
  float * F,
  float * O_buffer,
  float * O
){

  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);
  uint32_t W_o_pool = 0, H_o_pool = 0;
  assert(W_o >POOL_KERNEL);
  op_dim(W_o, POOL_STRIDE, POOL_KERNEL, W_o_pool);
  op_dim(H_o, POOL_STRIDE, POOL_KERNEL, H_o_pool);

  //3x3 Output Channel Block

  for(uint32_t j = 0; j < C_o; j += C_ob)
  {
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    uint32_t pool_block_offset = (j/C_ob)*H_o_pool*W_o_pool*C_ob;

    //First 3x3 Input Channel Block
    {
      // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float * filter_block_ptr = F + filter_i_c_block;
      //3x3 Output Rows
      for(uint32_t l = 0; l < H_o; l++)
      {

          uint32_t col_offset = l*W_o*C_ob ;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          // 3x3 pixel blocks
          for(uint32_t k = 0; k < W_o; k += W_ob)
          {

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;

            conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

          }// end 3x3 pixel blocks
      }// end 3x3 output rows
    }//end First 3x3 Input Channel Block

    // 3x3 Input channel blocks (N-2*C_ib updates)
    for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib)
    {
      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;
            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

        }
      }
    }//end 3x3 Input channel blocks

    // fused pooling
    //Last 3x3 Input Channel Block
    {
      uint32_t input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float * filter_block_ptr = F + filter_i_c_block;

      //First 3x3 Output Row
      {
        uint32_t col_offset = 0*W_o*C_ob;
        uint32_t input_col_offset = (0 * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;

        //first tile
        {
          conv_microkernel_pool_first_row_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                                        I_ptr, filter_block_ptr,
                                                                         O_buffer + col_offset + 0 *C_ob,
                                                                        O + pool_block_offset
                                                                      );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        // Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_first_row<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        // end Second to Penultimate tile

        //last tile
        {


          conv_microkernel_pool_first_row_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr, filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile

      }//end First 3x3 Output Row

      //Second 3x3 Output Row
      {
        // 3x3 params
        uint32_t col_offset = 1*W_o*C_ob;
        uint32_t input_col_offset = (1 * stride)*W_i*C_ob + input_block_offset;
        float * I_ptr = I + input_col_offset;

        //first tile
        {
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr,
                                          O_buffer + col_offset + 0*C_ob,
                                          O + pool_block_offset
                                        );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {
          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }//end Second 3x3 Output Row

      uint32_t pool_row_offset = 0 + pool_block_offset;
      uint32_t pool_row = 0;
      //Third to Penultimate 3x3 Output Rows
      for(uint32_t l = POOL_STRIDE; l < H_o-(POOL_KERNEL-1); l+= POOL_STRIDE)
      {
        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                               W_o_pool*C_ob,
                                                                I_ptr, filter_block_ptr,
                                                                O_buffer + col_offset + 0 *C_ob,
                                                                O + pool_row_offset
                                                                  );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                         W_o_pool*C_ob,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate tile

          //last tile
          {

            conv_microkernel_pool_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      W_o_pool*C_ob,
                                        I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }
          //end last tile
        }//end Even 3x3 Output Row

        pool_row_offset += W_o_pool*C_ib;
        pool_row++;

        //Odd 3x3 Output Row This should be the same as the Last row
        {
          // 3x3 params
          uint32_t col_offset = (l + 1)*W_o*C_ob;
          uint32_t input_col_offset = ((l + 1) * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                            I_ptr, filter_block_ptr, O_buffer + col_offset + 0*C_ob,
                                            O + pool_row_offset
                                          );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,
                                        filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate Tile

          //last tile
          {
            conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }//end last tile
        }//end Odd 3x3 Output Row

      }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      {// 3x3 params
        uint32_t col_offset = (pool_row*POOL_STRIDE + POOL_STRIDE)*W_o*C_ob;
        uint32_t input_col_offset = ((pool_row*POOL_STRIDE + POOL_STRIDE) * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;
        //first tile
        {
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr, O_buffer + col_offset + 0 *C_ob,
                                          O + pool_row_offset
                                        );
        }
        //end first tile

        uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,
                                      filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }// end Last 3x3 Output Row
    }// end Last 3x3 Input Channel Block
  }//end 3x3 Output Channel Blocks
}




// asssumes C_i > 32
// 3x3 output not fully computed, reuse a buffer num_threads*(H_o*W_o*16)
template <uint32_t stride, uint32_t H_f, uint32_t W_f>
void parallel_fused_pooling_direct_convolution(
  uint32_t C_i,
  uint32_t C_o,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
  float * I,
  float * F,
  float * O_buffers,
  float * O
){

  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);
  uint32_t W_o_pool = 0, H_o_pool = 0;
  assert(W_o >POOL_KERNEL);
  op_dim(W_o, POOL_STRIDE, POOL_KERNEL, W_o_pool);
  op_dim(H_o, POOL_STRIDE, POOL_KERNEL, H_o_pool);

  //3x3 Output Channel Block
  #pragma omp parallel for
  for(uint32_t j = 0; j < C_o; j += C_ob)
  {
    uint32_t tid = omp_get_thread_num();
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    uint32_t pool_block_offset = (j/C_ob)*H_o_pool*W_o_pool*C_ob;
    float * O_buffer = O_buffers + (tid)*(H_o*W_o*C_ob);
    //First 3x3 Input Channel Block
    {
      // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float * filter_block_ptr = F + filter_i_c_block;
      //3x3 Output Rows
      for(uint32_t l = 0; l < H_o; l++)
      {

          uint32_t col_offset = l*W_o*C_ob ;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          // 3x3 pixel blocks
          for(uint32_t k = 0; k < W_o; k += W_ob)
          {

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;

            conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

          }// end 3x3 pixel blocks
      }// end 3x3 output rows
    }//end First 3x3 Input Channel Block

    // 3x3 Input channel blocks (N-2*C_ib updates)
    for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib)
    {
      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;
            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

        }
      }
    }//end 3x3 Input channel blocks

    // fused pooling
    //Last 3x3 Input Channel Block
    {
      uint32_t input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float * filter_block_ptr = F + filter_i_c_block;

      //First 3x3 Output Row
      {
        uint32_t col_offset = 0*W_o*C_ob;
        uint32_t input_col_offset = (0 * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;

        //first tile
        {
          conv_microkernel_pool_first_row_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                                        I_ptr, filter_block_ptr,
                                                                         O_buffer + col_offset + 0 *C_ob,
                                                                        O + pool_block_offset
                                                                      );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        // Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_first_row<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        // end Second to Penultimate tile

        //last tile
        {


          conv_microkernel_pool_first_row_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr, filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile

      }//end First 3x3 Output Row

      //Second 3x3 Output Row
      {
        // 3x3 params
        uint32_t col_offset = 1*W_o*C_ob;
        uint32_t input_col_offset = (1 * stride)*W_i*C_ob + input_block_offset;
        float * I_ptr = I + input_col_offset;

        //first tile
        {
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr,
                                          O_buffer + col_offset + 0*C_ob,
                                          O + pool_block_offset
                                        );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {
          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }//end Second 3x3 Output Row

      uint32_t pool_row_offset = 0 + pool_block_offset;
      uint32_t pool_row = 0;
      //Third to Penultimate 3x3 Output Rows
      for(uint32_t l = POOL_STRIDE; l < H_o-(POOL_KERNEL-1); l+= POOL_STRIDE)
      {
        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                               W_o_pool*C_ob,
                                                                I_ptr, filter_block_ptr,
                                                                O_buffer + col_offset + 0 *C_ob,
                                                                O + pool_row_offset
                                                                  );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                         W_o_pool*C_ob,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate tile

          //last tile
          {

            conv_microkernel_pool_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      W_o_pool*C_ob,
                                        I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }
          //end last tile
        }//end Even 3x3 Output Row

        pool_row_offset += W_o_pool*C_ib;
        pool_row++;

        //Odd 3x3 Output Row This should be the same as the Last row
        {
          // 3x3 params
          uint32_t col_offset = (l + 1)*W_o*C_ob;
          uint32_t input_col_offset = ((l + 1) * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                            I_ptr, filter_block_ptr, O_buffer + col_offset + 0*C_ob,
                                            O + pool_row_offset
                                          );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,
                                        filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate Tile

          //last tile
          {
            conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }//end last tile
        }//end Odd 3x3 Output Row

      }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      {// 3x3 params
        uint32_t col_offset = (pool_row*POOL_STRIDE + POOL_STRIDE)*W_o*C_ob;
        uint32_t input_col_offset = ((pool_row*POOL_STRIDE + POOL_STRIDE) * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;
        //first tile
        {
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr, O_buffer + col_offset + 0 *C_ob,
                                          O + pool_row_offset
                                        );
        }
        //end first tile

        uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,
                                      filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }// end Last 3x3 Output Row
    }// end Last 3x3 Input Channel Block
  }//end 3x3 Output Channel Blocks
}

// asssumes C_i > 32
// 3x3 output not fully computed, use intermediate output of full size (H_o*W_*C_o)
template <uint32_t stride, uint32_t H_f, uint32_t W_f>
inline void parallel_fused_pooling_direct_convolution_not_buffered(
  uint32_t C_i,
  uint32_t C_o,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
  float * I,
  float * F,
  float * O_buffers,
  float * O
){

  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);
  uint32_t W_o_pool = 0, H_o_pool = 0;
  assert(W_o >POOL_KERNEL);
  op_dim(W_o, POOL_STRIDE, POOL_KERNEL, W_o_pool);
  op_dim(H_o, POOL_STRIDE, POOL_KERNEL, H_o_pool);

  //3x3 Output Channel Block
  #pragma omp parallel for
  for(uint32_t j = 0; j < C_o; j += C_ob)
  {
    uint32_t tid = omp_get_thread_num();
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    uint32_t pool_block_offset = (j/C_ob)*H_o_pool*W_o_pool*C_ob;
    float * O_buffer = O_buffers + (j/C_ob)*(H_o*W_o*C_ob);
    //First 3x3 Input Channel Block
    {
      // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float * filter_block_ptr = F + filter_i_c_block;
      //3x3 Output Rows
      for(uint32_t l = 0; l < H_o; l++)
      {

          uint32_t col_offset = l*W_o*C_ob ;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          // 3x3 pixel blocks
          for(uint32_t k = 0; k < W_o; k += W_ob)
          {

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;

            conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

          }// end 3x3 pixel blocks
      }// end 3x3 output rows
    }//end First 3x3 Input Channel Block

    // 3x3 Input channel blocks (N-2*C_ib updates)
    for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib)
    {
      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;
            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

        }
      }
    }//end 3x3 Input channel blocks

    // fused pooling
    //Last 3x3 Input Channel Block
    {
      uint32_t input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float * filter_block_ptr = F + filter_i_c_block;

      //First 3x3 Output Row
      {
        uint32_t col_offset = 0*W_o*C_ob;
        uint32_t input_col_offset = (0 * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;

        //first tile
        {
          conv_microkernel_pool_first_row_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                                        I_ptr, filter_block_ptr,
                                                                         O_buffer + col_offset + 0 *C_ob,
                                                                        O + pool_block_offset
                                                                      );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        // Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_first_row<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        // end Second to Penultimate tile

        //last tile
        {


          conv_microkernel_pool_first_row_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr, filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile

      }//end First 3x3 Output Row

      //Second 3x3 Output Row
      {
        // 3x3 params
        uint32_t col_offset = 1*W_o*C_ob;
        uint32_t input_col_offset = (1 * stride)*W_i*C_ob + input_block_offset;
        float * I_ptr = I + input_col_offset;

        //first tile
        {
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr,
                                          O_buffer + col_offset + 0*C_ob,
                                          O + pool_block_offset
                                        );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {
          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }//end Second 3x3 Output Row

      uint32_t pool_row_offset = 0 + pool_block_offset;
      uint32_t pool_row = 0;
      //Third to Penultimate 3x3 Output Rows
      for(uint32_t l = POOL_STRIDE; l < H_o-(POOL_KERNEL-1); l+= POOL_STRIDE)
      {
        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                               W_o_pool*C_ob,
                                                                I_ptr, filter_block_ptr,
                                                                O_buffer + col_offset + 0 *C_ob,
                                                                O + pool_row_offset
                                                                  );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                         W_o_pool*C_ob,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate tile

          //last tile
          {

            conv_microkernel_pool_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      W_o_pool*C_ob,
                                        I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }
          //end last tile
        }//end Even 3x3 Output Row

        pool_row_offset += W_o_pool*C_ib;
        pool_row++;

        //Odd 3x3 Output Row This should be the same as the Last row
        {
          // 3x3 params
          uint32_t col_offset = (l + 1)*W_o*C_ob;
          uint32_t input_col_offset = ((l + 1) * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                            I_ptr, filter_block_ptr, O_buffer + col_offset + 0*C_ob,
                                            O + pool_row_offset
                                          );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,
                                        filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate Tile

          //last tile
          {
            conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }//end last tile
        }//end Odd 3x3 Output Row

      }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      {// 3x3 params
        uint32_t col_offset = (pool_row*POOL_STRIDE + POOL_STRIDE)*W_o*C_ob;
        uint32_t input_col_offset = ((pool_row*POOL_STRIDE + POOL_STRIDE) * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;
        //first tile
        {
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr, O_buffer + col_offset + 0 *C_ob,
                                          O + pool_row_offset
                                        );
        }
        //end first tile

        uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,
                                      filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }// end Last 3x3 Output Row
    }// end Last 3x3 Input Channel Block
  }//end 3x3 Output Channel Blocks
}

// asssumes C_i > 32
// 3x3 outputfully computed, reuse use intermediate output of full size (H_o*W_*C_o)
template <uint32_t stride, uint32_t H_f, uint32_t W_f>
inline void parallel_fused_pooling_direct_convolution_complete(
  uint32_t C_i,
  uint32_t C_o,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
  float * I,
  float * F,
  float * O_buffers,
  float * O
){

  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);
  uint32_t W_o_pool = 0, H_o_pool = 0;
  assert(W_o >POOL_KERNEL);
  op_dim(W_o, POOL_STRIDE, POOL_KERNEL, W_o_pool);
  op_dim(H_o, POOL_STRIDE, POOL_KERNEL, H_o_pool);

  //3x3 Output Channel Block
  #pragma omp parallel for
  for(uint32_t j = 0; j < C_o; j += C_ob)
  {
    uint32_t tid = omp_get_thread_num();
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    uint32_t pool_block_offset = (j/C_ob)*H_o_pool*W_o_pool*C_ob;
    float * O_buffer = O_buffers + (j/C_ob)*(H_o*W_o*C_ob);
    //First 3x3 Input Channel Block
    {
      // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float * filter_block_ptr = F + filter_i_c_block;
      //3x3 Output Rows
      for(uint32_t l = 0; l < H_o; l++)
      {

          uint32_t col_offset = l*W_o*C_ob ;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          // 3x3 pixel blocks
          for(uint32_t k = 0; k < W_o; k += W_ob)
          {

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;

            conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

          }// end 3x3 pixel blocks
      }// end 3x3 output rows
    }//end First 3x3 Input Channel Block

    // 3x3 Input channel blocks (N-2*C_ib updates)
    for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib)
    {
      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;
            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

        }
      }
    }//end 3x3 Input channel blocks

    // fused pooling
    //Last 3x3 Input Channel Block
    {
      uint32_t input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float * filter_block_ptr = F + filter_i_c_block;

      //First 3x3 Output Row
      {
        uint32_t col_offset = 0*W_o*C_ob;
        uint32_t input_col_offset = (0 * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;

        //first tile
        {
          complete_conv_microkernel_pool_first_row_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                                        I_ptr, filter_block_ptr,
                                                                         O_buffer + col_offset + 0 *C_ob,
                                                                        O + pool_block_offset
                                                                      );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        // Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          complete_conv_microkernel_pool_first_row<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        // end Second to Penultimate tile

        //last tile
        {


          complete_conv_microkernel_pool_first_row_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr, filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile

      }//end First 3x3 Output Row

      //Second 3x3 Output Row
      {
        // 3x3 params
        uint32_t col_offset = 1*W_o*C_ob;
        uint32_t input_col_offset = (1 * stride)*W_i*C_ob + input_block_offset;
        float * I_ptr = I + input_col_offset;

        //first tile
        {
          complete_conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr,
                                          O_buffer + col_offset + 0*C_ob,
                                          O + pool_block_offset
                                        );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {
          complete_conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          complete_conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }//end Second 3x3 Output Row

      uint32_t pool_row_offset = 0 + pool_block_offset;
      uint32_t pool_row = 0;
      //Third to Penultimate 3x3 Output Rows
      for(uint32_t l = POOL_STRIDE; l < H_o-(POOL_KERNEL-1); l+= POOL_STRIDE)
      {
        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          float * I_ptr = I + input_col_offset;

          //first tile
          {
            complete_conv_microkernel_pool_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                               W_o_pool*C_ob,
                                                                I_ptr, filter_block_ptr,
                                                                O_buffer + col_offset + 0 *C_ob,
                                                                O + pool_row_offset
                                                                  );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            complete_conv_microkernel_pool<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                         W_o_pool*C_ob,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate tile

          //last tile
          {

            complete_conv_microkernel_pool_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      W_o_pool*C_ob,
                                        I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }
          //end last tile
        }//end Even 3x3 Output Row

        pool_row_offset += W_o_pool*C_ib;
        pool_row++;

        //Odd 3x3 Output Row This should be the same as the Last row
        {
          // 3x3 params
          uint32_t col_offset = (l + 1)*W_o*C_ob;
          uint32_t input_col_offset = ((l + 1) * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          //first tile
          {
            complete_conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                            I_ptr, filter_block_ptr, O_buffer + col_offset + 0*C_ob,
                                            O + pool_row_offset
                                          );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            complete_conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,
                                        filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate Tile

          //last tile
          {
            complete_conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }//end last tile
        }//end Odd 3x3 Output Row

      }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      {// 3x3 params
        uint32_t col_offset = (pool_row*POOL_STRIDE + POOL_STRIDE)*W_o*C_ob;
        uint32_t input_col_offset = ((pool_row*POOL_STRIDE + POOL_STRIDE) * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;
        //first tile
        {
          complete_conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr, O_buffer + col_offset + 0 *C_ob,
                                          O + pool_row_offset
                                        );
        }
        //end first tile

        uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          complete_conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,
                                      filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          complete_conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }// end Last 3x3 Output Row
    }// end Last 3x3 Input Channel Block
  }//end 3x3 Output Channel Blocks
}

void pooling(
    uint32_t C,
    uint32_t H_i,
    uint32_t W_i,
    float * I,
    float * O
)
{
  const int w_block = 1;
  uint32_t s = POOL_STRIDE;
  uint32_t W_o , H_o;
  op_dim(W_i, POOL_STRIDE, POOL_KERNEL, W_o);
  op_dim(H_i, POOL_STRIDE, POOL_KERNEL, H_o);
  // printf("\n %d %d to %d %d\n",H_i, W_i,H_o, W_o);
  uint32_t offset = 0;
  #pragma omp parallel for
  for(uint32_t i = 0; i < C; i+=C_ob)
  {
    uint32_t block_offset = (i/C_ob) * H_i * W_i * C_ob;
    // printf(" channel block %d: %d %d\n", i, offset, block_offset);

    for(uint32_t l = 0; l < H_o; l++)
    {
      uint32_t row_offset = (l*s)*W_i*C_ob + block_offset;
      for(uint32_t k = 0; k < W_o; k+=w_block)
      {
        uint32_t col_offset = (k*s)*C_ob + row_offset;
        __m256 r0, r1, r2, r3, r4, r5, r6, r7,r8, r9, r10, r11, r12,r13, r14, r15;
        r0 = _mm256_load_ps(I + col_offset + 0*C_ob);
        r1 = _mm256_load_ps(I + col_offset + 0*C_ob +  SIMD);
        // r2 = _mm256_load_ps(I + col_offset + 1*C_ob);
        // r3 = _mm256_load_ps(I + col_offset + 1*C_ob +  SIMD);
        // r4 = _mm256_load_ps(I + col_offset + 2*C_ob);
        // r5 = _mm256_load_ps(I + col_offset + 2*C_ob +  SIMD);
        // r6 = _mm256_load_ps(I + col_offset + 3*C_ob);
        // r7 = _mm256_load_ps(I + col_offset + 3*C_ob +  SIMD);
        for(uint32_t n = 0; n < POOL_KERNEL; n++)
        {
          uint32_t stencil_h = n*W_i*C_ob + col_offset;
          for(uint32_t m = 0; m < POOL_KERNEL; m++)
          {
            uint32_t stencil_w = m*C_ob + stencil_h;
            // printf("\t %d",stencil_w);
            r8 = _mm256_load_ps(I + stencil_w);
            stencil_w += SIMD;
            r9 = _mm256_load_ps(I + stencil_w + SIMD);
            // stencil_w += SIMD;
            // r10 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r11 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r12 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r13 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r14 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r15 = _mm256_load_ps(I + stencil_w);

            r0 = _mm256_max_ps(r0, r8);
            r1 = _mm256_max_ps(r1, r9);
            // r2 = _mm256_max_ps(r2, r10);
            // r3 = _mm256_max_ps(r3, r11);
            // r4 = _mm256_max_ps(r4, r12);
            // r5 = _mm256_max_ps(r5, r13);
            // r6 = _mm256_max_ps(r6, r14);
            // r7 = _mm256_max_ps(r7, r15);
          }
          // printf("\n");
        }
        // printf("\n");
        _mm256_store_ps(O+offset, r0);
        offset += SIMD;
        _mm256_store_ps(O+offset, r1);
        offset += SIMD;
        // _mm256_store_ps(O+offset, r2);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r3);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r4);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r5);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r6);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r7);
        // offset += SIMD;
      }
    }
  }

}


void pooling_fma(
    uint32_t C,
    uint32_t H_i,
    uint32_t W_i,
    float * I,
    float * O
)
{
  const int w_block = 1;
  uint32_t s = POOL_STRIDE;
  uint32_t W_o , H_o;
  op_dim(W_i, POOL_STRIDE, POOL_KERNEL, W_o);
  op_dim(H_i, POOL_STRIDE, POOL_KERNEL, H_o);
  // printf("\n %d %d to %d %d\n",H_i, W_i,H_o, W_o);
  uint32_t offset = 0;
  for(uint32_t i = 0; i < C; i+=C_ob)
  {
    uint32_t block_offset = (i/C_ob) * H_i * W_i * C_ob;
    // printf(" channel block %d: %d %d\n", i, offset, block_offset);

    for(uint32_t l = 0; l < H_o; l++)
    {
      uint32_t row_offset = (l*s)*W_i*C_ob + block_offset;
      for(uint32_t k = 0; k < W_o; k+=w_block)
      {
        uint32_t col_offset = (k*s)*C_ob + row_offset;
        __m256 r0, r1, r2, r3, r4, r5, r6, r7,r8, r9, r10, r11, r12,r13, r14, r15;
        r0 = _mm256_load_ps(I + col_offset);
        r1 = _mm256_load_ps(I + col_offset +  SIMD);
        // r2 = _mm256_setzero_ps();
        // r3 = _mm256_setzero_ps();
        // r4 = _mm256_setzero_ps();
        // r5 = _mm256_setzero_ps();
        // r6 = _mm256_setzero_ps();
        // r7 = _mm256_setzero_ps();
        for(uint32_t n = 0; n < POOL_KERNEL; n++)
        {
          uint32_t stencil_h = n*W_i*C_ob + col_offset;
          for(uint32_t m = 0; m < POOL_KERNEL; m++)
          {
            uint32_t stencil_w = m*C_ob + stencil_h;
            // printf("\t %d",stencil_w);
            r8 = _mm256_load_ps(I + stencil_w);
            stencil_w += SIMD;
            r9 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r10 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r11 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r12 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r13 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r14 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r15 = _mm256_load_ps(I + stencil_w);

            r0 = _mm256_max_ps(r0, r8);
            r1 = _mm256_max_ps(r1, r9);
            // r2 = _mm256_max_ps(r2, r10);
            // r3 = _mm256_max_ps(r3, r11);
            // r4 = _mm256_max_ps(r4, r12);
            // r5 = _mm256_max_ps(r5, r13);
            // r6 = _mm256_max_ps(r6, r14);
            // r7 = _mm256_max_ps(r7, r15);
          }
          // printf("\n");
        }
        // printf("\n");
        _mm256_store_ps(O+offset, r0);
        offset += SIMD;
        _mm256_store_ps(O+offset, r1);
        offset += SIMD;
        // _mm256_store_ps(O+offset, r2);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r3);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r4);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r5);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r6);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r7);
        // offset += SIMD;
      }
    }
  }

}
