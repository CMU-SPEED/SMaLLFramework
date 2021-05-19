
#include <immintrin.h>

#define SIMD 8
# define rank_k 32

// AMD Zen2 parameters
#define W_ob 6
#define C_ob  16
#define C_ib rank_k


#define filter_seed 2048
#define image_seed 1729




void print256_float(__m256 var)
{
    float val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n",
           val[0], val[1], val[2], val[3], val[4], val[5],
           val[6], val[7]);
}



template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_load_ps(O + (0 * C_ob));
   c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);
   c2 = _mm256_load_ps(O + (1 * C_ob));
   c3 = _mm256_load_ps(O + (1 * C_ob) + SIMD);
   c4 = _mm256_load_ps(O + (2 * C_ob));
   c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);
   c6 = _mm256_load_ps(O + (3 * C_ob));
   c7 = _mm256_load_ps(O + (3 * C_ob) + SIMD);
   c8 = _mm256_load_ps(O + (4 * C_ob));
   c9 = _mm256_load_ps(O + (4 * C_ob) + SIMD);
   c10 = _mm256_load_ps(O + (5 * C_ob));
   c11 = _mm256_load_ps(O + (5 * C_ob) + SIMD);
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
        int p_cur = ii ;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (p_cur));
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

  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
  _mm256_store_ps(O + (2 * C_ob), c4);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c5);
  _mm256_store_ps(O + (3 * C_ob), c6);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c7);
  _mm256_store_ps(O + (4 * C_ob), c8);
  _mm256_store_ps(O + (4 * C_ob + SIMD), c9);
  _mm256_store_ps(O + (5 * C_ob), c10);
  _mm256_store_ps(O + (5 * C_ob + SIMD), c11);

}

template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_start(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_setzero_ps();
   c1 = _mm256_setzero_ps();
   c2 = _mm256_setzero_ps();
   c3 = _mm256_setzero_ps();
   c4 = _mm256_setzero_ps();
   c5 = _mm256_setzero_ps();
   c6 = _mm256_setzero_ps();
   c7 = _mm256_setzero_ps();
   c8 = _mm256_setzero_ps();
   c9 = _mm256_setzero_ps();
   c10 = _mm256_setzero_ps();
   c11 = _mm256_setzero_ps();
  int updates = 0;
  // uint32_t step = C_ob;//stride*C_ob;
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

  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
  _mm256_store_ps(O + (2 * C_ob), c4);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c5);
  _mm256_store_ps(O + (3 * C_ob), c6);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c7);
  _mm256_store_ps(O + (4 * C_ob), c8);
  _mm256_store_ps(O + (4 * C_ob + SIMD), c9);
  _mm256_store_ps(O + (5 * C_ob), c10);
  _mm256_store_ps(O + (5 * C_ob + SIMD), c11);

}

template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_last(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * norm_weights,
                            float * O){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_load_ps(O + (0 * C_ob));
   c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);
   c2 = _mm256_load_ps(O + (1 * C_ob));
   c3 = _mm256_load_ps(O + (1 * C_ob) + SIMD);
   c4 = _mm256_load_ps(O + (2 * C_ob));
   c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);
   c6 = _mm256_load_ps(O + (3 * C_ob));
   c7 = _mm256_load_ps(O + (3 * C_ob) + SIMD);
   c8 = _mm256_load_ps(O + (4 * C_ob));
   c9 = _mm256_load_ps(O + (4 * C_ob) + SIMD);
   c10 = _mm256_load_ps(O + (5 * C_ob));
   c11 = _mm256_load_ps(O + (5 * C_ob) + SIMD);
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
        int p_cur = ii ;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (p_cur));
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

  //do normalization
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
  _mm256_store_ps(O + (2 * C_ob), c4);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c5);
  _mm256_store_ps(O + (3 * C_ob), c6);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c7);
  _mm256_store_ps(O + (4 * C_ob), c8);
  _mm256_store_ps(O + (4 * C_ob + SIMD), c9);
  _mm256_store_ps(O + (5 * C_ob), c10);
  _mm256_store_ps(O + (5 * C_ob + SIMD), c11);

}


// pooling aware convolution kernels
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_start_end(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_setzero_ps();
   c1 = _mm256_setzero_ps();
   c2 = _mm256_setzero_ps();
   c3 = _mm256_setzero_ps();
   c4 = _mm256_setzero_ps();
   c5 = _mm256_setzero_ps();
   c6 = _mm256_setzero_ps();
   c7 = _mm256_setzero_ps();
   c8 = _mm256_setzero_ps();
   c9 = _mm256_setzero_ps();
   c10 = _mm256_setzero_ps();
   c11 = _mm256_setzero_ps();
  int updates = 0;
  // uint32_t step = C_ob;//stride*C_ob;
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
        // c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        // c11 = _mm256_fmadd_ps(a_reg, b1, c11);
      }
    }
  }

  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
  _mm256_store_ps(O + (2 * C_ob), c4);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c5);
  _mm256_store_ps(O + (3 * C_ob), c6);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c7);
  _mm256_store_ps(O + (4 * C_ob), c8);
  _mm256_store_ps(O + (4 * C_ob + SIMD), c9);
  // _mm256_store_ps(O + (5 * C_ob), c10);
  // _mm256_store_ps(O + (5 * C_ob + SIMD), c11);

}

template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_end(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_load_ps(O + (0 * C_ob));
   c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);
   c2 = _mm256_load_ps(O + (1 * C_ob));
   c3 = _mm256_load_ps(O + (1 * C_ob) + SIMD);
   c4 = _mm256_load_ps(O + (2 * C_ob));
   c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);
   c6 = _mm256_load_ps(O + (3 * C_ob));
   c7 = _mm256_load_ps(O + (3 * C_ob) + SIMD);
   c8 = _mm256_load_ps(O + (4 * C_ob));
   c9 = _mm256_load_ps(O + (4 * C_ob) + SIMD);
   c10 = _mm256_load_ps(O + (5 * C_ob));
   c11 = _mm256_load_ps(O + (5 * C_ob) + SIMD);
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
        int p_cur = ii ;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (p_cur));
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
        // c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        // c11 = _mm256_fmadd_ps(a_reg, b1, c11);
      }
    }
  }

  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
  _mm256_store_ps(O + (2 * C_ob), c4);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c5);
  _mm256_store_ps(O + (3 * C_ob), c6);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c7);
  _mm256_store_ps(O + (4 * C_ob), c8);
  _mm256_store_ps(O + (4 * C_ob + SIMD), c9);
  // _mm256_store_ps(O + (5 * C_ob), c10);
  // _mm256_store_ps(O + (5 * C_ob + SIMD), c11);

}

// No cache hints

// template <uint32_t step, uint32_t H_f, uint32_t W_f>
// inline void conv_microkernel_pool_first_row_start(
//                             uint32_t input_col_stride,
//                             float * I,
//                             float * F,
//                             float * O_buffer,
//                             float *O
//                           )
// {
//
//   __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
//
//   c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
//    c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
//    c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
//    c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
//    c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
//    c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
//    c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
//    c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
//    c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
//    c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
//    c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
//    c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
//   int updates = 0;
//   // uint32_t step = stride*C_ob;
//   // int count = 0;
//   for(uint32_t n = 0; n < H_f; n++){
//
//     int filter_offset_h = n*W_f*C_ib*C_ob;
//     int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
//
//     for(uint32_t m = 0; m < W_f; m++){
//
//       int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
//       int input_stencil_w = m*C_ib + input_stencil_h;
//
//       for(uint32_t ii = 0 ; ii < C_ib; ii++){
//
//         // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
//         float *b = F + filter_offset_w;
//         float *a = I + input_stencil_w;
//
//         int p_cur = ii;
//         b0 = _mm256_load_ps(b + (ii * C_ob));
//         b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
//         a_reg = _mm256_broadcast_ss(a + (ii));
//         p_cur += step;
//         c0 = _mm256_fmadd_ps(a_reg, b0, c0);
//         c1 = _mm256_fmadd_ps(a_reg, b1, c1);
//         a_reg = _mm256_broadcast_ss(a + (p_cur));
//         p_cur += step;
//         c2 = _mm256_fmadd_ps(a_reg, b0, c2);
//         c3 = _mm256_fmadd_ps(a_reg, b1, c3);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c4 = _mm256_fmadd_ps(a_reg, b0, c4);
//         c5 = _mm256_fmadd_ps(a_reg, b1, c5);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c6 = _mm256_fmadd_ps(a_reg, b0, c6);
//         c7 = _mm256_fmadd_ps(a_reg, b1, c7);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c8 = _mm256_fmadd_ps(a_reg, b0, c8);
//         c9 = _mm256_fmadd_ps(a_reg, b1, c9);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c10 = _mm256_fmadd_ps(a_reg, b0, c10);
//         c11 = _mm256_fmadd_ps(a_reg, b1, c11);
//         // count++;
//       }
//     }
//   }
//
//
//   // horizontal pooling
//   c2 = _mm256_max_ps(c2,c0);
//   c3 = _mm256_max_ps(c3,c1);
//   c6 = _mm256_max_ps(c6,c4);
//   c7 = _mm256_max_ps(c7,c5);
//   c10 = _mm256_max_ps(c10,c8);
//   c11 = _mm256_max_ps(c11,c9);
//
//   c2 = _mm256_max_ps(c2,c4);
//   c3 = _mm256_max_ps(c3,c5);
//   c6 = _mm256_max_ps(c6,c8);
//   c7 = _mm256_max_ps(c7,c9);
//
//
//
//   // store to output of pooling layer
//
//   _mm256_store_ps(O + (0 * C_ob), c2);
//   _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);
//
//   _mm256_store_ps(O + (1 * C_ob), c6);
//   _mm256_store_ps(O + (1 * C_ob + SIMD), c7);
//
//   _mm256_store_ps(O + (2 * C_ob), c10);
//   _mm256_store_ps(O + (2 * C_ob + SIMD), c11);
//
//
// }
// // load partial output from previous tile
// // compute partial update for next tile
// template <uint32_t step, uint32_t H_f, uint32_t W_f>
// inline void conv_microkernel_pool_first_row(
//                             uint32_t input_col_stride,
//                             float * I,
//                             float * F,
//                             float * O_buffer,
//                             float *O
//                           )
// {
//
//   __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
//
//   c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
//    c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
//    c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
//    c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
//    c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
//    c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
//    c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
//    c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
//    c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
//    c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
//    c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
//    c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
//   int updates = 0;
//   // uint32_t step = stride*C_ob;
//   // int count = 0;
//   for(uint32_t n = 0; n < H_f; n++){
//
//     int filter_offset_h = n*W_f*C_ib*C_ob;
//     int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
//
//     for(uint32_t m = 0; m < W_f; m++){
//
//       int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
//       int input_stencil_w = m*C_ib + input_stencil_h;
//
//       for(uint32_t ii = 0 ; ii < C_ib; ii++){
//
//         // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
//         float *b = F + filter_offset_w;
//         float *a = I + input_stencil_w;
//
//         int p_cur = ii;
//         b0 = _mm256_load_ps(b + (ii * C_ob));
//         b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
//         a_reg = _mm256_broadcast_ss(a + (ii));
//         p_cur += step;
//         c0 = _mm256_fmadd_ps(a_reg, b0, c0);
//         c1 = _mm256_fmadd_ps(a_reg, b1, c1);
//         a_reg = _mm256_broadcast_ss(a + (p_cur));
//         p_cur += step;
//         c2 = _mm256_fmadd_ps(a_reg, b0, c2);
//         c3 = _mm256_fmadd_ps(a_reg, b1, c3);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c4 = _mm256_fmadd_ps(a_reg, b0, c4);
//         c5 = _mm256_fmadd_ps(a_reg, b1, c5);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c6 = _mm256_fmadd_ps(a_reg, b0, c6);
//         c7 = _mm256_fmadd_ps(a_reg, b1, c7);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c8 = _mm256_fmadd_ps(a_reg, b0, c8);
//         c9 = _mm256_fmadd_ps(a_reg, b1, c9);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c10 = _mm256_fmadd_ps(a_reg, b0, c10);
//         c11 = _mm256_fmadd_ps(a_reg, b1, c11);
//         // count++;
//       }
//     }
//   }
//
//   // horizontal pooling
//
//   // load partial update from previous
//   b0 = _mm256_load_ps(O);
//   b1 = _mm256_load_ps(O + SIMD);
//
//   c2 = _mm256_max_ps(c2,c0);
//   c3 = _mm256_max_ps(c3,c1);
//   c6 = _mm256_max_ps(c6,c4);
//   c7 = _mm256_max_ps(c7,c5);
//   c10 = _mm256_max_ps(c10,c8);
//   c11 = _mm256_max_ps(c11,c9);
//
//   c2 = _mm256_max_ps(c2,c4);
//   c3 = _mm256_max_ps(c3,c5);
//   c6 = _mm256_max_ps(c6,c8);
//   c7 = _mm256_max_ps(c7,c9);
//
//   //update previoud tile
//   c0 = _mm256_max_ps(c0, b0);
//   c1 = _mm256_max_ps(c1,b1);
//
//   // store to output of pooling layer
//   _mm256_store_ps(O + (0 * C_ob), c0);
//   _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
//
//   _mm256_store_ps(O + (1 * C_ob), c2);
//   _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
//
//   _mm256_store_ps(O + (2 * C_ob), c6);
//   _mm256_store_ps(O + (2 * C_ob + SIMD), c7);
//
//   _mm256_store_ps(O + (3 * C_ob), c10);
//   _mm256_store_ps(O + (3 * C_ob + SIMD), c11);
//
//
// }
//
// template <uint32_t step, uint32_t H_f, uint32_t W_f>
// inline void conv_microkernel_pool_first_row_end(
//                             uint32_t input_col_stride,
//                             float * I,
//                             float * F,
//                             float * O_buffer,
//                             float *O
//                           )
// {
//
//   __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
//
//   c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
//    c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
//    c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
//    c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
//    c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
//    c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
//    c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
//    c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
//    c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
//    c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
//    c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
//    c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
//   int updates = 0;
//   // uint32_t step = stride*C_ob;
//   // int count = 0;
//   for(uint32_t n = 0; n < H_f; n++){
//
//     int filter_offset_h = n*W_f*C_ib*C_ob;
//     int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
//
//     for(uint32_t m = 0; m < W_f; m++){
//
//       int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
//       int input_stencil_w = m*C_ib + input_stencil_h;
//
//       for(uint32_t ii = 0 ; ii < C_ib; ii++){
//
//         // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
//         float *b = F + filter_offset_w;
//         float *a = I + input_stencil_w;
//
//         int p_cur = ii;
//         b0 = _mm256_load_ps(b + (ii * C_ob));
//         b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
//         a_reg = _mm256_broadcast_ss(a + (ii));
//         p_cur += step;
//         c0 = _mm256_fmadd_ps(a_reg, b0, c0);
//         c1 = _mm256_fmadd_ps(a_reg, b1, c1);
//         a_reg = _mm256_broadcast_ss(a + (p_cur));
//         p_cur += step;
//         c2 = _mm256_fmadd_ps(a_reg, b0, c2);
//         c3 = _mm256_fmadd_ps(a_reg, b1, c3);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c4 = _mm256_fmadd_ps(a_reg, b0, c4);
//         c5 = _mm256_fmadd_ps(a_reg, b1, c5);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c6 = _mm256_fmadd_ps(a_reg, b0, c6);
//         c7 = _mm256_fmadd_ps(a_reg, b1, c7);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c8 = _mm256_fmadd_ps(a_reg, b0, c8);
//         c9 = _mm256_fmadd_ps(a_reg, b1, c9);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c10 = _mm256_fmadd_ps(a_reg, b0, c10);
//         c11 = _mm256_fmadd_ps(a_reg, b1, c11);
//         // count++;
//       }
//     }
//   }
//
//   // horizontal pooling
//   // load the partial update from the previous tile
//   b0 = _mm256_load_ps(O);
//   b1 = _mm256_load_ps(O + SIMD);
//
//   c2 = _mm256_max_ps(c2,c0);
//   c3 = _mm256_max_ps(c3,c1);
//   c6 = _mm256_max_ps(c6,c4);
//   c7 = _mm256_max_ps(c7,c5);
//   // c10 = _mm256_max_ps(c10,c8);
//   // c11 = _mm256_max_ps(c11,c9);
//
//   c2 = _mm256_max_ps(c2,c4);
//   c3 = _mm256_max_ps(c3,c5);
//   c6 = _mm256_max_ps(c6,c8);
//   c7 = _mm256_max_ps(c7,c9);
//
//   c0 = _mm256_max_ps(c0, b0);
//   c1 = _mm256_max_ps(c1,b1);
//
//   // store to output of pooling layer
//   _mm256_store_ps(O + (0 * C_ob), c0);
//   _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
//
//   _mm256_store_ps(O + (1 * C_ob), c2);
//   _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
//
//   _mm256_store_ps(O + (2 * C_ob), c6);
//   _mm256_store_ps(O + (2 * C_ob + SIMD), c7);
//
//
//
//
// }
//
//
// template <uint32_t step, uint32_t H_f, uint32_t W_f>
// inline void conv_microkernel_pool_accum_start(
//                             uint32_t input_col_stride,
//                             float * I,
//                             float * F,
//                             float * O_buffer,
//                             float *O
//                           )
// {
//
//   __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
//
//   c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
//    c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
//    c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
//    c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
//    c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
//    c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
//    c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
//    c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
//    c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
//    c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
//    c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
//    c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
//   int updates = 0;
//   // uint32_t step = stride*C_ob;
//   // int count = 0;
//   for(uint32_t n = 0; n < H_f; n++){
//
//     int filter_offset_h = n*W_f*C_ib*C_ob;
//     int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
//
//     for(uint32_t m = 0; m < W_f; m++){
//
//       int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
//       int input_stencil_w = m*C_ib + input_stencil_h;
//
//       for(uint32_t ii = 0 ; ii < C_ib; ii++){
//
//         // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
//         float *b = F + filter_offset_w;
//         float *a = I + input_stencil_w;
//
//         int p_cur = ii;
//         b0 = _mm256_load_ps(b + (ii * C_ob));
//         b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
//         a_reg = _mm256_broadcast_ss(a + (ii));
//         p_cur += step;
//         c0 = _mm256_fmadd_ps(a_reg, b0, c0);
//         c1 = _mm256_fmadd_ps(a_reg, b1, c1);
//         a_reg = _mm256_broadcast_ss(a + (p_cur));
//         p_cur += step;
//         c2 = _mm256_fmadd_ps(a_reg, b0, c2);
//         c3 = _mm256_fmadd_ps(a_reg, b1, c3);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c4 = _mm256_fmadd_ps(a_reg, b0, c4);
//         c5 = _mm256_fmadd_ps(a_reg, b1, c5);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c6 = _mm256_fmadd_ps(a_reg, b0, c6);
//         c7 = _mm256_fmadd_ps(a_reg, b1, c7);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c8 = _mm256_fmadd_ps(a_reg, b0, c8);
//         c9 = _mm256_fmadd_ps(a_reg, b1, c9);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c10 = _mm256_fmadd_ps(a_reg, b0, c10);
//         c11 = _mm256_fmadd_ps(a_reg, b1, c11);
//         // count++;
//       }
//     }
//   }
//
//   // horizontal pooling
//   // load partial updates from previous rows (only ones being finalized)
//   b0 = _mm256_load_ps(O + (0 * C_ob));
//   b1 = _mm256_load_ps(O + (1 * C_ob));
//   a_reg = _mm256_load_ps(O + (2 * C_ob));
//   __m256 temp = _mm256_load_ps(O + (0 * C_ob) + SIMD);
//
//   c2 = _mm256_max_ps(c2,c0);
//   c3 = _mm256_max_ps(c3,c1);
//   c6 = _mm256_max_ps(c6,c4);
//   c7 = _mm256_max_ps(c7,c5);
//   c10 = _mm256_max_ps(c10,c8);
//   c11 = _mm256_max_ps(c11,c9);
//
//   c0 = _mm256_load_ps(O + (1 * C_ob) + SIMD);
//   c1 = _mm256_load_ps(O + (2 * C_ob) + SIMD);
//
//   c2 = _mm256_max_ps(c2,c4);
//   c3 = _mm256_max_ps(c3,c5);
//   c6 = _mm256_max_ps(c6,c8);
//   c7 = _mm256_max_ps(c7,c9);
//
//   //accumulate with previous rows
//   c2 = _mm256_max_ps(c2, b0);
//   c3 = _mm256_max_ps(c3, temp);
//
//   c6 = _mm256_max_ps(c6, b1);
//   c7 = _mm256_max_ps(c7, c0);
//
//   c10 = _mm256_max_ps(c10, a_reg);
//   c11 = _mm256_max_ps(c11, c1);
//
//   // store to output of pooling layer
//
//   _mm256_store_ps(O + (0 * C_ob), c2);
//   _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);
//
//   _mm256_store_ps(O + (1 * C_ob), c6);
//   _mm256_store_ps(O + (1 * C_ob + SIMD), c7);
//
//   _mm256_store_ps(O + (2 * C_ob), c10);
//   _mm256_store_ps(O + (2 * C_ob + SIMD), c11);
//
//
// }
// template <uint32_t step, uint32_t H_f, uint32_t W_f>
// inline void conv_microkernel_pool_accum(
//                             uint32_t input_col_stride,
//                             float * I,
//                             float * F,
//                             float * O_buffer,
//                             float *O
//                           )
// {
//
//   __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
//
//   c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
//    c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
//    c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
//    c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
//    c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
//    c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
//    c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
//    c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
//    c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
//    c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
//    c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
//    c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
//   int updates = 0;
//   // uint32_t step = stride*C_ob;
//   // int count = 0;
//   for(uint32_t n = 0; n < H_f; n++){
//
//     int filter_offset_h = n*W_f*C_ib*C_ob;
//     int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
//
//     for(uint32_t m = 0; m < W_f; m++){
//
//       int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
//       int input_stencil_w = m*C_ib + input_stencil_h;
//
//       for(uint32_t ii = 0 ; ii < C_ib; ii++){
//
//         // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
//         float *b = F + filter_offset_w;
//         float *a = I + input_stencil_w;
//
//         int p_cur = ii;
//         b0 = _mm256_load_ps(b + (ii * C_ob));
//         b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
//         a_reg = _mm256_broadcast_ss(a + (ii));
//         p_cur += step;
//         c0 = _mm256_fmadd_ps(a_reg, b0, c0);
//         c1 = _mm256_fmadd_ps(a_reg, b1, c1);
//         a_reg = _mm256_broadcast_ss(a + (p_cur));
//         p_cur += step;
//         c2 = _mm256_fmadd_ps(a_reg, b0, c2);
//         c3 = _mm256_fmadd_ps(a_reg, b1, c3);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c4 = _mm256_fmadd_ps(a_reg, b0, c4);
//         c5 = _mm256_fmadd_ps(a_reg, b1, c5);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c6 = _mm256_fmadd_ps(a_reg, b0, c6);
//         c7 = _mm256_fmadd_ps(a_reg, b1, c7);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c8 = _mm256_fmadd_ps(a_reg, b0, c8);
//         c9 = _mm256_fmadd_ps(a_reg, b1, c9);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c10 = _mm256_fmadd_ps(a_reg, b0, c10);
//         c11 = _mm256_fmadd_ps(a_reg, b1, c11);
//         // count++;
//       }
//     }
//   }
//
//   // horizontal pooling previous tile
//   b0 = _mm256_load_ps(O);
//   b1 = _mm256_load_ps(O + SIMD);
//
//
//   a_reg = _mm256_load_ps(O + (1 * C_ob));
//   __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);
//
//   //
//   c2 = _mm256_max_ps(c2,c0);
//   c3 = _mm256_max_ps(c3,c1);
//   c6 = _mm256_max_ps(c6,c4);
//   c7 = _mm256_max_ps(c7,c5);
//   c10 = _mm256_max_ps(c10,c8);
//   c11 = _mm256_max_ps(c11,c9);
//
//   c2 = _mm256_max_ps(c2,c4);
//   c3 = _mm256_max_ps(c3,c5);
//   c6 = _mm256_max_ps(c6,c8);
//   c7 = _mm256_max_ps(c7,c9);
//
//
//   c4 = _mm256_load_ps(O + (2 * C_ob));
//   c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);
//
//   c8 = _mm256_load_ps(O + (3 * C_ob));
//   c9 = _mm256_load_ps(O + (3 * C_ob) + SIMD);
//
//
//   // accumulate with previous tile
//   c0 = _mm256_max_ps(c0, b0);
//   c1 = _mm256_max_ps(c1,b1);
//   //accumulate with previous row
//   c2 = _mm256_max_ps(c2, a_reg);
//   c3 = _mm256_max_ps(c3, temp);
//   c6 = _mm256_max_ps(c6,c4);
//   c7 = _mm256_max_ps(c7,c5);
//   c10 = _mm256_max_ps(c10,c8);
//   c11 = _mm256_max_ps(c11,c9);
//
//
//   // store to output of pooling layer
//   _mm256_store_ps(O + (0 * C_ob), c0);
//   _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
//
//   _mm256_store_ps(O + (1 * C_ob), c2);
//   _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
//
//   _mm256_store_ps(O + (2 * C_ob), c6);
//   _mm256_store_ps(O + (2 * C_ob + SIMD), c7);
//
//   _mm256_store_ps(O + (3 * C_ob), c10);
//   _mm256_store_ps(O + (3 * C_ob + SIMD), c11);
//
//
// }
// template <uint32_t step, uint32_t H_f, uint32_t W_f>
// inline void conv_microkernel_pool_accum_end(
//                             uint32_t input_col_stride,
//                             float * I,
//                             float * F,
//                             float * O_buffer,
//                             float *O
//                           )
// {
//
//   __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
//
//   c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
//    c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
//    c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
//    c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
//    c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
//    c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
//    c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
//    c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
//    c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
//    c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
//    c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
//    c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
//   int updates = 0;
//   // uint32_t step = stride*C_ob;
//   // int count = 0;
//   for(uint32_t n = 0; n < H_f; n++){
//
//     int filter_offset_h = n*W_f*C_ib*C_ob;
//     int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
//
//     for(uint32_t m = 0; m < W_f; m++){
//
//       int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
//       int input_stencil_w = m*C_ib + input_stencil_h;
//
//       for(uint32_t ii = 0 ; ii < C_ib; ii++){
//
//         // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
//         float *b = F + filter_offset_w;
//         float *a = I + input_stencil_w;
//
//         int p_cur = ii;
//         b0 = _mm256_load_ps(b + (ii * C_ob));
//         b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
//         a_reg = _mm256_broadcast_ss(a + (ii));
//         p_cur += step;
//         c0 = _mm256_fmadd_ps(a_reg, b0, c0);
//         c1 = _mm256_fmadd_ps(a_reg, b1, c1);
//         a_reg = _mm256_broadcast_ss(a + (p_cur));
//         p_cur += step;
//         c2 = _mm256_fmadd_ps(a_reg, b0, c2);
//         c3 = _mm256_fmadd_ps(a_reg, b1, c3);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c4 = _mm256_fmadd_ps(a_reg, b0, c4);
//         c5 = _mm256_fmadd_ps(a_reg, b1, c5);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c6 = _mm256_fmadd_ps(a_reg, b0, c6);
//         c7 = _mm256_fmadd_ps(a_reg, b1, c7);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c8 = _mm256_fmadd_ps(a_reg, b0, c8);
//         c9 = _mm256_fmadd_ps(a_reg, b1, c9);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c10 = _mm256_fmadd_ps(a_reg, b0, c10);
//         c11 = _mm256_fmadd_ps(a_reg, b1, c11);
//         // count++;
//       }
//     }
//   }
//
//   // horizontal pooling
//   b0 = _mm256_load_ps(O);
//   b1 = _mm256_load_ps(O + SIMD);
//
//   c2 = _mm256_max_ps(c2,c0);
//   c3 = _mm256_max_ps(c3,c1);
//   c6 = _mm256_max_ps(c6,c4);
//   c7 = _mm256_max_ps(c7,c5);
//   // c10 = _mm256_max_ps(c10,c8);
//   // c11 = _mm256_max_ps(c11,c9);
//
//   c2 = _mm256_max_ps(c2,c4);
//   c3 = _mm256_max_ps(c3,c5);
//   c6 = _mm256_max_ps(c6,c8);
//   c7 = _mm256_max_ps(c7,c9);
//
//   c0 = _mm256_max_ps(c0, b0);
//   c1 = _mm256_max_ps(c1,b1);
//
//   // accumulate with previous row
//   b0 = _mm256_load_ps(O + (1 * C_ob));
//   b1 = _mm256_load_ps(O + (2 * C_ob));
//   // a_reg = _mm256_load_ps(O + (3 * C_ob));
//   a_reg = _mm256_load_ps(O + (1 * C_ob) + SIMD);
//   __m256 temp = _mm256_load_ps(O + (2 * C_ob) + SIMD);
//
//   c2 = _mm256_max_ps(c2, b0);
//   c3 = _mm256_max_ps(c3, a_reg);
//
//   c6 = _mm256_max_ps(c6, b1);
//   c7 = _mm256_max_ps(c7, temp);
//
//   // c10 = _mm256_max_ps(c10, c0);
//   // c11 = _mm256_max_ps(c11, c1);
//
//
//   // store to output of pooling layer
//   _mm256_store_ps(O + (0 * C_ob), c0);
//   _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
//
//   _mm256_store_ps(O + (1 * C_ob), c2);
//   _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
//
//   _mm256_store_ps(O + (2 * C_ob), c6);
//   _mm256_store_ps(O + (2 * C_ob + SIMD), c7);
//
//
//
//
// }
//
// // accumulate to the previous pool output row
// //TODO: do these stores non temporally
// // write to new new output row
// template <uint32_t step, uint32_t H_f, uint32_t W_f>
// inline void conv_microkernel_pool_start(
//                             uint32_t input_col_stride,
//                             uint32_t pool_col_stride,
//                             float * I,
//                             float * F,
//                             float * O_buffer,
//                             float *O
//                           )
// {
//
//   __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
//   float * O_next_row = O + pool_col_stride;
//   c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
//    c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
//    c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
//    c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
//    c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
//    c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
//    c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
//    c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
//    c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
//    c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
//    c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
//    c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
//   int updates = 0;
//   // uint32_t step = stride*C_ob;
//   // int count = 0;
//   for(uint32_t n = 0; n < H_f; n++){
//
//     int filter_offset_h = n*W_f*C_ib*C_ob;
//     int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
//
//     for(uint32_t m = 0; m < W_f; m++){
//
//       int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
//       int input_stencil_w = m*C_ib + input_stencil_h;
//
//       for(uint32_t ii = 0 ; ii < C_ib; ii++){
//
//         // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
//         float *b = F + filter_offset_w;
//         float *a = I + input_stencil_w;
//
//         int p_cur = ii;
//         b0 = _mm256_load_ps(b + (ii * C_ob));
//         b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
//         a_reg = _mm256_broadcast_ss(a + (ii));
//         p_cur += step;
//         c0 = _mm256_fmadd_ps(a_reg, b0, c0);
//         c1 = _mm256_fmadd_ps(a_reg, b1, c1);
//         a_reg = _mm256_broadcast_ss(a + (p_cur));
//         p_cur += step;
//         c2 = _mm256_fmadd_ps(a_reg, b0, c2);
//         c3 = _mm256_fmadd_ps(a_reg, b1, c3);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c4 = _mm256_fmadd_ps(a_reg, b0, c4);
//         c5 = _mm256_fmadd_ps(a_reg, b1, c5);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c6 = _mm256_fmadd_ps(a_reg, b0, c6);
//         c7 = _mm256_fmadd_ps(a_reg, b1, c7);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c8 = _mm256_fmadd_ps(a_reg, b0, c8);
//         c9 = _mm256_fmadd_ps(a_reg, b1, c9);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c10 = _mm256_fmadd_ps(a_reg, b0, c10);
//         c11 = _mm256_fmadd_ps(a_reg, b1, c11);
//         // count++;
//       }
//     }
//   }
//
//   // horizontal pooling
//   // load partial updates from previous rows
//   b0 = _mm256_load_ps(O + (0 * C_ob));
//   b1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);
//   a_reg = _mm256_load_ps(O + (1 * C_ob));
//   __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);
//
//   c2 = _mm256_max_ps(c2,c0);
//   c3 = _mm256_max_ps(c3,c1);
//   c6 = _mm256_max_ps(c6,c4);
//   c7 = _mm256_max_ps(c7,c5);
//   c10 = _mm256_max_ps(c10,c8);
//   c11 = _mm256_max_ps(c11,c9);
//   c2 = _mm256_max_ps(c2,c4);
//   c3 = _mm256_max_ps(c3,c5);
//   c6 = _mm256_max_ps(c6,c8);
//   c7 = _mm256_max_ps(c7,c9);
//
//   // store partial updates to next row
//
//   _mm256_store_ps(O_next_row + (0 * C_ob), c2);
//   _mm256_store_ps(O_next_row + (0 * C_ob) + SIMD, c3);
//
//   _mm256_store_ps(O_next_row + (1 * C_ob), c6);
//   _mm256_store_ps(O_next_row + (1 * C_ob + SIMD), c7);
//
//   _mm256_store_ps(O_next_row + (2 * C_ob), c10);
//   _mm256_store_ps(O_next_row + (2 * C_ob + SIMD), c11);
//
//   //accumulate with previous rows
//   c2 = _mm256_max_ps(c2, b0);
//   c3 = _mm256_max_ps(c3, b1);
//
//   c6 = _mm256_max_ps(c6, a_reg);
//   c7 = _mm256_max_ps(c7, temp);
//
//   // c10 = _mm256_max_ps(c10, a_reg);
//   // c11 = _mm256_max_ps(c11, c1);
//
//   // store to output of pooling layer
//
//   _mm256_store_ps(O + (0 * C_ob), c2);
//   _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);
//
//   _mm256_store_ps(O + (1 * C_ob), c6);
//   _mm256_store_ps(O + (1 * C_ob + SIMD), c7);
//
//   //
//   // _mm256_store_ps(O + (2 * C_ob), c10);
//   // _mm256_store_ps(O + (2 * C_ob + SIMD), c11);
//
//
// }
// template <uint32_t step, uint32_t H_f, uint32_t W_f>
// inline void conv_microkernel_pool(
//                             uint32_t input_col_stride,
//                             uint32_t pool_col_stride,
//                             float * I,
//                             float * F,
//                             float * O_buffer,
//                             float *O
//                           )
// {
//
//   __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
//   float * O_next_row = O + pool_col_stride;
//   c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
//    c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
//    c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
//    c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
//    c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
//    c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
//    c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
//    c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
//    c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
//    c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
//    c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
//    c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
//   int updates = 0;
//   // uint32_t step = stride*C_ob;
//   // int count = 0;
//   for(uint32_t n = 0; n < H_f; n++){
//
//     int filter_offset_h = n*W_f*C_ib*C_ob;
//     int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
//
//     for(uint32_t m = 0; m < W_f; m++){
//
//       int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
//       int input_stencil_w = m*C_ib + input_stencil_h;
//
//       for(uint32_t ii = 0 ; ii < C_ib; ii++){
//
//         // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
//         float *b = F + filter_offset_w;
//         float *a = I + input_stencil_w;
//
//         int p_cur = ii;
//         b0 = _mm256_load_ps(b + (ii * C_ob));
//         b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
//         a_reg = _mm256_broadcast_ss(a + (ii));
//         p_cur += step;
//         c0 = _mm256_fmadd_ps(a_reg, b0, c0);
//         c1 = _mm256_fmadd_ps(a_reg, b1, c1);
//         a_reg = _mm256_broadcast_ss(a + (p_cur));
//         p_cur += step;
//         c2 = _mm256_fmadd_ps(a_reg, b0, c2);
//         c3 = _mm256_fmadd_ps(a_reg, b1, c3);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c4 = _mm256_fmadd_ps(a_reg, b0, c4);
//         c5 = _mm256_fmadd_ps(a_reg, b1, c5);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c6 = _mm256_fmadd_ps(a_reg, b0, c6);
//         c7 = _mm256_fmadd_ps(a_reg, b1, c7);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c8 = _mm256_fmadd_ps(a_reg, b0, c8);
//         c9 = _mm256_fmadd_ps(a_reg, b1, c9);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c10 = _mm256_fmadd_ps(a_reg, b0, c10);
//         c11 = _mm256_fmadd_ps(a_reg, b1, c11);
//         // count++;
//       }
//     }
//   }
//
//   // horizontal pooling previous tile
//   b0 = _mm256_load_ps(O_next_row);
//   b1 = _mm256_load_ps(O_next_row + SIMD);
//
//
//   a_reg = _mm256_load_ps(O + (1 * C_ob));
//   __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);
//
//   //
//   c2 = _mm256_max_ps(c2,c0);
//   c3 = _mm256_max_ps(c3,c1);
//   c6 = _mm256_max_ps(c6,c4);
//   c7 = _mm256_max_ps(c7,c5);
//   c10 = _mm256_max_ps(c10,c8);
//   c11 = _mm256_max_ps(c11,c9);
//
//   c2 = _mm256_max_ps(c2,c4);
//   c3 = _mm256_max_ps(c3,c5);
//   c6 = _mm256_max_ps(c6,c8);
//   c7 = _mm256_max_ps(c7,c9);
//
//
//   // accumulate with previous tile
//   c0 = _mm256_max_ps(c0, b0);
//   c1 = _mm256_max_ps(c1,b1);
//
//   //Store Partial Outputs to next row
//   _mm256_store_ps(O_next_row + (0 * C_ob), c0);
//   _mm256_store_ps(O_next_row + (0 * C_ob) + SIMD, c1);
//
//   _mm256_store_ps(O_next_row + (1 * C_ob), c2);
//   _mm256_store_ps(O_next_row + (1 * C_ob + SIMD), c3);
//
//   _mm256_store_ps(O_next_row + (2 * C_ob), c6);
//   _mm256_store_ps(O_next_row + (2 * C_ob + SIMD), c7);
//
//   _mm256_store_ps(O_next_row + (3 * C_ob), c10);
//   _mm256_store_ps(O_next_row + (3 * C_ob + SIMD), c11);
//
//   //Load partial outputs from previous row
//   c4 = _mm256_load_ps(O + (2 * C_ob));
//   c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);
//
//   c8 = _mm256_load_ps(O + (3 * C_ob));
//   c9 = _mm256_load_ps(O + (3 * C_ob) + SIMD);
//
//   b0 = _mm256_load_ps(O);
//   b1 = _mm256_load_ps(O + SIMD);
//
//
//   //accumulate with previous row
//   c2 = _mm256_max_ps(c2, a_reg);
//   c3 = _mm256_max_ps(c3, temp);
//   c6 = _mm256_max_ps(c6,c4);
//   c7 = _mm256_max_ps(c7,c5);
//   c10 = _mm256_max_ps(c10,c8);
//   c11 = _mm256_max_ps(c11,c9);
//   c0 = _mm256_max_ps(c0, b0);
//   c1 = _mm256_max_ps(c1, b1);
//
//   // store to output of pooling layer
//   _mm256_store_ps(O + (0 * C_ob), c0);
//   _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
//
//   _mm256_store_ps(O + (1 * C_ob), c2);
//   _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
//
//   _mm256_store_ps(O + (2 * C_ob), c6);
//   _mm256_store_ps(O + (2 * C_ob + SIMD), c7);
//
//   _mm256_store_ps(O + (3 * C_ob), c10);
//   _mm256_store_ps(O + (3 * C_ob + SIMD), c11);
//
//
// }
// template <uint32_t step, uint32_t H_f, uint32_t W_f>
// inline void conv_microkernel_pool_end(
//                             uint32_t input_col_stride,
//                             uint32_t pool_col_stride,
//                             float * I,
//                             float * F,
//                             float * O_buffer,
//                             float * O
//                           )
// {
//
//   __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
//   float * O_next_row = O + pool_col_stride;
//   c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
//    c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
//    c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
//    c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
//    c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
//    c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
//    c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
//    c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
//    c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
//    c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
//    c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
//    c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
//   int updates = 0;
//   // uint32_t step = stride*C_ob;
//   // int count = 0;
//   for(uint32_t n = 0; n < H_f; n++){
//
//     int filter_offset_h = n*W_f*C_ib*C_ob;
//     int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
//
//     for(uint32_t m = 0; m < W_f; m++){
//
//       int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
//       int input_stencil_w = m*C_ib + input_stencil_h;
//
//       for(uint32_t ii = 0 ; ii < C_ib; ii++){
//
//         // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
//         float *b = F + filter_offset_w;
//         float *a = I + input_stencil_w;
//
//         int p_cur = ii;
//         b0 = _mm256_load_ps(b + (ii * C_ob));
//         b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
//         a_reg = _mm256_broadcast_ss(a + (ii));
//         p_cur += step;
//         c0 = _mm256_fmadd_ps(a_reg, b0, c0);
//         c1 = _mm256_fmadd_ps(a_reg, b1, c1);
//         a_reg = _mm256_broadcast_ss(a + (p_cur));
//         p_cur += step;
//         c2 = _mm256_fmadd_ps(a_reg, b0, c2);
//         c3 = _mm256_fmadd_ps(a_reg, b1, c3);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c4 = _mm256_fmadd_ps(a_reg, b0, c4);
//         c5 = _mm256_fmadd_ps(a_reg, b1, c5);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c6 = _mm256_fmadd_ps(a_reg, b0, c6);
//         c7 = _mm256_fmadd_ps(a_reg, b1, c7);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c8 = _mm256_fmadd_ps(a_reg, b0, c8);
//         c9 = _mm256_fmadd_ps(a_reg, b1, c9);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur += step;
//         c10 = _mm256_fmadd_ps(a_reg, b0, c10);
//         c11 = _mm256_fmadd_ps(a_reg, b1, c11);
//         // count++;
//       }
//     }
//   }
//
//   // horizontal pooling
//   b0 = _mm256_load_ps(O_next_row);
//   b1 = _mm256_load_ps(O_next_row + SIMD);
//
//
//   a_reg = _mm256_load_ps(O);
//   __m256 temp = _mm256_load_ps(O + SIMD);
//   c2 = _mm256_max_ps(c2,c0);
//   c3 = _mm256_max_ps(c3,c1);
//   c6 = _mm256_max_ps(c6,c4);
//   c7 = _mm256_max_ps(c7,c5);
//
//   c2 = _mm256_max_ps(c2,c4);
//   c3 = _mm256_max_ps(c3,c5);
//   c6 = _mm256_max_ps(c6,c8);
//   c7 = _mm256_max_ps(c7,c9);
//
//
//
//   // accumulate with previous tile
//   c0 = _mm256_max_ps(c0, b0);
//   c1 = _mm256_max_ps(c1,b1);
//
//
//
//   // store partial updates to next row
//   _mm256_store_ps(O_next_row + (0 * C_ob), c0);
//   _mm256_store_ps(O_next_row + (0 * C_ob) + SIMD, c1);
//
//   _mm256_store_ps(O_next_row + (1 * C_ob), c2);
//   _mm256_store_ps(O_next_row + (1 * C_ob + SIMD), c3);
//
//   _mm256_store_ps(O_next_row + (2 * C_ob), c6);
//   _mm256_store_ps(O_next_row + (2 * C_ob + SIMD), c7);
//
//   // load partial updates from previous row
//   c4 = _mm256_load_ps(O + (1 * C_ob));
//   c8 = _mm256_load_ps(O + (2 * C_ob));
//
//   c5 = _mm256_load_ps(O + (1 * C_ob) + SIMD);
//   c9 = _mm256_load_ps(O + (2 * C_ob) + SIMD);
//
//   c2 = _mm256_max_ps(c2, c4);
//   c3 = _mm256_max_ps(c3, c5);
//
//   c6 = _mm256_max_ps(c6, c8);
//   c7 = _mm256_max_ps(c7, c9);
//
//   c0 = _mm256_max_ps(c0,a_reg);
//   c1 = _mm256_max_ps(c1, temp);
//
//   // c10 = _mm256_max_ps(c10, c0);
//   // c11 = _mm256_max_ps(c11, c1);
//
//
//   // store to output of pooling layer
//   _mm256_store_ps(O + (0 * C_ob), c0);
//   _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
//
//   _mm256_store_ps(O + (1 * C_ob), c2);
//   _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
//
//   _mm256_store_ps(O + (2 * C_ob), c6);
//   _mm256_store_ps(O + (2 * C_ob + SIMD), c7);
//
//
//
//
// }





// // TODO: could pack?
// // access to A at a stride of 1
// inline void conv_microkernel_gemm(
//                             uint32_t input_col_stride,
//                             uint32_t H_f,
//                             uint32_t W_f,
//                             float * I,
//                             float * F,
//                             float * O){
//
//   __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
//
//   c0 = _mm256_load_ps(O+ (0 * C_ob));
//    c1 = _mm256_load_ps(O+ (0 * C_ob) + SIMD);
//    c2 = _mm256_load_ps(O+ (1 * C_ob));
//    c3 = _mm256_load_ps(O+ (1 * C_ob) + SIMD);
//    c4 = _mm256_load_ps(O+ (2 * C_ob));
//    c5 = _mm256_load_ps(O+ (2 * C_ob) + SIMD);
//    c6 = _mm256_load_ps(O+ (3 * C_ob));
//    c7 = _mm256_load_ps(O+ (3 * C_ob) + SIMD);
//    c8 = _mm256_load_ps(O+ (4 * C_ob));
//    c9 = _mm256_load_ps(O+ (4 * C_ob) + SIMD);
//    c10 = _mm256_load_ps(O+ (5 * C_ob));
//    c11 = _mm256_load_ps(O+ (5 * C_ob) + SIMD);
//   int updates = 0;
//   // int count = 0;
//   for(uint32_t n = 0; n < H_f; n++){
//
//     int filter_offset_h = n*W_f*C_ib*C_ob;
//     int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
//
//     for(uint32_t m = 0; m < W_f; m++){
//
//       int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
//       int input_stencil_w = m*C_ib + input_stencil_h;
//       int p_cur = 0;
//       for(uint32_t ii = 0 ; ii < C_ib; ii++){
//
//         // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
//         float *b = F + filter_offset_w;
//         float *a = I + input_stencil_w;
//
//         b0 = _mm256_load_ps(b + (ii * C_ob));
//         b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
//         a_reg = _mm256_broadcast_ss(a + (p_cur));
//         p_cur++;//ii + C_ob;
//         c0 = _mm256_fmadd_ps(a_reg, b0, c0);
//         c1 = _mm256_fmadd_ps(a_reg, b1, c1);
//         a_reg = _mm256_broadcast_ss(a + (p_cur));
//         p_cur++;// C_ob;
//         c2 = _mm256_fmadd_ps(a_reg, b0, c2);
//         c3 = _mm256_fmadd_ps(a_reg, b1, c3);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur++;// C_ob;
//         c4 = _mm256_fmadd_ps(a_reg, b0, c4);
//         c5 = _mm256_fmadd_ps(a_reg, b1, c5);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur++;// C_ob;
//         c6 = _mm256_fmadd_ps(a_reg, b0, c6);
//         c7 = _mm256_fmadd_ps(a_reg, b1, c7);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur++;// C_ob;
//         c8 = _mm256_fmadd_ps(a_reg, b0, c8);
//         c9 = _mm256_fmadd_ps(a_reg, b1, c9);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//         p_cur++;// C_ob;
//         c10 = _mm256_fmadd_ps(a_reg, b0, c10);
//         c11 = _mm256_fmadd_ps(a_reg, b1, c11);
//         // count++;
//       }
//     }
//   }
//
//   _mm256_store_ps(O + (0 * C_ob), c0);
//   _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
//   _mm256_store_ps(O + (1 * C_ob), c2);
//   _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
//   _mm256_store_ps(O + (2 * C_ob), c4);
//   _mm256_store_ps(O + (2 * C_ob + SIMD), c5);
//   _mm256_store_ps(O + (3 * C_ob), c6);
//   _mm256_store_ps(O + (3 * C_ob + SIMD), c7);
//   _mm256_store_ps(O + (4 * C_ob), c8);
//   _mm256_store_ps(O + (4 * C_ob + SIMD), c9);
//   _mm256_store_ps(O + (5 * C_ob), c10);
//   _mm256_store_ps(O + (5 * C_ob + SIMD), c11);
//
// }
//
// // accesses to A at a stride of 1
// inline void conv_microkernel_start_gemm(
//                             uint32_t input_col_stride,
//                             uint32_t H_f,
//                             uint32_t W_f,
//                             float * I,
//                             float * F,
//                             float * O){
//
//   __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
//
//   c0 = _mm256_setzero_ps();//_mm256_load_ps(O+ (0 * C_ob));
//    c1 = _mm256_setzero_ps();//_mm256_load_ps(O+ (0 * C_ob) + SIMD);
//    c2 = _mm256_setzero_ps();//_mm256_load_ps(O+ (1 * C_ob));
//    c3 = _mm256_setzero_ps();//_mm256_load_ps(O+ (1 * C_ob) + SIMD);
//    c4 = _mm256_setzero_ps();//_mm256_load_ps(O+ (2 * C_ob));
//    c5 = _mm256_setzero_ps();//_mm256_load_ps(O+ (2 * C_ob) + SIMD);
//    c6 = _mm256_setzero_ps();//_mm256_load_ps(O+ (3 * C_ob));
//    c7 = _mm256_setzero_ps();//_mm256_load_ps(O+ (3 * C_ob) + SIMD);
//    c8 = _mm256_setzero_ps();//_mm256_load_ps(O+ (4 * C_ob));
//    c9 = _mm256_setzero_ps();//_mm256_load_ps(O+ (4 * C_ob) + SIMD);
//    c10 = _mm256_setzero_ps();//_mm256_load_ps(O+ (5 * C_ob));
//    c11 = _mm256_setzero_ps();//_mm256_load_ps(O+ (5 * C_ob) + SIMD);
//   int updates = 0;
//   // int count = 0;
//   for(uint32_t n = 0; n < H_f; n++){
//
//     int filter_offset_h = n*W_f*C_ib*C_ob;
//     int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
//
//     for(uint32_t m = 0; m < W_f; m++){
//
//       int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
//       int input_stencil_w = m*C_ib + input_stencil_h;
//       int p_cur = 0;
//       for(uint32_t ii = 0 ; ii < C_ib; ii++){
//
//         // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
//         float *b = F + filter_offset_w;
//         float *a = I + input_stencil_w;
//
//         b0 = _mm256_load_ps(b + (ii * C_ob));
//         b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
//         a_reg = _mm256_broadcast_ss(a + (p_cur));
//         p_cur++;//ii + C_ob;
//         c0 = _mm256_fmadd_ps(a_reg, b0, c0);
//         c1 = _mm256_fmadd_ps(a_reg, b1, c1);
//         a_reg = _mm256_broadcast_ss(a + (p_cur));
//         p_cur++;// C_ob;
//         c2 = _mm256_fmadd_ps(a_reg, b0, c2);
//         c3 = _mm256_fmadd_ps(a_reg, b1, c3);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//
//         p_cur++;// C_ob;
//         c4 = _mm256_fmadd_ps(a_reg, b0, c4);
//         c5 = _mm256_fmadd_ps(a_reg, b1, c5);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//
//         p_cur++;// C_ob;
//         c6 = _mm256_fmadd_ps(a_reg, b0, c6);
//         c7 = _mm256_fmadd_ps(a_reg, b1, c7);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//
//         p_cur++;// C_ob;
//         c8 = _mm256_fmadd_ps(a_reg, b0, c8);
//         c9 = _mm256_fmadd_ps(a_reg, b1, c9);
//         a_reg = _mm256_broadcast_ss(a +  (p_cur));
//
//         p_cur++;// C_ob;
//         c10 = _mm256_fmadd_ps(a_reg, b0, c10);
//         c11 = _mm256_fmadd_ps(a_reg, b1, c11);
//
//         // count++;
//       }
//     }
//   }
// // store_C(C_ob,O+ block_offset + col_offset + k*C_ob);
//
//   _mm256_store_ps(O + (0 * C_ob), c0);
//   _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
//   _mm256_store_ps(O + (1 * C_ob), c2);
//   _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
//   _mm256_store_ps(O + (2 * C_ob), c4);
//   _mm256_store_ps(O + (2 * C_ob + SIMD), c5);
//   _mm256_store_ps(O + (3 * C_ob), c6);
//   _mm256_store_ps(O + (3 * C_ob + SIMD), c7);
//   _mm256_store_ps(O + (4 * C_ob), c8);
//   _mm256_store_ps(O + (4 * C_ob + SIMD), c9);
//   _mm256_store_ps(O + (5 * C_ob), c10);
//   _mm256_store_ps(O + (5 * C_ob + SIMD), c11);
//
// }
//
//
//
// // TODO: gemm style to put on Tensor Core
// inline void kernel_conv
// (
//  int m,
//  int n,
//  int k,
//  float*      a,
//  float*      b,
//  float*      c
// ){
//
//   __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
//
//   c0 = _mm256_load_ps(c + (0 * n));
//    c1 = _mm256_load_ps(c + (0 * n) + SIMD);
//    c2 = _mm256_load_ps(c + (1 * n));
//    c3 = _mm256_load_ps(c + (1 * n) + SIMD);
//    c4 = _mm256_load_ps(c + (2 * n));
//    c5 = _mm256_load_ps(c + (2 * n) + SIMD);
//    c6 = _mm256_load_ps(c + (3 * n));
//    c7 = _mm256_load_ps(c + (3 * n) + SIMD);
//    c8 = _mm256_load_ps(c + (4 * n));
//    c9 = _mm256_load_ps(c + (4 * n) + SIMD);
//    c10 = _mm256_load_ps(c + (5 * n));
//    c11 = _mm256_load_ps(c + (5 * n) + SIMD);
//
//       for (int p = 0; p != k; p++)
//       {
//               b0 = _mm256_load_ps(b + (p * n));
//       	      b1 = _mm256_load_ps(b + (p * n + SIMD));
//       	      a_reg = _mm256_broadcast_ss(a + (p));
//               int p_cur = p + n;
//        	      c0 = _mm256_fmadd_ps(a_reg, b0, c0);
//               c1 = _mm256_fmadd_ps(a_reg, b1, c1);
//       	      a_reg = _mm256_broadcast_ss(a + (p_cur));
//               p_cur += n;
//               c2 = _mm256_fmadd_ps(a_reg, b0, c2);
//               c3 = _mm256_fmadd_ps(a_reg, b1, c3);
//               a_reg = _mm256_broadcast_ss(a +  (p_cur));
//               p_cur += n;
//               c4 = _mm256_fmadd_ps(a_reg, b0, c4);
//               c5 = _mm256_fmadd_ps(a_reg, b1, c5);
//               a_reg = _mm256_broadcast_ss(a +  (p_cur));
//               p_cur += n;
//               c6 = _mm256_fmadd_ps(a_reg, b0, c6);
//               c7 = _mm256_fmadd_ps(a_reg, b1, c7);
//               a_reg = _mm256_broadcast_ss(a +  (p_cur));
//               p_cur += n;
//               c8 = _mm256_fmadd_ps(a_reg, b0, c8);
//               c9 = _mm256_fmadd_ps(a_reg, b1, c9);
//               a_reg = _mm256_broadcast_ss(a +  (p_cur));
//               p_cur += n;
//               c10 = _mm256_fmadd_ps(a_reg, b0, c10);
//               c11 = _mm256_fmadd_ps(a_reg, b1, c11);
//
//        }
//
//        _mm256_store_ps(c + (0 * n), c0);
//        _mm256_store_ps(c + (0 * n) + SIMD, c1);
//        _mm256_store_ps(c + (1 * n), c2);
//        _mm256_store_ps(c + (1 * n + SIMD), c3);
//        _mm256_store_ps(c + (2 * n), c4);
//        _mm256_store_ps(c + (2 * n + SIMD), c5);
//        _mm256_store_ps(c + (3 * n), c6);
//        _mm256_store_ps(c + (3 * n + SIMD), c7);
//        _mm256_store_ps(c + (4 * n), c8);
//        _mm256_store_ps(c + (4 * n + SIMD), c9);
//        _mm256_store_ps(c + (5 * n), c10);
//        _mm256_store_ps(c + (5 * n + SIMD), c11);
// }
//
//
// inline void kernel_conv_start
// (
//  int m,
//  int n,
//  int k,
//  float*      a,
//  float*      b,
//  float*      c
// ){
//
//   __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
//
//   // c0 = _mm256_load_ps(c + (0 * n));
//   //  c1 = _mm256_load_ps(c + (0 * n) + SIMD);
//   //  c2 = _mm256_load_ps(c + (1 * n));
//   //  c3 = _mm256_load_ps(c + (1 * n) + SIMD);
//   //  c4 = _mm256_load_ps(c + (2 * n));
//   //  c5 = _mm256_load_ps(c + (2 * n) + SIMD);
//   //  c6 = _mm256_load_ps(c + (3 * n));
//   //  c7 = _mm256_load_ps(c + (3 * n) + SIMD);
//   //  c8 = _mm256_load_ps(c + (4 * n));
//   //  c9 = _mm256_load_ps(c + (4 * n) + SIMD);
//   //  c10 = _mm256_load_ps(c + (5 * n));
//   //  c11 = _mm256_load_ps(c + (5 * n) + SIMD);
//
//       for (int p = 0; p != k; p++)
//       {
//               b0 = _mm256_load_ps(b + (p * n));
//       	      b1 = _mm256_load_ps(b + (p * n + SIMD));
//       	      a_reg = _mm256_broadcast_ss(a + (p));
//               int p_cur = p + n;
//        	      c0 = _mm256_fmadd_ps(a_reg, b0, c0);
//               c1 = _mm256_fmadd_ps(a_reg, b1, c1);
//       	      a_reg = _mm256_broadcast_ss(a + (p_cur));
//               p_cur += n;
//               c2 = _mm256_fmadd_ps(a_reg, b0, c2);
//               c3 = _mm256_fmadd_ps(a_reg, b1, c3);
//               a_reg = _mm256_broadcast_ss(a +  (p_cur));
//               p_cur += n;
//               c4 = _mm256_fmadd_ps(a_reg, b0, c4);
//               c5 = _mm256_fmadd_ps(a_reg, b1, c5);
//               a_reg = _mm256_broadcast_ss(a +  (p_cur));
//               p_cur += n;
//               c6 = _mm256_fmadd_ps(a_reg, b0, c6);
//               c7 = _mm256_fmadd_ps(a_reg, b1, c7);
//               a_reg = _mm256_broadcast_ss(a +  (p_cur));
//               p_cur += n;
//               c8 = _mm256_fmadd_ps(a_reg, b0, c8);
//               c9 = _mm256_fmadd_ps(a_reg, b1, c9);
//               a_reg = _mm256_broadcast_ss(a +  (p_cur));
//               p_cur += n;
//               c10 = _mm256_fmadd_ps(a_reg, b0, c10);
//               c11 = _mm256_fmadd_ps(a_reg, b1, c11);
//
//        }
//
//        _mm256_store_ps(c + (0 * n), c0);
//        _mm256_store_ps(c + (0 * n) + SIMD, c1);
//        _mm256_store_ps(c + (1 * n), c2);
//        _mm256_store_ps(c + (1 * n + SIMD), c3);
//        _mm256_store_ps(c + (2 * n), c4);
//        _mm256_store_ps(c + (2 * n + SIMD), c5);
//        _mm256_store_ps(c + (3 * n), c6);
//        _mm256_store_ps(c + (3 * n + SIMD), c7);
//        _mm256_store_ps(c + (4 * n), c8);
//        _mm256_store_ps(c + (4 * n + SIMD), c9);
//        _mm256_store_ps(c + (5 * n), c10);
//        _mm256_store_ps(c + (5 * n + SIMD), c11);
// }
