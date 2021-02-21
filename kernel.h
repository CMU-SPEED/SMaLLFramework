// #include <stdio.h>
// #include <stdlib.h>
#include <immintrin.h>

#define SIMD 8
# define rank_k 16

// AMD Zen2 parameters
#define W_ob 6
#define C_ob 16
#define C_ib rank_k


#define filter_seed 2048
#define image_seed 1729


#define POOL_KERNEL 3
#define POOL_STRIDE 2



// no hints
inline void fused_microkernel(
                         float * I,
                         uint32_t C_o_1x1,
                         float * F_1x1,
                         uint32_t out_panel_size,
                         float * O_1x1
){
  // printf("fused microkernel");
  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  // filter format for the 1x1 : C_i/C_ib x C_o/C_ob x C_ib x C_ob
  float * O = O_1x1;
  for(uint32_t j = 0; j < C_o_1x1; j+= C_ob){
    // float * F_ptr = F + j * C_ob * C_ob;
    float *b = F_1x1 + (j/C_ob) * C_ob * C_ob;
    float *a = I;
    c0 = _mm256_load_ps(O+ (0 * C_ob));
     c1 = _mm256_load_ps(O+ (0 * C_ob) + SIMD);
     c2 = _mm256_load_ps(O+ (1 * C_ob));
     c3 = _mm256_load_ps(O+ (1 * C_ob) + SIMD);
     c4 = _mm256_load_ps(O+ (2 * C_ob));
     c5 = _mm256_load_ps(O+ (2 * C_ob) + SIMD);
     c6 = _mm256_load_ps(O+ (3 * C_ob));
     c7 = _mm256_load_ps(O+ (3 * C_ob) + SIMD);
     c8 = _mm256_load_ps(O+ (4 * C_ob));
     c9 = _mm256_load_ps(O+ (4 * C_ob) + SIMD);
     c10 = _mm256_load_ps(O+ (5 * C_ob));
     c11 = _mm256_load_ps(O+ (5 * C_ob) + SIMD);
    // Reuse this tile as much as possible
    for(uint32_t ii = 0 ; ii < C_ob; ii++){
      b0 = _mm256_load_ps(b + (ii * C_ob));
      b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
      a_reg = _mm256_broadcast_ss(a + (ii));
      uint32_t p_cur = ii + C_ob;
      c0 = _mm256_fmadd_ps(a_reg, b0, c0);
      c1 = _mm256_fmadd_ps(a_reg, b1, c1);
      a_reg = _mm256_broadcast_ss(a + (p_cur));
      p_cur += C_ob;
      c2 = _mm256_fmadd_ps(a_reg, b0, c2);
      c3 = _mm256_fmadd_ps(a_reg, b1, c3);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c4 = _mm256_fmadd_ps(a_reg, b0, c4);
      c5 = _mm256_fmadd_ps(a_reg, b1, c5);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c6 = _mm256_fmadd_ps(a_reg, b0, c6);
      c7 = _mm256_fmadd_ps(a_reg, b1, c7);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c8 = _mm256_fmadd_ps(a_reg, b0, c8);
      c9 = _mm256_fmadd_ps(a_reg, b1, c9);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c10 = _mm256_fmadd_ps(a_reg, b0, c10);
      c11 = _mm256_fmadd_ps(a_reg, b1, c11);
      // count++;
      // print256_float(c0);
    }
    // printf("end of 1st update\n");print256_float(c0);
    // Store it out
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

    // printf("1x1 index: %d \n", O - O_1x1);
    O = O + out_panel_size;
  }
}

inline void fused_microkernel_start(
                         float * I,
                         uint32_t C_o_1x1,
                         float * F_1x1,
                         uint32_t out_panel_size,
                         float * O_1x1
){


  // printf("fused microkernel");
  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  // filter format for the 1x1 : C_i/C_ib x C_o/C_ob x C_ib x C_ob
  float * O = O_1x1;
  for(uint32_t j = 0; j < C_o_1x1; j+= C_ob){
    // float * F_ptr = F + j * C_ob * C_ob;
    float *b = F_1x1 + (j/C_ob) * C_ob * C_ob;
    float *a = I;
    c0 = _mm256_setzero_ps();//_mm256_load_ps(O+ (0 * C_ob));
     c1 = _mm256_setzero_ps();//_mm256_load_ps(O+ (0 * C_ob) + SIMD);
     c2 = _mm256_setzero_ps();//_mm256_load_ps(O+ (1 * C_ob));
     c3 = _mm256_setzero_ps();//_mm256_load_ps(O+ (1 * C_ob) + SIMD);
     c4 = _mm256_setzero_ps();//_mm256_load_ps(O+ (2 * C_ob));
     c5 = _mm256_setzero_ps();//_mm256_load_ps(O+ (2 * C_ob) + SIMD);
     c6 = _mm256_setzero_ps();//_mm256_load_ps(O+ (3 * C_ob));
     c7 = _mm256_setzero_ps();//_mm256_load_ps(O+ (3 * C_ob) + SIMD);
     c8 = _mm256_setzero_ps();//_mm256_load_ps(O+ (4 * C_ob));
     c9 = _mm256_setzero_ps();//_mm256_load_ps(O+ (4 * C_ob) + SIMD);
     c10 = _mm256_setzero_ps();//_mm256_load_ps(O+ (5 * C_ob));
     c11 = _mm256_setzero_ps();//_mm256_load_ps(O+ (5 * C_ob) + SIMD);
    // Reuse this tile as much as possible
    for(uint32_t ii = 0 ; ii < C_ob; ii++){
      b0 = _mm256_load_ps(b + (ii * C_ob));
      b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
      a_reg = _mm256_broadcast_ss(a + (ii));
      uint32_t p_cur = ii + C_ob;
      c0 = _mm256_fmadd_ps(a_reg, b0, c0);
      c1 = _mm256_fmadd_ps(a_reg, b1, c1);
      a_reg = _mm256_broadcast_ss(a + (p_cur));
      p_cur += C_ob;
      c2 = _mm256_fmadd_ps(a_reg, b0, c2);
      c3 = _mm256_fmadd_ps(a_reg, b1, c3);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c4 = _mm256_fmadd_ps(a_reg, b0, c4);
      c5 = _mm256_fmadd_ps(a_reg, b1, c5);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c6 = _mm256_fmadd_ps(a_reg, b0, c6);
      c7 = _mm256_fmadd_ps(a_reg, b1, c7);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c8 = _mm256_fmadd_ps(a_reg, b0, c8);
      c9 = _mm256_fmadd_ps(a_reg, b1, c9);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c10 = _mm256_fmadd_ps(a_reg, b0, c10);
      c11 = _mm256_fmadd_ps(a_reg, b1, c11);
      // count++;
      // print256_float(c0);
    }
    // printf("end of 1st update\n");print256_float(c0);
    // Store it out
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

    // printf("1x1 index: %d \n", O - O_1x1);
    O = O + out_panel_size;
  }
}

inline void conv_microkernel(
                            uint32_t input_col_stride,
                            uint32_t H_f,
                            uint32_t W_f,
                            uint32_t stride,
                            float * I,
                            float * F,
                            float * O){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_load_ps(O+ (0 * C_ob));
   c1 = _mm256_load_ps(O+ (0 * C_ob) + SIMD);
   c2 = _mm256_load_ps(O+ (1 * C_ob));
   c3 = _mm256_load_ps(O+ (1 * C_ob) + SIMD);
   c4 = _mm256_load_ps(O+ (2 * C_ob));
   c5 = _mm256_load_ps(O+ (2 * C_ob) + SIMD);
   c6 = _mm256_load_ps(O+ (3 * C_ob));
   c7 = _mm256_load_ps(O+ (3 * C_ob) + SIMD);
   c8 = _mm256_load_ps(O+ (4 * C_ob));
   c9 = _mm256_load_ps(O+ (4 * C_ob) + SIMD);
   c10 = _mm256_load_ps(O+ (5 * C_ob));
   c11 = _mm256_load_ps(O+ (5 * C_ob) + SIMD);
  int updates = 0;
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
        p_cur += stride*C_ob;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += stride*C_ob;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += stride*C_ob;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += stride*C_ob;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += stride*C_ob;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += stride*C_ob;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }
  // printf("%d updates \n ", count);
// store_C(C_ob,O+ block_offset + col_offset + k*C_ob);

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


inline void conv_microkernel_start(
                            uint32_t input_col_stride,
                            uint32_t H_f,
                            uint32_t W_f,
                            uint32_t stride,
                            float * I,
                            float * F,
                            float * O){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_setzero_ps();//_mm256_load_ps(O+ (0 * C_ob));
   c1 = _mm256_setzero_ps();//_mm256_load_ps(O+ (0 * C_ob) + SIMD);
   c2 = _mm256_setzero_ps();//_mm256_load_ps(O+ (1 * C_ob));
   c3 = _mm256_setzero_ps();//_mm256_load_ps(O+ (1 * C_ob) + SIMD);
   c4 = _mm256_setzero_ps();//_mm256_load_ps(O+ (2 * C_ob));
   c5 = _mm256_setzero_ps();//_mm256_load_ps(O+ (2 * C_ob) + SIMD);
   c6 = _mm256_setzero_ps();//_mm256_load_ps(O+ (3 * C_ob));
   c7 = _mm256_setzero_ps();//_mm256_load_ps(O+ (3 * C_ob) + SIMD);
   c8 = _mm256_setzero_ps();//_mm256_load_ps(O+ (4 * C_ob));
   c9 = _mm256_setzero_ps();//_mm256_load_ps(O+ (4 * C_ob) + SIMD);
   c10 = _mm256_setzero_ps();//_mm256_load_ps(O+ (5 * C_ob));
   c11 = _mm256_setzero_ps();//_mm256_load_ps(O+ (5 * C_ob) + SIMD);
  int updates = 0;
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
        p_cur += stride*C_ob;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += stride*C_ob;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += stride*C_ob;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += stride*C_ob;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += stride*C_ob;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += stride*C_ob;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }
  // printf("%d updates \n ", count);
// store_C(C_ob,O+ block_offset + col_offset + k*C_ob);

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


inline void conv_microkernel_pool_first_row_start(
                            uint32_t input_col_stride,
                            uint32_t H_f,
                            uint32_t W_f,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          ){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
   c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
   c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
   c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
   c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
   c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
   c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
   c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
   c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
   c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
   c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
   c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
  int updates = 0;
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

        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        int p_cur = ii + C_ob;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += C_ob;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
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

inline void conv_microkernel_pool_first_row(
                            uint32_t input_col_stride,
                            uint32_t H_f,
                            uint32_t W_f,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          ){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
   c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
   c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
   c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
   c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
   c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
   c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
   c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
   c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
   c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
   c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
   c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
  int updates = 0;
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

        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        int p_cur = ii + C_ob;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += C_ob;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
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
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

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

  _mm256_store_ps(O + (3 * C_ob), c10);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c11);


}

inline void conv_microkernel_pool_first_row_end(
                            uint32_t input_col_stride,
                            uint32_t H_f,
                            uint32_t W_f,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          ){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
   c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
   c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
   c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
   c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
   c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
   c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
   c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
   c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
   c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
   c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
   c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
  int updates = 0;
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

        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        int p_cur = ii + C_ob;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += C_ob;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
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
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

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

inline void conv_microkernel_pool_accum_start(
                            uint32_t input_col_stride,
                            uint32_t H_f,
                            uint32_t W_f,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          ){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
   c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
   c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
   c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
   c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
   c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
   c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
   c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
   c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
   c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
   c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
   c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
  int updates = 0;
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

        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        int p_cur = ii + C_ob;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += C_ob;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
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

inline void conv_microkernel_pool_accum(
                            uint32_t input_col_stride,
                            uint32_t H_f,
                            uint32_t W_f,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          ){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
   c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
   c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
   c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
   c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
   c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
   c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
   c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
   c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
   c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
   c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
   c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
  int updates = 0;
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

        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        int p_cur = ii + C_ob;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += C_ob;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
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
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

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

  _mm256_store_ps(O + (3 * C_ob), c10);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c11);


}

inline void conv_microkernel_pool_accum_end(
                            uint32_t input_col_stride,
                            uint32_t H_f,
                            uint32_t W_f,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          ){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_load_ps(O_buffer+ (0 * C_ob));
   c1 = _mm256_load_ps(O_buffer+ (0 * C_ob) + SIMD);
   c2 = _mm256_load_ps(O_buffer+ (1 * C_ob));
   c3 = _mm256_load_ps(O_buffer+ (1 * C_ob) + SIMD);
   c4 = _mm256_load_ps(O_buffer+ (2 * C_ob));
   c5 = _mm256_load_ps(O_buffer+ (2 * C_ob) + SIMD);
   c6 = _mm256_load_ps(O_buffer+ (3 * C_ob));
   c7 = _mm256_load_ps(O_buffer+ (3 * C_ob) + SIMD);
   c8 = _mm256_load_ps(O_buffer+ (4 * C_ob));
   c9 = _mm256_load_ps(O_buffer+ (4 * C_ob) + SIMD);
   c10 = _mm256_load_ps(O_buffer+ (5 * C_ob));
   c11 = _mm256_load_ps(O_buffer+ (5 * C_ob) + SIMD);
  int updates = 0;
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

        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (ii));
        int p_cur = ii + C_ob;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += C_ob;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += C_ob;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
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
  c10 = _mm256_max_ps(c10,c8);
  c11 = _mm256_max_ps(c11,c9);

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



// TODO: could pack?
// access to A at a stride of 1
inline void conv_microkernel_gemm(
                            uint32_t input_col_stride,
                            uint32_t H_f,
                            uint32_t W_f,
                            float * I,
                            float * F,
                            float * O){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_load_ps(O+ (0 * C_ob));
   c1 = _mm256_load_ps(O+ (0 * C_ob) + SIMD);
   c2 = _mm256_load_ps(O+ (1 * C_ob));
   c3 = _mm256_load_ps(O+ (1 * C_ob) + SIMD);
   c4 = _mm256_load_ps(O+ (2 * C_ob));
   c5 = _mm256_load_ps(O+ (2 * C_ob) + SIMD);
   c6 = _mm256_load_ps(O+ (3 * C_ob));
   c7 = _mm256_load_ps(O+ (3 * C_ob) + SIMD);
   c8 = _mm256_load_ps(O+ (4 * C_ob));
   c9 = _mm256_load_ps(O+ (4 * C_ob) + SIMD);
   c10 = _mm256_load_ps(O+ (5 * C_ob));
   c11 = _mm256_load_ps(O+ (5 * C_ob) + SIMD);
  int updates = 0;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;
      int p_cur = 0;
      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur++;//ii + C_ob;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur++;// C_ob;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur++;// C_ob;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur++;// C_ob;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur++;// C_ob;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur++;// C_ob;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // count++;
      }
    }
  }
  // printf("%d updates \n ", count);
// store_C(C_ob,O+ block_offset + col_offset + k*C_ob);

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

// accesses to A at a stride of 1
inline void conv_microkernel_start_gemm(
                            uint32_t input_col_stride,
                            uint32_t H_f,
                            uint32_t W_f,
                            float * I,
                            float * F,
                            float * O){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_setzero_ps();//_mm256_load_ps(O+ (0 * C_ob));
   c1 = _mm256_setzero_ps();//_mm256_load_ps(O+ (0 * C_ob) + SIMD);
   c2 = _mm256_setzero_ps();//_mm256_load_ps(O+ (1 * C_ob));
   c3 = _mm256_setzero_ps();//_mm256_load_ps(O+ (1 * C_ob) + SIMD);
   c4 = _mm256_setzero_ps();//_mm256_load_ps(O+ (2 * C_ob));
   c5 = _mm256_setzero_ps();//_mm256_load_ps(O+ (2 * C_ob) + SIMD);
   c6 = _mm256_setzero_ps();//_mm256_load_ps(O+ (3 * C_ob));
   c7 = _mm256_setzero_ps();//_mm256_load_ps(O+ (3 * C_ob) + SIMD);
   c8 = _mm256_setzero_ps();//_mm256_load_ps(O+ (4 * C_ob));
   c9 = _mm256_setzero_ps();//_mm256_load_ps(O+ (4 * C_ob) + SIMD);
   c10 = _mm256_setzero_ps();//_mm256_load_ps(O+ (5 * C_ob));
   c11 = _mm256_setzero_ps();//_mm256_load_ps(O+ (5 * C_ob) + SIMD);
  int updates = 0;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;
      int p_cur = 0;
      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F + filter_offset_w;
        float *a = I + input_stencil_w;

        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        // print256_float(a_reg, "a");
        p_cur++;//ii + C_ob;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        // print256_float(a_reg,"a");
        p_cur++;// C_ob;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        // print256_float(a_reg,"a");

        p_cur++;// C_ob;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        // print256_float(a_reg,"a");

        p_cur++;// C_ob;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        // print256_float(a_reg,"a");

        p_cur++;// C_ob;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        // print256_float(a_reg,"a");

        p_cur++;// C_ob;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
        // print256_float(a_reg,"a");

        // count++;
      }
    }
  }
  // print256_float(c0);
  // print256_float(c11);
  // printf("%d updates \n ", count);
// store_C(C_ob,O+ block_offset + col_offset + k*C_ob);

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



// TODO: gemm style to put on Tensor Core
inline void kernel_conv
(
 int m,
 int n,
 int k,
 float*      a,
 float*      b,
 float*      c
){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_load_ps(c + (0 * n));
   c1 = _mm256_load_ps(c + (0 * n) + SIMD);
   c2 = _mm256_load_ps(c + (1 * n));
   c3 = _mm256_load_ps(c + (1 * n) + SIMD);
   c4 = _mm256_load_ps(c + (2 * n));
   c5 = _mm256_load_ps(c + (2 * n) + SIMD);
   c6 = _mm256_load_ps(c + (3 * n));
   c7 = _mm256_load_ps(c + (3 * n) + SIMD);
   c8 = _mm256_load_ps(c + (4 * n));
   c9 = _mm256_load_ps(c + (4 * n) + SIMD);
   c10 = _mm256_load_ps(c + (5 * n));
   c11 = _mm256_load_ps(c + (5 * n) + SIMD);

      for (int p = 0; p != k; p++)
      {
              b0 = _mm256_load_ps(b + (p * n));
      	      b1 = _mm256_load_ps(b + (p * n + SIMD));
      	      a_reg = _mm256_broadcast_ss(a + (p));
              int p_cur = p + n;
       	      c0 = _mm256_fmadd_ps(a_reg, b0, c0);
              c1 = _mm256_fmadd_ps(a_reg, b1, c1);
      	      a_reg = _mm256_broadcast_ss(a + (p_cur));
              p_cur += n;
              c2 = _mm256_fmadd_ps(a_reg, b0, c2);
              c3 = _mm256_fmadd_ps(a_reg, b1, c3);
              a_reg = _mm256_broadcast_ss(a +  (p_cur));
              p_cur += n;
              c4 = _mm256_fmadd_ps(a_reg, b0, c4);
              c5 = _mm256_fmadd_ps(a_reg, b1, c5);
              a_reg = _mm256_broadcast_ss(a +  (p_cur));
              p_cur += n;
              c6 = _mm256_fmadd_ps(a_reg, b0, c6);
              c7 = _mm256_fmadd_ps(a_reg, b1, c7);
              a_reg = _mm256_broadcast_ss(a +  (p_cur));
              p_cur += n;
              c8 = _mm256_fmadd_ps(a_reg, b0, c8);
              c9 = _mm256_fmadd_ps(a_reg, b1, c9);
              a_reg = _mm256_broadcast_ss(a +  (p_cur));
              p_cur += n;
              c10 = _mm256_fmadd_ps(a_reg, b0, c10);
              c11 = _mm256_fmadd_ps(a_reg, b1, c11);

       }

       _mm256_store_ps(c + (0 * n), c0);
       _mm256_store_ps(c + (0 * n) + SIMD, c1);
       _mm256_store_ps(c + (1 * n), c2);
       _mm256_store_ps(c + (1 * n + SIMD), c3);
       _mm256_store_ps(c + (2 * n), c4);
       _mm256_store_ps(c + (2 * n + SIMD), c5);
       _mm256_store_ps(c + (3 * n), c6);
       _mm256_store_ps(c + (3 * n + SIMD), c7);
       _mm256_store_ps(c + (4 * n), c8);
       _mm256_store_ps(c + (4 * n + SIMD), c9);
       _mm256_store_ps(c + (5 * n), c10);
       _mm256_store_ps(c + (5 * n + SIMD), c11);
}


inline void kernel_conv_start
(
 int m,
 int n,
 int k,
 float*      a,
 float*      b,
 float*      c
){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  // c0 = _mm256_load_ps(c + (0 * n));
  //  c1 = _mm256_load_ps(c + (0 * n) + SIMD);
  //  c2 = _mm256_load_ps(c + (1 * n));
  //  c3 = _mm256_load_ps(c + (1 * n) + SIMD);
  //  c4 = _mm256_load_ps(c + (2 * n));
  //  c5 = _mm256_load_ps(c + (2 * n) + SIMD);
  //  c6 = _mm256_load_ps(c + (3 * n));
  //  c7 = _mm256_load_ps(c + (3 * n) + SIMD);
  //  c8 = _mm256_load_ps(c + (4 * n));
  //  c9 = _mm256_load_ps(c + (4 * n) + SIMD);
  //  c10 = _mm256_load_ps(c + (5 * n));
  //  c11 = _mm256_load_ps(c + (5 * n) + SIMD);

      for (int p = 0; p != k; p++)
      {
              b0 = _mm256_load_ps(b + (p * n));
      	      b1 = _mm256_load_ps(b + (p * n + SIMD));
      	      a_reg = _mm256_broadcast_ss(a + (p));
              int p_cur = p + n;
       	      c0 = _mm256_fmadd_ps(a_reg, b0, c0);
              c1 = _mm256_fmadd_ps(a_reg, b1, c1);
      	      a_reg = _mm256_broadcast_ss(a + (p_cur));
              p_cur += n;
              c2 = _mm256_fmadd_ps(a_reg, b0, c2);
              c3 = _mm256_fmadd_ps(a_reg, b1, c3);
              a_reg = _mm256_broadcast_ss(a +  (p_cur));
              p_cur += n;
              c4 = _mm256_fmadd_ps(a_reg, b0, c4);
              c5 = _mm256_fmadd_ps(a_reg, b1, c5);
              a_reg = _mm256_broadcast_ss(a +  (p_cur));
              p_cur += n;
              c6 = _mm256_fmadd_ps(a_reg, b0, c6);
              c7 = _mm256_fmadd_ps(a_reg, b1, c7);
              a_reg = _mm256_broadcast_ss(a +  (p_cur));
              p_cur += n;
              c8 = _mm256_fmadd_ps(a_reg, b0, c8);
              c9 = _mm256_fmadd_ps(a_reg, b1, c9);
              a_reg = _mm256_broadcast_ss(a +  (p_cur));
              p_cur += n;
              c10 = _mm256_fmadd_ps(a_reg, b0, c10);
              c11 = _mm256_fmadd_ps(a_reg, b1, c11);

       }

       _mm256_store_ps(c + (0 * n), c0);
       _mm256_store_ps(c + (0 * n) + SIMD, c1);
       _mm256_store_ps(c + (1 * n), c2);
       _mm256_store_ps(c + (1 * n + SIMD), c3);
       _mm256_store_ps(c + (2 * n), c4);
       _mm256_store_ps(c + (2 * n + SIMD), c5);
       _mm256_store_ps(c + (3 * n), c6);
       _mm256_store_ps(c + (3 * n + SIMD), c7);
       _mm256_store_ps(c + (4 * n), c8);
       _mm256_store_ps(c + (4 * n + SIMD), c9);
       _mm256_store_ps(c + (5 * n), c10);
       _mm256_store_ps(c + (5 * n + SIMD), c11);
}
