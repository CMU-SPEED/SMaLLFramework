
#include <immintrin.h>

#define SIMD 8
# define rank_k 16

// AMD Zen2 parameters
#define W_ob 6
#define C_ob 16
#define C_ib C_ob

void print256_float(__m256 var)
{
    float val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n",
           val[0], val[1], val[2], val[3], val[4], val[5],
           val[6], val[7]);
}


#include "scalar_intrinsics.h"




template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_kernel(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_TILE_C(O);

  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;
      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

        int p_cur = ii ;

        FMA_TILE_C(step, a, b, p_cur);

      }
    }
  }


STORE_TILE_C(O);

}

template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_kernel_start(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  ZERO_TILE_C();

  int updates = 0;
  // uint32_t step = C_ob;//stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;
      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

        int p_cur = ii;

        FMA_TILE_C(step, a, b, p_cur);

      }
    }
  }


STORE_TILE_C(O);

}


// cleanup convolution kernels
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_kernel_start_end(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O,
                            uint32_t W_last
                           ){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  ZERO_TILE_C();

  int updates = 0;
  // uint32_t step = C_ob;//stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;
      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

        int p_cur = ii;

        FMA_END_C(step, a, b, p_cur, W_last);

      }
    }
  }

  STORE_END_C(O, W_last);


}

template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_kernel_end(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O,
                            uint32_t W_last
                          ){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_LAST_C(O, W_last);

  int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;
      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

        int p_cur = ii ;

        FMA_END_C(step, a, b, p_cur, W_last);
      }
    }
  }

  STORE_END_C(O, W_last);


}


//padding kernels
template <uint32_t step, uint32_t padding, H_f, uint32_t W_f>
inline void padded_conv_kernel_row(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O,
                            uint32_t W_o_full
                           ){

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_ptr = O;
  ZERO_TILE_C();

  int updates = 0;
  // uint32_t step = C_ob;//stride*C_ob;
  // int count = 0;
  float * b = padding*W_f*C_ib*C_ob;
  float * a = padding*input_col_stride;
  for(uint32_t n = padding; n < H_f ; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = padding; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;
      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

        int p_cur = ii;

        FMA_END_C(step, a, b, p_cur, W_last);

      }
    }
  }

  STORE_END_C(O_ptr, W_last);


  //most of the row

  for(uint32_t k = padding; k < W_o_full - padding; k++){

  }

}
// No cache hints
