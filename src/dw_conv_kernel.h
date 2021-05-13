// Header File For different Versions of Fusing dw_conw with a Convolution
#define DW_UNROLL 8

#define DW_W_ob 5

#define DW_KERNEL 3
#define DW_STRIDE 2


#include "intrinsics.h"
// void print256_float(__m256 var)
// {
//     float val[8];
//     memcpy(val, &var, sizeof(val));
//     printf("Numerical: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n",
//            val[0], val[1], val[2], val[3], val[4], val[5],
//            val[6], val[7]);
// }


// kernels

#define DW_DUFF(F,I, w_final, O)\
  float * I_ptr = I;\
  float * O_ptr = O;\
  /*First accumulation*/\
  I_ptr = I;\
  for(uint32_t kk = 0 ; kk < w_final; kk++){\
  /**/      \
  f0 = _mm256_load_ps(F);\
  f1 = _mm256_load_ps(F + SIMD);\
  I_ptr = I + kk*DW_STRIDE*C_ob;\
  i0 = _mm256_load_ps(I_ptr);\
  i1 = _mm256_load_ps(I_ptr + SIMD);\
  /**/\
  c0 = _mm256_mul_ps(f0, i0);\
  c1 = _mm256_mul_ps(f1, i1);\
  /**/\
  float * F_ptr = F;\
  for(uint32_t m = 1; m < DW_KERNEL; m++)\
    {\
    I_ptr += C_ob;\
    F_ptr += C_ob;\
    f0 = _mm256_load_ps(F_ptr);\
    f1 = _mm256_load_ps(F_ptr + SIMD);\
    /**/\
    i0 = _mm256_load_ps(I_ptr);\
    i1 = _mm256_load_ps(I_ptr + SIMD);\
    /**/\
    c0 = _mm256_fmadd_ps(f0, i0, c0);\
    c1 = _mm256_fmadd_ps(f1, i1, c1);\
    }\
  _mm256_store_ps(O_ptr, c0);\
  _mm256_store_ps(O_ptr + SIMD, c1);\
  O_ptr +=  C_ob;\
  }\

#define DW_DUFF_ACCUM(F,I, w_final, O)\
float * I_ptr = I;\
float * O_ptr = O;\
/*First accumulation*/\
for(uint32_t kk = 0 ; kk < w_final; kk++){\
I_ptr = I + kk*DW_STRIDE*C_ob;\
/**/      \
c0 = _mm256_load_ps(O_ptr);\
c1 = _mm256_load_ps(O_ptr + SIMD);\
float * F_ptr = F;\
for(uint32_t m = 0; m < DW_KERNEL; m++)\
  {\
  f0 = _mm256_load_ps(F_ptr);\
  f1 = _mm256_load_ps(F_ptr + SIMD);\
  /**/\
  i0 = _mm256_load_ps(I_ptr);\
  i1 = _mm256_load_ps(I_ptr + SIMD);\
  /**/\
  c0 = _mm256_fmadd_ps(f0, i0, c0);\
  c1 = _mm256_fmadd_ps(f1, i1, c1);\
  I_ptr += C_ob;\
  F_ptr += C_ob;\
  }\
_mm256_store_ps(O_ptr, c0);\
_mm256_store_ps(O_ptr + SIMD, c1);\
O_ptr +=  C_ob;\
}\

inline void dw_first_row(
                            float * I,
                            float * F,
                            float *O
                          )
{

  __m256 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, f0, f1, f2, f3, i0, i1;




  DW_START(F, I);

  for(uint32_t m = 1; m < DW_KERNEL; m++)
  {
      I += C_ob;
      F += C_ob;
      DW_FMA(F, I);
  }

  // store to output of dw_conw layer
  STORE_10_DW(O, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);



}
// load partial output from previous tile
// compute partial update for next tile

inline void dw_first_row_end(
                            float * I,
                            float * F,
                            uint32_t w_final,
                            float *O
                          )
{

  __m256 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, f0, f1, f2, f3, i0, i1;


  // horizontal dw_conw
  DW_DUFF(F, I, w_final, O);



}



inline void dw_accum(
                            float * I,
                            float * F,
                            float *O
                          )
{

  __m256 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, f0, f1, f2, f3, i0, i1;


  LOAD_10_C(O);
  // horizontal dw_conw
  // load partial updates from previous rows (only ones being finalized)
  for(uint32_t m = 0; m < DW_KERNEL; m++)
  {
    DW_FMA(F, I);
    F += C_ob;
    I += C_ob;
  }

  // store to output of dw_conw layer
  STORE_10_DW(O, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);


}




inline void dw_accum_end(
                            float * I,
                            float * F,
                            uint32_t w_final,
                            float *O
                          )
{

  __m256 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, f0, f1, f2, f3, i0, i1;


  // horizontal dw_conw
  DW_DUFF_ACCUM(F, I, w_final, O);



}

// accumulate to the previous dw output row
//TODO: do these stores non temporally
// write to new new output row


inline void dw(
                            uint32_t dw_col_stride,
                            float * I,
                            float * F_0,
                            float * F_1,
                            float *O
                          )
{

  __m256 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, f0, f1, f2, f3, i0, i1;
  // printf("kern row 0 %f\t",*O);
  dw_accum(I, F_0, O);
  // printf("kern row 0 %f\n\n\n",*O);

  float * O_next_row = O + dw_col_stride;
  // printf("kern row 1 %f\t",*O_next_row);
  dw_first_row(I, F_1, O_next_row);
  // printf("kern row 1 %f\n\n\n",*O_next_row);

}


inline void dw_end(
                            uint32_t dw_col_stride,
                            float * I,
                            float * F_0,
                            float * F_1,
                            uint32_t w_final,
                            float *O
                          )
{  __m256 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, f0, f1, f2, f3, i0, i1;
  dw_accum_end(I, F_0,w_final, O);

  float * O_next_row = O + dw_col_stride;
  dw_first_row_end(I, F_1,w_final, O_next_row);

}







// template <uint32_t step>
// inline void dw_3_row_tile(
//                             uint32_t ip_row_stride,
//                             float * I,
//                             float * F,
//                             float *O
//                         )
// {
//   __m256 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, f0, f1, f2, f3, i0, i1;
//
//     LOAD_10_C(O);
//     // horizontal dw_conw
//     // load partial updates from previous rows (only ones being finalized)
//     for(uint32_t n = 1; m < DW_KERNEL; n++)
//     {
//       I_ptr = I + n*ip_row_stride;
//       for(uint32_t m = 0; m < DW_KERNEL; m++)
//       {
//         DW_FMA(F, I);
//         F += C_ob;
//         I_ptr += C_ob;
//       }
//
//     }
//
//     // store to output of dw_conw layer
//     STORE_10_DW(O, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
//
//
//
// }

template <uint32_t step>
inline void dw_3_rows(
                          float * conv_row_0,
                          float * conv_row_1,
                          float * conv_row_2,
                          uint32_t W_o,
                          float *   O
                        )
{
  __m256 c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13, c14, c15;
    float * dw_O = O;
    for(uint32_t i = 0; i < W_o - W_ob ; i+=W_ob){
      LOAD_6_C(conv_row_0, step);
        MAX_3_row(conv_row_0 + 1*C_ob, step);
        MAX_3_row(conv_row_0 + 2*C_ob, step);

        MAX_3_row(conv_row_1 + 0*C_ob, step);
        MAX_3_row(conv_row_1 + 1*C_ob, step);
        MAX_3_row(conv_row_1 + 2*C_ob, step);

        MAX_3_row(conv_row_2 + 0*C_ob, step);
        MAX_3_row(conv_row_2 + 1*C_ob, step);
        MAX_3_row(conv_row_2 + 2*C_ob, step);

      STORE_6_C(dw_O, c0, c1, c2, c3, c4, c5);
      dw_O += 3*C_ob;
      conv_row_0 += 3*step;
      conv_row_1 += 3*step;
      conv_row_2 += 3*step;

    }
    // last tile of conv output
    {
      LOAD_4_C(conv_row_0, step);
        MAX_3_row(conv_row_0 + 1*C_ob, step);
        MAX_3_row(conv_row_0 + 2*C_ob, step);

        MAX_3_row(conv_row_1 + 0*C_ob, step);
        MAX_3_row(conv_row_1 + 1*C_ob, step);
        MAX_3_row(conv_row_1 + 2*C_ob, step);

        MAX_3_row(conv_row_2 + 0*C_ob, step);
        MAX_3_row(conv_row_2 + 1*C_ob, step);
        MAX_3_row(conv_row_2 + 2*C_ob, step);

      STORE_4_C(dw_O, c0, c1, c2, c3);


    }


}
template <uint32_t step>
inline void dw_3_rows_strided(
                          float * conv_row_0,
                          float * conv_row_1,
                          float * conv_row_2,
                          uint32_t W_o,
                          float *   O
                        )
{
  __m256 c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13, c14, c15;
    float * dw_O = O;
    for(uint32_t i = 0; i < W_o - W_ob ; i+=W_ob){
      LOAD_6_C(conv_row_0, step);
        MAX_3_row(conv_row_0 + 1*C_ob, step);
        MAX_3_row(conv_row_0 + 2*C_ob, step);

        MAX_3_row(conv_row_1 + 0*C_ob, step);
        MAX_3_row(conv_row_1 + 1*C_ob, step);
        MAX_3_row(conv_row_1 + 2*C_ob, step);

        MAX_3_row(conv_row_2 + 0*C_ob, step);
        MAX_3_row(conv_row_2 + 1*C_ob, step);
        MAX_3_row(conv_row_2 + 2*C_ob, step);

      STORE_6_C(dw_O, c0, c1, c2, c3, c4, c5);
      dw_O += 3*C_ob;
      conv_row_0 += 3*step;
      conv_row_1 += 3*step;
      conv_row_2 += 3*step;

    }
    // last tile of conv output
    {
      LOAD_4_C(conv_row_0, step);
        MAX_3_row(conv_row_0 + 1*C_ob, step);
        MAX_3_row(conv_row_0 + 2*C_ob, step);

        MAX_3_row(conv_row_1 + 0*C_ob, step);
        MAX_3_row(conv_row_1 + 1*C_ob, step);
        MAX_3_row(conv_row_1 + 2*C_ob, step);

        MAX_3_row(conv_row_2 + 0*C_ob, step);
        MAX_3_row(conv_row_2 + 1*C_ob, step);
        MAX_3_row(conv_row_2 + 2*C_ob, step);

      STORE_4_C(dw_O, c0, c1, c2, c3);


    }


}




















// microkernel fused kernels
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_dw_first_row_start(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);
  int updates = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = n * input_col_stride;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){
        int p_cur = ii;
        FMA_12_C(step, a, b, p_cur);
      }
    }
  }


  MAX_START();

  // store to output of dw_conw layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);



}
// load partial output from previous tile
// compute partial update for next tile
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_dw_first_row(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);
  int updates = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = n * input_col_stride;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        int p_cur = ii;
        FMA_12_C(step, a, b, p_cur);
      }
    }
  }

  // horizontal dw_conw
  MAX(O);

  // store to output of dw_conw layer
  STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11);



}

template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_dw_first_row_end(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);
  int updates = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = n * input_col_stride;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        int p_cur = ii;
        FMA_10_C(step, a, b, p_cur);
      }
    }
  }

  // horizontal dw_conw
  MAX_END(O);

  // store to output of dw_conw layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



}


template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_dw_accum_start(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);
  int updates = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = n * input_col_stride;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        int p_cur = ii;
        FMA_12_C(step, a, b, p_cur);
      }
    }
  }

  // horizontal dw_conw
  // load partial updates from previous rows (only ones being finalized)
  ACCUM_MAX_START(O);

  // store to output of dw_conw layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);



}
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_dw_accum(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);
  int updates = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = n * input_col_stride;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        int p_cur = ii;
        FMA_12_C(step, a, b, p_cur);
      }
    }
  }

  // horizontal dw_conw previous tile
  ACCUM_MAX(O);



  // store to output of dw_conw layer
  STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11);



}
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_dw_accum_end(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);
  int updates = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = n * input_col_stride ;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){


        int p_cur = ii;
        FMA_10_C(step, a, b, p_cur);
      }
    }
  }

  // horizontal dw_conw
  ACCUM_MAX_END(O);

  // store to output of dw_conw layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



}

// accumulate to the previous dw output row
//TODO: do these stores non temporally
// write to new new output row
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_dw_start(
                            uint32_t input_col_stride,
                            uint32_t dw_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + dw_col_stride;
  LOAD_12_C(O_buffer);
  int updates = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = n * input_col_stride;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        int p_cur = ii;
        FMA_12_C(step, a, b, p_cur);
      }
    }
  }

  // horizontal dw_conw
;

  MAX_START();


  // store partial updates to next row
  STORE_6_C(O_next_row, c2, c3, c6, c7, c10, c11);


  ACCUM_START(O);


  // store to output of dw_conw layer
  STORE_4_C(O, c2, c3, c6, c7);




}
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_dw(
                            uint32_t input_col_stride,
                            uint32_t dw_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + dw_col_stride;
  LOAD_12_C(O_buffer);
  int updates = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = n * input_col_stride;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        int p_cur = ii;
        FMA_12_C(step, a, b, p_cur);
      }
    }
  }
  //Local Updates
  MAX(O_next_row);

  //Store Partial Outputs to next row
  STORE_8_C(O_next_row, c0, c1, c2, c3, c6, c7, c10, c11);

  // Accumulate with updates to previous row
  ACCUM(O);

  // store to output of dw_conw layer
  STORE_6_C(O, c0, c1, c2, c3, c6, c7);



}
template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void conv_microkernel_dw_end(
                            uint32_t input_col_stride,
                            uint32_t dw_col_stride,
                            float * I,
                            float * F,
                            float * O_buffer,
                            float * O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + dw_col_stride;
  LOAD_12_C(O_buffer);
  int updates = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*C_ib*C_ob;
    int input_stencil_h = n * input_col_stride;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
      int input_stencil_w = m*C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;

      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        int p_cur = ii;
        FMA_10_C(step, a, b, p_cur);
      }
    }
  }
  //Local Updates
  MAX_END(O_next_row);

  // store partial updates to next row
  STORE_6_C(O_next_row, c0, c1, c2, c3, c6, c7);

  // accumulate with previous row
  ACCUM_END(O);

  // store to output of dw_conw layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



}
