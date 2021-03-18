// Header File For different Versions of Fusing Pooling with a Convolution
//average pooling

#define POOL_UNROLL 8

#define POOL_KERNEL 3
#define POOL_STRIDE 2

float weight = 1.0/9.0;
#include "intrinsics.h"

// kernels

inline void pool_first_row_start(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);



  FMA_START();

  // store to output of pooling layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);



}
// load partial output from previous tile
// compute partial update for next tile

inline void pool_first_row(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);


  // horizontal pooling
  FMA(O);

  // store to output of pooling layer
  STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11);



}


inline void pool_first_row_end(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);


  // horizontal pooling
  FMA_END(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



}



inline void pool_accum_start(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);


  // horizontal pooling
  // load partial updates from previous rows (only ones being finalized)
  ACCUM_FMA_START(O);

  // store to output of pooling layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);



}

inline void pool_accum(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);


  // horizontal pooling previous tile
  ACCUM_FMA(O);



  // store to output of pooling layer
  STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11);



}

inline void pool_accum_end(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);

  // horizontal pooling
  ACCUM_FMA_END(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);

}

// accumulate to the previous pool output row
//TODO: do these stores non temporally
// write to new new output row

inline void pool_start(
                            uint32_t pool_col_stride,

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_12_C(O_buffer);
  int updates = 0;

  FMA_START();


  // store partial updates to next row
  STORE_6_C(O_next_row, c2, c3, c6, c7, c10, c11);


  ACCUM_AVG_START(O);


  // store to output of pooling layer
  STORE_4_C(O, c2, c3, c6, c7);




}

inline void pool(
                            uint32_t pool_col_stride,

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_12_C(O_buffer);

  //Local Updates
  FMA(O_next_row);

  //Store Partial Outputs to next row
  STORE_8_C(O_next_row, c0, c1, c2, c3, c6, c7, c10, c11);

  // Accumulate with updates to previous row
  ACCUM_AVG(O);

  // store to output of pooling layer
  STORE_8_C(O, c0, c1, c2, c3, c6, c7,c10,c11);



}

inline void pool_end(
                            uint32_t pool_col_stride,

                            float * O_buffer,
                            float * O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_12_C(O_buffer);

  //Local Updates
  FMA_END(O_next_row);

  // store partial updates to next row
  STORE_6_C(O_next_row, c0, c1, c2, c3, c6, c7);

  // accumulate with previous row
  // printf("Accumm\n");
  ACCUM_AVG_END(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



}





// fused kernels
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


  FMA_START();

  // store to output of pooling layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);



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

  // horizontal pooling
  FMA(O);

  // store to output of pooling layer
  STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11);



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

  // horizontal pooling
  FMA_END(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



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

  // horizontal pooling
  // load partial updates from previous rows (only ones being finalized)
  ACCUM_FMA_START(O);

  // store to output of pooling layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);



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

  // horizontal pooling previous tile
  ACCUM_FMA(O);



  // store to output of pooling layer
  STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11);



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

  // horizontal pooling
  ACCUM_FMA_END(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



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

  // horizontal pooling
;

  FMA_START();


  // store partial updates to next row
  STORE_6_C(O_next_row, c2, c3, c6, c7, c10, c11);


  ACCUM_START(O);


  // store to output of pooling layer
  STORE_4_C(O, c2, c3, c6, c7);




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
  FMA(O_next_row);

  //Store Partial Outputs to next row
  STORE_8_C(O_next_row, c0, c1, c2, c3, c6, c7, c10, c11);

  // Accumulate with updates to previous row
  ACCUM(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2, c3, c6, c7);



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
  FMA_END(O_next_row);

  // store partial updates to next row
  STORE_6_C(O_next_row, c0, c1, c2, c3, c6, c7);

  // accumulate with previous row
  ACCUM_END(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



}
