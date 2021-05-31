// Header File For different Versions of Fusing Pooling with a Convolution
#define POOL_UNROLL 8

#define POOL_KERNEL 3
#define POOL_STRIDE 2

#define GPOOL_W_ob 6

#include "intrinsics.h"


inline void global_pool(
                          float * I,
                          __m256 &o0, __m256 &o1,
                          __m256 &o2, __m256 &o3,
                          __m256 &o4, __m256 &o5
                       )
{
    __m256 c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11;
  LOAD_12_C(I);
  // print256_float(c0);
  // printf("\t");print256_float(o0);
  ADD_12_C();
  // printf("\t");print256_float(o0);

}


inline void global_pool_reduce(
                              uint32_t w_final,
                              float *I,
                              __m256 o0, __m256 o1,
                              __m256 o2, __m256 o3,
                              __m256 o4, __m256 o5,
                              float *O
                                )
{
      __m256 c0,c1,c2,c3,c4,c5,c6,c7,c8,c9;
    //
    // printf("pool reduce \t\t");print256_float(o0);
    // printf("pool reduce \t\t");print256_float(o2);
    // printf("pool reduce \t\t");print256_float(o4);
    REDUCE_2_C();
    for(uint32_t i = 0; i < w_final; i++)
    {
      LOAD_2_C(I, o2, o3);
      REDUCE_2_C();
    }
    // printf("\t\t\t\t");print256_float(o0);


    STORE_2_C(O, o0, o1);

}

// kernels
#if POOLING == 1
inline void pool_first_row_start(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);



  MAX_START();

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

  LOAD_14_C(O_buffer);


  // horizontal pooling
  MAX_3();

  // store to output of pooling layer
  STORE_6_C(O,c2, c3, c6, c7, c10, c11);


}


inline void pool_first_row_end(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);


  // horizontal pooling
  MAX_2();

  // store to output of pooling layer
  STORE_4_C(O,c2,c3, c6, c7);



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
  ACCUM_MAX_START(O);

  // store to output of pooling layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);


}

inline void pool_accum(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_14_C(O_buffer);


  // horizontal pooling previous tile
  MAX_3();

  ACCUM_3(O);

  // store to output of pooling layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);


}

inline void pool_accum_end(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);

  // horizontal pooling
  MAX_2();
  ACCUM_2(O);

  // store to output of pooling layer
  STORE_4_C(O, c2,c3, c6, c7);

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
  LOAD_14_C(O_buffer);
  int updates = 0;

  MAX_3();
  // store partial updates to next row
  STORE_6_C(O_next_row, c2, c3, c6, c7, c10, c11);


  ACCUM_3(O);

  // store to output of pooling layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);




}

inline void pool(
                            uint32_t pool_col_stride,

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_14_C(O_buffer);

  MAX_3();
  // store partial updates to next row
  STORE_6_C(O_next_row, c2, c3, c6, c7, c10, c11);


  ACCUM_3(O);

  // store to output of pooling layer
  STORE_6_C(O,  c2, c3, c6, c7, c10, c11);


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
  int updates = 0;

  MAX_2();
  // store partial updates to next row
  STORE_4_C(O_next_row, c2, c3, c6, c7);


  ACCUM_2(O);

  // store to output of pooling layer
  STORE_4_C(O,  c2, c3, c6, c7);



}

//avg pooling kernels
#elif POOLING == 2
inline void pool_first_row(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_14_C(O_buffer);


  // horizontal pooling
  AVG_3();

  // store to output of pooling layer
  STORE_6_C(O,c2, c3, c6, c7, c10, c11);


}


inline void pool_first_row_end(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);


  // horizontal pooling
  AVG_2();

  // store to output of pooling layer
  STORE_4_C(O,c2,c3, c6, c7);



}

inline void pool_accum(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_14_C(O_buffer);


  // horizontal pooling previous tile
  AVG_3();

  ACCUM_AVG_3(O);

  // store to output of pooling layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);


}


inline void pool_accum_end(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);

  // horizontal pooling
  AVG_2();
  ACCUM_AVG_2(O);

  // store to output of pooling layer
  STORE_4_C(O, c2,c3, c6, c7);

}

// accumulate to the previous pool output row
//TODO: do these stores non temporally
// write to new new output row


inline void pool(
                            uint32_t pool_col_stride,

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_14_C(O_buffer);

  AVG_3();
  // store partial updates to next row
  STORE_6_C(O_next_row, c2, c3, c6, c7, c10, c11);


  ACCUM_AVG_3(O);

  // store to output of pooling layer
  STORE_6_C(O,  c2, c3, c6, c7, c10, c11);


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
  int updates = 0;

  AVG_2();
  // store partial updates to next row
  STORE_4_C(O_next_row, c2, c3, c6, c7);


  ACCUM_AVG_2(O);

  // store to output of pooling layer
  STORE_4_C(O,  c2, c3, c6, c7);



}
//end avg pooling kernels

#endif




template <uint32_t step>
inline void pool_3_rows(
                          uint32_t col_stride,
                          uint32_t W_o,
                          float * O_buffer,
                          float *   O
                        )
{
  __m256 c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13, c14, c15;
    float * conv_O = O_buffer;
    float * pool_O = O;
    for(uint32_t i = 0; i < W_o - W_ob ; i+=W_ob){
      LOAD_6_C(conv_O, step);
        MAX_3_row(conv_O + 1*C_ob, step);
        MAX_3_row(conv_O + 2*C_ob, step);
        float * conv_row_O = conv_O + col_stride;
        MAX_3_row(conv_row_O + 0*C_ob, step);
        MAX_3_row(conv_row_O + 1*C_ob, step);
        MAX_3_row(conv_row_O + 2*C_ob, step);
        conv_row_O += col_stride;
        MAX_3_row(conv_row_O + 0*C_ob, step);
        MAX_3_row(conv_row_O + 1*C_ob, step);
        MAX_3_row(conv_row_O + 2*C_ob, step);

      STORE_6_C(pool_O, c0, c1, c2, c3, c4, c5);
      pool_O += 3*C_ob;
      conv_O += 3*step;

    }
    // last tile of conv output
    {
      LOAD_4_C(conv_O, step);
        MAX_3_row(conv_O + 1*C_ob, step);
        MAX_3_row(conv_O + 2*C_ob, step);
        float * conv_row_O = conv_O + col_stride;
        MAX_3_row(conv_row_O + 0*C_ob, step);
        MAX_3_row(conv_row_O + 1*C_ob, step);
        MAX_3_row(conv_row_O + 2*C_ob, step);
        conv_row_O += col_stride;
        MAX_3_row(conv_row_O + 0*C_ob, step);
        MAX_3_row(conv_row_O + 1*C_ob, step);
        MAX_3_row(conv_row_O + 2*C_ob, step);

      STORE_4_C(pool_O, c0, c1, c2, c3);


    }


}


template <uint32_t step>
inline void pool_3_rows_strided(
                          float * conv_row_0,
                          float * conv_row_1,
                          float * conv_row_2,
                          uint32_t W_o,
                          float *   O
                        )
{
  __m256 c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13, c14, c15;
    float * pool_O = O;
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

      STORE_6_C(pool_O, c0, c1, c2, c3, c4, c5);
      pool_O += 3*C_ob;
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

      STORE_4_C(pool_O, c0, c1, c2, c3);


    }


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


  MAX_START();

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
  MAX(O);

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
  MAX_END(O);

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
  ACCUM_MAX_START(O);

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
  ACCUM_MAX(O);



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
  ACCUM_MAX_END(O);

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

  MAX_START();


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
  MAX(O_next_row);

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
  MAX_END(O_next_row);

  // store partial updates to next row
  STORE_6_C(O_next_row, c0, c1, c2, c3, c6, c7);

  // accumulate with previous row
  ACCUM_END(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



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


  STORE_12_C(O_buffer);
  MAX_START();



  // store to output of pooling layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);



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

  STORE_12_C(O_buffer);
  // horizontal pooling

  // load partial update from previous
  MAX(O);

  // store to output of pooling layer
  STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11);



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

  STORE_10_C(O_buffer);
  // horizontal pooling

  MAX_END(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



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

  STORE_12_C(O_buffer);
  // horizontal pooling
  // load partial updates from previous rows (only ones being finalized)
  ACCUM_MAX_START(O);

  // store to output of pooling layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);



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

  STORE_12_C(O_buffer);
  // horizontal pooling previous tile
  ACCUM_MAX(O);



  // store to output of pooling layer
  STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11);



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

  STORE_10_C(O_buffer);
  // horizontal pooling
  ACCUM_MAX_END(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



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

  STORE_12_C(O_buffer);
  // horizontal pooling
  // load partial updates from previous rows


  MAX_START();


  // store partial updates to next row

  STORE_6_C(O_next_row, c2, c3, c6, c7, c10, c11);

  ACCUM_START(O);


  // store to output of pooling layer
  STORE_4_C(O, c2, c3, c6, c7);




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

  STORE_12_C(O_buffer);

  //Updates within this row
  MAX(O_next_row);

  //Store Partial Outputs to next row
  STORE_8_C(O_next_row, c0, c1, c2, c3, c6, c7, c10, c11);

  //Accumulate with outputs to previos output row
  ACCUM(O);

  // store to output of pooling layer
  STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11);



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

  STORE_10_C(O_buffer);
  // Local Updates within this row
  MAX_END(O_next_row);

  // store partial updates to next row
  STORE_6_C(O_next_row, c0, c1, c2, c3, c6, c7);

  //accumulate with updates to previous rows
  ACCUM_END(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);

}


// end fully computed kernels
