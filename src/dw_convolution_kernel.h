// Header File For different Versions of Fusing Pooling with a Convolution
#define POOL_UNROLL 8

#define POOL_KERNEL 3
#define POOL_STRIDE 2


#include "intrinsics.h"


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
