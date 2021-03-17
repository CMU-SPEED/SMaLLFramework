// Header File For different Versions of Fusing Pooling with a Convolution
//average pooling

#define POOL_UNROLL 8

#define POOL_KERNEL 3
#define POOL_STRIDE 2


// #include "intrinsics.h"

// kernels


inline void avg_pool_first_row_start(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);



  ADD_START();

  // store to output of pooling layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);



}
// load partial output from previous tile
// compute partial update for next tile

inline void avg_pool_first_row(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);


  // horizontal pooling
  ADD(O);

  // store to output of pooling layer
  STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11);



}


inline void avg_pool_first_row_end(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);


  // horizontal pooling
  ADD_END(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



}



inline void avg_pool_accum_start(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);


  // horizontal pooling
  // load partial updates from previous rows (only ones being finalized)
  ACCUM_ADD_START(O);

  // store to output of pooling layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);



}

inline void avg_pool_accum(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);


  // horizontal pooling previous tile
  ACCUM_ADD(O);



  // store to output of pooling layer
  STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11);



}

inline void avg_pool_accum_end(

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);

  // horizontal pooling
  ACCUM_ADD_END(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);

}

// accumulate to the previous pool output row
//TODO: do these stores non temporally
// write to new new output row

inline void avg_pool_start(
                            uint32_t pool_col_stride,

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_12_C(O_buffer);
  int updates = 0;

  ADD_START();


  // store partial updates to next row
  STORE_6_C(O_next_row, c2, c3, c6, c7, c10, c11);


  LAST_ADD_MUL_START(O);


  // store to output of pooling layer
  STORE_4_C(O, c2, c3, c6, c7);




}

inline void avg_pool(
                            uint32_t pool_col_stride,

                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_12_C(O_buffer);

  //Local Updates
  ADD(O_next_row);

  //Store Partial Outputs to next row
  STORE_8_C(O_next_row, c0, c1, c2, c3, c6, c7, c10, c11);

  // Accumulate with updates to previous row
  ACCUM(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2, c3, c6, c7);



}

inline void avg_pool_end(
                            uint32_t pool_col_stride,

                            float * O_buffer,
                            float * O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  float * O_next_row = O + pool_col_stride;
  LOAD_12_C(O_buffer);

  //Local Updates
  ADD_END(O_next_row);

  // store partial updates to next row
  STORE_6_C(O_next_row, c0, c1, c2, c3, c6, c7);

  // accumulate with previous row
  ACCUM(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



}

inline void avg_pool_last_start(
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);
  int updates = 0;

  ADD_START();

  LAST_ADD_MUL_START(O);


  // store to output of pooling layer
  STORE_6_C(O, c2, c3, c6, c7, c10, c11);




}

inline void avg_pool_last(
                            float * O_buffer,
                            float *O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);

  //Local Updates
  ADD(O);

  // Accumulate with updates to previous row
  LAST_ADD_MUL(O);

  // store to output of pooling layer
  STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11);



}

inline void avg_pool_last_end(
                            float * O_buffer,
                            float * O
                          )
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  LOAD_12_C(O_buffer);

  //Local Updates
  ADD_END(O);

  // store partial updates to next row
  // accumulate with previous row
  LAST_ADD_MUL(O);

  // store to output of pooling layer
  STORE_6_C(O, c0, c1, c2,c3, c6, c7);



}
