// this file implements simple direct convolution with no fusion


#include<stdint.h>
#include "kernel.h"



#define op_dim(IN_dim, stride, K_dim, OUT_dim){\
  OUT_dim = (IN_dim - K_dim)/stride + 1;\
}
//Assume padding to maintain the same input
template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t stride, uint32_t H_f, uint32_t W_f>
void direct_convolution(
  uint32_t C_i,
  uint32_t C_o,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
  float * I,
  float * F,
  float * O
){
  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o_full = 0;
  op_dim(W_i, stride,W_f,W_o_full);
  uint32_t W_o = (W_o_full/_W_ob)*_W_ob;
  uint32_t W_last = W_o_full % _W_ob;
  // printf("W_of_full: %d W_o : %d W_last: %d\n ", W_o_full, W_o, W_last);

  #if PARALLEL==1
  #pragma omp parallel for
  #endif
  for(uint32_t j = 0; j < C_o ; j += _C_ob){

    uint32_t block_offset = (j/_C_ob)*H_o*W_o_full*_C_ob;
    uint32_t filter_o_c_block = (j/_C_ob)*(C_i/_C_ib)*H_f*W_f*_C_ib*_C_ob;
    float * O_buffer = O + block_offset;
    // These are all 0
    uint32_t input_block_offset = (0/_C_ib)*H_i*W_i*_C_ib;
    uint32_t filter_i_c_block = (0/_C_ib)*H_f*W_f*_C_ib*_C_ob + filter_o_c_block;

    float *filter_block_ptr = F + filter_i_c_block;

    for(uint32_t l = 0; l < H_o; l++){

        uint32_t col_offset = l*W_o_full*_C_ob;
        uint32_t input_col_offset = (l * stride)*W_i*_C_ob + input_block_offset;

        uint32_t input_row_offset = 0;
        float * I_ptr = I + input_col_offset;

        float * O_ptr = O_buffer + col_offset;
        for(uint32_t k = 0; k < W_o; k += _W_ob){



          conv_kernel_start<_W_ob, _C_ob, _C_ib, stride*_C_ob, H_f, W_f>(W_i*_C_ib, I_ptr, filter_block_ptr, O_ptr);

          I_ptr += stride*_W_ob*_C_ob;
          O_ptr += _W_ob*_C_ob;
      }

      conv_kernel_start_end<_W_ob, _C_ob, _C_ib, stride * _C_ob, H_f, W_f>(W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
    }

    for(uint32_t i = _C_ib; i < C_i; i += _C_ib){
      // printf("second ip block \n");

      uint32_t input_block_offset = (i/_C_ib) * H_i * W_i * _C_ib;
      uint32_t filter_i_c_block = (i/_C_ib) * H_f * W_f * _C_ib * _C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++)
      {

          uint32_t col_offset = l * W_o_full * _C_ob;
          uint32_t input_col_offset = (l * stride) * W_i * _C_ob + input_block_offset;

          float * I_ptr = I + input_col_offset;
          float * O_ptr = O_buffer + col_offset;

          for(uint32_t k = 0; k < W_o; k += _W_ob){

            // uint32_t input_row_offset = (k * stride)*_C_ob;
            // float * I_ptr = I + input_row_offset + input_col_offset;

            conv_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ob, H_f, W_f>(W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr);

            I_ptr += stride*_W_ob*_C_ob;
            O_ptr += _W_ob*_C_ob;

        }
        // printf("%.2f %.2f %.2f %.2f \n", I_ptr[0], I_ptr[16], I_ptr[32], filter_block_ptr[0]);

        conv_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ob, H_f, W_f>(W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
      }
    }
  }
  
}

template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t stride, uint32_t H_f, uint32_t W_f>
void direct_dw_convolution(
    uint32_t C_i,
    uint32_t C_o,
    // uint32_t H_f,
    // uint32_t W_f,
    uint32_t H_i,
    uint32_t W_i,
    // uint32_t stride,
    float *I,
    float *F,
    float *O)
{
  uint32_t H_o = 0;
  op_dim(H_i, stride, H_f, H_o);
  uint32_t W_o_full = 0;
  op_dim(W_i, stride, W_f, W_o_full);
  uint32_t W_o = (W_o_full / _W_ob) * _W_ob;
  uint32_t W_last = W_o_full % _W_ob;
  // printf("W_of_full: %d W_o : %d W_last: %d\n ", W_o_full, W_o, W_last);

#if PARALLEL == 1
#pragma omp parallel for
#endif
  for (uint32_t j = 0; j < C_o; j += _C_ob)
  {

    uint32_t block_offset = (j / _C_ob) * H_o * W_o_full * _C_ob;
    uint32_t filter_o_c_block = (j / _C_ob) * (C_i / _C_ib) * H_f * W_f * _C_ib * _C_ob;
    float *O_buffer = O + block_offset;

    //Indexing on the Inputs change
    uint32_t input_block_offset = (j / _C_ob) * H_i * W_i * _C_ob;
    uint32_t filter_i_c_block = /*(0 / _C_ib) * H_f * W_f * _C_ib * _C_ob + */ filter_o_c_block;

    float *filter_block_ptr = F + filter_i_c_block;

    for (uint32_t l = 0; l < H_o; l++)
    {

      uint32_t col_offset = l * W_o_full * _C_ob;
      uint32_t input_col_offset = (l * stride) * W_i * _C_ob + input_block_offset;

      uint32_t input_row_offset = 0;
      float *I_ptr = I + input_col_offset;

      float *O_ptr = O_buffer + col_offset;
      for (uint32_t k = 0; k < W_o; k += _W_ob)
      {

        dw_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ob, H_f, W_f>(W_i * _C_ob, I_ptr, filter_block_ptr, O_ptr);

        I_ptr += stride * _W_ob * _C_ob;
        O_ptr += _W_ob * _C_ob;
      }
      // printf("%.2f %.2f %.2f %.2f \n", I_ptr[0], I_ptr[16], I_ptr[32], filter_block_ptr[0]);
      dw_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ob, H_f, W_f>(W_i * _C_ob, I_ptr, filter_block_ptr, O_ptr, W_last);
    }

    //This will not exist
    for (uint32_t i = _C_ib; i < C_i; i += _C_ib)
    {
      printf("second ip block \n");

      uint32_t input_block_offset = (i / _C_ib) * H_i * W_i * _C_ib;
      uint32_t filter_i_c_block = (i / _C_ib) * H_f * W_f * _C_ib * _C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for (uint32_t l = 0; l < H_o; l++)
      {

        uint32_t col_offset = l * W_o_full * _C_ob;
        uint32_t input_col_offset = (l * stride) * W_i * _C_ob + input_block_offset;

        float *I_ptr = I + input_col_offset;
        float *O_ptr = O_buffer + col_offset;

        for (uint32_t k = 0; k < W_o; k += _W_ob)
        {

          // uint32_t input_row_offset = (k * stride)*_C_ob;
          // float * I_ptr = I + input_row_offset + input_col_offset;

          dw_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ob, H_f, W_f>(W_i * _C_ob, I_ptr, filter_block_ptr, O_ptr);

          I_ptr += stride * _W_ob * _C_ob;
          O_ptr += _W_ob * _C_ob;
        }
        dw_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ob, H_f, W_f>(W_i * _C_ob, I_ptr, filter_block_ptr, O_ptr, W_last);
      }
    }
  }
}

template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t stride, uint32_t H_f, uint32_t W_f>
void direct_group_convolution(
    uint32_t C_i,
    uint32_t C_o,
    // uint32_t H_f,
    // uint32_t W_f,
    uint32_t H_i,
    uint32_t W_i,
    // uint32_t stride,
    float *I,
    float *F,
    float *O)
{
  uint32_t H_o = 0;
  op_dim(H_i, stride, H_f, H_o);
  uint32_t W_o_full = 0;
  op_dim(W_i, stride, W_f, W_o_full);
  uint32_t W_o = (W_o_full / _W_ob) * _W_ob;
  uint32_t W_last = W_o_full % _W_ob;
  // printf("W_of_full: %d W_o : %d W_last: %d\n ", W_o_full, W_o, W_last);

#if PARALLEL == 1
#pragma omp parallel for
#endif
  for (uint32_t j = 0; j < C_o; j += _C_ob)
  {

    uint32_t block_offset = (j / _C_ob) * H_o * W_o_full * _C_ob;
    uint32_t filter_o_c_block = (j / _C_ob) * (C_i / _C_ib) * H_f * W_f * _C_ib * _C_ob;
    float *O_buffer = O + block_offset;
    // These are all 0
    uint32_t input_block_offset = (j / _C_ib) * H_i * W_i * _C_ib;
    uint32_t filter_i_c_block = (0 / _C_ib) * H_f * W_f * _C_ib * _C_ob + filter_o_c_block;

    float *filter_block_ptr = F + filter_i_c_block;

    for (uint32_t l = 0; l < H_o; l++)
    {

      uint32_t col_offset = l * W_o_full * _C_ob;
      uint32_t input_col_offset = (l * stride) * W_i * _C_ob + input_block_offset;

      uint32_t input_row_offset = 0;
      float *I_ptr = I + input_col_offset;

      float *O_ptr = O_buffer + col_offset;
      for (uint32_t k = 0; k < W_o; k += _W_ob)
      {

        conv_kernel_start<_W_ob, _C_ob, _C_ib, stride * _C_ob, H_f, W_f>(W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr);

        I_ptr += stride * _W_ob * _C_ob;
        O_ptr += _W_ob * _C_ob;
      }

      conv_kernel_start_end<_W_ob, _C_ob, _C_ib, stride * _C_ob, H_f, W_f>(W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
    }
    //Should get optimized out
    for (uint32_t i = _C_ib; i < C_i; i += _C_ib)
    {
      // printf("second ip block \n");

      uint32_t input_block_offset = (i / _C_ib) * H_i * W_i * _C_ib;
      uint32_t filter_i_c_block = (i / _C_ib) * H_f * W_f * _C_ib * _C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for (uint32_t l = 0; l < H_o; l++)
      {

        uint32_t col_offset = l * W_o_full * _C_ob;
        uint32_t input_col_offset = (l * stride) * W_i * _C_ob + input_block_offset;

        float *I_ptr = I + input_col_offset;
        float *O_ptr = O_buffer + col_offset;

        for (uint32_t k = 0; k < W_o; k += _W_ob)
        {

          // uint32_t input_row_offset = (k * stride)*_C_ob;
          // float * I_ptr = I + input_row_offset + input_col_offset;

          conv_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ob, H_f, W_f>(W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr);

          I_ptr += stride * _W_ob * _C_ob;
          O_ptr += _W_ob * _C_ob;
        }
        // printf("%.2f %.2f %.2f %.2f \n", I_ptr[0], I_ptr[16], I_ptr[32], filter_block_ptr[0]);

        conv_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ob, H_f, W_f>(W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
      }
    }
  }
}
