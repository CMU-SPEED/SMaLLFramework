// this file implements simple direct convolution with no fusion


#include<stdint.h>
#include "kernel.h"



#define op_dim(IN_dim, stride, K_dim, OUT_dim){\
  OUT_dim = (IN_dim - K_dim)/stride + 1;\
}

//Assume padding to maintain the same input

template <uint32_t stride, uint32_t H_f, uint32_t W_f>
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
  uint32_t W_o = (W_o_full/W_ob)*W_ob;
  uint32_t W_last = W_o_full % W_ob;
  // printf("W_of_full: %d W_o : %d W_last: %d\n ", W_o_full, W_o, W_last);

  #if PARALLEL==1
  #pragma omp parallel for
  #endif
  for(uint32_t j = 0; j < C_o ; j += C_ob){

    uint32_t block_offset = (j/C_ob)*H_o*W_o_full*C_ob;
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    float * O_buffer = O + block_offset;
    // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float *filter_block_ptr = F + filter_i_c_block;

    for(uint32_t l = 0; l < H_o; l++){

        uint32_t col_offset = l*W_o_full*C_ob;
        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

        uint32_t input_row_offset = 0;
        float * I_ptr = I + input_col_offset;

        float * O_ptr = O_buffer + col_offset;
        for(uint32_t k = 0; k < W_o; k += W_ob){



          conv_kernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_ptr);

          I_ptr += stride*W_ob*C_ob;
          O_ptr += W_ob*C_ob;
      }

      conv_kernel_start_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);

    }

    for(uint32_t i = C_ib; i < C_i; i += C_ib){

      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o_full*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          float * I_ptr = I + input_col_offset;
          float * O_ptr = O_buffer + col_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            // uint32_t input_row_offset = (k * stride)*C_ob;
            // float * I_ptr = I + input_row_offset + input_col_offset;

            conv_kernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_ptr);

            I_ptr += stride*W_ob*C_ob;
            O_ptr += W_ob*C_ob;

        }
        conv_kernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);

      }
    }
  }

}


