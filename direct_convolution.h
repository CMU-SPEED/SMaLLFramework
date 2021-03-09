// this file implements simple direct convolution with no fusion


#include<stdint.h>
#include "kernel.h"



#define op_dim(IN_dim, stride, K_dim, OUT_dim){\
  OUT_dim = (IN_dim - K_dim )/stride + 1;\
}


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
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);
  #pragma omp parallel for
  for(uint32_t j = 0; j < C_o ; j += C_ob){

    uint32_t block_offset = (j/C_ob)*H_o*W_o*C_ob;
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    // float * O_buffer = O + block_offset;
    // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float *filter_block_ptr = F + filter_i_c_block;

    for(uint32_t l = 0; l < H_o; l++){

        uint32_t col_offset = l*W_o*C_ob;
        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

        for(uint32_t k = 0; k < W_o; k += W_ob){

          uint32_t input_row_offset = (k * stride)*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;


          conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O + col_offset + k *C_ob);


      }
    }

    for(uint32_t i = C_ib; i < C_i; i += C_ib){

      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob + block_offset;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;

            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O + col_offset + k *C_ob);


        }
      }
    }
  }

}

template <uint32_t stride, uint32_t H_f, uint32_t W_f>
inline void direct_convolution_pooling_aware(
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
  // printf("%d\n", H_o);

  H_o -= (H_o%2==0);
  // printf("%d\n", H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);
  #pragma omp parallel for
  for(uint32_t j = 0; j < C_o ; j += C_ob){

    uint32_t block_offset = (j/C_ob)*H_o*W_o*C_ob;
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    // float * O_buffer = O + block_offset;
    // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float *filter_block_ptr = F + filter_i_c_block;

    for(uint32_t l = 0; l < H_o; l++){

        uint32_t col_offset = l*W_o*C_ob;
        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

        for(uint32_t k = 0; k < W_o; k += W_ob){

          uint32_t input_row_offset = (k * stride)*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;


          conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O + col_offset + k *C_ob);


      }
    }

    for(uint32_t i = C_ib; i < C_i; i += C_ib){

      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob + block_offset;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;

            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O + col_offset + k *C_ob);


        }
      }
    }
  }

}

// template <uint32_t stride, uint32_t H_f, uint32_t W_f>
// inline void direct_convolution_norm(
//   uint32_t C_i,
//   uint32_t C_o,
//   // uint32_t H_f,
//   // uint32_t W_f,
//   uint32_t H_i,
//   uint32_t W_i,
//   // uint32_t stride,
//   float * I,
//   float * F,
//   float * norm_weights,
//   float * O
// ){
//
//
//   uint32_t H_o = 0;
//   op_dim(H_i, stride,H_f,H_o);
//   uint32_t W_o = 0;
//   op_dim(W_i, stride,W_f,W_o);
//   #pragma omp parallel for
//   for(uint32_t j = 0; j < C_o; j += C_ob){
//
//     uint32_t block_offset = (j/C_ob)*H_o*W_o*C_ob;
//     uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
//
//     // These are all 0
//     uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
//     uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//
//     float *filter_block_ptr = F + filter_i_c_block;
//
//     for(uint32_t l = 0; l < H_o; l++){
//
//         uint32_t col_offset = l*W_o*C_ob + block_offset;
//         uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//
//         for(uint32_t k = 0; k < W_o; k += W_ob){
//
//           uint32_t input_row_offset = (k * stride)*C_ob;
//           float * I_ptr = I + input_row_offset + input_col_offset;
//
//
//           conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O + col_offset + k *C_ob);
//
//
//       }
//     }
//
//     for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib){
//
//       uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
//       uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//       float *filter_block_ptr = F + filter_i_c_block;
//
//       for(uint32_t l = 0; l < H_o; l++){
//
//           uint32_t col_offset = l*W_o*C_ob + block_offset;
//           uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//
//           for(uint32_t k = 0; k < W_o; k += W_ob){
//
//             uint32_t input_row_offset = (k * stride)*C_ob;
//             float * I_ptr = I + input_row_offset + input_col_offset;
//
//             conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O + col_offset + k *C_ob);
//
//
//         }
//       }
//     }
//
//     //last channel
//     {
//         uint32_t input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
//         uint32_t filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//         float *filter_block_ptr = F + filter_i_c_block;
//         uint32_t norm_block_offset = 2*(j)*(C_ob);
//         for(uint32_t l = 0; l < H_o; l++){
//
//             uint32_t col_offset = l*W_o*C_ob + block_offset;
//             uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//
//             for(uint32_t k = 0; k < W_o; k += W_ob){
//
//               uint32_t input_row_offset = (k * stride)*C_ob;
//               float * I_ptr = I + input_row_offset + input_col_offset;
//
//               conv_microkernel_last<stride*C_ob, H_f, W_f>(W_i*C_ib,
//                                                            I_ptr, filter_block_ptr,
//                                                            norm_weights + norm_block_offset,
//                                                            O + col_offset + k *C_ob);
//
//
//           }
//         }
//       }
//   }
//
// }


//fused pooling
// 3 x 2 x 16 register tile (still contiguous)
// Only works for 3x3 or 2x2
// #define POOL_KERNEL 3
// #define POOL_STRIDE 2






// pool horizontally and reuse the partially computed output for the height of the pooling layer
// Affected by how wide the output is?
// Need to buffer 2 rows of the pooling output?
//ASSUMPTION: Assumes 3x3 pooling with stride 2
