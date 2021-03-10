// this is a copy of the original header file modified to explain how things work
// I would read the 2 files side by side
#include<stdint.h>
#include "kernel.h"

void print256_float(__m256 var)
{
    /*

    Print the Contents of a vector register

    */
}

// Macro to get output dims from input config
#define op_dim(IN_dim, stride, K_dim, OUT_dim){\
  OUT_dim = (IN_dim - K_dim )/stride + 1;\
}

void direct_convolution_gemm(
  uint32_t C_i,
  uint32_t C_o,
  uint32_t H_f,
  uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  uint32_t stride,
  float * I,
  float * F,
  float * O
){

  // uint32_t C_o = out_dims[1];
  // uint32_t H_o = out_dims[2];
  // uint32_t W_o = out_dims[3];
  // // printf("C_o : %d, H_o: %d, W_o:%d\n", C_o, H_o, W_o);
  // uint32_t C_i = in_dims[1];
  // uint32_t H_i = in_dims[2];
  // uint32_t W_i = in_dims[3];
  //
  //
  // uint32_t H_f = filter_dims[2];
  // uint32_t W_f = filter_dims[3];

  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);
  // printf("%d->%d \t %d->%d\n\n\n\n",H_i, H_o, W_i, W_o);
  for(uint32_t j = 0; j < C_o; j += C_ob){

    uint32_t block_offset = (j/C_ob)*H_o*W_o*C_ob;
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;

    for(uint32_t i = 0; i < C_i; i += C_ib){

      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob + block_offset;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            // if(i!=0)
            //   load_C(C_ob,O+col_offset + k*C_ob);
            // else
            //   zero_tile();
            uint32_t updates = 0;
            for(uint32_t n = 0; n < H_f; n++){

              uint32_t filter_offset_h = n*W_f*C_ib*C_ob;
              uint32_t input_stencil_h = input_col_offset + n*W_i*C_ib + input_row_offset;

              for(uint32_t m = 0; m < W_f; m++){

                uint32_t filter_offset_w = m*C_ib*C_ob + filter_offset_h;
                uint32_t input_stencil_w = m*C_ib + input_stencil_h;

                for(uint32_t ii = 0 ; ii < C_ib; ii+= rank_k){
                  kernel_conv(W_ob,C_ob,rank_k,I+input_stencil_w, filter_block_ptr + filter_offset_w, O+col_offset + k*C_ob);
                   // print("Last index: %d",O+col_offset + k*C_ob + 5*C_ob + C_ob );
              }
            }
          }
          // store_C(C_ob,O+ block_offset + col_offset + k*C_ob);

        }
      }
    }
  }

  // O[(C_o-C_ob)*H_o*W_o*C_ob +  (H_o - 1)*W_ob*C_ob + (W_ob - 1)*C_ob + (C_ob - 1)] = 1234.567;
}



inline void direct_convolution(
  uint32_t C_i,
  uint32_t C_o,
  uint32_t H_f,
  uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  uint32_t stride,
  float * I,
  float * F,
  float * O
){


  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);

  for(uint32_t j = 0; j < C_o; j += C_ob){

    uint32_t block_offset = (j/C_ob)*H_o*W_o*C_ob;
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;

    // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
    float *filter_block_ptr = F + filter_i_c_block;

    for(uint32_t l = 0; l < H_o; l++){

        uint32_t col_offset = l*W_o*C_ob + block_offset;
        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

        for(uint32_t k = 0; k < W_o; k += W_ob){

          uint32_t input_row_offset = (k * stride)*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;

          #if(GEMM)
          conv_microkernel_start_gemm(W_i*C_ib, H_f, W_f, I_ptr, filter_block_ptr, O + col_offset + k *C_ob);
          #else
          conv_microkernel_start(W_i*C_ib, H_f, W_f, I_ptr, filter_block_ptr, O + col_offset + k *C_ob);
          #endif

      }
    }
    // printf("shata iteration\n");
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
            #if(GEMM)
            conv_microkernel_gemm(W_i*C_ib, H_f, W_f, I_ptr, filter_block_ptr, O + col_offset + k *C_ob);
            #else
            conv_microkernel(W_i*C_ib, H_f, W_f, I_ptr, filter_block_ptr, O + col_offset + k *C_ob);
            #endif

        }
      }
    }
  }

  // O[(C_o-C_ob)*H_o*W_o*C_ob +  (H_o - 1)*W_ob*C_ob + (W_ob - 1)*C_ob + (C_ob - 1)] = 1234.567;
}


// minimum 32 input channels
void fused_direct_convolution(
  uint32_t C_i,
  uint32_t C_o,
  uint32_t C_o_1x1,
  uint32_t H_f,
  uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  uint32_t stride,
  float * I,
  float * F,
  float * F_1x1,
  float * O_buffer,
  float * O_1x1
){

  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);

  // buffer is reused for different blocks

  // First 3x3 output block; don't load the 1x1 output (first update)

    // First 3x3 input block; dont' load 3x3 output buffer
    {
      float *filter_block_ptr = F;  // Using this variable for consistency
      // Loop over 3x3 output height
      for(uint32_t l = 0; l < H_o; l++){
          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob;

          // Loop over 3x3 output width
          for(uint32_t k = 0; k < W_o; k += W_ob){
            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;
            // Partially update a tile of the output (without loading the tile from memory)
            //            |           |
            //            V           V
            // (W_f*H_f*C_ib times) (W_ob x C_ob)
            conv_microkernel_start(W_i*C_ib, H_f, W_f, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

        }
      }
    }// end First  3x3 input block

    // Second -> penultimate 3x3 input block
    for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib){
      // printf("shata iteration\n");
      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;
            conv_microkernel(W_i*C_ib, H_f, W_f, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

        }
      }
    }// End Second -> penultimate 3x3 input block

    // Last 3x3 input
    {
      uint32_t input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob;
      float * filter_block_ptr = F + filter_i_c_block;
      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob;

          uint32_t o_1x1_row_offset = l * W_o * C_ob;
          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;

            uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
            conv_microkernel(W_i*C_ib, H_f, W_f, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
            fused_microkernel_start(O_buffer + col_offset + k *C_ob,
                                    C_o_1x1,
                                    F_1x1,
                                    H_o*W_o*C_ob,
                                    O_1x1 +
                                    o_1x1_col_offset);
        }

      }
    }


  float * F_1x1_ptr;
  // printf("filter offset\n");
  for(uint32_t j = C_ob; j < C_o; j += C_ob){
    F_1x1_ptr = F_1x1 + (j/C_ob)*(C_o_1x1/C_ob)*(C_ob)*C_ob;
    // printf("%d \n", F_1x1_ptr-F);
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;

    // uint32_t filer_1x1_block = (j/C_ob)*C_ob*C_o_1x1;

    // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
    float *filter_block_ptr = F + filter_i_c_block;

    for(uint32_t l = 0; l < H_o; l++){

        uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

        for(uint32_t k = 0; k < W_o; k += W_ob){

          uint32_t input_row_offset = (k * stride)*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;

          conv_microkernel_start(W_i*C_ib, H_f, W_f, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

      }
    }

    for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib){
      // printf("shata iteration\n");
      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;
            conv_microkernel(W_i*C_ib, H_f, W_f, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

        }
      }
    }

    //for(uint32_t i = C_ib; i < C_i; i += C_ib) (last iteration)

    input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
    filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
    filter_block_ptr = F + filter_i_c_block;
    for(uint32_t l = 0; l < H_o; l++){

        uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

        uint32_t o_1x1_row_offset = l * W_o * C_ob;
        for(uint32_t k = 0; k < W_o; k += W_ob){

          uint32_t input_row_offset = (k * stride)*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;

          uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
          conv_microkernel(W_i*C_ib, H_f, W_f, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
          fused_microkernel(O_buffer + col_offset + k *C_ob,
                                  C_o_1x1,
                                  F_1x1_ptr,
                                  H_o*W_o*C_ob,
                                  O_1x1 +
                                  o_1x1_col_offset);
      }
    }
  }
}
