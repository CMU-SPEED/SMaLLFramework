// this file implements simple direct convolution with no fusion


#include<stdint.h>
#include "kernel.h"



#define op_dim(IN_dim, stride, K_dim, OUT_dim){\
  OUT_dim = (IN_dim - K_dim )/stride + 1;\
}

// Ignore For now

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
  #pragma omp parallel for num_threads(6)
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


          conv_microkernel_start(W_i*C_ib, H_f, W_f, stride, I_ptr, filter_block_ptr, O + col_offset + k *C_ob);


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

            conv_microkernel(W_i*C_ib, H_f, W_f, stride, I_ptr, filter_block_ptr, O + col_offset + k *C_ob);


        }
      }
    }
  }

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

  // First output block; don't load the 1x1 output

    // uint32_t block_offset = (j/C_ob)*H_o*W_o*C_ob;
    // uint32_t filter_o_c_block = (0/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;

    //uint32_t filer_1x1_block = (0/C_ob)*C_ob*C_o_1x1;
    // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob /*+ filter_o_c_block*/;
    float *filter_block_ptr = F + filter_i_c_block;

    for(uint32_t l = 0; l < H_o; l++){

        uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

        for(uint32_t k = 0; k < W_o; k += W_ob){

          uint32_t input_row_offset = (k * stride)*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;

          conv_microkernel_start(W_i*C_ib, H_f, W_f, stride, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

      }
    }

    for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib){
      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob /*+ filter_o_c_block*/;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;
            conv_microkernel(W_i*C_ib, H_f, W_f, stride, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

        }
      }
    }

    //for(uint32_t i = C_ib; i < C_i; i += C_ib) (last iteration)

    input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
    filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob /*+ filter_o_c_block*/;
    filter_block_ptr = F + filter_i_c_block;
    for(uint32_t l = 0; l < H_o; l++){

        uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

        uint32_t o_1x1_row_offset = l * W_o * C_ob;
        for(uint32_t k = 0; k < W_o; k += W_ob){

          uint32_t input_row_offset = (k * stride)*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;

          uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
          conv_microkernel(W_i*C_ib, H_f, W_f, stride, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
          fused_microkernel_start(O_buffer + col_offset + k *C_ob,
                                  C_o_1x1,
                                  F_1x1,
                                  H_o*W_o*C_ob,
                                  O_1x1 +
                                  o_1x1_col_offset);
      }

    }
    //}


  float * F_1x1_ptr;
  for(uint32_t j = C_ob; j < C_o; j += C_ob){
    F_1x1_ptr = F_1x1 + (j/C_ob)*(C_o_1x1/C_ob)*(C_ob)*C_ob;
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;


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

          conv_microkernel_start(W_i*C_ib, H_f, W_f, stride, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

      }
    }

    for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib){
      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;
            conv_microkernel(W_i*C_ib, H_f, W_f, stride, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

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
          conv_microkernel(W_i*C_ib, H_f, W_f, stride, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
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

//fused pooling
// 3 x 2 x 16 register tile (still contiguous)
// Only works for 3x3 or 2x2
// #define POOL_KERNEL 3
// #define POOL_STRIDE 2

#define POOL_UNROLL 8




// pool horizontally and reuse the partially computed output for the height of the pooling layer
// Affected by how wide the output is?
// Need to buffer 2 rows of the pooling output?
//ASSUMPTION: Assumes 3x3 pooling with stride 2

//TODO: Work with more than 1 output row
void fused_pooling_direct_convolution(
  uint32_t C_i,
  uint32_t C_o,
  uint32_t H_f,
  uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  uint32_t stride,
  float * I,
  float * F,
  float * O_buffer,
  float * O
){

  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);
  uint32_t W_o_pool = 0, H_o_pool = 0;
  assert(W_o >POOL_KERNEL);
  op_dim(W_o, POOL_STRIDE, POOL_KERNEL, W_o_pool);
  op_dim(H_o, POOL_STRIDE, POOL_KERNEL, H_o_pool);

  //3x3 Output Channel Blocks
  for(uint32_t j = 0; j < C_o; j += C_ob)
  {

    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    uint32_t pool_block_offset = (j/C_ob)*H_o_pool*W_o_pool*C_ob;

    //First 3x3 Input Channel Block
    {
      // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float * filter_block_ptr = F + filter_i_c_block;
      //3x3 Output Rows
      for(uint32_t l = 0; l < H_o; l++)
      {

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          // 3x3 pixel blocks
          for(uint32_t k = 0; k < W_o; k += W_ob)
          {

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;

            conv_microkernel_start(W_i*C_ib, H_f, W_f, stride, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

          }// end 3x3 pixel blocks
      }// end 3x3 output rows
    }//end First 3x3 Input Channel Block

    // 3x3 Input channel blocks (N-2*C_ib updates)
    for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib)
    {
      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;
            conv_microkernel(W_i*C_ib, H_f, W_f, stride, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

        }
      }
    }//end 3x3 Input channel blocks

    //Last 3x3 Input Channel Block
    {
      uint32_t input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float * filter_block_ptr = F + filter_i_c_block;

      //First 3x3 Output Row
      {
        uint32_t col_offset = 0*W_o*C_ob;
        uint32_t input_col_offset = (0 * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;

        //first tile
        {
          conv_microkernel_pool_first_row_start(W_i*C_ib,
                                          H_f, W_f,
                                          stride,
                                          I_ptr, filter_block_ptr, O_buffer + col_offset + 0 *C_ob,
                                          O + pool_block_offset
                                        );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        // Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_first_row(W_i*C_ib,
                                      H_f, W_f,
                                      stride,
                                      I_ptr,
                                      filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        // end Second to Penultimate tile

        //last tile
        {


          conv_microkernel_pool_first_row_end(W_i*C_ib,
                                      H_f, W_f,
                                      stride,
                                      I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile

      }//end First 3x3 Output Row

      //Second 3x3 Output Row
      {
        // 3x3 params
        uint32_t col_offset = 1*W_o*C_ob;
        uint32_t input_col_offset = (1 * stride)*W_i*C_ob + input_block_offset;
        float * I_ptr = I + input_col_offset;

        //first tile
        {
          // uint32_t input_row_offset = (0 * stride)*C_ob;

          conv_microkernel_pool_accum_start(W_i*C_ib,
                                          H_f, W_f,
                                          stride,
                                          I_ptr, filter_block_ptr, O_buffer + col_offset + 0*C_ob,
                                          O + pool_block_offset
                                        );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;
        // printf("pool op offset %d \t", pool_col_offset/C_ob);
        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          // input_row_offset = (stride)*C_ob;
          // I_ptr = I + input_row_offset + input_col_offset;
          // printf("k %d\n",k);
          // uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
          conv_microkernel_pool_accum(W_i*C_ib,
                                      H_f, W_f,
                                      stride,
                                      I_ptr,
                                      filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          // printf("pool op offset %d \t", pool_col_offset/C_ob);
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {
          // uint32_t input_row_offset = ((W_o - W_ob) * stride)*C_ob;
          // float * I_ptr = I + input_row_offset + input_col_offset;

          // uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
          conv_microkernel_pool_accum_end(W_i*C_ib,
                                      H_f, W_f,
                                      stride,
                                      I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }//end Second 3x3 Output Row

      // Third to Penultimate 3x3 Output Rows
      // for(uint32_t l = POOL_STRIDE; l < H_o-(POOL_KERNEL - POOL_STRIDE); l+= POOL_STRIDE)
      // {
      //
      //
      //
      //   uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
      //   uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
      //
      //   //accumulate previous and start a new row
      //   for(uint32_t k = 0; k < W_o; k += W_ob)
      //   {
      //
      //     uint32_t input_row_offset = (k * stride)*C_ob;
      //     float * I_ptr = I + input_row_offset + input_col_offset;
      //
      //     // uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
      //     conv_microkernel_pool(W_i*C_ib, H_f, W_f, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
      //
      //   }
      //
      //
      //   // Reuse Previous Pool
      //   col_offset = (l + 1)*W_o*C_ob /*+ block_offset*/;
      //   input_col_offset = ((l + 1) *  stride)*W_i*C_ob + input_block_offset;
      //
      //   // uint32_t o_1x1_row_offset = l * W_o * C_ob;
      //   for(uint32_t k = 0; k < W_o; k += W_ob)
      //   {
      //
      //     uint32_t input_row_offset = (k * stride)*C_ob;
      //     float * I_ptr = I + input_row_offset + input_col_offset;
      //
      //     // uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
      //     conv_microkernel_pool_accum(W_i*C_ib, H_f, W_f, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
      //
      //   }
      //
      //
      // }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      // printf("\n\n\nLast 3x3 Output Row\n\n");
      {// 3x3 params
        uint32_t col_offset = (H_o - (POOL_KERNEL - POOL_STRIDE))*W_o*C_ob;
        uint32_t input_col_offset = ((H_o - (POOL_KERNEL - POOL_STRIDE)) * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;
        //first tile
        {
          // uint32_t input_row_offset = (0 * stride)*C_ob;
          // float * I_ptr = I + input_row_offset + input_col_offset;

          conv_microkernel_pool_accum_start(W_i*C_ib,
                                          H_f, W_f,
                                          stride,
                                          I_ptr, filter_block_ptr, O_buffer + col_offset + 0 *C_ob,
                                          O + pool_block_offset
                                        );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {
          // printf("\n3x3 6 pixels %d \n", k);
          // input_row_offset = (stride)*C_ob;
          // I_ptr = I + input_row_offset + input_col_offset;

          // uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
          conv_microkernel_pool_accum(W_i*C_ib,
                                      H_f, W_f,
                                      stride,
                                      I_ptr,
                                      filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {
          // uint32_t input_row_offset = ((W_o - W_ob) * stride)*C_ob;
          // I_ptr = I + input_row_offset + input_col_offset;

          // uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
          conv_microkernel_pool_accum_end(W_i*C_ib,
                                      H_f, W_f,
                                      stride,
                                      I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }// end Last 3x3 Output Row
    }// end Last 3x3 Input Channel Block
  }//end 3x3 Output Channel Blocks
}


void pooling(
    uint32_t C,
    uint32_t H_i,
    uint32_t W_i,
    float * I,
    float * O
)
{
  const int w_block = 1;
  uint32_t s = POOL_STRIDE;
  uint32_t W_o , H_o;
  op_dim(W_i, POOL_STRIDE, POOL_KERNEL, W_o);
  op_dim(H_i, POOL_STRIDE, POOL_KERNEL, H_o);
  // printf("\n %d %d to %d %d\n",H_i, W_i,H_o, W_o);
  uint32_t offset = 0;
  for(uint32_t i = 0; i < C; i+=C_ob)
  {
    uint32_t block_offset = (i/C_ob) * H_i * W_i * C_ob;
    // printf(" channel block %d: %d %d\n", i, offset, block_offset);

    for(uint32_t l = 0; l < H_o; l++)
    {
      uint32_t row_offset = (l*s)*W_i*C_ob + block_offset;
      for(uint32_t k = 0; k < W_o; k+=w_block)
      {
        uint32_t col_offset = (k*s)*C_ob + row_offset;
        __m256 r0, r1, r2, r3, r4, r5, r6, r7,r8, r9, r10, r11, r12,r13, r14, r15;
        r0 = _mm256_load_ps(I + col_offset);
        r1 = _mm256_load_ps(I + col_offset +  SIMD);
        // r2 = _mm256_setzero_ps();
        // r3 = _mm256_setzero_ps();
        // r4 = _mm256_setzero_ps();
        // r5 = _mm256_setzero_ps();
        // r6 = _mm256_setzero_ps();
        // r7 = _mm256_setzero_ps();
        for(uint32_t n = 0; n < POOL_KERNEL; n++)
        {
          uint32_t stencil_h = n*W_i*C_ob + col_offset;
          for(uint32_t m = 0; m < POOL_KERNEL; m++)
          {
            uint32_t stencil_w = m*C_ob + stencil_h;
            // printf("\t %d",stencil_w);
            r8 = _mm256_load_ps(I + stencil_w);
            stencil_w += SIMD;
            r9 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r10 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r11 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r12 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r13 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r14 = _mm256_load_ps(I + stencil_w);
            // stencil_w += SIMD;
            // r15 = _mm256_load_ps(I + stencil_w);

            r0 = _mm256_max_ps(r0, r8);
            r1 = _mm256_max_ps(r1, r9);
            // r2 = _mm256_max_ps(r2, r10);
            // r3 = _mm256_max_ps(r3, r11);
            // r4 = _mm256_max_ps(r4, r12);
            // r5 = _mm256_max_ps(r5, r13);
            // r6 = _mm256_max_ps(r6, r14);
            // r7 = _mm256_max_ps(r7, r15);
          }
          // printf("\n");
        }
        // printf("\n");
        _mm256_store_ps(O+offset, r0);
        offset += SIMD;
        _mm256_store_ps(O+offset, r1);
        offset += SIMD;
        // _mm256_store_ps(O+offset, r2);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r3);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r4);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r5);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r6);
        // offset += SIMD;
        // _mm256_store_ps(O+offset, r7);
        // offset += SIMD;
      }
    }
  }

}
