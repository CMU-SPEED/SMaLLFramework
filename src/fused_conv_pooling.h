#include "pooling_kernel.h"

// end fully computed kernels

// asssumes C_i > 32
template <uint32_t stride, uint32_t H_f, uint32_t W_f>
inline void fused_pooling_direct_convolution(
  uint32_t C_i,
  uint32_t C_o,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
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

  //3x3 Output Channel Block

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

          uint32_t col_offset = l*W_o*C_ob ;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          // 3x3 pixel blocks
          for(uint32_t k = 0; k < W_o; k += W_ob)
          {

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;

            conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

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
            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

        }
      }
    }//end 3x3 Input channel blocks

    // fused pooling
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
          conv_microkernel_pool_first_row_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                                        I_ptr, filter_block_ptr,
                                                                         O_buffer + col_offset + 0 *C_ob,
                                                                        O + pool_block_offset
                                                                      );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        // Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_first_row<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        // end Second to Penultimate tile

        //last tile
        {


          conv_microkernel_pool_first_row_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr, filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
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
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr,
                                          O_buffer + col_offset + 0*C_ob,
                                          O + pool_block_offset
                                        );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {
          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }//end Second 3x3 Output Row

      uint32_t pool_row_offset = 0 + pool_block_offset;
      uint32_t pool_row = 0;
      //Third to Penultimate 3x3 Output Rows
      for(uint32_t l = POOL_STRIDE; l < H_o-(POOL_KERNEL-1); l+= POOL_STRIDE)
      {
        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                               W_o_pool*C_ob,
                                                                I_ptr, filter_block_ptr,
                                                                O_buffer + col_offset + 0 *C_ob,
                                                                O + pool_row_offset
                                                                  );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                         W_o_pool*C_ob,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate tile

          //last tile
          {

            conv_microkernel_pool_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      W_o_pool*C_ob,
                                        I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }
          //end last tile
        }//end Even 3x3 Output Row

        pool_row_offset += W_o_pool*C_ib;
        pool_row++;

        //Odd 3x3 Output Row This should be the same as the Last row
        {
          // 3x3 params
          uint32_t col_offset = (l + 1)*W_o*C_ob;
          uint32_t input_col_offset = ((l + 1) * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                            I_ptr, filter_block_ptr, O_buffer + col_offset + 0*C_ob,
                                            O + pool_row_offset
                                          );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
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
            conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }//end last tile
        }//end Odd 3x3 Output Row

      }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      {// 3x3 params
        uint32_t col_offset = (pool_row*POOL_STRIDE + POOL_STRIDE)*W_o*C_ob;
        uint32_t input_col_offset = ((pool_row*POOL_STRIDE + POOL_STRIDE) * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;
        //first tile
        {
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr, O_buffer + col_offset + 0 *C_ob,
                                          O + pool_row_offset
                                        );
        }
        //end first tile

        uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,

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

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }// end Last 3x3 Output Row
    }// end Last 3x3 Input Channel Block
  }//end 3x3 Output Channel Blocks
}

// asssumes C_i > 32
// 3x3 output not fully computed, use intermediate output of full size (H_o*W_*C_o)
template <uint32_t stride, uint32_t H_f, uint32_t W_f>
void fused_pooling_direct_convolution_not_buffered(
  uint32_t C_i,
  uint32_t C_o,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
  float * I,
  float * F,
  float * O_buffers,
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
  H_o -= (H_o % 2 == 0);
  //3x3 Output Channel Block

  for(uint32_t j = 0; j < C_o; j += C_ob)
  {
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    uint32_t pool_block_offset = (j/C_ob)*H_o_pool*W_o_pool*C_ob;
    float * O_buffer = O_buffers + (j/C_ob)*(H_o*W_o*C_ob);
    //First 3x3 Input Channel Block
    {
      // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float * filter_block_ptr = F + filter_i_c_block;
      //3x3 Output Rows
      for(uint32_t l = 0; l < H_o; l++)
      {

          uint32_t col_offset = l*W_o*C_ob ;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          // 3x3 pixel blocks
          for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
          {


            conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
            I_ptr += (W_ob*stride)*C_ob;
          }// end 3x3 pixel blocks

          conv_microkernel_start_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);

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
          float * I_ptr = I + input_col_offset;

          for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){

            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
            I_ptr += (W_ob * stride)*C_ob;
        }
        conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);

      }
    }//end 3x3 Input channel blocks

    // fused pooling
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
          conv_microkernel_pool_first_row_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                                        I_ptr, filter_block_ptr,
                                                                         O_buffer + col_offset + 0 *C_ob,
                                                                        O + pool_block_offset
                                                                      );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        // Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_first_row<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        // end Second to Penultimate tile

        //last tile
        {


          conv_microkernel_pool_first_row_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr, filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
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
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr,
                                          O_buffer + col_offset + 0*C_ob,
                                          O + pool_block_offset
                                        );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {
          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }//end Second 3x3 Output Row

      uint32_t pool_row_offset = 0 + pool_block_offset;
      uint32_t pool_row = 0;
      //Third to Penultimate 3x3 Output Rows
      for(uint32_t l = POOL_STRIDE; l < H_o-(POOL_KERNEL-1); l+= POOL_STRIDE)
      {
        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                               W_o_pool*C_ob,
                                                                I_ptr, filter_block_ptr,
                                                                O_buffer + col_offset + 0 *C_ob,
                                                                O + pool_row_offset
                                                                  );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                         W_o_pool*C_ob,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate tile

          //last tile
          {

            conv_microkernel_pool_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      W_o_pool*C_ob,
                                        I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }
          //end last tile
        }//end Even 3x3 Output Row

        pool_row_offset += W_o_pool*C_ib;
        pool_row++;

        //Odd 3x3 Output Row This should be the same as the Last row
        {
          // 3x3 params
          uint32_t col_offset = (l + 1)*W_o*C_ob;
          uint32_t input_col_offset = ((l + 1) * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                            I_ptr, filter_block_ptr, O_buffer + col_offset + 0*C_ob,
                                            O + pool_row_offset
                                          );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
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
            conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }//end last tile
        }//end Odd 3x3 Output Row

      }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      {// 3x3 params
        uint32_t col_offset = (pool_row*POOL_STRIDE + POOL_STRIDE)*W_o*C_ob;
        uint32_t input_col_offset = ((pool_row*POOL_STRIDE + POOL_STRIDE) * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;
        //first tile
        {
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr, O_buffer + col_offset + 0 *C_ob,
                                          O + pool_row_offset
                                        );
        }
        //end first tile

        uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,

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

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }// end Last 3x3 Output Row
    }// end Last 3x3 Input Channel Block
  }//end 3x3 Output Channel Blocks
}


// asssumes C_i > 32
// 3x3 outputfully computed, reuse use intermediate output of full size (H_o*W_*C_o)
template <uint32_t stride, uint32_t H_f, uint32_t W_f>
void fused_pooling_direct_convolution_complete(
  uint32_t C_i,
  uint32_t C_o,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
  float * I,
  float * F,
  float * O_buffers,
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
  H_o -= (H_o % 2 == 0);
  //3x3 Output Channel Block
  for(uint32_t j = 0; j < C_o; j += C_ob)
  {

    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    uint32_t pool_block_offset = (j/C_ob)*H_o_pool*W_o_pool*C_ob;
    float * O_buffer = O_buffers + (j/C_ob)*(H_o*W_o*C_ob);
    //First 3x3 Input Channel Block
    {
      // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float * filter_block_ptr = F + filter_i_c_block;
      //3x3 Output Rows
      for(uint32_t l = 0; l < H_o; l++)
      {

          uint32_t col_offset = l*W_o*C_ob ;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          // 3x3 pixel blocks
          for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
          {


            conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
            I_ptr += (W_ob*stride)*C_ob;
          }// end 3x3 pixel blocks

          conv_microkernel_start_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);

      }// end 3x3 output rows
    }//end First 3x3 Input Channel Block

    // 3x3 Input channel blocks (N-2*C_ib updates)
    for(uint32_t i = C_ib; i < C_i  - C_ib; i += C_ib)
    {
      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){

            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
            I_ptr += (W_ob * stride)*C_ob;
        }
        conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);

      }
    }//end 3x3 Input channel blocks

    // fused pooling
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
          complete_conv_microkernel_pool_first_row_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                                        I_ptr, filter_block_ptr,
                                                                         O_buffer + col_offset + 0 *C_ob,
                                                                        O + pool_block_offset
                                                                      );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        // Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          complete_conv_microkernel_pool_first_row<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        // end Second to Penultimate tile

        //last tile
        {


          complete_conv_microkernel_pool_first_row_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr, filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
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
          complete_conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr,
                                          O_buffer + col_offset + 0*C_ob,
                                          O + pool_block_offset
                                        );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {
          complete_conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          complete_conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }//end Second 3x3 Output Row

      uint32_t pool_row_offset = 0 + pool_block_offset;
      uint32_t pool_row = 0;
      //Third to Penultimate 3x3 Output Rows
      for(uint32_t l = POOL_STRIDE; l < H_o-(POOL_KERNEL-1); l+= POOL_STRIDE)
      {
        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          float * I_ptr = I + input_col_offset;

          //first tile
          {
            complete_conv_microkernel_pool_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                               W_o_pool*C_ob,
                                                                I_ptr, filter_block_ptr,
                                                                O_buffer + col_offset + 0 *C_ob,
                                                                O + pool_row_offset
                                                                  );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            complete_conv_microkernel_pool<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                         W_o_pool*C_ob,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate tile

          //last tile
          {

            complete_conv_microkernel_pool_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      W_o_pool*C_ob,
                                        I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }
          //end last tile
        }//end Even 3x3 Output Row

        pool_row_offset += W_o_pool*C_ib;
        pool_row++;

        //Odd 3x3 Output Row This should be the same as the Last row
        {
          // 3x3 params
          uint32_t col_offset = (l + 1)*W_o*C_ob;
          uint32_t input_col_offset = ((l + 1) * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          //first tile
          {
            complete_conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                            I_ptr, filter_block_ptr, O_buffer + col_offset + 0*C_ob,
                                            O + pool_row_offset
                                          );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            complete_conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
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
            complete_conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }//end last tile
        }//end Odd 3x3 Output Row

      }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      {// 3x3 params
        uint32_t col_offset = (pool_row*POOL_STRIDE + POOL_STRIDE)*W_o*C_ob;
        uint32_t input_col_offset = ((pool_row*POOL_STRIDE + POOL_STRIDE) * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;
        //first tile
        {
          complete_conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr, O_buffer + col_offset + 0 *C_ob,
                                          O + pool_row_offset
                                        );
        }
        //end first tile

        uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          complete_conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,

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

          complete_conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }// end Last 3x3 Output Row
    }
    // end Last 3x3 Input Channel Block
  }//end 3x3 Output Channel Blocks
}



// asssumes C_i > 32
// 3x3 output not fully computed, reuse a buffer num_threads*(H_o*W_o*16)
template <uint32_t stride, uint32_t H_f, uint32_t W_f>
void parallel_fused_pooling_direct_convolution(
  uint32_t C_i,
  uint32_t C_o,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
  float * I,
  float * F,
  float * O_buffers,
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
  H_o -= (H_o % 2 == 0);
  //3x3 Output Channel Block
  #pragma omp parallel for
  for(uint32_t j = 0; j < C_o; j += C_ob)
  {
    uint32_t tid = omp_get_thread_num();
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    uint32_t pool_block_offset = (j/C_ob)*H_o_pool*W_o_pool*C_ob;
    float * O_buffer = O_buffers + (tid)*(H_o*W_o*C_ob);
    //First 3x3 Input Channel Block
    {
      // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float * filter_block_ptr = F + filter_i_c_block;
      //3x3 Output Rows
      for(uint32_t l = 0; l < H_o; l++)
      {

          uint32_t col_offset = l*W_o*C_ob ;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
            float * I_ptr = I + input_col_offset;
          // 3x3 pixel blocks
          for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
          {

            // uint32_t input_row_offset = (k * stride)*C_ob;


            conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
            I_ptr += (W_ob * stride) * C_ob;
          }// end 3x3 pixel blocks

          conv_microkernel_start_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob) *C_ob);


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
          float * I_ptr = I + input_col_offset;
          for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){

            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
            I_ptr += (W_ob * stride)*C_ob;
        }
        conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);

      }
    }//end 3x3 Input channel blocks

    // fused pooling
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
          conv_microkernel_pool_first_row_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                                        I_ptr, filter_block_ptr,
                                                                         O_buffer + col_offset + 0 *C_ob,
                                                                        O + pool_block_offset
                                                                      );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        // Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_first_row<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        // end Second to Penultimate tile

        //last tile
        {


          conv_microkernel_pool_first_row_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr, filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
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
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr,
                                          O_buffer + col_offset + 0*C_ob,
                                          O + pool_block_offset
                                        );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {
          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }//end Second 3x3 Output Row

      uint32_t pool_row_offset = 0 + pool_block_offset;
      uint32_t pool_row = 0;
      //Third to Penultimate 3x3 Output Rows
      for(uint32_t l = POOL_STRIDE; l < H_o-(POOL_KERNEL-1); l+= POOL_STRIDE)
      {
        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                               W_o_pool*C_ob,
                                                                I_ptr, filter_block_ptr,
                                                                O_buffer + col_offset + 0 *C_ob,
                                                                O + pool_row_offset
                                                                  );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                         W_o_pool*C_ob,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate tile

          //last tile
          {

            conv_microkernel_pool_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      W_o_pool*C_ob,
                                        I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }
          //end last tile
        }//end Even 3x3 Output Row

        pool_row_offset += W_o_pool*C_ib;
        pool_row++;

        //Odd 3x3 Output Row This should be the same as the Last row
        {
          // 3x3 params
          uint32_t col_offset = (l + 1)*W_o*C_ob;
          uint32_t input_col_offset = ((l + 1) * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                            I_ptr, filter_block_ptr, O_buffer + col_offset + 0*C_ob,
                                            O + pool_row_offset
                                          );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
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
            conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }//end last tile
        }//end Odd 3x3 Output Row

      }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      {// 3x3 params
        uint32_t col_offset = (pool_row*POOL_STRIDE + POOL_STRIDE)*W_o*C_ob;
        uint32_t input_col_offset = ((pool_row*POOL_STRIDE + POOL_STRIDE) * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;
        //first tile
        {
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr, O_buffer + col_offset + 0 *C_ob,
                                          O + pool_row_offset
                                        );
        }
        //end first tile

        uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,

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

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }// end Last 3x3 Output Row
    }// end Last 3x3 Input Channel Block
  }//end 3x3 Output Channel Blocks
}


// asssumes C_i > 32
// 3x3 output not fully computed, use intermediate output of full size (H_o*W_*C_o)
template <uint32_t stride, uint32_t H_f, uint32_t W_f>
void parallel_fused_pooling_direct_convolution_not_buffered(
  uint32_t C_i,
  uint32_t C_o,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
  float * I,
  float * F,
  float * O_buffers,
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
  H_o -= (H_o % 2 == 0);
  //3x3 Output Channel Block
  #pragma omp parallel for
  for(uint32_t j = 0; j < C_o; j += C_ob)
  {
    uint32_t tid = omp_get_thread_num();
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    uint32_t pool_block_offset = (j/C_ob)*H_o_pool*W_o_pool*C_ob;
    float * O_buffer = O_buffers + (j/C_ob)*(H_o*W_o*C_ob);
    //First 3x3 Input Channel Block
    {
      // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float * filter_block_ptr = F + filter_i_c_block;
      //3x3 Output Rows
      for(uint32_t l = 0; l < H_o; l++)
      {

          uint32_t col_offset = l*W_o*C_ob ;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          // 3x3 pixel blocks
          for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
          {


            conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
            I_ptr += (W_ob*stride)*C_ob;
          }// end 3x3 pixel blocks

          conv_microkernel_start_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);

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
          float * I_ptr = I + input_col_offset;

          for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){

            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
            I_ptr += (W_ob * stride)*C_ob;
        }
        conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);

      }
    }//end 3x3 Input channel blocks

    // fused pooling
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
          conv_microkernel_pool_first_row_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                                        I_ptr, filter_block_ptr,
                                                                         O_buffer + col_offset + 0 *C_ob,
                                                                        O + pool_block_offset
                                                                      );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        // Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_first_row<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        // end Second to Penultimate tile

        //last tile
        {


          conv_microkernel_pool_first_row_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr, filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
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
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr,
                                          O_buffer + col_offset + 0*C_ob,
                                          O + pool_block_offset
                                        );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {
          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }//end Second 3x3 Output Row

      uint32_t pool_row_offset = 0 + pool_block_offset;
      uint32_t pool_row = 0;
      //Third to Penultimate 3x3 Output Rows
      for(uint32_t l = POOL_STRIDE; l < H_o-(POOL_KERNEL-1); l+= POOL_STRIDE)
      {
        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                               W_o_pool*C_ob,
                                                                I_ptr, filter_block_ptr,
                                                                O_buffer + col_offset + 0 *C_ob,
                                                                O + pool_row_offset
                                                                  );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                         W_o_pool*C_ob,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate tile

          //last tile
          {

            conv_microkernel_pool_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      W_o_pool*C_ob,
                                        I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }
          //end last tile
        }//end Even 3x3 Output Row

        pool_row_offset += W_o_pool*C_ib;
        pool_row++;

        //Odd 3x3 Output Row This should be the same as the Last row
        {
          // 3x3 params
          uint32_t col_offset = (l + 1)*W_o*C_ob;
          uint32_t input_col_offset = ((l + 1) * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          //first tile
          {
            conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                            I_ptr, filter_block_ptr, O_buffer + col_offset + 0*C_ob,
                                            O + pool_row_offset
                                          );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
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
            conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }//end last tile
        }//end Odd 3x3 Output Row

      }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      {// 3x3 params
        uint32_t col_offset = (pool_row*POOL_STRIDE + POOL_STRIDE)*W_o*C_ob;
        uint32_t input_col_offset = ((pool_row*POOL_STRIDE + POOL_STRIDE) * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;
        //first tile
        {
          conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr, O_buffer + col_offset + 0 *C_ob,
                                          O + pool_row_offset
                                        );
        }
        //end first tile

        uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,

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

          conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }// end Last 3x3 Output Row
    }// end Last 3x3 Input Channel Block
  }//end 3x3 Output Channel Blocks
}

template <uint32_t stride, uint32_t H_f, uint32_t W_f>
void parallel_fused_pooling_direct_convolution_complete(
  uint32_t C_i,
  uint32_t C_o,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
  float * I,
  float * F,
  float * O_buffers,
  float * O
){

  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
    H_o -= (H_o % 2 == 0);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);
  uint32_t W_o_pool = 0, H_o_pool = 0;
  assert(W_o >POOL_KERNEL);
  op_dim(W_o, POOL_STRIDE, POOL_KERNEL, W_o_pool);
  op_dim(H_o, POOL_STRIDE, POOL_KERNEL, H_o_pool);

  //Conv Output Channel Block
  #pragma omp parallel for
  for(uint32_t j = 0; j < C_o; j += C_ob)
  {

    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    uint32_t pool_block_offset = (j/C_ob)*H_o_pool*W_o_pool*C_ob;
    float * O_buffer = O_buffers + (j/C_ob)*(H_o*W_o*C_ob);
    //First Conv Input Channel Block
    {
      // These are all 0
      uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

      float * filter_block_ptr = F + filter_i_c_block;
      //Conv Output Rows
      for(uint32_t l = 0; l < H_o; l++)
      {

          uint32_t col_offset = l*W_o*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;
          // Conv pixel Blocks
          for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
          {

            conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
            I_ptr += (W_ob * stride)*C_ob;

          }// end Conv pixel Blocks

          conv_microkernel_start_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);

      }//end Conv Output Rows
    }//end First Conv Input Channel Block

    // Conv Input Channel Blocks 2 -> (N - 1)
    for(uint32_t i = C_ib; i < C_i - C_ib; i += C_ib)
    {
      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){

            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
            I_ptr += (W_ob * stride)*C_ob;
        }
        conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);

      }
    }//end Conv Input Channel Blocks 2 -> (N - 1)

    // fused pooling
    //Last Conv Input Channel Block
    {
      uint32_t input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float * filter_block_ptr = F + filter_i_c_block;

      //First Conv Output Row
      {
        uint32_t col_offset = 0*W_o*C_ob;
        uint32_t input_col_offset = (0 * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;

        //first tile
        {
          complete_conv_microkernel_pool_first_row_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                                        I_ptr, filter_block_ptr,
                                                                         O_buffer + col_offset + 0 *C_ob,
                                                                        O + pool_block_offset
                                                                      );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        // Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          complete_conv_microkernel_pool_first_row<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        // end Second to Penultimate tile

        //last tile
        {


          complete_conv_microkernel_pool_first_row_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr, filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile

      }//end First Conv Output Row

      //Second Conv Output Row
      {
        // Conv params
        uint32_t col_offset = 1*W_o*C_ob;
        uint32_t input_col_offset = (1 * stride)*W_i*C_ob + input_block_offset;
        float * I_ptr = I + input_col_offset;

        //first tile
        {
          complete_conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr,
                                          O_buffer + col_offset + 0*C_ob,
                                          O + pool_block_offset
                                        );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {
          complete_conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + k *C_ob,
                                      O + pool_col_offset
                                    );
          pool_col_offset += 3*C_ob;
          I_ptr += (W_ob*stride)*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          complete_conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      I_ptr,filter_block_ptr,
                                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }//end Second Conv Output Row

      uint32_t pool_row_offset = 0 + pool_block_offset;
      uint32_t pool_row = 0;
      //Third to Penultimate Conv Output Rows
      for(uint32_t l = POOL_STRIDE; l < H_o-(POOL_KERNEL-1); l+= POOL_STRIDE)
      {
        //Even Conv Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          float * I_ptr = I + input_col_offset;

          //first tile
          {
            complete_conv_microkernel_pool_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                               W_o_pool*C_ob,
                                                                I_ptr, filter_block_ptr,
                                                                O_buffer + col_offset + 0 *C_ob,
                                                                O + pool_row_offset
                                                                  );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            complete_conv_microkernel_pool<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                                         W_o_pool*C_ob,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + k *C_ob,
                                        O + pool_col_offset
                                      );
            pool_col_offset += 3*C_ob;
            I_ptr += (W_ob*stride)*C_ob;
          }
          //end Second to Penultimate tile

          //last tile
          {

            complete_conv_microkernel_pool_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                      W_o_pool*C_ob,
                                        I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }
          //end last tile
        }//end Even Conv Output Row

        pool_row_offset += W_o_pool*C_ib;
        pool_row++;

        //Odd Conv Output Row This should be the same as the Last row
        {
          // Conv params
          uint32_t col_offset = (l + 1)*W_o*C_ob;
          uint32_t input_col_offset = ((l + 1) * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          //first tile
          {
            complete_conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                            I_ptr, filter_block_ptr, O_buffer + col_offset + 0*C_ob,
                                            O + pool_row_offset
                                          );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
          I_ptr += (W_ob*stride)*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            complete_conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
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
            complete_conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                        I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                        O + pool_col_offset
                                      );

          }//end last tile
        }//end Odd Conv Output Row

      }// end Third to Penultimate Conv Output Rows

      // Last Conv Output Row
      {// Conv params
        uint32_t col_offset = (pool_row*POOL_STRIDE + POOL_STRIDE)*W_o*C_ob;
        uint32_t input_col_offset = ((pool_row*POOL_STRIDE + POOL_STRIDE) * stride)*W_i*C_ob + input_block_offset;

        float * I_ptr = I + input_col_offset;
        //first tile
        {
          complete_conv_microkernel_pool_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
                                          I_ptr, filter_block_ptr, O_buffer + col_offset + 0 *C_ob,
                                          O + pool_row_offset
                                        );
        }
        //end first tile

        uint32_t pool_col_offset = pool_row_offset + 2*C_ob;
        I_ptr += (W_ob*stride)*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          complete_conv_microkernel_pool_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,

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

          complete_conv_microkernel_pool_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,

                                      I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
                                      O + pool_col_offset
                                    );

        }//end last tile
      }// end Last Conv Output Row
    }
    //end Last Conv Input Channel Block
  }//end Conv Output Channel Blocks
}



void parallel_pooling(
    uint32_t C,
    uint32_t H_o,
    uint32_t W_o,
    float * I,
    float * O
)
{
  const int w_block = 1;
  uint32_t s = POOL_STRIDE;
  uint32_t W_o_pool , H_o_pool;
  op_dim(W_o, POOL_STRIDE, POOL_KERNEL, W_o_pool);
  op_dim(H_o, POOL_STRIDE, POOL_KERNEL, H_o_pool);
  // printf("\n %d %d to %d %d\n",H_i, W_i,H_o, W_o);
  uint32_t offset = 0;
  for(uint32_t i = 0; i < C; i+=C_ob)
  {
    uint32_t block_offset = (i/C_ob) * H_o * W_o * C_ob;
    float * O_buffer  = I + block_offset;
    uint32_t pool_block_offset = (i/C_ob)*H_o_pool*W_o_pool*C_ob;

    // printf(" channel block %d: %d %d\n", i, offset, block_offset);

    {

      //First 3x3 Output Row
      {
        uint32_t col_offset = 0*W_o*C_ob;


        //first tile
        {
          pool_first_row_start(
                                 O_buffer + col_offset + 0 *C_ob,
                                 O + pool_block_offset
                               );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;

        // Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          pool_first_row(
                          O_buffer + col_offset + k *C_ob,
                          O + pool_col_offset
                        );
          pool_col_offset += 3*C_ob;

        }
        // end Second to Penultimate tile

        //last tile
        {


          pool_first_row_end(
                              O_buffer + col_offset + (W_o - W_ob)*C_ob,
                              O + pool_col_offset
                            );

        }//end last tile

      }//end First 3x3 Output Row

      //Second 3x3 Output Row
      {
        // 3x3 params
        uint32_t col_offset = 1*W_o*C_ob;

        //first tile
        {
          pool_accum_start(
                            O_buffer + col_offset + 0*C_ob,
                            O + pool_block_offset
                          );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {
          pool_accum(
                        O_buffer + col_offset + k *C_ob,
                        O + pool_col_offset
                      );
          pool_col_offset += 3*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          pool_accum_end(
                        O_buffer + col_offset + (W_o - W_ob)*C_ob,
                        O + pool_col_offset
                      );

        }//end last tile
      }//end Second 3x3 Output Row

      uint32_t pool_row_offset = 0 + pool_block_offset;
      uint32_t pool_row = 0;
      //Third to Penultimate 3x3 Output Rows
      for(uint32_t l = POOL_STRIDE; l < H_o-(POOL_KERNEL-1); l+= POOL_STRIDE)
      {
        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;

          //first tile
          {
            pool_start(
                        W_o_pool*C_ob,
                        O_buffer + col_offset + 0 *C_ob,
                        O + pool_row_offset
                      );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            pool(
                 W_o_pool*C_ob,
                 O_buffer + col_offset + k *C_ob,
                 O + pool_col_offset
                );
            pool_col_offset += 3*C_ob;
          }
          //end Second to Penultimate tile

          //last tile
          {

            pool_end(
                      W_o_pool*C_ob,
                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                      O + pool_col_offset
                    );

          }
          //end last tile
        }//end Even 3x3 Output Row

        pool_row_offset += W_o_pool*C_ib;
        pool_row++;

        //Odd 3x3 Output Row This should be the same as the Last row
        {
          // 3x3 params
          uint32_t col_offset = (l + 1)*W_o*C_ob;

          //first tile
          {
            pool_accum_start(
                              O_buffer + col_offset + 0*C_ob,
                              O + pool_row_offset
                            );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            pool_accum(
                        O_buffer + col_offset + k *C_ob,
                        O + pool_col_offset
                      );
            pool_col_offset += 3*C_ob;
          }
          //end Second to Penultimate Tile

          //last tile
          {
            pool_accum_end(
                            O_buffer + col_offset + (W_o - W_ob)*C_ob,
                            O + pool_col_offset
                          );

          }//end last tile
        }//end Odd 3x3 Output Row

      }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      {// 3x3 params
        uint32_t col_offset = (pool_row*POOL_STRIDE + POOL_STRIDE)*W_o*C_ob;

        //first tile
        {
          pool_accum_start(
                            O_buffer + col_offset + 0 *C_ob,
                            O + pool_row_offset
                          );
        }
        //end first tile

        uint32_t pool_col_offset = pool_row_offset + 2*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          pool_accum(
                      O_buffer + col_offset + k *C_ob,
                      O + pool_col_offset
                    );
          pool_col_offset += 3*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          pool_accum_end(
                          O_buffer + col_offset + (W_o - W_ob)*C_ob,
                          O + pool_col_offset
                        );

        }//end last tile
      }// end Last 3x3 Output Row
    }
  }

}
void pooling(
    uint32_t C,
    uint32_t H_o,
    uint32_t W_o,
    float * I,
    float * O
)
{
  const int w_block = 1;
  uint32_t s = POOL_STRIDE;
  uint32_t W_o_pool , H_o_pool;
  op_dim(W_o, POOL_STRIDE, POOL_KERNEL, W_o_pool);
  op_dim(H_o, POOL_STRIDE, POOL_KERNEL, H_o_pool);
  // printf("\n %d %d to %d %d\n",H_i, W_i,H_o, W_o);
  uint32_t offset = 0;
  for(uint32_t i = 0; i < C; i+=C_ob)
  {
    uint32_t block_offset = (i/C_ob) * H_o * W_o * C_ob;
    float * O_buffer  = I + block_offset;
    uint32_t pool_block_offset = (i/C_ob)*H_o_pool*W_o_pool*C_ob;

    // printf(" channel block %d: %d %d\n", i, offset, block_offset);

    {

      //First 3x3 Output Row
      {
        uint32_t col_offset = 0*W_o*C_ob;


        //first tile
        {
          pool_first_row_start(
                                 O_buffer + col_offset + 0 *C_ob,
                                 O + pool_block_offset
                               );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;

        // Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          pool_first_row(
                          O_buffer + col_offset + k *C_ob,
                          O + pool_col_offset
                        );
          pool_col_offset += 3*C_ob;

        }
        // end Second to Penultimate tile

        //last tile
        {


          pool_first_row_end(
                              O_buffer + col_offset + (W_o - W_ob)*C_ob,
                              O + pool_col_offset
                            );

        }//end last tile

      }//end First 3x3 Output Row

      //Second 3x3 Output Row
      {
        // 3x3 params
        uint32_t col_offset = 1*W_o*C_ob;

        //first tile
        {
          pool_accum_start(
                            O_buffer + col_offset + 0*C_ob,
                            O + pool_block_offset
                          );
        }//end first tile

        uint32_t pool_col_offset = pool_block_offset + 2*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {
          pool_accum(
                        O_buffer + col_offset + k *C_ob,
                        O + pool_col_offset
                      );
          pool_col_offset += 3*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          pool_accum_end(
                        O_buffer + col_offset + (W_o - W_ob)*C_ob,
                        O + pool_col_offset
                      );

        }//end last tile
      }//end Second 3x3 Output Row

      uint32_t pool_row_offset = 0 + pool_block_offset;
      uint32_t pool_row = 0;
      //Third to Penultimate 3x3 Output Rows
      for(uint32_t l = POOL_STRIDE; l < H_o-(POOL_KERNEL-1); l+= POOL_STRIDE)
      {
        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;

          //first tile
          {
            pool_start(
                        W_o_pool*C_ob,
                        O_buffer + col_offset + 0 *C_ob,
                        O + pool_row_offset
                      );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            pool(
                 W_o_pool*C_ob,
                 O_buffer + col_offset + k *C_ob,
                 O + pool_col_offset
                );
            pool_col_offset += 3*C_ob;
          }
          //end Second to Penultimate tile

          //last tile
          {

            pool_end(
                      W_o_pool*C_ob,
                      O_buffer + col_offset + (W_o - W_ob)*C_ob,
                      O + pool_col_offset
                    );

          }
          //end last tile
        }//end Even 3x3 Output Row

        pool_row_offset += W_o_pool*C_ib;
        pool_row++;

        //Odd 3x3 Output Row This should be the same as the Last row
        {
          // 3x3 params
          uint32_t col_offset = (l + 1)*W_o*C_ob;

          //first tile
          {
            pool_accum_start(
                              O_buffer + col_offset + 0*C_ob,
                              O + pool_row_offset
                            );
          }//end first tile

          uint32_t pool_col_offset = pool_row_offset + 2*C_ob;

          //Second to Penultimate tile
          for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
          {

            pool_accum(
                        O_buffer + col_offset + k *C_ob,
                        O + pool_col_offset
                      );
            pool_col_offset += 3*C_ob;
          }
          //end Second to Penultimate Tile

          //last tile
          {
            pool_accum_end(
                            O_buffer + col_offset + (W_o - W_ob)*C_ob,
                            O + pool_col_offset
                          );

          }//end last tile
        }//end Odd 3x3 Output Row

      }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      {// 3x3 params
        uint32_t col_offset = (pool_row*POOL_STRIDE + POOL_STRIDE)*W_o*C_ob;

        //first tile
        {
          pool_accum_start(
                            O_buffer + col_offset + 0 *C_ob,
                            O + pool_row_offset
                          );
        }
        //end first tile

        uint32_t pool_col_offset = pool_row_offset + 2*C_ob;

        //Second to Penultimate tile
        for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
        {

          pool_accum(
                      O_buffer + col_offset + k *C_ob,
                      O + pool_col_offset
                    );
          pool_col_offset += 3*C_ob;
        }
        //end Second to Penultimate Tile

        //last tile
        {

          pool_accum_end(
                          O_buffer + col_offset + (W_o - W_ob)*C_ob,
                          O + pool_col_offset
                        );

        }//end last tile
      }// end Last 3x3 Output Row
    }
  }

}
