#include "pooling_kernel.h"

template <uint32_t stride, uint32_t H_f, uint32_t W_f>
 void l_fused_gpool_conv_direct_convolution(
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
)
{

  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);
  uint32_t W_o_gpool = 1, H_o_gpool = 1;



  // W_o -= (W_o%2==0);

  uint32_t W_o_int = (W_o/(GPOOL_W_ob))*GPOOL_W_ob;
  uint32_t w_final = W_o - (W_o_int);
  //3x3 Output Channel Block
  #if PARALLEL==1
  #pragma omp parallel for
  #endif
  for(uint32_t j = 0; j < C_o; j += C_ob)
  {
    __m256 o0,  o1,
           o2,   o3,
           o4,   o5;

    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    uint32_t gpool_block_offset = (j/C_ob)*H_o_gpool*W_o_gpool*C_ob;
    float * gpool_O_ptr  = O + gpool_block_offset;
    float divisor = 1.0/(H_o*W_o);
    #if PARALLEL==1
    int tid = omp_get_thread_num();
    float * O_buffer = O_buffers + (tid)*(H_o*W_o*(C_ob));
    #else
    float * O_buffer = O_buffers;
    #endif
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

      for(uint32_t l = 0; l < H_o - 1; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          float * I_ptr = I + input_col_offset;

          for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){

            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
            I_ptr += (W_ob * stride)*C_ob;
        }
        conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);

      }
      #if PREFETCH
      __builtin_prefetch((const void*)(O_buffer),0,0);
      #endif
      uint32_t col_offset = (H_o - 1)*W_o*C_ob /*+ block_offset*/;
      uint32_t input_col_offset = ((H_o - 1) * stride)*W_i*C_ob + input_block_offset;
      float * I_ptr = I + input_col_offset;

      for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){

        conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
        I_ptr += (W_ob * stride)*C_ob;
    }
    conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
    }//end 3x3 Input channel blocks

    // fused gpool_conv
    //Last 3x3 Input Channel Block
    {
      uint32_t input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float * filter_block_ptr = F + filter_i_c_block;

      //First 3x3 Output Row
      {
        uint32_t col_offset = 0*W_o*C_ob;
        uint32_t input_col_offset = (0 * stride)*W_i*C_ob + input_block_offset;
        float * O_col = O_buffer  + col_offset;
        float * I_ptr = I + input_col_offset;

          for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
          {

            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + k *C_ob);
            I_ptr += (W_ob * stride)*C_ob;
          }
          conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + (W_o - W_ob)*C_ob);


        //first tile

        float * gpool_I_ptr = O_col;

        ZERO_2_C();
        ZERO_4_C();
        for(uint32_t k = 0; k < W_o_int ; k += GPOOL_W_ob)
        {
          global_pool(
                          gpool_I_ptr, // I
                          o0,  o1,
                          o2,   o3,
                          o4,   o5
                        );

        gpool_I_ptr += (GPOOL_W_ob)*C_ob;
        }


        // end gpool_first_row to Penultimate tile

        //last tile


          global_pool_reduce(
                              w_final,
                              gpool_I_ptr,
                              o0,  o1,
                              o2,   o3,
                              o4,   o5,
                              gpool_O_ptr
                            );



      }//end First 3x3 Output Row

      //2nd to Last 3x3 Output Rows
      for(uint32_t l = 1; l < H_o; l++)
      {

        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
          float * O_col = O_buffer +  col_offset;
          float * I_ptr = I + input_col_offset;

          for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
          {

            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + k *C_ob);
            I_ptr += (W_ob * stride)*C_ob;
          }
          conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + (W_o - W_ob)*C_ob);


          float * gpool_I_ptr = O_col;

          // First to Penultimate tile



          LOAD_2_C(gpool_O_ptr, o0, o1);
          ZERO_4_C();
          for(uint32_t k = 0; k < W_o_int ; k += GPOOL_W_ob)
          {
            global_pool(
                            gpool_I_ptr, // I
                            o0,  o1,
                            o2,   o3,
                            o4,   o5
                          );


          gpool_I_ptr += (GPOOL_W_ob)*C_ob;
          }


          // end gpool_first_row to Penultimate tile

          //last tile


            global_pool_reduce(
                                w_final,
                                gpool_I_ptr,
                                o0,  o1,
                                o2,   o3,
                                o4,   o5,
                                gpool_O_ptr
                              );

        }//end Even 3x3 Output Row



      }// end 2nd to Last 3x3 Output Rows

    }// end Last 3x3 Input Channel Block
    DIVIDE(gpool_O_ptr, divisor);
  }//end 3x3 Output Channel Blocks
}


 void gpooling(
  uint32_t C_o,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_o,
  uint32_t W_o,
  // uint32_t stride,
  float * I,
  float * O
)
{


  uint32_t W_o_gpool = 1, H_o_gpool = 1;



  // W_o -= (W_o%2==0);

  uint32_t W_o_int = (W_o/(GPOOL_W_ob))*GPOOL_W_ob;
  uint32_t w_final = W_o - (W_o_int);
  // printf("w_fin: %d", w_final);
  float divisor = 1.0/(H_o*W_o);
  //3x3 Output Channel Block
  #if PARALLEL==1
  #pragma omp parallel for
  #endif
  for(uint32_t j = 0; j < C_o; j += C_ob)
  {
    __m256 o0,  o1,
           o2,   o3,
           o4,   o5;

    uint32_t gpool_block_offset = (j/C_ob)*H_o_gpool*W_o_gpool*C_ob;
    float * gpool_O_ptr  = O + gpool_block_offset;
    float * O_buffer = I + (j/C_ob)*H_o*W_o*C_ob;


    // fused gpool_conv
    //Last 3x3 Input Channel Block
    {
      //First 3x3 Output Row
      {
        uint32_t col_offset = 0*W_o*C_ob;
        float * O_col = O_buffer  + col_offset;



        //first tile
        float * gpool_I_ptr = O_col;

        ZERO_2_C();
        ZERO_4_C();
        for(uint32_t k = 0; k < W_o_int ; k += GPOOL_W_ob)
        {
          global_pool(
                          gpool_I_ptr, // I
                          o0,  o1,
                          o2,   o3,
                          o4,   o5
                        );

        gpool_I_ptr += (GPOOL_W_ob)*C_ob;
        }


        // end gpool_first_row to Penultimate tile

        //last tile


          global_pool_reduce(
                              w_final,
                              gpool_I_ptr,
                              o0,  o1,
                              o2,   o3,
                              o4,   o5,
                              gpool_O_ptr
                            );




      }//end First 3x3 Output Row

      //2nd to Last 3x3 Output Rows
      for(uint32_t l = 1; l < H_o; l++)
      {

        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;
          float * O_col = O_buffer +  col_offset;


          float * gpool_I_ptr = O_col;

          // First to Penultimate tile



          LOAD_2_C(gpool_O_ptr, o0, o1);
          ZERO_4_C();
          for(uint32_t k = 0; k < W_o_int ; k += GPOOL_W_ob)
          {
            global_pool(
                            gpool_I_ptr, // I
                            o0,  o1,
                            o2,   o3,
                            o4,   o5
                          );


          gpool_I_ptr += (GPOOL_W_ob)*C_ob;
          }


          // end gpool_first_row to Penultimate tile

          //last tile


            global_pool_reduce(
                                w_final,
                                gpool_I_ptr,
                                o0,  o1,
                                o2,   o3,
                                o4,   o5,
                                gpool_O_ptr
                              );



        }//end Even 3x3 Output Row


      }// end 2nd to Last 3x3 Output Rows

    }// end Last 3x3 Input Channel Block

    DIVIDE(gpool_O_ptr, divisor);
  }//end 3x3 Output Channel Blocks
}
