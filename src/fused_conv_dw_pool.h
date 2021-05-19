#include "dw_conv_kernel.h"

// end fully computed kernels

// asssumes C_i > 32
// template <uint32_t stride, uint32_t H_f, uint32_t W_f>
// void fused_dw_conv_direct_convolution(
//   uint32_t C_i,
//   uint32_t C_o,
//   // uint32_t H_f,
//   // uint32_t W_f,
//   uint32_t H_i,
//   uint32_t W_i,
//   // uint32_t stride,
//   float * I,
//   float * F,
//   float * O_buffers,
//   float * O
// ){
//
//   uint32_t H_o = 0;
//   op_dim(H_i, stride,H_f,H_o);
//   uint32_t W_o = 0;
//   op_dim(W_i, stride,W_f,W_o);
//   uint32_t W_o_dw = 0, H_o_dw = 0;
//   assert(W_o >DW_KERNEL);
//   op_dim(W_o, DW_STRIDE, DW_KERNEL, W_o_dw);
//   op_dim(H_o, DW_STRIDE, DW_KERNEL, H_o_dw);
//
//   H_o -= (H_o%2==0);
//   // W_o -= (W_o%2==0);
//
//   //3x3 Output Channel Block
//   #if PARALLEL==1
//   #pragma omp parallel for
//   #endif
//   for(uint32_t j = 0; j < C_o; j += C_ob)
//   {
//     uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
//     uint32_t dw_block_offset = (j/C_ob)*H_o_dw*W_o_dw*C_ob;
//     #if PARALLEL==1
//     int tid = omp_get_thread_num();
//     float * O_buffer = O_buffers + (tid)*(H_o*W_o*(C_ob));
//     #else
//     float * O_buffer = O_buffers;
//     #endif
//     //First 3x3 Input Channel Block
//     {
//       // These are all 0
//     uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
//     uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//
//     float * filter_block_ptr = F + filter_i_c_block;
//       //3x3 Output Rows
//       for(uint32_t l = 0; l < H_o; l++)
//       {
//
//           uint32_t col_offset = l*W_o*C_ob ;
//           uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//           float * I_ptr = I + input_col_offset;
//
//           // 3x3 pixel blocks
//           for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
//           {
//
//
//             conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//             I_ptr += (W_ob*stride)*C_ob;
//           }// end 3x3 pixel blocks
//
//           conv_microkernel_start_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//
//       }// end 3x3 output rows
//     }//end First 3x3 Input Channel Block
//
//     // 3x3 Input channel blocks (N-2*C_ib updates)
//     for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib)
//     {
//       uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
//       uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//       float *filter_block_ptr = F + filter_i_c_block;
//
//       for(uint32_t l = 0; l < H_o - 1; l++){
//
//           uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
//           uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//           float * I_ptr = I + input_col_offset;
//
//           for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){
//
//             conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//             I_ptr += (W_ob * stride)*C_ob;
//         }
//         conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//
//       }
//       #if PREFETCH
//       __builtin_prefetch((const void*)(O_buffer),0,0);
//       #endif
//       uint32_t col_offset = (H_o - 1)*W_o*C_ob /*+ block_offset*/;
//       uint32_t input_col_offset = ((H_o - 1) * stride)*W_i*C_ob + input_block_offset;
//       float * I_ptr = I + input_col_offset;
//
//       for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){
//
//         conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//         I_ptr += (W_ob * stride)*C_ob;
//     }
//     conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//     }//end 3x3 Input channel blocks
//
//     // fused dw_conv
//     //Last 3x3 Input Channel Block
//     {
//       uint32_t input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
//       uint32_t filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//       float * filter_block_ptr = F + filter_i_c_block;
//
//       //First 3x3 Output Row
//       {
//         uint32_t col_offset = 0*W_o*C_ob;
//         uint32_t input_col_offset = (0 * stride)*W_i*C_ob + input_block_offset;
//
//         float * I_ptr = I + input_col_offset;
//
//         //first tile
//         {
//           conv_microkernel_dw_first_row_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
//                                                                         I_ptr, filter_block_ptr,
//                                                                          O_buffer + col_offset + 0 *C_ob,
//                                                                         O + dw_block_offset
//                                                                       );
//         }//end first tile
//
//         uint32_t dw_col_offset = dw_block_offset + 2*C_ob;
//         I_ptr += (W_ob*stride)*C_ob;
//
//         // First to Penultimate tile
//         for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
//         {
//
//           conv_microkernel_dw_first_row<stride*C_ob, H_f, W_f>(W_i*C_ib,
//                                       I_ptr,filter_block_ptr,
//                                       O_buffer + col_offset + k *C_ob,
//                                       O + dw_col_offset
//                                     );
//           dw_col_offset += 3*C_ob;
//           I_ptr += (W_ob*stride)*C_ob;
//         }
//         // end First to Penultimate tile
//
//         //last tile
//         {
//
//
//           conv_microkernel_dw_first_row_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
//                                       I_ptr, filter_block_ptr,
//                                       O_buffer + col_offset + (W_o - W_ob)*C_ob,
//                                       O + dw_col_offset
//                                     );
//
//         }//end last tile
//
//       }//end First 3x3 Output Row
//
//       //Second 3x3 Output Row
//       {
//         // 3x3 params
//         uint32_t col_offset = 1*W_o*C_ob;
//         uint32_t input_col_offset = (1 * stride)*W_i*C_ob + input_block_offset;
//         float * I_ptr = I + input_col_offset;
//
//         //first tile
//         {
//           conv_microkernel_dw_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
//                                           I_ptr, filter_block_ptr,
//                                           O_buffer + col_offset + 0*C_ob,
//                                           O + dw_block_offset
//                                         );
//         }//end first tile
//
//         uint32_t dw_col_offset = dw_block_offset + 2*C_ob;
//         I_ptr += (W_ob*stride)*C_ob;
//
//         //First to Penultimate tile
//         for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
//         {
//           conv_microkernel_dw_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
//                                       I_ptr,filter_block_ptr,
//                                       O_buffer + col_offset + k *C_ob,
//                                       O + dw_col_offset
//                                     );
//           dw_col_offset += 3*C_ob;
//           I_ptr += (W_ob*stride)*C_ob;
//         }
//         //end First to Penultimate Tile
//
//         //last tile
//         {
//
//           conv_microkernel_dw_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
//                                       I_ptr,filter_block_ptr,
//                                       O_buffer + col_offset + (W_o - W_ob)*C_ob,
//                                       O + dw_col_offset
//                                     );
//
//         }//end last tile
//       }//end Second 3x3 Output Row
//
//       uint32_t dw_row_offset = 0 + dw_block_offset;
//       uint32_t dw_row = 0;
//       //Third to Penultimate 3x3 Output Rows
//       for(uint32_t l = DW_STRIDE; l < H_o-(DW_KERNEL-1); l+= DW_STRIDE)
//       {
//         //Even 3x3 Output Row
//         {
//           uint32_t col_offset = l*W_o*C_ob;
//           uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//
//           float * I_ptr = I + input_col_offset;
//
//           //first tile
//           {
//             conv_microkernel_dw_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
//                                                                W_o_dw*C_ob,
//                                                                 I_ptr, filter_block_ptr,
//                                                                 O_buffer + col_offset + 0 *C_ob,
//                                                                 O + dw_row_offset
//                                                                   );
//           }//end first tile
//
//           uint32_t dw_col_offset = dw_row_offset + 2*C_ob;
//           I_ptr += (W_ob*stride)*C_ob;
//
//           //First to Penultimate tile
//           for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
//           {
//
//             conv_microkernel_dw<stride*C_ob, H_f, W_f>(W_i*C_ib,
//                                                          W_o_dw*C_ob,
//                                         I_ptr,filter_block_ptr, O_buffer + col_offset + k *C_ob,
//                                         O + dw_col_offset
//                                       );
//             dw_col_offset += 3*C_ob;
//             I_ptr += (W_ob*stride)*C_ob;
//           }
//           //end First to Penultimate tile
//
//           //last tile
//           {
//
//             conv_microkernel_dw_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
//                                       W_o_dw*C_ob,
//                                         I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
//                                         O + dw_col_offset
//                                       );
//
//           }
//           //end last tile
//         }//end Even 3x3 Output Row
//
//         dw_row_offset += W_o_dw*C_ib;
//         dw_row++;
//
//         //Odd 3x3 Output Row This should be the same as the Last row
//         {
//           // 3x3 params
//           uint32_t col_offset = (l + 1)*W_o*C_ob;
//           uint32_t input_col_offset = ((l + 1) * stride)*W_i*C_ob + input_block_offset;
//           float * I_ptr = I + input_col_offset;
//
//           //first tile
//           {
//             conv_microkernel_dw_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
//
//                                             I_ptr, filter_block_ptr, O_buffer + col_offset + 0*C_ob,
//                                             O + dw_row_offset
//                                           );
//           }//end first tile
//
//           uint32_t dw_col_offset = dw_row_offset + 2*C_ob;
//           I_ptr += (W_ob*stride)*C_ob;
//
//           //First to Penultimate tile
//           for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
//           {
//
//             conv_microkernel_dw_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
//                                         I_ptr,
//                                         filter_block_ptr, O_buffer + col_offset + k *C_ob,
//                                         O + dw_col_offset
//                                       );
//             dw_col_offset += 3*C_ob;
//             I_ptr += (W_ob*stride)*C_ob;
//           }
//           //end First to Penultimate Tile
//
//           //last tile
//           {
//             conv_microkernel_dw_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
//                                         I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
//                                         O + dw_col_offset
//                                       );
//
//           }//end last tile
//         }//end Odd 3x3 Output Row
//
//       }// end Third to Penultimate 3x3 Output Rows
//
//       // Last 3x3 Output Row
//       {// 3x3 params
//         #if PREFETCH
//         __builtin_prefetch((const void*)(O_buffer),0,0);
//         #endif
//         uint32_t col_offset = (dw_row*DW_STRIDE + DW_STRIDE)*W_o*C_ob;
//         uint32_t input_col_offset = ((dw_row*DW_STRIDE + DW_STRIDE) * stride)*W_i*C_ob + input_block_offset;
//
//         float * I_ptr = I + input_col_offset;
//         //first tile
//         {
//           conv_microkernel_dw_accum_start<stride*C_ob, H_f, W_f>(W_i*C_ib,
//                                           I_ptr, filter_block_ptr, O_buffer + col_offset + 0 *C_ob,
//                                           O + dw_row_offset
//                                         );
//         }
//         //end first tile
//
//         uint32_t dw_col_offset = dw_row_offset + 2*C_ob;
//         I_ptr += (W_ob*stride)*C_ob;
//
//         //First to Penultimate tile
//         for(uint32_t k = W_ob; k < (W_o - W_ob) ; k += W_ob)
//         {
//
//           conv_microkernel_dw_accum<stride*C_ob, H_f, W_f>(W_i*C_ib,
//
//                                       I_ptr,
//                                       filter_block_ptr, O_buffer + col_offset + k *C_ob,
//                                       O + dw_col_offset
//                                     );
//           dw_col_offset += 3*C_ob;
//           I_ptr += (W_ob*stride)*C_ob;
//         }
//         //end First to Penultimate Tile
//
//         //last tile
//         {
//
//           conv_microkernel_dw_accum_end<stride*C_ob, H_f, W_f>(W_i*C_ib,
//
//                                       I_ptr,filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob,
//                                       O + dw_col_offset
//                                     );
//
//         }//end last tile
//       }// end Last 3x3 Output Row
//     }// end Last 3x3 Input Channel Block
//   }//end 3x3 Output Channel Blocks
// }


template <uint32_t stride, uint32_t H_f, uint32_t W_f>
 void l_fused_dw_conv_direct_convolution(
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
  float * dw_F,
  float * O
)
{

  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);
  uint32_t W_o_dw = 0, H_o_dw = 0;
  assert(W_o >DW_KERNEL);
  op_dim(W_o, DW_STRIDE, DW_KERNEL, W_o_dw);
  op_dim(H_o, DW_STRIDE, DW_KERNEL, H_o_dw);

  H_o -= (H_o%2==0);
  // W_o -= (W_o%2==0);

  uint32_t W_o_int = (W_o_dw/(DW_W_ob))*DW_W_ob;
  uint32_t w_final = W_o_dw - (W_o_int);
  //3x3 Output Channel Block
  #if PARALLEL==1
  #pragma omp parallel for
  #endif
  for(uint32_t j = 0; j < C_o; j += C_ob)
  {
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
    uint32_t dw_block_offset = (j/C_ob)*H_o_dw*W_o_dw*C_ob;
    float * dw_filter_block_ptr = dw_F + (j/C_ob)*DW_KERNEL*DW_KERNEL*C_ob;
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

    // fused dw_conv
    //Last 3x3 Input Channel Block
    {
      uint32_t input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float * filter_block_ptr = F + filter_i_c_block;

      float * dw_filter_row = dw_filter_block_ptr;
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

        uint32_t dw_col_offset = dw_block_offset;

        // First to Penultimate tile


        float * dw_I_ptr = O_col;
        float * dw_O_ptr  = O + dw_col_offset;
        for(uint32_t k = 0; k < W_o_int ; k += DW_W_ob)
        {
          dw_first_row(
                          dw_I_ptr, // I
                          dw_filter_row, // F
                          dw_O_ptr // O
                        );
          dw_O_ptr += DW_W_ob*C_ob;
          dw_I_ptr += (DW_W_ob*DW_STRIDE)*C_ob;
        }
        // end dw_first_row to Penultimate tile

        //last tile

        if(w_final)
        {

          dw_first_row_end(
                              dw_I_ptr,
                              dw_filter_row,
                              w_final,
                              dw_O_ptr
                            );


        }//end last tile


      }//end First 3x3 Output Row

      dw_filter_row += DW_KERNEL*C_ob;
      //Second 3x3 Output Row
      {
        // 3x3 params
        uint32_t col_offset =  1*W_o*C_ob;
        uint32_t input_col_offset = (1 * stride)*W_i*C_ob + input_block_offset;
        float * I_ptr = I + input_col_offset;
        float * O_col = O_buffer + col_offset;
        for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
        {

          conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + k *C_ob);
          I_ptr += (W_ob * stride)*C_ob;
        }
        conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + (W_o - W_ob)*C_ob);



        uint32_t dw_col_offset = dw_block_offset;


        float * dw_I_ptr = O_col;
        float * dw_O_ptr = O + dw_col_offset;
        //First to Penultimate tile
        for(uint32_t k = 0; k < W_o_int ; k += DW_W_ob)
        {

          dw_accum(
                    dw_I_ptr, // I
                    dw_filter_row, // F
                    dw_O_ptr // O
                  );
            dw_O_ptr += DW_W_ob*C_ob;
            dw_I_ptr += (DW_W_ob*DW_STRIDE)*C_ob;
        }
        //end First to Penultimate Tile

        //last tile

        if(w_final)
        {

          dw_accum_end(
                        dw_I_ptr,
                        dw_filter_row,
                        w_final,
                        dw_O_ptr
                      );


        }//end last tile

      }//end Second 3x3 Output Row

      uint32_t dw_row_offset = 0 + dw_block_offset;
      uint32_t dw_row = 0;
      float * dw_filter_row_0 = dw_filter_block_ptr;
      float * dw_filter_row_1 = dw_filter_block_ptr + 1*(DW_KERNEL)*C_ob;
      float * dw_filter_row_2 = dw_filter_block_ptr + 2*(DW_KERNEL)*C_ob;
      //Third to Penultimate 3x3 Output Rows
      for(uint32_t l = DW_STRIDE; l < H_o-(DW_KERNEL-1); l+= DW_STRIDE)
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


          uint32_t dw_col_offset = dw_row_offset;

          float * dw_I_ptr = O_col;
          float * dw_O_ptr = O + dw_col_offset;

          //First to Penultimate tile
          for(uint32_t k = 0; k < W_o_int ; k += DW_W_ob)
          {

            dw(   W_o_dw*C_ob,
                  dw_I_ptr,
                  dw_filter_row_2,
                  dw_filter_row_0,
                  dw_O_ptr
                );

                dw_O_ptr += DW_W_ob*C_ob;
                dw_I_ptr += (DW_W_ob*DW_STRIDE)*C_ob;

          }
          //end First to Penultimate tile


          //last tile
          if(w_final)
          {


            dw_end(   W_o_dw*C_ob,
                      dw_I_ptr,
                      dw_filter_row_2,
                      dw_filter_row_0,
                      w_final,
                      dw_O_ptr
                    );

          }
          //end last tile
        }//end Even 3x3 Output Row

        dw_row_offset += W_o_dw*C_ob;
        dw_row++;

        //Odd 3x3 Output Row This should be the same as the Last row
        {

            // 3x3 params
            uint32_t col_offset = (l + 1)*W_o*C_ob;
            uint32_t input_col_offset = ((l + 1) * stride)*W_i*C_ob + input_block_offset;
            float * I_ptr = I + input_col_offset;
            float * O_col = O_buffer + col_offset;

            for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
            {

              conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + k *C_ob);
              I_ptr += (W_ob * stride)*C_ob;
            }
            conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + (W_o - W_ob)*C_ob);



            uint32_t dw_col_offset = dw_row_offset;


            float * dw_I_ptr = O_col;
            float * dw_O_ptr = O + dw_col_offset;

            //First to Penultimate tile
            for(uint32_t k = 0; k < W_o_int ; k += DW_W_ob)
            {

              dw_accum(
                        dw_I_ptr, // I
                        dw_filter_row_1, // F
                        dw_O_ptr // O
                      );
                dw_O_ptr += DW_W_ob*C_ob;
                dw_I_ptr += (DW_W_ob*DW_STRIDE)*C_ob;
            }

            //end First to Penultimate Tile

            //last tile
            if(w_final)
            {
              dw_accum_end(
                            dw_I_ptr,
                            dw_filter_row_1,
                            w_final,
                            dw_O_ptr
                          );

            }//end last tile

        }//end Odd 3x3 Output Row

      }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      {// 3x3 params
        #if PREFETCH
        __builtin_prefetch((const void*)(O_buffer),0,0);
        #endif
        uint32_t col_offset = (dw_row*DW_STRIDE + DW_STRIDE)*W_o*C_ob;
        uint32_t input_col_offset = ((dw_row*DW_STRIDE + DW_STRIDE) * stride)*W_i*C_ob + input_block_offset;
        float * O_col = O_buffer + col_offset;
        float * I_ptr = I + input_col_offset;

        for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
        {
          conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + k *C_ob);
          I_ptr += (W_ob * stride)*C_ob;
        }
        conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + (W_o - W_ob)*C_ob);


        uint32_t dw_col_offset = dw_row_offset;


        float * dw_I_ptr = O_col;
        float * dw_O_ptr = O + dw_col_offset;
        //First to Penultimate tile
        for(uint32_t k = 0; k < W_o_int ; k += DW_W_ob)
        {

          dw_accum(
                    dw_I_ptr, // I
                    dw_filter_row_2, // F
                    dw_O_ptr // O
                  );
            dw_O_ptr += DW_W_ob*C_ob;
            dw_I_ptr += (DW_W_ob*DW_STRIDE)*C_ob;
        }
        //end First to Penultimate Tile

        //last tile


        if(w_final)
        {

          dw_accum_end(
                        dw_I_ptr,
                        dw_filter_row_2,
                        w_final,
                        dw_O_ptr
                      );

        }//end last tile
      }// end Last 3x3 Output Row
    }// end Last 3x3 Input Channel Block
  }//end 3x3 Output Channel Blocks
}

void dw_conv(
    uint32_t C_o,
    uint32_t H_o,
    uint32_t W_o,
    float * I,
    float * dw_F,
    float * O
)
{

  uint32_t W_o_dw = 0, H_o_dw = 0;
  assert(W_o >DW_KERNEL);
  op_dim(W_o, DW_STRIDE, DW_KERNEL, W_o_dw);
  op_dim(H_o, DW_STRIDE, DW_KERNEL, H_o_dw);

  uint32_t input_block_offset = (W_o)*(H_o)*C_ob;
  H_o -= (H_o%2==0);
  // W_o -= (W_o%2==0);

  uint32_t W_o_int = (W_o_dw/(DW_W_ob))*DW_W_ob;
  uint32_t w_final = W_o_dw - (W_o_int);
  //3x3 Output Channel Block
  #if PARALLEL==1
  // printf("parallel\n");
  #pragma omp parallel for
  #endif
  for(uint32_t j = 0; j < C_o; j += C_ob)
  {
    uint32_t dw_block_offset = (j/C_ob)*H_o_dw*W_o_dw*C_ob;
    float * dw_filter_block_ptr = dw_F + (j/C_ob)*DW_KERNEL*DW_KERNEL*C_ob;
    // #if PARALLEL==1
    // int tid = omp_get_thread_num();
    // float * O_buffer = O_buffers + (tid)*(H_o*W_o*(C_ob));
    // #else
    // float * O_buffer = O_buffers;
    // #endif
    float * O_buffer = I + (j/C_ob)*input_block_offset;

    //dw_conv
    {

      float * dw_filter_row = dw_filter_block_ptr;
      //First 3x3 Output Row
      {
        uint32_t col_offset = 0*W_o*C_ob;
        float * O_col = O_buffer  + col_offset;

        uint32_t dw_col_offset = dw_block_offset;

        // First to Penultimate tile


        float * dw_I_ptr = O_col;
        float * dw_O_ptr  = O + dw_col_offset;
        for(uint32_t k = 0; k < W_o_int ; k += DW_W_ob)
        {
          dw_first_row(
                          dw_I_ptr, // I
                          dw_filter_row, // F
                          dw_O_ptr // O
                        );
          dw_O_ptr += DW_W_ob*C_ob;
          dw_I_ptr += (DW_W_ob*DW_STRIDE)*C_ob;
        }
        // end dw_first_row to Penultimate tile

        //last tile

        if(w_final)
        {

          dw_first_row_end(
                              dw_I_ptr,
                              dw_filter_row,
                              w_final,
                              dw_O_ptr
                            );


        }//end last tile


      }//end First 3x3 Output Row

      dw_filter_row += DW_KERNEL*C_ob;
      //Second 3x3 Output Row
      {
        // 3x3 params
        uint32_t col_offset =  1*W_o*C_ob;
        float * O_col = O_buffer + col_offset;

        uint32_t dw_col_offset = dw_block_offset;


        float * dw_I_ptr = O_col;
        float * dw_O_ptr = O + dw_col_offset;
        //First to Penultimate tile
        for(uint32_t k = 0; k < W_o_int ; k += DW_W_ob)
        {

          dw_accum(
                    dw_I_ptr, // I
                    dw_filter_row, // F
                    dw_O_ptr // O
                  );
            dw_O_ptr += DW_W_ob*C_ob;
            dw_I_ptr += (DW_W_ob*DW_STRIDE)*C_ob;
        }
        //end First to Penultimate Tile

        //last tile

        if(w_final)
        {

          dw_accum_end(
                        dw_I_ptr,
                        dw_filter_row,
                        w_final,
                        dw_O_ptr
                      );


        }//end last tile

      }//end Second 3x3 Output Row

      uint32_t dw_row_offset = 0 + dw_block_offset;
      uint32_t dw_row = 0;
      float * dw_filter_row_0 = dw_filter_block_ptr;
      float * dw_filter_row_1 = dw_filter_block_ptr + 1*(DW_KERNEL)*C_ob;
      float * dw_filter_row_2 = dw_filter_block_ptr + 2*(DW_KERNEL)*C_ob;
      //Third to Penultimate 3x3 Output Rows
      for(uint32_t l = DW_STRIDE; l < H_o-(DW_KERNEL-1); l+= DW_STRIDE)
      {

        //Even 3x3 Output Row
        {
          uint32_t col_offset = l*W_o*C_ob;
          float * O_col = O_buffer +  col_offset;


          uint32_t dw_col_offset = dw_row_offset;

          float * dw_I_ptr = O_col;
          float * dw_O_ptr = O + dw_col_offset;

          //First to Penultimate tile
          for(uint32_t k = 0; k < W_o_int ; k += DW_W_ob)
          {

            dw(   W_o_dw*C_ob,
                  dw_I_ptr,
                  dw_filter_row_2,
                  dw_filter_row_0,
                  dw_O_ptr
                );

                dw_O_ptr += DW_W_ob*C_ob;
                dw_I_ptr += (DW_W_ob*DW_STRIDE)*C_ob;

          }
          //end First to Penultimate tile


          //last tile
          if(w_final)
          {


            dw_end(   W_o_dw*C_ob,
                      dw_I_ptr,
                      dw_filter_row_2,
                      dw_filter_row_0,
                      w_final,
                      dw_O_ptr
                    );

          }
          //end last tile
        }//end Even 3x3 Output Row

        dw_row_offset += W_o_dw*C_ob;
        dw_row++;

        //Odd 3x3 Output Row This should be the same as the Last row
        {

            // 3x3 params
            uint32_t col_offset = (l + 1)*W_o*C_ob;
            float * O_col = O_buffer + col_offset;



            uint32_t dw_col_offset = dw_row_offset;


            float * dw_I_ptr = O_col;
            float * dw_O_ptr = O + dw_col_offset;

            //First to Penultimate tile
            for(uint32_t k = 0; k < W_o_int ; k += DW_W_ob)
            {

              dw_accum(
                        dw_I_ptr, // I
                        dw_filter_row_1, // F
                        dw_O_ptr // O
                      );
                dw_O_ptr += DW_W_ob*C_ob;
                dw_I_ptr += (DW_W_ob*DW_STRIDE)*C_ob;
            }

            //end First to Penultimate Tile

            //last tile

            if(w_final)
            {
              dw_accum_end(
                            dw_I_ptr,
                            dw_filter_row_1,
                            w_final,
                            dw_O_ptr
                          );

            }//end last tile

        }//end Odd 3x3 Output Row

      }// end Third to Penultimate 3x3 Output Rows

      // Last 3x3 Output Row
      {// 3x3 params
        #if PREFETCH
        __builtin_prefetch((const void*)(O_buffer),0,0);
        #endif
        uint32_t col_offset = (dw_row*DW_STRIDE + DW_STRIDE)*W_o*C_ob;
        float * O_col = O_buffer + col_offset;

        uint32_t dw_col_offset = dw_row_offset;


        float * dw_I_ptr = O_col;
        float * dw_O_ptr = O + dw_col_offset;
        //First to Penultimate tile
        for(uint32_t k = 0; k < W_o_int ; k += DW_W_ob)
        {

          dw_accum(
                    dw_I_ptr, // I
                    dw_filter_row_2, // F
                    dw_O_ptr // O
                  );
            dw_O_ptr += DW_W_ob*C_ob;
            dw_I_ptr += (DW_W_ob*DW_STRIDE)*C_ob;
        }
        //end First to Penultimate Tile

        //last tile


        if(w_final)
        {

          dw_accum_end(
                        dw_I_ptr,
                        dw_filter_row_2,
                        w_final,
                        dw_O_ptr
                      );

        }//end last tile
      }// end Last 3x3 Output Row
    }// end Last 3x3 Input Channel Block
  }//end 3x3 Output Channel Blocks
}

// template <uint32_t stride, uint32_t H_f, uint32_t W_f>
//  void j_fused_dw_conv_direct_convolution(
//   uint32_t C_i,
//   uint32_t C_o,
//   // uint32_t H_f,
//   // uint32_t W_f,
//   uint32_t H_i,
//   uint32_t W_i,
//   // uint32_t stride,
//   float * I,
//   float * F,
//   float * O_buffers,
//   float * O
// ){
//
//   uint32_t H_o = 0;
//   op_dim(H_i, stride,H_f,H_o);
//   uint32_t W_o = 0;
//   op_dim(W_i, stride,W_f,W_o);
//   uint32_t W_o_dw = 0, H_o_dw = 0;
//   assert(W_o >DW_KERNEL);
//   op_dim(W_o, DW_STRIDE, DW_KERNEL, W_o_dw);
//   op_dim(H_o, DW_STRIDE, DW_KERNEL, H_o_dw);
//
//   H_o -= (H_o%2==0);
//   // W_o -= (W_o%2==0);
//
//   //3x3 Output Channel Block
//   #if PARALLEL==1
//   #pragma omp parallel for
//   #endif
//   for(uint32_t j = 0; j < C_o; j += C_ob)
//   {
//     uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
//     uint32_t dw_block_offset = (j/C_ob)*H_o_dw*W_o_dw*C_ob;
//     #if PARALLEL==1
//     int tid = omp_get_thread_num();
//     float * O_buffer = O_buffers + (tid)*(H_o*W_o*(C_ob));
//     #else
//     float * O_buffer = O_buffers;
//     #endif
//     //First 3x3 Input Channel Block
//     {
//       // These are all 0
//     uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
//     uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//
//     float * filter_block_ptr = F + filter_i_c_block;
//       //3x3 Output Rows
//       for(uint32_t l = 0; l < H_o; l++)
//       {
//
//           uint32_t col_offset = l*W_o*C_ob ;
//           uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//           float * I_ptr = I + input_col_offset;
//
//           // 3x3 pixel blocks
//           for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
//           {
//
//
//             conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//             I_ptr += (W_ob*stride)*C_ob;
//           }// end 3x3 pixel blocks
//
//           conv_microkernel_start_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//
//       }// end 3x3 output rows
//     }//end First 3x3 Input Channel Block
//
//     // 3x3 Input channel blocks (N-2*C_ib updates)
//     for(uint32_t i = C_ib; i < C_i; i += C_ib)
//     {
//       uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
//       uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//       float *filter_block_ptr = F + filter_i_c_block;
//
//       for(uint32_t l = 0; l < H_o - 1; l++){
//
//           uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
//           uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//           float * I_ptr = I + input_col_offset;
//
//           for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){
//
//             conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//             I_ptr += (W_ob * stride)*C_ob;
//         }
//         conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//
//       }
//       #if PREFETCH
//       __builtin_prefetch((const void*)(O_buffer),0,0);
//       #endif
//       uint32_t col_offset = (H_o - 1)*W_o*C_ob /*+ block_offset*/;
//       uint32_t input_col_offset = ((H_o - 1) * stride)*W_i*C_ob + input_block_offset;
//       float * I_ptr = I + input_col_offset;
//
//       for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){
//
//         conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//         I_ptr += (W_ob * stride)*C_ob;
//     }
//     conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//     }//end 3x3 Input channel blocks
//
//     // fused dw_conv
//     //Last 3x3 Input Channel Block
//     {
//
//
//       //First 3x3 Output Row
//       {
//         uint32_t col_offset = 0*W_o*C_ob;
//         float * O_col = O_buffer  + col_offset;
//
//         //first tile
//         uint32_t dw_col_offset = dw_block_offset;
//
//         // First to Penultimate tile
//         for(uint32_t k = 0; k < (W_o - W_ob) ; k += W_ob)
//         {
//
//           dw_first_row(
//                           O_col + k *C_ob,
//                           O + dw_col_offset
//                         );
//           dw_col_offset += 3*C_ob;
//         }
//         // end First to Penultimate tile
//
//         //last tile
//         {
//
//
//           dw_first_row_end(
//                               O_col + (W_o - W_ob)*C_ob,
//                               O + dw_col_offset
//                             );
//
//         }//end last tile
//
//       }//end First 3x3 Output Row
//
//       //Second 3x3 Output Row
//       {
//         // 3x3 params
//         uint32_t col_offset =  1*W_o*C_ob;
//         float * O_col = O_buffer + col_offset;
//         uint32_t dw_col_offset = dw_block_offset;
//
//
//         //First to Penultimate tile
//         for(uint32_t k = 0; k < (W_o - W_ob) ; k += W_ob)
//         {
//
//         dw_accum(
//                     O_col + k *C_ob,
//                     O + dw_col_offset
//                   );
//           dw_col_offset += 3*C_ob;
//         }
//         //end First to Penultimate Tile
//
//         //last tile
//         {
//           dw_accum_end(
//                             O_col + (W_o - W_ob)*C_ob,
//                             O + dw_col_offset
//                         );
//
//         }//end last tile
//       }//end Second 3x3 Output Row
//
//       uint32_t dw_row_offset = 0 + dw_block_offset;
//       uint32_t dw_row = 0;
//       //Third to Penultimate 3x3 Output Rows
//       for(uint32_t l = DW_STRIDE; l < H_o-(DW_KERNEL-1); l+= DW_STRIDE)
//       {
//         //Even 3x3 Output Row
//         {
//           uint32_t col_offset = l*W_o*C_ob;
//           float * O_col = O_buffer +  col_offset;
//           uint32_t dw_col_offset = dw_row_offset;
//           //First to Penultimate tile
//           for(uint32_t k = 0; k < (W_o - W_ob) ; k += W_ob)
//           {
//
//             dw( W_o_dw*C_ob,
//                   O_col + k *C_ob,
//                   O + dw_col_offset
//                 );
//             dw_col_offset += 3*C_ob;
//
//           }
//           //end First to Penultimate tile
//
//           //last tile
//           {
//
//             dw_end(   W_o_dw*C_ob,
//                         O_col + (W_o - W_ob)*C_ob,
//                         O + dw_col_offset
//                     );
//
//           }
//           //end last tile
//         }//end Even 3x3 Output Row
//
//         dw_row_offset += W_o_dw*C_ib;
//         dw_row++;
//
//         //Odd 3x3 Output Row This should be the same as the Last row
//         {
//
//             // 3x3 params
//             uint32_t col_offset = (l + 1)*W_o*C_ob;
//             float * O_col = O_buffer + col_offset;
//
//
//
//
//
//             uint32_t dw_col_offset = dw_row_offset ;
//
//             //First to Penultimate tile
//             for(uint32_t k = 0; k < (W_o - W_ob) ; k += W_ob)
//             {
//
//             dw_accum(
//                         O_col + k *C_ob,
//                         O + dw_col_offset
//                       );
//               dw_col_offset += 3*C_ob;
//             }
//             //end First to Penultimate Tile
//
//             //last tile
//             {
//               dw_accum_end(
//                                 O_col + (W_o - W_ob)*C_ob,
//                                 O + dw_col_offset
//                             );
//
//             }//end last tile
//         }//end Odd 3x3 Output Row
//
//       }// end Third to Penultimate 3x3 Output Rows
//
//       // Last 3x3 Output Row
//       {// 3x3 params
//         #if PREFETCH
//         __builtin_prefetch((const void*)(O_buffer),0,0);
//         #endif
//         uint32_t col_offset = (dw_row*DW_STRIDE + DW_STRIDE)*W_o*C_ob;
//         float * O_col = O_buffer + col_offset;
//
//         uint32_t dw_col_offset = dw_row_offset;
//
//
//         //First to Penultimate tile
//         for(uint32_t k = 0; k < (W_o - W_ob) ; k += W_ob)
//         {
//
//         dw_accum(
//                     O_col + k *C_ob,
//                     O + dw_col_offset
//                   );
//           dw_col_offset += 3*C_ob;
//         }
//         //end First to Penultimate Tile
//
//         //last tile
//         {
//           dw_accum_end(
//                             O_col + (W_o - W_ob)*C_ob,
//                             O + dw_col_offset
//                         );
//
//         }//end last tile
//       }// end Last 3x3 Output Row
//     }// end Last 3x3 Input Channel Block
//   }//end 3x3 Output Channel Blocks
// }


// template <uint32_t stride, uint32_t H_f, uint32_t W_f>
//  void j_fused_H_tile_dw_conv_direct_convolution(
//   uint32_t C_i,
//   uint32_t C_o,
//   // uint32_t H_f,
//   // uint32_t W_f,
//   uint32_t H_i,
//   uint32_t W_i,
//   // uint32_t stride,
//   float * I,
//   float * F,
//   float * O_buffers,
//   float * O
// ){
//
//   uint32_t H_o = 0;
//   op_dim(H_i, stride,H_f,H_o);
//   uint32_t W_o = 0;
//   op_dim(W_i, stride,W_f,W_o);
//   uint32_t W_o_dw = 0, H_o_dw = 0;
//   assert(W_o >DW_KERNEL);
//   op_dim(W_o, DW_STRIDE, DW_KERNEL, W_o_dw);
//   op_dim(H_o, DW_STRIDE, DW_KERNEL, H_o_dw);
//
//   H_o -= (H_o%2==0);
//   // W_o -= (W_o%2==0);
//
//   //3x3 Output Channel Block
//   #if PARALLEL==1
//   #pragma omp parallel for
//   #endif
//   for(uint32_t j = 0; j < C_o; j += C_ob)
//   {
//     uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
//     uint32_t dw_block_offset = (j/C_ob)*H_o_dw*W_o_dw*C_ob;
//     #if PARALLEL==1
//     int tid = omp_get_thread_num();
//     float * O_buffer = O_buffers + (tid)*(H_o*W_o*(C_ob));
//     #else
//     float * O_buffer = O_buffers;
//     #endif
//
//
//     float *O_dw = O + dw_block_offset;
//     //First 3x3 Input Channel Block
//     {
//       // These are all 0
//     uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
//     uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//
//     float * filter_block_ptr = F + filter_i_c_block;
//       //3x3 Output Rows
//       for(uint32_t l = 0; l < H_o; l++)
//       {
//
//           uint32_t col_offset = l*W_o*C_ob ;
//           uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//           float * I_ptr = I + input_col_offset;
//
//           // 3x3 pixel blocks
//           for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
//           {
//
//
//             conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//             I_ptr += (W_ob*stride)*C_ob;
//           }// end 3x3 pixel blocks
//
//           conv_microkernel_start_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//
//       }// end 3x3 output rows
//     }//end First 3x3 Input Channel Block
//
//     // 3x3 Input channel blocks (N-2*C_ib updates)
//     for(uint32_t i = C_ib; i < C_i; i += C_ib)
//     {
//       uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
//       uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//       float *filter_block_ptr = F + filter_i_c_block;
//
//       for(uint32_t l = 0; l < H_o - 1; l++){
//
//           uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
//           uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//           float * I_ptr = I + input_col_offset;
//
//           for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){
//
//             conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//             I_ptr += (W_ob * stride)*C_ob;
//         }
//         conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//
//       }
//       #if PREFETCH
//       __builtin_prefetch((const void*)(O_buffer),0,0);
//       #endif
//       uint32_t col_offset = (H_o - 1)*W_o*C_ob /*+ block_offset*/;
//       uint32_t input_col_offset = ((H_o - 1) * stride)*W_i*C_ob + input_block_offset;
//       float * I_ptr = I + input_col_offset;
//
//       for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){
//
//         conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//         I_ptr += (W_ob * stride)*C_ob;
//     }
//     conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//     }//end 3x3 Input channel blocks
//
//     // fused dw_conv
//     //Last 3x3 Input Channel Block
//     {

//
//
//
//
//         dw_3_rows<DW_STRIDE*C_ob>(W_o*C_ob, W_o, O_buffer, O_dw);
//         uint32_t dw_row_offset = W_o_dw*C_ob + dw_block_offset;
//         uint32_t dw_row = 1;
//         float * O_conv_row = O_buffer + (dw_row*DW_STRIDE)*W_o*C_ob;
//         float * O_dw_row = O_dw + dw_row*W_o_dw*C_ob;
//         //Third to Penultimate 3x3 Output Rows
//         for(uint32_t l = DW_KERNEL; l < H_o ; l+= DW_STRIDE)
//         {
//           dw_3_rows<DW_STRIDE*C_ob>(W_o*C_ob, W_o, O_conv_row, O_dw_row);
//           dw_row_offset += W_o_dw*C_ib;
//           dw_row++;
//           O_dw_row += W_o_dw*C_ib;
//           O_conv_row += (DW_STRIDE)*W_o*C_ob;
//
//         }// end Third to Penultimate 3x3 Output Rows
//
//
//     }// end Last 3x3 Input Channel Block
//   }//end 3x3 Output Channel Blocks
// }

// template <uint32_t stride, uint32_t H_f, uint32_t W_f>
//  void H_tile_fused_dw_conv_direct_convolution(
//   uint32_t C_i,
//   uint32_t C_o,
//   // uint32_t H_f,
//   // uint32_t W_f,
//   uint32_t H_i,
//   uint32_t W_i,
//   // uint32_t stride,
//   float * I,
//   float * F,
//   float * O_buffers,
//   float * O
// ){
//
//   uint32_t H_o = 0;
//   op_dim(H_i, stride,H_f,H_o);
//   uint32_t W_o = 0;
//   op_dim(W_i, stride,W_f,W_o);
//   uint32_t W_o_dw = 0, H_o_dw = 0;
//   assert(W_o >DW_KERNEL);
//   op_dim(W_o, DW_STRIDE, DW_KERNEL, W_o_dw);
//   op_dim(H_o, DW_STRIDE, DW_KERNEL, H_o_dw);
//
//   H_o -= (H_o%2==0);
//   // W_o -= (W_o%2==0);
//
//   //3x3 Output Channel Block
//   #if PARALLEL==1
//   #pragma omp parallel for
//   #endif
//   for(uint32_t j = 0; j < C_o; j += C_ob)
//   {
//     uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
//     uint32_t dw_block_offset = (j/C_ob)*H_o_dw*W_o_dw*C_ob;
//     float * O_dw = O + dw_block_offset;
//     #if PARALLEL==1
//     int tid = omp_get_thread_num();
//     float * O_buffer = O_buffers + (tid)*(H_o*W_o*(C_ob));
//     #else
//     float * O_buffer = O_buffers;
//     #endif
//     //First 3x3 Input Channel Block
//     {
//       // These are all 0
//     uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
//     uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//
//     float * filter_block_ptr = F + filter_i_c_block;
//       //3x3 Output Rows
//       for(uint32_t l = 0; l < H_o; l++)
//       {
//
//           uint32_t col_offset = l*W_o*C_ob ;
//           uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//           float * I_ptr = I + input_col_offset;
//
//           // 3x3 pixel blocks
//           for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
//           {
//
//
//             conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//             I_ptr += (W_ob*stride)*C_ob;
//           }// end 3x3 pixel blocks
//
//           conv_microkernel_start_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//
//       }// end 3x3 output rows
//     }//end First 3x3 Input Channel Block
//
//     // 3x3 Input channel blocks (N-2*C_ib updates)
//     for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib)
//     {
//       uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
//       uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//       float *filter_block_ptr = F + filter_i_c_block;
//
//       for(uint32_t l = 0; l < H_o - 1; l++){
//
//           uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
//           uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//           float * I_ptr = I + input_col_offset;
//
//           for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){
//
//             conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//             I_ptr += (W_ob * stride)*C_ob;
//         }
//         conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//
//       }
//       #if PREFETCH
//       __builtin_prefetch((const void*)(O_buffer),0,0);
//       #endif
//       uint32_t col_offset = (H_o - 1)*W_o*C_ob /*+ block_offset*/;
//       uint32_t input_col_offset = ((H_o - 1) * stride)*W_i*C_ob + input_block_offset;
//       float * I_ptr = I + input_col_offset;
//
//       for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){
//
//         conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//         I_ptr += (W_ob * stride)*C_ob;
//     }
//     conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//     }//end 3x3 Input channel blocks
//
//     // fused dw_conv
//     //Last 3x3 Input Channel Block
//     {
//       uint32_t input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
//       uint32_t filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//       float * filter_block_ptr = F + filter_i_c_block;
//
//       //First 3 conv Output Rows
//       {
//
//         for(uint32_t l = 0; l < 3; l++){
//           uint32_t col_offset = l*W_o*C_ob;
//           uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//           float * O_col = O_buffer  + col_offset;
//           float * I_ptr = I + input_col_offset;
//
//
//             for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
//             {
//
//               conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + k *C_ob);
//               I_ptr += (W_ob * stride)*C_ob;
//             }
//             conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + (W_o - W_ob)*C_ob);
//
//
//           //first tile
//
//           uint32_t dw_col_offset = dw_block_offset;
//
//         }
//
//         dw_3_rows<DW_STRIDE*C_ob>(W_o*C_ob, W_o, O_buffer, O_dw);
//
//       }//end 3 conv Output Rows
//
//
//
//       uint32_t dw_row_offset = dw_block_offset + W_o_dw*C_ob;
//       uint32_t dw_row = 1;
//       float * O_dw_row = O + dw_row_offset;
//       float * O_conv_row = O_buffer + (dw_row * DW_STRIDE)*W_o*C_ob;
//       //Third to last 3x3 Output Rows
//       for(uint32_t l = DW_KERNEL; l < H_o; l+= DW_STRIDE)
//       {
//
//          //conv 2 rows
//         for(uint32_t ll = 0; ll < DW_STRIDE; ll++)
//         {
//           uint32_t col_offset = (l+ll)*W_o*C_ob;
//           uint32_t input_col_offset = ((l+ll) * stride)*W_i*C_ob + input_block_offset;
//           float * O_col = O_buffer +  col_offset;
//           float * I_ptr = I + input_col_offset;
//
//           for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
//           {
//
//             conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + k *C_ob);
//             I_ptr += (W_ob * stride)*C_ob;
//           }
//           conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_col + (W_o - W_ob)*C_ob);
//         }// end conv 2 more rows
//
//         //dw 3 rows starting with the conv output from previous iteration
//         dw_3_rows<DW_STRIDE*C_ob>(W_o*C_ob, W_o, O_conv_row, O_dw_row);
//
//
//         dw_row_offset += W_o_dw*C_ob;
//         O_conv_row += (DW_STRIDE)*W_o*C_ob;
//         O_dw_row += W_o_dw*C_ob;
//         dw_row++;
//
//
//       }// end Third to Last 3x3 Output Rows
//
//
//
//     }// end Last 3x3 Input Channel Block
//   }//end 3x3 Output Channel Blocks
// }

// template <uint32_t stride, uint32_t H_f, uint32_t W_f>
//  void H_loop_fused_dw_conv_direct_convolution(
//   uint32_t C_i,
//   uint32_t C_o,
//   // uint32_t H_f,
//   // uint32_t W_f,
//   uint32_t H_i,
//   uint32_t W_i,
//   // uint32_t stride,
//   float * I,
//   float * F,
//   float * O_buffers,
//   float * O
// ){
//
//   uint32_t H_o = 0;
//   op_dim(H_i, stride,H_f,H_o);
//   uint32_t W_o = 0;
//   op_dim(W_i, stride,W_f,W_o);
//   uint32_t W_o_dw = 0, H_o_dw = 0;
//   assert(W_o >DW_KERNEL);
//   op_dim(W_o, DW_STRIDE, DW_KERNEL, W_o_dw);
//   op_dim(H_o, DW_STRIDE, DW_KERNEL, H_o_dw);
//
//   H_o -= (H_o%2==0);
//   // W_o -= (W_o%2==0);
//
//   //3x3 Output Channel Block
//   #if PARALLEL==1
//   #pragma omp parallel for
//   #endif
//   for(uint32_t j = 0; j < C_o; j += C_ob)
//   {
//     uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
//     uint32_t dw_block_offset = (j/C_ob)*H_o_dw*W_o_dw*C_ob;
//     float * O_dw = O + dw_block_offset;
//     #if PARALLEL==1
//     int tid = omp_get_thread_num();
//     float * O_buffer = O_buffers + (tid)*(H_o*W_o*(C_ob));
//     #else
//     float * O_buffer = O_buffers;
//     #endif
//     //Start H_tile
//     {
//       //First 3x3 Input Channel Block
//       {
//         // These are all 0
//       uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
//       uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//
//       float * filter_block_ptr = F + filter_i_c_block;
//         //3x3 Output Rows
//         for(uint32_t l = 0; l < DW_KERNEL; l++)
//         {
//
//             uint32_t col_offset = l*W_o*C_ob ;
//             uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//             float * I_ptr = I + input_col_offset;
//
//             // 3x3 pixel blocks
//             for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
//             {
//
//
//               conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//               I_ptr += (W_ob*stride)*C_ob;
//             }// end 3x3 pixel blocks
//
//             conv_microkernel_start_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//
//         }// end 3x3 output rows
//       }//end First 3x3 Input Channel Block
//
//       // 3x3 Input channel blocks (N-2*C_ib updates)
//       for(uint32_t i = C_ib; i < C_i; i += C_ib)
//       {
//         uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
//         uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//         float *filter_block_ptr = F + filter_i_c_block;
//
//         for(uint32_t l = 0; l < DW_KERNEL; l++){
//
//             uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
//             uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
//             float * I_ptr = I + input_col_offset;
//
//             for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){
//
//               conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
//               I_ptr += (W_ob * stride)*C_ob;
//           }
//           conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + (W_o - W_ob)*C_ob);
//
//         }
//
//       }//end 3x3 Input channel blocks
//
//     // fused dw_conv
//       dw_3_rows<DW_STRIDE*C_ob>(W_o*C_ob, W_o, O_buffer, O_dw);
//     }//end First H_tile
//     float * O_dw_row = O_dw + W_o_dw*C_ob;
//     uint32_t dw_row = 1;
//     for(uint32_t l = DW_KERNEL; l < H_o; l+=DW_STRIDE)
//     {
//       uint32_t offsets[2];
//       uint32_t buffer_offset_0 = (dw_row+1)%3;
//       offsets[0] = (dw_row+2)%3;
//       offsets[1] = (dw_row+3)%3;

//       //First 3x3 Input Channel Block
//       {
//         // These are all 0
//         uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
//         uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//
//         float * filter_block_ptr = F + filter_i_c_block;
//           //3x3 Output Rows
//           for(uint32_t ll = 0; ll < DW_STRIDE; ll++)
//           {
//
//               uint32_t col_offset = offsets[ll]*W_o*C_ob ;
//               uint32_t input_col_offset = ((l+ll) * stride)*W_i*C_ob + input_block_offset;
//               float * I_ptr = I + input_col_offset;
//               float * O_row = O_buffer + col_offset;
//               // 3x3 pixel blocks
//               for(uint32_t k = 0; k < W_o - W_ob; k += W_ob)
//               {
//
//
//                 conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_row+ k *C_ob);
//                 I_ptr += (W_ob*stride)*C_ob;
//               }// end 3x3 pixel blocks
//
//               conv_microkernel_start_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_row + (W_o - W_ob)*C_ob);
//
//           }// end 3x3 output rows
//         }//end First 3x3 Input Channel Block
//
//       // 3x3 Input channel blocks (N-2*C_ib updates)
//       for(uint32_t i = C_ib; i < C_i; i += C_ib)
//       {
//         uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
//         uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
//         float *filter_block_ptr = F + filter_i_c_block;
//
//         for(uint32_t ll = 0; ll < DW_STRIDE; ll++)
//         {
//
//             uint32_t col_offset = offsets[ll]*W_o*C_ob /*+ block_offset*/;
//             uint32_t input_col_offset = ((l+ll) * stride)*W_i*C_ob + input_block_offset;
//             float * I_ptr = I + input_col_offset;
//             float * O_row = O_buffer + col_offset;
//             for(uint32_t k = 0; k < W_o - W_ob; k += W_ob){
//
//               conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_row + k *C_ob);
//               I_ptr += (W_ob * stride)*C_ob;
//           }
//           conv_microkernel_end<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_row + (W_o - W_ob)*C_ob);
//
//         }
//
//       }//end 3x3 Input channel blocks
//
//       float * conv_row_0 = O_buffer + buffer_offset_0*W_o*C_ob;
//       float * conv_row_1 = O_buffer + offsets[0]*W_ob*C_ob;
//       float * conv_row_2 = O_buffer + offsets[1]*W_ob*C_ob;

//       dw_3_rows_strided<DW_STRIDE*C_ob>(conv_row_0, conv_row_1, conv_row_2, W_o, O_dw_row);
//       O_dw_row += W_o_dw*C_ob;
//       dw_row++;
//     }
//   }//end 3x3 Output Channel Blocks
// }

// asssumes C_i > 32
// 3x3 output not fully computed, use intermediate output of full size (H_o*W_*C_o)





// void H_tile_dw_conv(
//     uint32_t C,
//     uint32_t H_o,
//     uint32_t W_o,
//     float * I,
//     float * O
// )
// {
//   const int w_block = 1;
//   uint32_t s = DW_STRIDE;
//   uint32_t W_o_dw , H_o_dw;
//   op_dim(W_o, DW_STRIDE, DW_KERNEL, W_o_dw);
//   op_dim(H_o, DW_STRIDE, DW_KERNEL, H_o_dw);
//
//   // H_o -= (H_o%2==0);
//   uint32_t offset = 0;
//   #if PARALLEL==1
//   #pragma omp parallel for
//   #endif
//   for(uint32_t i = 0; i < C; i+=C_ob)
//   {
//     uint32_t block_offset = (i/C_ob) * H_o * W_o * C_ob;
//     float * O_buffer  = I + block_offset;
//     uint32_t dw_block_offset = (i/C_ob)*H_o_dw*W_o_dw*C_ob;
//     float *O_dw = O + dw_block_offset;
//
//
//     {
//
//
//       dw_3_rows<DW_STRIDE*C_ob>(W_o*C_ob, W_o, O_buffer, O_dw);
//       uint32_t dw_row_offset = W_o_dw*C_ob + dw_block_offset;
//       uint32_t dw_row = 1;
//       float * O_conv_row = O_buffer + (dw_row*DW_STRIDE)*W_o*C_ob;
//       float * O_dw_row = O_dw + dw_row*W_o_dw*C_ob;
//       //Third to Penultimate 3x3 Output Rows
//       for(uint32_t l = DW_KERNEL; l < H_o ; l+= DW_STRIDE)
//       {
//         dw_3_rows<DW_STRIDE*C_ob>(W_o*C_ob, W_o, O_conv_row, O_dw_row);
//         dw_row_offset += W_o_dw*C_ib;
//         dw_row++;
//         O_dw_row += W_o_dw*C_ib;
//         O_conv_row += (DW_STRIDE)*W_o*C_ob;
//
//       }// end Third to Penultimate 3x3 Output Rows
//
//     }
//   }
// }
