// this file implements simple direct convolution with no fusion

#include <stdint.h>
#include "kernel_naive.h"

// #define op_dim(IN_dim, stride, K_dim, OUT_dim)   \
//     {                                            \
//         OUT_dim = (IN_dim - K_dim) / stride + 1; \
//     }
// Assume padding to maintain the same input
// assumes channel stride = C_f
// assumes output initialized to 0
template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t stride, char op>
void direct_convolution_naive(
    uint32_t channel_stride,
    uint32_t H_f,
    uint32_t W_f,
    uint32_t C_f,
    uint32_t C_o,
    uint32_t G,
    uint32_t H_i,
    uint32_t W_i,
    float *I,
    float *F,
    float *O
    )
{
    uint32_t H_o = 0;
    op_dim(H_i, stride, H_f, H_o);
    uint32_t W_o_full = 0;
    op_dim(W_i, stride, W_f, W_o_full);
    uint32_t W_o = (W_o_full / _W_ob) * _W_ob;
    uint32_t W_last = W_o_full % _W_ob;
// printf("W_of_full: %d W_o : %d W_last: %d\n ", W_o_full, W_o, W_last);

// printf(" input dims : %d %d \n", H_i, W_i);
#if PARALLEL == 1
#pragma omp parallel for
#endif
    for (uint32_t j = 0; j < C_o * G; j += _C_ob)
    {

        uint32_t block_offset = (j / _C_ob) * H_o * W_o_full * _C_ob;
        uint32_t filter_o_c_block = (j / _C_ob) * (C_f / _C_ib) * H_f * W_f * _C_ib * _C_ob;
        float *O_buffer = O + block_offset;
        uint32_t group_offset = j / C_o;
        // printf("%d  filter block %d %f \n", group_offset, filter_o_c_block, F[filter_o_c_block]);
        // fflush(0);
        for (uint32_t i = 0; i < C_f; i += _C_ib)
        {
            // printf("second ip block \n");
            uint32_t input_block_offset;
            if (_C_ib == 1)
            {
                input_block_offset = ((i / _C_ib) * H_i * W_i * _C_ib) + (group_offset * H_i * W_i);
            }
            else
            {
                input_block_offset = ((i / _C_ib) * H_i * W_i * _C_ib) + (group_offset * channel_stride * H_i * W_i);
            }
            uint32_t filter_i_c_block = (i / _C_ib) * H_f * W_f * _C_ib * _C_ob + filter_o_c_block;
            float *filter_block_ptr = F + filter_i_c_block;
            for (uint32_t l = 0; l < H_o; l++)
            {
                uint32_t col_offset = l * W_o_full * _C_ob;
                uint32_t input_col_offset = (l * stride) * W_i * _C_ob + input_block_offset;
                float *I_ptr = I + input_col_offset;
                float *O_ptr = O_buffer + col_offset;
                // printf(" I_ptr %f \n", I_ptr[0]);
                for (uint32_t k = 0; k < W_o; k += _W_ob)
                {
                    // uint32_t input_row_offset = (k * stride)*_C_ob;
                    // float * I_ptr = I + input_row_offset + input_col_offset;
                    if (_C_ib == 1)
                    {
                        if(op == 'c'){
                            dw_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ob>(H_f, W_f, W_i * _C_ob, I_ptr, filter_block_ptr, O_ptr);
                        }
                        else if( op == 'p'){
                            
                        }
                    }
                    else
                    {
                        conv_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ob>(H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr);
                    }
                    I_ptr += stride * _W_ob * _C_ob;
                    O_ptr += _W_ob * _C_ob;
                }
                // printf("%.2f %.2f %.2f %.2f \n", I_ptr[0], I_ptr[16], I_ptr[32], filter_block_ptr[0]);
                if (_C_ib == 1)
                {
                    dw_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ob>(H_f, W_f, W_i * _C_ob, I_ptr, filter_block_ptr, O_ptr, W_last);
                }
                else
                {
                    conv_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ob>(H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
                }
            }
            // printf("op[0]: %f update %d \n", O_buffer[0], i);
        }
    }
}

// template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t stride, uint32_t channel_stride, uint32_t H_f, uint32_t W_f, uint32_t C_f>
// void direct_convolution(
//     uint32_t C_o,
//     uint32_t G,
//     uint32_t H_i,
//     uint32_t W_i,
//     float *I,
//     float *F,
//     float *O)
// {
//   uint32_t H_o = 0;
//   op_dim(H_i, stride, H_f, H_o);
//   uint32_t W_o_full = 0;
//   op_dim(W_i, stride, W_f, W_o_full);
//   uint32_t W_o = (W_o_full / _W_ob) * _W_ob;
//   uint32_t W_last = W_o_full % _W_ob;
// // printf("W_of_full: %d W_o : %d W_last: %d\n ", W_o_full, W_o, W_last);

// // printf(" input dims : %d %d \n", H_i, W_i);
// #if PARALLEL == 1
// #pragma omp parallel for
// #endif
//   for (uint32_t j = 0; j < C_o * G; j += _C_ob)
//   {

//     uint32_t block_offset = (j / _C_ob) * H_o * W_o_full * _C_ob;
//     uint32_t filter_o_c_block = (j / _C_ob) * (C_f / _C_ib) * H_f * W_f * _C_ib * _C_ob;
//     float *O_buffer = O + block_offset;
//     uint32_t group_offset = j / C_o;
//     // printf("%d  filter block %d %f \n", group_offset, filter_o_c_block, F[filter_o_c_block]);
//     fflush(0);

//     {
//       uint32_t i = 0;
//       // printf("second ip block \n");
//       uint32_t input_block_offset;
//       if (_C_ib == 1)
//       {
//         input_block_offset = ((i / _C_ib) * H_i * W_i * _C_ib) + (group_offset * H_i * W_i);
//       }
//       else
//       {
//         input_block_offset = ((i / _C_ib) * H_i * W_i * _C_ib) + (group_offset * channel_stride * H_i * W_i);
//       }
//       uint32_t filter_i_c_block = (i / _C_ib) * H_f * W_f * _C_ib * _C_ob + filter_o_c_block;
//       float *filter_block_ptr = F + filter_i_c_block;
//       for (uint32_t l = 0; l < H_o; l++)
//       {
//         uint32_t col_offset = l * W_o_full * _C_ob;
//         uint32_t input_col_offset = (l * stride) * W_i * _C_ob + input_block_offset;
//         float *I_ptr = I + input_col_offset;
//         float *O_ptr = O_buffer + col_offset;
//         // printf(" I_ptr %f \n", I_ptr[0]);
//         for (uint32_t k = 0; k < W_o; k += _W_ob)
//         {
//           // uint32_t input_row_offset = (k * stride)*_C_ob;
//           // float * I_ptr = I + input_row_offset + input_col_offset;
//           if (_C_ib == 1)
//           {
//             dw_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ob,>(H_f, W_f, W_i * _C_ob, I_ptr, filter_block_ptr, O_ptr);
//           }
//           else
//           {
//             conv_kernel_start<_W_ob, _C_ob, _C_ib, stride * _C_ob,>(H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr);
//           }
//           I_ptr += stride * _W_ob * _C_ob;
//           O_ptr += _W_ob * _C_ob;
//         }
//         // printf("%.2f %.2f %.2f %.2f \n", I_ptr[0], I_ptr[16], I_ptr[32], filter_block_ptr[0]);
//         if (_C_ib == 1)
//         {
//           dw_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ob,>(H_f, W_f, W_i * _C_ob, I_ptr, filter_block_ptr, O_ptr, W_last);
//         }
//         else
//         {
//           conv_kernel_start_end<_W_ob, _C_ob, _C_ib, stride * _C_ob,>(H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
//         }
//       }
//       // printf("op[0]: %f update %d \n", O_buffer[0], i);
//     }

//     for (uint32_t i = _C_ib; i < C_f; i += _C_ib)
//     {
//       // printf("second ip block \n");
//       uint32_t input_block_offset;
//       if (_C_ib == 1)
//       {
//         input_block_offset = ((i / _C_ib) * H_i * W_i * _C_ib) + (group_offset * H_i * W_i);
//       }
//       else
//       {
//         input_block_offset = ((i / _C_ib) * H_i * W_i * _C_ib) + (group_offset * channel_stride * H_i * W_i);
//       }
//       uint32_t filter_i_c_block = (i / _C_ib) * H_f * W_f * _C_ib * _C_ob + filter_o_c_block;
//       float *filter_block_ptr = F + filter_i_c_block;
//       for (uint32_t l = 0; l < H_o; l++)
//       {
//         uint32_t col_offset = l * W_o_full * _C_ob;
//         uint32_t input_col_offset = (l * stride) * W_i * _C_ob + input_block_offset;
//         float *I_ptr = I + input_col_offset;
//         float *O_ptr = O_buffer + col_offset;
//         // printf(" I_ptr %f \n", I_ptr[0]);
//         for (uint32_t k = 0; k < W_o; k += _W_ob)
//         {
//           // uint32_t input_row_offset = (k * stride)*_C_ob;
//           // float * I_ptr = I + input_row_offset + input_col_offset;
//           conv_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ob,>(H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr);

//           I_ptr += stride * _W_ob * _C_ob;
//           O_ptr += _W_ob * _C_ob;
//         }
//         // printf("%.2f %.2f %.2f %.2f \n", I_ptr[0], I_ptr[16], I_ptr[32], filter_block_ptr[0]);

//         conv_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ob,>(H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);

//       }
//       // printf("op[0]: %f update %d \n", O_buffer[0], i);
//     }
//   }
// }
