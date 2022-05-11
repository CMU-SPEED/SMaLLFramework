// this file implements simple direct convolution with no fusion

#include <stdint.h>

#include "kernel.h"

// assumes channels are the fastest dimension
template <uint32_t _W_ob, uint32_t _C_ob, uint32_t stride>
void initial_direct_convolution(
    uint32_t channel_stride,
    uint32_t H_f,
    uint32_t W_f,
    uint32_t C_f,
    uint32_t C_o,
    uint32_t G,
    uint32_t H_i,
    uint32_t W_i,
    uint32_t padding,
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

    // uint32_t H_o_padding = 0, W_o_padding = 0;
    // if (padding == 'f')
    // {
    //     H_o_padding = (H_i - H_o) / 2;
    //     W_o_padding = (W_i - W_o) / 2;
    // }
    // printf("%d %d %d\n", _W_ob, _C_ob, _C_ib);

// printf("W_of_full: %d W_o : %d W_last: %d\n ", W_o_full, W_o, W_last);

// printf(" input dims : %d %d \n", H_i, W_i);
#if PARALLEL == 1
#pragma omp parallel for
#endif
    for (uint32_t j = 0; j < C_o * G; j += _C_ob)
    {

        uint32_t block_offset = (j / _C_ob) * H_o * W_o_full * _C_ob;
        uint32_t filter_o_c_block = (j / _C_ob) * (C_f / C_f) * H_f * W_f * C_f * _C_ob;
        // printf("filter group offset %d  %d %d %d    ", filter_o_c_block, _C_ob, C_f, _C_ib);
        float *O_buffer = O + block_offset;
        // printf("output block %d \n", O_buffer - O);
        uint32_t group_offset = j / C_o;
        // printf("%d  filter block %d %f \n", group_offset, filter_o_c_block, F[filter_o_c_block]);
        // fflush(0);

            uint32_t first = 1;
            // printf("first\n");
            uint32_t input_block_offset;
      
            input_block_offset = 0; //((i / _C_ib) * H_i * W_i * _C_ib) + (group_offset * channel_stride * H_i * W_i);
        
            uint32_t filter_i_c_block = 0 +  filter_o_c_block; //(i / _C_ib) * H_f * W_f * _C_ib * _C_ob + filter_o_c_block;
            // printf("filter input offset %d ", filter_i_c_block - filter_o_c_block);

            float *filter_block_ptr = F + filter_i_c_block;
            // front padding row
            //  if(stride == 1)
            //  {
            //      for(uint32_t l_padded = 0; l_padded < H_o_padding; l_padded++)
            //      {
            //          for (uint32_t k = 0; k < W_o; k += _W_ob)
            //          {

            //         }
            // }
            // Set the output pointer to the full section
            // end front padding
            for (uint32_t l = 0; l < H_o; l++)
            {
                uint32_t col_offset = l * W_o_full * _C_ob;
                uint32_t input_col_offset = (l * stride) * W_i * C_f + input_block_offset;
                float *I_ptr = I + input_col_offset;
                float *O_ptr = O_buffer + col_offset;
                for (uint32_t k = 0; k < W_o; k += _W_ob)
                {
                    // uint32_t input_row_offset = (k * stride)*_C_ob;
                    // float * I_ptr = I + input_row_offset + input_col_offset;
                    initial_conv_kernel_combined<_W_ob, _C_ob>(first, C_f, stride * C_f, H_f, W_f, W_i * C_f, I_ptr, filter_block_ptr, O_ptr);
                    I_ptr += stride * _W_ob * C_f;
                    O_ptr += _W_ob * _C_ob;
                }
                initial_conv_kernel_end_combined<_W_ob, _C_ob>(first, C_f, stride * C_f, H_f, W_f, W_i * C_f, I_ptr, filter_block_ptr, O_ptr, W_last);
            }

            // back padding row
    }
}

template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t stride, char op>
void direct_convolution(
    uint32_t channel_stride,
    uint32_t H_f,
    uint32_t W_f,
    uint32_t C_f,
    uint32_t C_o,
    uint32_t G,
    uint32_t H_i,
    uint32_t W_i,
    uint32_t padding,
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

    // uint32_t H_o_padding = 0, W_o_padding = 0;
    // if (padding == 'f')
    // {
    //     H_o_padding = (H_i - H_o) / 2;
    //     W_o_padding = (W_i - W_o) / 2;
    // }
    // printf("%d %d %d\n", _W_ob, _C_ob, _C_ib);

// printf("W_of_full: %d W_o : %d W_last: %d\n ", W_o_full, W_o, W_last);

// printf(" input dims : %d %d \n", H_i, W_i);
#if PARALLEL == 1
#pragma omp parallel for
#endif
    for (uint32_t j = 0; j < C_o * G; j += _C_ob)
    {

        uint32_t block_offset = (j / _C_ob) * H_o * W_o_full * _C_ob;
        uint32_t filter_o_c_block = (j / _C_ob) * (C_f / _C_ib) * H_f * W_f * _C_ib * _C_ob;
        // printf("filter group offset %d  %d %d %d    ", filter_o_c_block, _C_ob, C_f, _C_ib);
        float *O_buffer = O + block_offset;
        // printf("output block %d \n", O_buffer - O);
        uint32_t group_offset = j / C_o;
        // printf("%d  filter block %d %f \n", group_offset, filter_o_c_block, F[filter_o_c_block]);
        // fflush(0);
        for (uint32_t i = 0; i < C_f; i += _C_ib)
        {
            // printf("second ip block \n");
            uint32_t first = i==0;
            // printf("first\n");
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
            // printf("filter input offset %d ", filter_i_c_block - filter_o_c_block);

            float *filter_block_ptr = F + filter_i_c_block;
            // front padding row
            //  if(stride == 1)
            //  {
            //      for(uint32_t l_padded = 0; l_padded < H_o_padding; l_padded++)
            //      {
            //          for (uint32_t k = 0; k < W_o; k += _W_ob)
            //          {

            //         }
            // }
            // Set the output pointer to the full section
            // end front padding
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
                        if (op == 'c')
                        {
                            dw_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ob>(H_f, W_f, W_i * _C_ob, I_ptr, filter_block_ptr, O_ptr);
                        }
                        else if (op == 'p')
                        {
                            pool_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ob>(H_f, W_f, W_i * _C_ob, I_ptr, O_ptr);
                        }
                        else if (op == 'a')
                        {
                            activation_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ob>(H_f, W_f, W_i * _C_ob, I_ptr, O_ptr);
                        }
                    }
                    else
                    {
                        conv_kernel_combined<_W_ob, _C_ob, _C_ib, stride * _C_ib>(first, H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr);
                    }
                    I_ptr += stride * _W_ob * _C_ob;
                    O_ptr += _W_ob * _C_ob;
                }
                // printf("%.2f %.2f %.2f %.2f \n", I_ptr[0], I_ptr[16], I_ptr[32], filter_block_ptr[0]);
                if (_C_ib == 1)
                {
                    // printf("calling dwise  %d\n", O_ptr - O);
                    if (op == 'c')
                    {
                        dw_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ob>(H_f, W_f, W_i * _C_ob, I_ptr, filter_block_ptr, O_ptr, W_last);
                    }
                    else if (op == 'p')
                    {
                        pool_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ob>(H_f, W_f, W_i * _C_ob, I_ptr, O_ptr, W_last);
                    }
                    else if (op == 'a')
                    {
                        activation_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ob>(H_f, W_f, W_i * _C_ob, I_ptr, O_ptr, W_last);
                    }
                    // printf("filter: %.5f %d input: %.5f\n",filter_block_ptr[0], filter_block_ptr - F, I_ptr[0]);
                    // printf("%.5f %.5f  %.5f  %.5f %.5f %.5f %.5f %.5f \n", O_ptr[0], O_ptr[1], O_ptr[2], O_ptr[3], O_ptr[4], O_ptr[5], O_ptr[6], O_ptr[7]);
                }
                else
                {
                    conv_kernel_end_combined<_W_ob, _C_ob, _C_ib, stride * _C_ib>(first, H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
                }
            }
            // printf("op[0]: %f update %d \n", O_buffer[0], i);

            // back padding row
        }


    }
}

template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t stride, char op>
void direct_convolution_partial(
    uint32_t channel_stride,
    uint32_t H_f,
    uint32_t W_f,
    uint32_t C_f,
    uint32_t C_o,
    uint32_t G,
    uint32_t H_i,
    uint32_t W_i,
    uint32_t padding,
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

    // if (padding == 'f')
    // {
    //     uint32_t H_o_padding = (H_i - H_o) / 2;  // NOTE: scoped to this block only
    //     uint32_t W_o_padding = (W_i - W_o) / 2;
    // }
    // else if (padding == 's')
    // {
    //     uint32_t H_o_padding = (H_i - H_o) / 2;
    //     uint32_t W_o_padding = (W_i - W_o) / 2;
    // }
// printf("W_of_full: %d W_o : %d W_las t: %d\n ", W_o_full, W_o, W_last);

// printf(" input dims : %d %d \n", H_i, W_i);
#if PARALLEL == 1
#pragma omp parallel for
#endif
    for (uint32_t j = 0; j < C_o * G; j += _C_ob)
    {

        uint32_t block_offset = (j / _C_ob) * H_o * W_o_full * _C_ob;
        uint32_t filter_o_c_block = (j / _C_ob) * (C_f / _C_ib) * H_f * W_f * _C_ib * _C_ob;
        float *O_buffer = O + block_offset;
        // uint32_t group_offset = j / C_o;

        for (uint32_t i = 0; i < C_f; i += _C_ib)
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

                    conv_kernel_combined<_W_ob, _C_ob, _C_ib, stride * _C_ib>(0, H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr);
                    // conv_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ib, H_f, W_f>(W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr);
                    // printf("\t \t %d \n", k);
                    I_ptr += stride * _W_ob * _C_ob;
                    O_ptr += _W_ob * _C_ob;
                }
                // printf(" input channel: %d  row: %d %.2f %.2f %.2f %.2f \n", i, l, I_ptr[0], I_ptr[_C_ob*stride], I_ptr[2*_C_ob*stride], filter_block_ptr[0]);
                // printf("\t %d \n", l);

                conv_kernel_end_combined<_W_ob, _C_ob, _C_ib, stride * _C_ib>(0, H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
                // printf(" input channel: %d  row: %d %.2f %.2f %.2f %.2f \n", i, l, I_ptr[0], I_ptr[_C_ob * stride], I_ptr[2 * _C_ob * stride], filter_block_ptr[0]);

                // conv_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ib, H_f, W_f>(W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
            }
            // printf("%d \n", i);
        }
    }
}


/*
The code verision below has two separate kernels, 
one to deal with the first iteration 
and another to handle all subsequent ones
*/

//template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t stride, char op>
// void direct_convolution(
//     uint32_t channel_stride,
//     uint32_t H_f,
//     uint32_t W_f,
//     uint32_t C_f,
//     uint32_t C_o,
//     uint32_t G,
//     uint32_t H_i,
//     uint32_t W_i,
//     uint32_t padding,
//     float *I,
//     float *F,
//     float *O)
// {
//     uint32_t H_o = 0;
//     op_dim(H_i, stride, H_f, H_o);
//     uint32_t W_o_full = 0;
//     op_dim(W_i, stride, W_f, W_o_full);
//     uint32_t W_o = (W_o_full / _W_ob) * _W_ob;
//     uint32_t W_last = W_o_full % _W_ob;

//     // uint32_t H_o_padding = 0, W_o_padding = 0;
//     // if (padding == 'f')
//     // {
//     //     H_o_padding = (H_i - H_o) / 2;
//     //     W_o_padding = (W_i - W_o) / 2;
//     // }

// // printf("W_of_full: %d W_o : %d W_las t: %d\n ", W_o_full, W_o, W_last);

// // printf(" input dims : %d %d \n", H_i, W_i);
// #if PARALLEL == 1
// #pragma omp parallel for
// #endif
//     for (uint32_t j = 0; j < C_o * G; j += _C_ob)
//     {

//         uint32_t block_offset = (j / _C_ob) * H_o * W_o_full * _C_ob;
//         uint32_t filter_o_c_block = (j / _C_ob) * (C_f / _C_ib) * H_f * W_f * _C_ib * _C_ob;
//         // printf("filter group offset %d  %d %d %d    ", filter_o_c_block, _C_ob, C_f, _C_ib);
//         float *O_buffer = O + block_offset;
//         // printf("output block %d \n", O_buffer - O);
//         uint32_t group_offset = j / C_o;
//         // printf("%d  filter block %d %f \n", group_offset, filter_o_c_block, F[filter_o_c_block]);
//         // fflush(0);
//         // for (uint32_t i = 0; i < C_f; i += _C_ib)
//         {
//             uint32_t i = 0;
//             // printf("second ip block \n");
//             uint32_t input_block_offset;
//             if (_C_ib == 1)
//             {
//                 input_block_offset = ((i / _C_ib) * H_i * W_i * _C_ib) + (group_offset * H_i * W_i);
//             }
//             else
//             {
//                 input_block_offset = ((i / _C_ib) * H_i * W_i * _C_ib) + (group_offset * channel_stride * H_i * W_i);
//             }
//             uint32_t filter_i_c_block = (i / _C_ib) * H_f * W_f * _C_ib * _C_ob + filter_o_c_block;
//             // printf("filter input offset %d ", filter_i_c_block - filter_o_c_block);

//             float *filter_block_ptr = F + filter_i_c_block;
//             // front padding row
//             //  if(stride == 1)
//             //  {
//             //      for(uint32_t l_padded = 0; l_padded < H_o_padding; l_padded++)
//             //      {
//             //          for (uint32_t k = 0; k < W_o; k += _W_ob)
//             //          {

//             //         }
//             // }
//             // Set the output pointer to the full section
//             // end front padding
//             for (uint32_t l = 0; l < H_o; l++)
//             {
//                 uint32_t col_offset = l * W_o_full * _C_ob;
//                 uint32_t input_col_offset = (l * stride) * W_i * _C_ob + input_block_offset;
//                 float *I_ptr = I + input_col_offset;
//                 float *O_ptr = O_buffer + col_offset;
//                 // printf(" I_ptr %f \n", I_ptr[0]);
//                 for (uint32_t k = 0; k < W_o; k += _W_ob)
//                 {
//                     // uint32_t input_row_offset = (k * stride)*_C_ob;
//                     // float * I_ptr = I + input_row_offset + input_col_offset;
//                     if (_C_ib == 1)
//                     {
//                         if (op == 'c')
//                         {
//                             dw_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ib>(H_f, W_f, W_i * _C_ob, I_ptr, filter_block_ptr, O_ptr);
//                         }
//                         else if (op == 'p')
//                         {
//                             pool_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ib>(H_f, W_f, W_i * _C_ob, I_ptr, O_ptr);
//                         }
//                         else if (op == 'a')
//                         {
//                             activation_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ib>(H_f, W_f, W_i * _C_ob, I_ptr, O_ptr);
//                         }
//                     }
//                     else
//                     {
//                         conv_kernel_start<_W_ob, _C_ob, _C_ib, stride * _C_ib>(H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr);
//                     }
//                     I_ptr += stride * _W_ob * _C_ob;
//                     O_ptr += _W_ob * _C_ob;
//                 }
//                 // printf("%.2f %.2f %.2f %.2f \n", I_ptr[0], I_ptr[16], I_ptr[32], filter_block_ptr[0]);
//                 if (_C_ib == 1)
//                 {
//                     // printf("calling dwise  %d\n", O_ptr - O);
//                     if (op == 'c')
//                     {
//                         dw_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ib>(H_f, W_f, W_i * _C_ob, I_ptr, filter_block_ptr, O_ptr, W_last);
//                     }
//                     else if (op == 'p')
//                     {
//                         pool_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ib>(H_f, W_f, W_i * _C_ob, I_ptr, O_ptr, W_last);
//                     }
//                     else if (op == 'a')
//                     {
//                         activation_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ib>(H_f, W_f, W_i * _C_ob, I_ptr, O_ptr, W_last);
//                     }
//                     // printf("filter: %.5f %d input: %.5f\n",filter_block_ptr[0], filter_block_ptr - F, I_ptr[0]);
//                     // printf("%.5f %.5f  %.5f  %.5f %.5f %.5f %.5f %.5f \n", O_ptr[0], O_ptr[1], O_ptr[2], O_ptr[3], O_ptr[4], O_ptr[5], O_ptr[6], O_ptr[7]);
//                 }
//                 else
//                 {
//                     conv_kernel_start_end<_W_ob, _C_ob, _C_ib, stride * _C_ib>(H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
//                 }
//             }
//             // printf("op[0]: %f update %d \n", O_buffer[0], i);

//             // back padding row
//         }

//         for (uint32_t i = _C_ib; i < C_f; i += _C_ib)
//         {
//             // printf("second ip block \n");

//             uint32_t input_block_offset = (i / _C_ib) * H_i * W_i * _C_ib;
//             uint32_t filter_i_c_block = (i / _C_ib) * H_f * W_f * _C_ib * _C_ob + filter_o_c_block;
//             float *filter_block_ptr = F + filter_i_c_block;

//             for (uint32_t l = 0; l < H_o; l++)
//             {

//                 uint32_t col_offset = l * W_o_full * _C_ob;
//                 uint32_t input_col_offset = (l * stride) * W_i * _C_ob + input_block_offset;

//                 float *I_ptr = I + input_col_offset;
//                 float *O_ptr = O_buffer + col_offset;

//                 for (uint32_t k = 0; k < W_o; k += _W_ob)
//                 {

//                     // uint32_t input_row_offset = (k * stride)*_C_ob;
//                     // float * I_ptr = I + input_row_offset + input_col_offset;

//                     conv_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ib>(H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr);
//                     // conv_kernel<_W_ob, _C_ob, _C_ib, stride * _C_ib, H_f, W_f>(W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr);
//                     // printf("\t \t %d \n", k);
//                     I_ptr += stride * _W_ob * _C_ob;
//                     O_ptr += _W_ob * _C_ob;
//                 }
//                 // printf(" input channel: %d  row: %d %.2f %.2f %.2f %.2f \n", i, l, I_ptr[0], I_ptr[_C_ob*stride], I_ptr[2*_C_ob*stride], filter_block_ptr[0]);
//                 // printf("\t %d \n", l);

//                 conv_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ib>(H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
//                 // printf(" input channel: %d  row: %d %.2f %.2f %.2f %.2f \n", i, l, I_ptr[0], I_ptr[_C_ob * stride], I_ptr[2 * _C_ob * stride], filter_block_ptr[0]);

//                 // conv_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ib, H_f, W_f>(W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
//             }
//             // printf("%d \n", i);
//         }
//     }
// }
