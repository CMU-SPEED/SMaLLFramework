// this file implements simple direct convolution with no fusion

#include <stdint.h>
#include<math.h>
#include "kernel.h"

// // #ifndef op_dim
// #define op_dim(IN_dim, stride, K_dim, OUT_dim)         \
//     {                                                  \
//         int out_elems = (IN_dim - K_dim) / stride + 1; \
        printf("number of elements %d \n", out_elems);                    \
//         OUT_dim = (out_elems > 0) ? out_elems : 0;     \
//     }
// // #endif

#define CALC_PADDING(I_dim, K_dim, stride, padding_front, padding_back)                      \
    {                                                                                        \
        uint32_t padding;                                                                    \
        if (I_dim % stride == 0)                                                             \
        {                                                                                    \
            padding = (K_dim - stride > 0) ? K_dim - stride : 0;                 \
        }                                                                                    \
        else                                                                                 \
        {                                                                                    \
            padding = (K_dim - (I_dim % stride) > 0) ? K_dim - (I_dim% stride) : 0; \
        }                                                                                    \
        padding_front = padding / 2;                                                         \
        padding_back = padding - padding_front;                                              \
    }

// assumes channels are the fastest dimension
template <uint32_t _W_ob, uint32_t _C_ob>
void initial_direct_convolution(
    uint32_t channel_stride,
    uint32_t stride,
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
    uint32_t H_padding = 0, W_padding = 0;
    uint32_t H_f_padding = 0, W_f_padding = 0;
    uint32_t H_f_elements = 0, W_f_elements = 0;
    uint32_t H_b_padding = 0, W_b_padding = 0;
    uint32_t H_b_elements = 0, W_b_elements = 0;
    uint32_t H_pad_row = 0, W_pad_row = 0;
    //Output Elements with padding
    uint32_t H_o_w_pad = 0, W_o_w_pad = 0;
    //Output Elements using the full filter
    uint32_t H_o = 0, W_o_full = 0;

//TODO: Change the interface to do this at a different (higher) level of abstraction
    if (padding == 'f')
    {
        CALC_PADDING(H_i, H_f, stride, H_f_padding, H_b_padding);
        CALC_PADDING(W_i, W_f, stride, W_f_padding, W_b_padding);
    }

    //Total output elements
    // To calculate offsets to next output row, next output block
    op_dim((H_i+H_f_padding+H_b_padding), stride, H_f, H_o_w_pad);
    op_dim((W_i+W_f_padding+W_b_padding), stride, W_f, W_o_w_pad);
    // printf("W_o_w_pad : %d H_o_w_pad: %d\n ", W_o_w_pad, H_o_w_pad);

    H_f_elements = H_f_padding / stride + (H_f_padding % stride != 0);
    W_f_elements = W_f_padding / stride + (W_f_padding % stride != 0);
    // printf("W_f_elements : %d H_f_elements: %d\n ", W_f_elements, H_f_elements);

    H_pad_row = H_f_elements*stride - H_f_padding;
    W_pad_row = W_f_elements*stride - W_f_padding;

    // printf("starting index of full elements\n H: %d W: %d \n", H_pad_row, W_pad_row);

    // Full kernel output elements
    op_dim(H_i - H_pad_row, stride, H_f, H_o);
    op_dim(W_i - W_pad_row, stride, W_f, W_o_full);
    // printf(" H_o: %d W_o_full: %d\n ", H_o, W_o_full);

    //back padding elements
    uint32_t H_back_index, W_back_index;
    
    H_back_index = H_pad_row + stride * (H_o);
    W_back_index = W_pad_row + stride * (W_o_full);
    // printf("starting index of back padding elements\n H: %d W: %d \n", H_back_index, W_back_index);

    op_dim((H_i + H_b_padding - H_back_index), stride, H_f, H_b_elements);
    op_dim((W_i + W_b_padding - W_back_index), stride, W_f, W_b_elements);
    // printf("W_b_elements : %d H_b_elements: %d\n ", W_b_elements, H_b_elements);

    // setting up microkernel specific parameters
    uint32_t W_o = (W_o_full / _W_ob) * _W_ob;
    uint32_t W_last = W_o_full % _W_ob;

    // printf("W_o : %d W_last: %d\n", W_o, W_last);

    // printf("%d %d %d\n", _W_ob, _C_ob, _C_ib);


// printf(" input dims : %d %d \n", H_i, W_i);
#if PARALLEL == 1
#pragma omp parallel for
#endif
        for (uint32_t j = 0; j < C_o * G; j += _C_ob)
    {

        uint32_t first = 1;
        uint32_t input_block_offset;

        input_block_offset = 0;

        //Include padding outputs
        uint32_t block_offset = (j / _C_ob) * H_o_w_pad * W_o_w_pad * _C_ob;
        float *O_buffer = O + block_offset;
        

        //Unaffected by padding
        uint32_t filter_o_c_block = (j / _C_ob) * (C_f / C_f) * H_f * W_f * C_f * _C_ob;
        uint32_t group_offset = j / C_o;
        uint32_t filter_i_c_block = 0 + filter_o_c_block; //(i / _C_ib) * H_f * W_f * _C_ib * _C_ob + filter_o_c_block;

        float *I_buffer = I + input_block_offset;
        float *filter_block_ptr = F + filter_i_c_block;

        float * I_row = I_buffer;
        float * O_row = O_buffer;
        // front padding rows
        initial_conv_kernel_padding_top_combined<_W_ob, _C_ob>(first, 
                                                                stride, 
                                                                C_f, 
                                                                stride*C_f,
                                                                H_f, W_f, 
                                                                W_i*C_f, 
                                                                I_row, filter_block_ptr, O_row,
                                                                H_f_padding,
                                                                // W_pad_row,
                                                                W_o, W_last, 
                                                                W_f_padding, W_b_padding,
                                                                W_pad_row,
                                                                W_o_w_pad
                                                            );
        // printf("Padding output elements: %.2f \n", O_row[0]);
        // Set Input pointer to the full section
        I_row = I_buffer + (H_pad_row)*W_i * C_f;
        // Set the output pointer to the full section
        O_row = O_buffer + H_f_elements * W_o_w_pad * C_ob;
        // end front padding
        for (uint32_t l = 0; l < H_o; l++)
        {
            uint32_t col_offset = l * W_o_full * _C_ob;
            uint32_t input_col_offset = (l * stride) *W_i *C_f;
            float *I_ptr = I_row + input_col_offset;
            // printf("row index %d \n",)
            float *O_ptr = O_row + col_offset;

            //left padding elements
            // initial_conv_kernel_padding_left_combined<_W_ob, _C_ob>(first, stride, C_f, stride * C_f, H_f, W_f, W_i * C_f, I_ptr, filter_block_ptr, O_ptr, W_f_padding);

            //Set pointer to full elements
            I_ptr += W_pad_row*C_f;
            O_ptr += W_f_elements*C_ob;
            for (uint32_t k = 0; k < W_o; k += _W_ob)
            {
                // uint32_t input_row_offset = (k * stride)*_C_ob;
                // float * I_ptr = I + input_row_offset + input_col_offset;
                initial_conv_kernel_combined<_W_ob, _C_ob>(first, C_f, stride * C_f, H_f, W_f, W_i * C_f, I_ptr, filter_block_ptr, O_ptr);
                I_ptr += stride * _W_ob * C_f;
                O_ptr += _W_ob * _C_ob;
            }
            //cleanup elements and back padding
            // printf("i row offset %d %.2f \n",(I_ptr-I_row)/C_f, I_ptr[0]);
            initial_conv_kernel_padding_right_combined<_W_ob, _C_ob>(first, stride, C_f, stride * C_f, H_f, W_f, W_i * C_f, I_ptr, filter_block_ptr, O_ptr, W_last, W_b_padding);
        
        } 
            
        // O_buffer = O_buffer + (H_f_padding + H_o )* W_o_w_pad * C_ob;
        
        // back padding row
            I_row = I_buffer + (H_back_index) * W_i * C_f;
            // printf("%d == %d? %d\n", I_buffer + (H_o * stride) * W_i * C_f, I_buffer + H_back_index * W_i* C_f);
            // printf("I_row: %d \n", (I_row-I_buffer)/(W_i*C_f));
            O_row = O_buffer + (H_o + H_f_elements)*W_o_w_pad * C_ob;
            // printf("O offset = %d %d\n", (O_row - O) / (W_o_w_pad * _C_ob), (O_row - O));
            initial_conv_kernel_padding_bottom_combined<_W_ob, _C_ob>(first,
                                                                         stride,
                                                                         C_f,
                                                                         stride * C_f,
                                                                         H_f, W_f,
                                                                         W_i * C_f,
                                                                         I_row, filter_block_ptr, O_row,
                                                                         H_b_elements,
                                                                         W_o, W_last,
                                                                         W_f_elements, W_b_elements,
                                                                         W_pad_row,
                                                                         W_o_w_pad);
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
    // uint32_t H_o = 0;
    // op_dim(H_i, stride, H_f, H_o);
    // uint32_t W_o_full = 0;
    // op_dim(W_i, stride, W_f, W_o_full);
    // uint32_t W_o = (W_o_full / _W_ob) * _W_ob;
    // uint32_t W_last = W_o_full % _W_ob;

    // uint32_t H_padding = 0, W_padding = 0;
    // if (padding == 'f')
    // {
    //     H_padding = (H_i - H_o) / 2;
    //     W_padding = (W_i - W_o) / 2;
    // }
    // printf("%d %d %d\n", _W_ob, _C_ob, _C_ib);

// printf("W_of_full: %d W_o : %d W_last: %d\n ", W_o_full, W_o, W_last);
// printf("C_f %d G %d C_o %d\n", C_f, G, C_o);
// printf(" output dims : %d %d \n", H_o, W_o_full);



//Calculating padding and output parameters

    uint32_t H_padding = 0, W_padding = 0;
    uint32_t H_f_padding = 0, W_f_padding = 0;
    uint32_t H_f_elements = 0, W_f_elements = 0;
    uint32_t H_b_padding = 0, W_b_padding = 0;
    uint32_t H_b_elements = 0, W_b_elements = 0;
    uint32_t H_pad_row = 0, W_pad_row = 0;
    // Output Elements with padding
    uint32_t H_o_w_pad = 0, W_o_w_pad = 0;
    // Output Elements using the full filter
    uint32_t H_o = 0, W_o_full = 0;

    // TODO: Change the interface to do this at a different (higher) level of abstraction
    if (padding == 'f')
    {
        CALC_PADDING(H_i, H_f, stride, H_f_padding, H_b_padding);
        CALC_PADDING(W_i, W_f, stride, W_f_padding, W_b_padding);
    }

    // printf("%d %d\n", H_f_padding, W_f_elements);
    // Total output elements
    //  To calculate offsets to next output row, next output block
    op_dim((H_i + H_f_padding + H_b_padding), stride, H_f, H_o_w_pad);
    op_dim((W_i + W_f_padding + W_b_padding), stride, W_f, W_o_w_pad);
    // printf("W_o_w_pad : %d H_o_w_pad: %d\n ", W_o_w_pad, H_o_w_pad);

    H_f_elements = H_f_padding / stride + (H_f_padding % stride != 0);
    W_f_elements = W_f_padding / stride + (W_f_padding % stride != 0);
    // printf("W_f_elements : %d H_f_elements: %d\n ", W_f_elements, H_f_elements);
    // printf("W_f_padding : %d H_f_padding: %d\n ", W_f_padding, H_f_padding);

    H_pad_row = H_f_elements * stride - H_f_padding;
    W_pad_row = W_f_elements * stride - W_f_padding;

    // printf("starting index of full elements\n H: %d W: %d \n", H_pad_row, W_pad_row);

    // Full kernel output elements
    op_dim(H_i - H_pad_row, stride, H_f, H_o);
    op_dim(W_i - W_pad_row, stride, W_f, W_o_full);

    // printf(" H_o: %d W_o_full: %d\n ", H_o, W_o_full);

    // back padding elements
    uint32_t H_back_index, W_back_index;

    H_back_index = H_pad_row + stride * (H_o);
    W_back_index = W_pad_row + stride * (W_o_full);

    // printf("starting index of back padding elements\n H: %d W: %d \n", H_back_index, W_back_index);

    op_dim((H_i + H_b_padding - H_back_index), stride, H_f, H_b_elements);
    op_dim((W_i + W_b_padding - W_back_index), stride, W_f, W_b_elements);
    // printf("W_b_elements : %d H_b_elements: %d\n ", W_b_elements, H_b_elements);

    // setting up microkernel specific parameters
    uint32_t W_o = (W_o_full / _W_ob) * _W_ob;
    uint32_t W_last = W_o_full % _W_ob;

    // printf("W_o : %d W_last: %d\n", W_o, W_last);

#if PARALLEL == 1
#pragma omp parallel for
#endif
    for (uint32_t j = 0; j < C_o * G; j += _C_ob)
    {

        uint32_t block_offset = (j / _C_ob) * H_o_w_pad * W_o_w_pad * _C_ob;
        float *O_buffer = O + block_offset;

        //Unaffected by padding
        uint32_t group_offset = j / C_o;
        uint32_t filter_o_c_block = (j / _C_ob) * (C_f / _C_ib) * H_f * W_f * _C_ib * _C_ob;

        for (uint32_t i = 0; i < C_f; i += _C_ib)
        {
            uint32_t first = i==0;
            uint32_t input_block_offset;
            
            // if (_C_ib == 1)
            // {
            //     input_block_offset = ((i / _C_ib) * H_i * W_i * _C_ib) + (group_offset * H_i * W_i);
            // }
            // else
            // {
            // }
            input_block_offset = ((i / _C_ib) * H_i * W_i * _C_ib) + (group_offset * channel_stride * H_i * W_i);
            float *I_buffer = I + input_block_offset;
            float * I_row_ptr = I_buffer;

            uint32_t filter_i_c_block = (i / _C_ib) * H_f * W_f * _C_ib * _C_ob + filter_o_c_block;
            float *filter_block_ptr = F + filter_i_c_block;

            //begin top padding
            float * O_row = O_buffer;
            
            if(_C_ib == 1)
            {
                if (op == 'c')
                {
                    dw_kernel_padding_top<_W_ob, _C_ob, _C_ib, stride * _C_ob>(stride, H_f, W_f, W_i * _C_ob, I_row_ptr, filter_block_ptr, O_row, H_f_padding, W_o, W_last, W_f_padding, W_b_padding, W_pad_row, W_o_w_pad);
                }
            }
            else
            {
                conv_kernel_padding_top_combined<_W_ob, _C_ob, _C_ib, stride * _C_ob>(
                    first,
                    stride,
                    H_f, W_f,
                    W_i * _C_ib,
                    I_row_ptr,
                    filter_block_ptr,
                    O_row,
                    H_f_padding,
                    W_o,
                    W_last,
                    W_f_padding,
                    W_b_padding,
                    W_pad_row,
                    W_o_w_pad);
            }
            // Set the pointers to the full section
            I_row_ptr = I_buffer + (H_pad_row)*W_i * _C_ob;
            
            // end front padding


            for (uint32_t l = 0; l < H_o; l++)
            {
                // printf(" block: %d row: %d \n", j, l);
                uint32_t col_offset = l * W_o_full * _C_ob;
                uint32_t input_col_offset = (l * stride) * W_i * _C_ob ;
                float *I_ptr = I_row_ptr + input_col_offset;
                float *O_ptr = O_buffer + col_offset;
                // printf(" I_ptr %f \n", I_ptr[0]);

                if (_C_ib == 1)
                {
                    if (op == 'c')
                    {
                        dw_kernel_padding_left<_W_ob, _C_ob, _C_ib, stride * _C_ob>(stride, H_f, W_f, W_i * _C_ob, I_row_ptr, filter_block_ptr, O_row, W_f_padding);
                    }
                }
                else
                {
                    conv_kernel_padding_left_combined<_W_ob, _C_ob, _C_ib, stride * _C_ib>(first,
                                                                                             stride,
                                                                                             H_f, W_f,
                                                                                             W_i * C_f,
                                                                                             I_row_ptr, filter_block_ptr, O_row,
                                                                                             W_f_padding);
                }

                I_ptr += W_pad_row*_C_ob;
                O_ptr += W_f_elements*_C_ob;
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

                        dw_kernel_padding_right<_W_ob, _C_ob, _C_ib, stride * _C_ob>(stride, H_f, W_f, W_i * _C_ob, I_ptr, filter_block_ptr, O_ptr, W_last, W_b_padding);
                    }
                    else if (op == 'p')
                    {
                        pool_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ob>(H_f, W_f, W_i * _C_ob, I_ptr, O_ptr, W_last, W_padding);
                    }
                    else if (op == 'a')
                    {
                        activation_kernel_end<_W_ob, _C_ob, _C_ib, stride * _C_ob>(H_f, W_f, W_i * _C_ob, I_ptr, O_ptr, W_last, W_padding);
                    }
                    // printf("filter: %.5f %d input: %.5f\n",filter_block_ptr[0], filter_block_ptr - F, I_ptr[0]);
                    // printf("%.5f %.5f  %.5f  %.5f %.5f %.5f %.5f %.5f \n", O_ptr[0], O_ptr[1], O_ptr[2], O_ptr[3], O_ptr[4], O_ptr[5], O_ptr[6], O_ptr[7]);
                }
                else
                {
                    conv_kernel_padding_right_combined<_W_ob, _C_ob, _C_ib, stride * _C_ib>(first, stride, H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last, W_b_padding);
                }
            }
            // printf("op[0]: %f update %d \n", O_buffer[0], i);

            // back padding row

            I_row_ptr = I_buffer + (H_back_index)*W_i * _C_ob;
            // printf("%d == %d? %d\n", I_buffer + (H_o * stride) * W_i * C_f, I_buffer + H_back_index * W_i* C_f);
            // printf("I_row: %d \n", (I_row-I_buffer)/(W_i*C_f));
            O_row = O_buffer + (H_o + H_f_elements) * W_o_w_pad * _C_ob;
            // printf("O offset = %d %d\n", (O_row - O) / (W_o_w_pad * _C_ob), (O_row - O));
            if(_C_ib == 1)
            {
                if (op == 'c')
                {
                    dw_kernel_padding_bottom<_W_ob, _C_ob, _C_ib, stride * _C_ob>(stride, H_f, W_f, W_i * _C_ob, I_row_ptr, filter_block_ptr, O_row, H_b_padding, W_o, W_last, W_f_padding, W_b_padding, W_pad_row, W_o_w_pad);
                                }
            }
            else
            {
                conv_kernel_padding_bottom_combined<_W_ob, _C_ob, _C_ib, stride * _C_ib>(first,
                                                                                         stride,

                                                                                         H_f, W_f,
                                                                                         W_i * C_f,
                                                                                         I_row_ptr, filter_block_ptr, O_row,
                                                                                         H_b_padding,
                                                                                         W_o, W_last,
                                                                                         W_f_padding, W_b_padding,
                                                                                         W_pad_row,
                                                                                         W_o_w_pad);
            }
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

    uint32_t H_padding = 0, W_padding = 0;
    if (padding == 'f')
    {
        H_padding = (H_i - H_o) / 2;
        W_padding = (W_i - W_o) / 2;
    }

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

                conv_kernel_end_combined<_W_ob, _C_ob, _C_ib, stride * _C_ib>(0, H_f, W_f, W_i * _C_ib, I_ptr, filter_block_ptr, O_ptr, W_last, W_padding);
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
and another to handle all subsequent ones.

The switch to a single kernel means that there are fewer kernels to implement over all
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

//     // uint32_t H_padding = 0, W_padding = 0;
//     // if (padding == 'f')
//     // {
//     //     H_padding = (H_i - H_o) / 2;
//     //     W_padding = (W_i - W_o) / 2;
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
//             //      for(uint32_t l_padded = 0; l_padded < H_padding; l_padded++)
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
