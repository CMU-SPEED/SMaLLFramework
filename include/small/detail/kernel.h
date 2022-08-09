#pragma once

// #include <immintrin.h>

// activations kernels
template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void activation_kernel(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *O)
{
    DEF_TILE_C(_W_ob, _C_ob);
    ZERO_TILE_C(_W_ob, _C_ob);

    // int updates = 0;
    // uint32_t step = _C_ob;//stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        // int filter_offset_h = n * W_f * _C_ib * _C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            // int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
            int input_stencil_w = m * _C_ob + input_stencil_h;

            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                // int p_cur = ii;

                MAX_TILE_C(step, a, _W_ob, _C_ob);
            }
        }
    }

    STORE_TILE_C(O, _W_ob, _C_ob);
}

template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void activation_kernel_end(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *O,
    uint32_t W_last,
    uint32_t W_padding)
{

    DEF_TILE_C_END(_W_ob, _C_ob);
    ZERO_END_C(_W_ob, _C_ob);

    // int updates = 0;
    // uint32_t step = _C_ob;//stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int input_stencil_w = m * _C_ob + input_stencil_h;

            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                // int p_cur = ii;
                MAX_END_C(step, a, W_last, _C_ob);
            }
        }
    }

    STORE_END_C(O, _W_ob, _C_ob, W_last);
}

template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void activation_kernel_padding_end(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *O,
    uint32_t H_padding,
    uint32_t W_o,
    uint32_t W_last,
    uint32_t W_padding)
{

    DEF_TILE_C_END(_W_ob, _C_ob);
    ZERO_END_C(_W_ob, _C_ob);

    // int updates = 0;
    // uint32_t step = _C_ob;//stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int input_stencil_w = m * _C_ob + input_stencil_h;

            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                // int p_cur = ii;
                MAX_END_C(step, a, W_last, _C_ob);
            }
        }
    }

    STORE_END_C(O, _W_ob, _C_ob, W_last);
}

// pooling kernels
template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void pool_kernel(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *O)
{
    DEF_TILE_C(_W_ob, _C_ob);
    LOAD_TILE_C_strided(I, step, _W_ob, _C_ob);

    // int updates = 0;
    // uint32_t step = _C_ob;//stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        //int filter_offset_h = n * W_f * _C_ib * _C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            //int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
            int input_stencil_w = m * _C_ob + input_stencil_h;

            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                // int p_cur = ii;

                MAX_TILE_C(step, a, _W_ob, _C_ob);
            }
        }
    }

    STORE_TILE_C(O, _W_ob, _C_ob);
}

template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void pool_kernel_end(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *O,
    uint32_t W_last,
    uint32_t W_padding)
{
    DEF_TILE_C_END(_W_ob, _C_ob);
    LOAD_LAST_C_strided(I, step, _W_ob, _C_ob, W_last);

    // int updates = 0;
    // uint32_t step = _C_ob;//stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int input_stencil_w = m * _C_ob + input_stencil_h;

            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                // int p_cur = ii;
                MAX_END_C(step, a, W_last, _C_ob);
            }
        }
    }

    STORE_END_C(O, _W_ob, _C_ob, W_last);
}

template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void pool_kernel_padding_end(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *O,
    uint32_t H_padding,
    uint32_t W_o,
    uint32_t W_last,
    uint32_t W_padding)
{
    DEF_TILE_C_END(_W_ob, _C_ob);
    LOAD_LAST_C_strided(I, step, _W_ob, _C_ob, W_last);

    // int updates = 0;
    // uint32_t step = _C_ob;//stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int input_stencil_w = m * _C_ob + input_stencil_h;

            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                // int p_cur = ii;
                MAX_END_C(step, a, W_last, _C_ob);
            }
        }
    }

    STORE_END_C(O, _W_ob, _C_ob, W_last);
}

// depthwise kernels
#define COMPUTE_DW_PADDING(H_lb, H_ub, W_lb, W_ub, W_f, input_channels, W_elements)                        \
    for (uint32_t n = H_lb; n < H_ub; n++)                                                                 \
    {                                                                                                      \
        int filter_offset_h = n * W_f * input_channels * _C_ob;                                            \
        int input_stencil_h = /*input_col_offset +*/ (n - H_lb) * input_col_stride /*+ input_row_offset*/; \
        for (uint32_t m = W_lb; m < W_ub; m++)                                                             \
        {                                                                                                  \
            int filter_offset_w = m * input_channels * _C_ob + filter_offset_h;                            \
            int input_stencil_w = (m - W_lb) * input_channels + input_stencil_h;                           \
            float *b = F + filter_offset_w;                                                                \
            float *a = I_ptr + input_stencil_w;                                                            \
            for (uint32_t ii = 0; ii < input_channels; ii++)                                               \
            {                                                                                              \
                int p_cur = ii;                                                                            \
                /* if (W_elements == _W_ob) */                                                             \
                /*  FMA_TILE_C(step, a, b, p_cur, _W_ob, _C_ob);*/                                         \
                /* else */                                                                                 \
                DW_END_C(step, a, b, W_elements, _C_ob);\
            }                                                                                              \
        }                                                                                                  \
    }

template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void dw_kernel_padding_top(
    uint32_t stride,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t H_f_padding,
    // uint32_t W_pad_row,
    uint32_t W_o,
    uint32_t W_last,
    uint32_t W_f_padding,
    uint32_t W_b_padding,
    uint32_t W_pad_row,
    uint32_t W_total)
{
    // printf("calling top padding element computation \n");
    DEF_TILE_C_END(_W_ob, _C_ob);

    uint32_t H_i_valid = H_f_padding; // Which filter rows are multiplied with non-zeros
    uint32_t W_i_valid = W_f_padding; // Which filter elements are multiplied with non-zeros

    float *I_row = I;
    // Iterate over each padded output row
    uint32_t l = 0;
    for (uint32_t l_p = 0; l_p < H_f_padding; l_p += stride)
    {
        // printf("row : %d\n", l_p + 1);
        float *O_row = O + l * (W_total * _C_ob);
        float *O_ptr = O_row;
        float *I_ptr = I_row;

        // padded row elements
        // uint32_t k = 0;
        for (uint32_t k_p = 0; k_p < W_f_padding; k_p += stride)
        {

            ZERO_END_C(1, _C_ob);

            COMPUTE_DW_PADDING(H_i_valid, H_f, W_i_valid, W_f, W_f, _C_ib, 1);

            STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

            W_i_valid -= stride;
            O_ptr += _C_ob;
            // k++;
        }

        // full row elements
        I_ptr = I_row + W_pad_row * _C_ib;
        {
            DEF_TILE_C(_W_ob, _C_ob);

            for (uint32_t k = 0; k < W_o; k += _W_ob)
            {

                ZERO_TILE_C(_W_ob, _C_ob);
                for (uint32_t n = H_i_valid; n < H_f; n++)
                {

                    int filter_offset_h = n * W_f * _C_ib * _C_ob;
                    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

                    for (uint32_t m = 0; m < W_f; m++)
                    {

                        int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
                        int input_stencil_w = m * _C_ob + input_stencil_h;

                        float *b = F + filter_offset_w;
                        float *a = I + input_stencil_w;
                        for (uint32_t ii = 0; ii < _C_ib; ii++)
                        {

                            // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                            // int p_cur = ii;

                            DW_TILE_C(step, a, b, _W_ob, _C_ob);
                        }
                    }
                }

                STORE_TILE_C(O_ptr, _W_ob, _C_ob);

                O_ptr += _W_ob * _C_ob;
                I_ptr += stride * _W_ob * _C_ib;
            }
        }

        // clean up tile elements
        //  O_ptr = O_row + (k + W_o)*_C_ob;
        I_ptr = I_row + (W_pad_row + stride * W_o) * _C_ib;

        ZERO_END_C(W_last, _C_ob);


        COMPUTE_DW_PADDING(H_i_valid, H_f, 0, W_f, W_f, _C_ib, W_last);

        STORE_END_C(O_ptr, _W_ob, _C_ob, W_last);

        // back padded row elements
        O_ptr += W_last * _C_ob;
        I_ptr += (W_last * stride) * _C_ib;
        W_i_valid = W_f - W_b_padding;

        for (uint32_t k_p = 0; k_p < W_b_padding; k_p += stride)
        {

            ZERO_END_C(1, _C_ob);


            COMPUTE_DW_PADDING(H_i_valid, H_f, 0, W_i_valid, W_f, _C_ib, 1);

            STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

            W_i_valid -= stride;
            O_ptr += _C_ob;
            I_ptr += stride * _C_ib;
        }

        H_i_valid -= stride;
        l++;
        I_row += stride * input_col_stride;
    }
}

template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void dw_kernel_padding_left(
    uint32_t stride,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t W_padding)
{
    DEF_TILE_C_END(_W_ob, _C_ob);
    // left padding elements
    float *O_ptr = O;
    float *I_ptr = I;

    int W_i_valid = W_padding;

    for (uint32_t k_p = 0; k_p < W_padding; k_p += stride)
    {

        ZERO_END_C(1, _C_ob);


        COMPUTE_DW_PADDING(0, H_f, W_i_valid, W_f, W_f, _C_ib, 1);

        STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

        W_i_valid -= stride;
        O_ptr += _C_ob;
        I_ptr += stride * _C_ib;
    }
}

template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void dw_kernel(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O)
{
    DEF_TILE_C(_W_ob, _C_ob);
    ZERO_TILE_C(_W_ob, _C_ob);

    // int updates = 0;
    // uint32_t step = _C_ob;//stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int filter_offset_h = n * W_f * _C_ib * _C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
            //This is C_ob because the microkernel stretches across groups
            int input_stencil_w = m * _C_ob + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                // int p_cur = ii;

                DW_TILE_C(step, a, b, _W_ob, _C_ob);
            }
        }
    }

    STORE_TILE_C(O, _W_ob, _C_ob);
    // printf("output %f \n", O[0]);
    // fflush(0);
}

template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void dw_kernel_end(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t W_last,
    uint32_t W_padding)
{

    DEF_TILE_C_END(_W_ob, _C_ob);
    ZERO_END_C(W_last, _C_ob);


    // int updates = 0;
    // uint32_t step = _C_ob;//stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int filter_offset_h = n * W_f * _C_ib * _C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
        for (uint32_t m = 0; m < W_f; m++)
        {

            int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
            int input_stencil_w = m * _C_ob + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
                // int p_cur = ii;
                DW_END_C(step, a, b, W_last, _C_ob);
            }
        }
    }

    STORE_END_C(O, _W_ob, _C_ob, W_last);
    // printf("%.5f %.5f  %.5f  %.5f %.5f %.5f %.5f %.5f \n", O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7]);

    float * O_padded = O +  W_last*_C_ob;
    // DEF_TILE_C_END(_W_ob, _C_ob);
    ZERO_END_C(W_padding, _C_ob);
    for (uint32_t n = 0; n < H_f; n++)
    {

        int filter_offset_h = n * W_f * _C_ib * _C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
        for (uint32_t m = 0; m < (W_f - W_padding); m++)
        {

            int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
            int input_stencil_w = m * _C_ob + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
                // int p_cur = ii;
                DW_END_C(step, a, b, W_padding, _C_ob);
            }
        }
    }
    STORE_END_C(O_padded, _W_ob, _C_ob, W_padding);
}

template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void dw_kernel_padding_right(
    uint32_t stride,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t W_last,
    uint32_t W_padding)
{

    DEF_TILE_C_END(_W_ob, _C_ob);
    ZERO_END_C(W_last, _C_ob);

    // int updates = 0;
    // uint32_t step = _C_ob;//stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int filter_offset_h = n * W_f * _C_ib * _C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
        for (uint32_t m = 0; m < W_f; m++)
        {

            int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
            int input_stencil_w = m * _C_ob + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
                // int p_cur = ii;
                DW_END_C(step, a, b, W_last, _C_ob);
            }
        }
    }

    STORE_END_C(O, _W_ob, _C_ob, W_last);
    // printf("%.5f %.5f  %.5f  %.5f %.5f %.5f %.5f %.5f \n", O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7]);

    //right padding elements
    float *O_ptr = O + W_last * C_ob;
    float *I_ptr = I + (W_last * stride) * _C_ib;

    int W_i_valid = W_f - W_padding;

    for (uint32_t k_p = 0; k_p < W_padding; k_p += stride)
    {

        ZERO_END_C(1, _C_ob);

        COMPUTE_DW_PADDING(0, H_f, 0, W_i_valid, W_f, _C_ib, 1);

        STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

        W_i_valid -= stride;
        O_ptr += _C_ob;
        I_ptr += stride * _C_ib;
    }
}

template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void dw_kernel_padding_bottom(
    uint32_t stride,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t H_b_padding,
    uint32_t W_o,
    uint32_t W_last,
    uint32_t W_f_padding,
    uint32_t W_b_padding,
    uint32_t W_pad_row,
    uint32_t W_total)
{
    // printf("calling bottom padding element computation \n");
    DEF_TILE_C_END(_W_ob, _C_ob);

    uint32_t H_i_valid = H_f - H_b_padding; // Which filter rows are multiplied with non-zeros
    uint32_t W_i_valid = W_f_padding;       // Which filter elements are multiplied with non-zeros

    // printf("W_i_valid:%d \t H_i_valid:%d \n", W_i_valid, H_i_valid);
    float *I_row = I;
    // Iterate over each padded output row
    uint32_t l = 0;
    for (uint32_t l_p = 0; l_p < H_b_padding; l_p += stride)
    {
        // printf("row : %d\n", l_p + 1);
        float *O_row = O + l * (W_total * _C_ob);
        float *O_ptr = O_row;
        float *I_ptr = I_row;

        // padded row elements
        // uint32_t k = 0;
        for (uint32_t k_p = 0; k_p < W_f_padding; k_p += stride)
        {
            // printf("\t col : %d\n", k_p + 1);

            // O_ptr = O_row + k * _C_ob;

            ZERO_END_C(1, _C_ob);


            COMPUTE_DW_PADDING(0, H_i_valid, W_i_valid, W_f, W_f, _C_ib, 1);

            STORE_END_C(O_ptr, _W_ob, _C_ob, 1);
            // printf("Output after padding %.2f \n", O_ptr[0]);
            W_i_valid -= stride;
            O_ptr += _C_ob;
            // k++;
        }

        // int k_w_output = W_f_padding / stride + (W_f_padding % stride != 0);
        // printf("calc v sim %d %d %d  %d \n", k, k_w_output, k == k_w_output, k_w_output + );

        // full row elements
        I_ptr = I_row + W_pad_row * _C_ib;
        // float *I_ptr = I_row + W_pad_row * _C_ib;
        {
            // float * O_ptr = O + k * _C_ob;
            DEF_TILE_C(_W_ob, _C_ob);

            for (uint32_t k = 0; k < W_o; k += _W_ob)
            {

                ZERO_TILE_C(_W_ob, _C_ob);

                for (uint32_t n = 0; n < H_i_valid; n++)
                {

                    int filter_offset_h = n * W_f * _C_ib * _C_ob;
                    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

                    for (uint32_t m = 0; m < W_f; m++)
                    {

                        int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
                        int input_stencil_w = m * _C_ib + input_stencil_h;

                        float *b = F + filter_offset_w;
                        float *a = I_ptr + input_stencil_w;
                        for (uint32_t ii = 0; ii < _C_ib / UNROLL; ii++)
                        {

                            // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                            int p_cur = ii;

                            // FMA_TILE_C(step, a, b, p_cur, _W_ob, _C_ob);
                            DW_END_C(step, a, b, W_last, _C_ob);
                        }
                    }
                }

                STORE_TILE_C(O_ptr, _W_ob, _C_ob);

                O_ptr += _W_ob * _C_ob;
                I_ptr += stride * _W_ob * _C_ib;
            }
        }

        // clean up tile elements
        //  O_ptr = O_row + (k_w_output + W_o)*_C_ob;

        ZERO_END_C(W_last, _C_ob);

        COMPUTE_DW_PADDING(0, H_i_valid, 0, W_f, W_f, _C_ib, W_last);

        STORE_END_C(O_ptr, _W_ob, _C_ob, W_last);

        // back padded row elements
        W_i_valid = W_f - W_b_padding;

        I_ptr += (W_last * stride) * _C_ib;
        // printf("I ptr index : %d\n", (I_ptr - I_row)/_C_ib);
        O_ptr += (W_last)*_C_ob;
        // printf("O ptr index : %d\n", (O_ptr - O) / _C_ob);
        for (uint32_t k_p = 0; k_p < W_b_padding; k_p += stride)
        {
            // float *O_ptr = O_row + k * _C_ob;

            ZERO_END_C(1, _C_ob);

            COMPUTE_DW_PADDING(0, H_i_valid, 0, W_i_valid, W_f, _C_ib, 1);
            STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

            W_i_valid -= stride;
            O_ptr += _C_ob;
            I_ptr += stride * _C_ib;
        }

        H_i_valid -= stride;
        l++;
        I_row += stride * input_col_stride;
    }
}

// template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
// inline void dw_kernel_padding_end(
//     uint32_t H_f, uint32_t W_f,
//     uint32_t input_col_stride,
//     float *I,
//     float *F,
//     float *O,
//     uint32_t H_padding,
//     uint32_t W_o,
//     uint32_t W_last,
//     uint32_t W_padding)
// {


//     DEF_TILE_C_END(_W_ob, _C_ob);
//     ZERO_END_C(W_last, _C_ob);

//     // int updates = 0;
//     // uint32_t step = _C_ob;//stride*_C_ob;
//     // int count = 0;
//     for (uint32_t n = 0; n < H_f; n++)
//     {

//         int filter_offset_h = n * W_f * _C_ib * _C_ob;
//         int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
//         for (uint32_t m = 0; m < W_f; m++)
//         {

//             int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
//             int input_stencil_w = m * _C_ob + input_stencil_h;

//             float *b = F + filter_offset_w;
//             float *a = I + input_stencil_w;
//             for (uint32_t ii = 0; ii < _C_ib; ii++)
//             {

//                 // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
//                 // int p_cur = ii;
//                 DW_END_C(step, a, b, W_last, _C_ob);
//             }
//         }
//     }

//     STORE_END_C(O, _W_ob, _C_ob, W_last);
//     // printf("%.5f %.5f  %.5f  %.5f %.5f %.5f %.5f %.5f \n", O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7]);

//     float *O_padded = O + W_last * _C_ob;
//     // DEF_TILE_C_END(_W_ob, _C_ob);
//     ZERO_END_C(W_padding, _C_ob);
//     for (uint32_t n = 0; n < H_f; n++)
//     {

//         int filter_offset_h = n * W_f * _C_ib * _C_ob;
//         int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;
//         for (uint32_t m = 0; m < (W_f - W_padding); m++)
//         {

//             int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
//             int input_stencil_w = m * _C_ob + input_stencil_h;

//             float *b = F + filter_offset_w;
//             float *a = I + input_stencil_w;
//             for (uint32_t ii = 0; ii < _C_ib; ii++)
//             {

//                 // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
//                 // int p_cur = ii;
//                 DW_END_C(step, a, b, W_padding, _C_ob);
//             }
//         }
//     }
//     STORE_END_C(O_padded, _W_ob, _C_ob, W_padding);
// }










// convolution kernels
// printf("H_lb:%d, W_lb:%d \nH_ub:%d, W_ub:%d\n", H_lb, W_lb, H_ub, W_ub);

#define COMPUTE_W_PADDING(H_lb, H_ub, W_lb, W_ub, W_f, input_channels, W_elements)                         \
    for (uint32_t n = H_lb; n < H_ub; n++)                                                                 \
    {                                                                                                      \
        int filter_offset_h = n * W_f * input_channels * _C_ob;                                            \
        int input_stencil_h = /*input_col_offset +*/ (n - H_lb) * input_col_stride /*+ input_row_offset*/; \
        for (uint32_t m = W_lb; m < W_ub; m++)                                                             \
        {                                                                                                  \
            int filter_offset_w = m * input_channels * _C_ob + filter_offset_h;                            \
            int input_stencil_w = (m - W_lb) * input_channels + input_stencil_h;                           \
            float *b = F + filter_offset_w;                                                                \
            float *a = I_ptr + input_stencil_w;                                                            \
            for (uint32_t ii = 0; ii < input_channels; ii++)                                               \
            {                                                                                              \
                int p_cur = ii;                                                                            \
                /* if (W_elements == _W_ob) */                                                               \
                /*  FMA_TILE_C(step, a, b, p_cur, _W_ob, _C_ob);*/                                         \
                /* else */\
                FMA_END_C(step, a, b, p_cur, _W_ob, _C_ob, W_elements);                                         \
            }                                                                                              \
        }                                                                                                  \
    }



// all initial cases specialize for problem size I_C < C_ib
template <uint32_t _W_ob, uint32_t _C_ob>
inline void initial_conv_kernel_padding_top_combined(
    uint32_t first,
    uint32_t stride,
    uint32_t _C_ib, uint32_t step,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t H_f_padding,
    // uint32_t W_pad_row,
    uint32_t W_o,
    uint32_t W_last,
    uint32_t W_f_padding,
    uint32_t W_b_padding,
    uint32_t W_pad_row,
    uint32_t W_total)
{
    // printf("calling top padding element computation \n");
    DEF_TILE_C_END(_W_ob, _C_ob);

    uint32_t H_i_valid = H_f_padding; // Which filter rows are multiplied with non-zeros
    uint32_t W_i_valid = W_f_padding; // Which filter elements are multiplied with non-zeros

    float *I_row = I;
    // Iterate over each padded output row
    uint32_t l = 0;
    for (uint32_t l_p = 0; l_p < H_f_padding; l_p+=stride)
    {
        // printf("row : %d\n", l_p + 1);
        float * O_row = O +  l *(W_total * _C_ob);
        float * O_ptr = O_row;
        float * I_ptr = I_row;


        // padded row elements
        // uint32_t k = 0;
        for (uint32_t k_p = 0; k_p < W_f_padding; k_p+=stride)
        {

            if (first)
            {
                ZERO_END_C(1, _C_ob);
            }
            else
            {
                LOAD_LAST_C(O_ptr, _W_ob, _C_ob, 1);
            }


            COMPUTE_W_PADDING(H_i_valid, H_f, W_i_valid, W_f, W_f, _C_ib, 1);

            STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

            W_i_valid -= stride;
            O_ptr += _C_ob;
            // k++;
        }

        //full row elements
        I_ptr = I_row + W_pad_row * _C_ib;
        {
            DEF_TILE_C(_W_ob, _C_ob);

            for (uint32_t k = 0; k < W_o; k += _W_ob)
            {
                if (first)
                {
                    ZERO_TILE_C(_W_ob, _C_ob);
                }
                else
                {
                    LOAD_TILE_C(O_ptr, _W_ob, _C_ob);
                }
                for (uint32_t n = H_i_valid; n < H_f; n++)
                {

                    int filter_offset_h = n * W_f * _C_ib * _C_ob;
                    int input_stencil_h = /*input_col_offset +*/ (n - H_i_valid) * input_col_stride /*+ input_row_offset*/;

                    for (uint32_t m = 0; m < W_f; m++)
                    {

                        int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
                        int input_stencil_w = m * _C_ib + input_stencil_h;

                        float *b = F + filter_offset_w;
                        float *a = I_ptr + input_stencil_w;
                        for (uint32_t ii = 0; ii < _C_ib / UNROLL; ii++)
                        {

                            // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                            int p_cur = ii;

                            FMA_TILE_C(step, a, b, p_cur, _W_ob, _C_ob);
                        }
                    }
                }

                STORE_TILE_C(O_ptr, _W_ob, _C_ob);

                O_ptr += _W_ob*_C_ob;
                I_ptr += stride*_W_ob*_C_ib;
            }
        }


        //clean up tile elements
        // O_ptr = O_row + (k + W_o)*_C_ob;
        I_ptr = I_row + (W_pad_row + stride*W_o)* _C_ib;
        if (first)
        {
            ZERO_END_C(W_last, _C_ob);
        }
        else
        {
            LOAD_LAST_C(O_ptr, _W_ob, _C_ob, W_last);
        }

        COMPUTE_W_PADDING(H_i_valid, H_f, 0, W_f, W_f, _C_ib, W_last);

        STORE_END_C(O_ptr, _W_ob, _C_ob, W_last);

        // back padded row elements
        O_ptr += W_last*_C_ob;
        I_ptr += (W_last*stride)*_C_ib;
        W_i_valid = W_f - W_b_padding;

        for (uint32_t k_p = 0; k_p < W_b_padding; k_p+=stride)
        {
            if (first)
            {
                ZERO_END_C(1, _C_ob);
            }
            else
            {
                LOAD_LAST_C(O_ptr, _W_ob, _C_ob, 1);
            }

            COMPUTE_W_PADDING(H_i_valid, H_f, 0, W_i_valid, W_f, _C_ib, 1);

            STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

            W_i_valid -= stride;
            O_ptr += _C_ob;
            I_ptr += stride*_C_ib;
        }

        H_i_valid -= stride;
        l++;
        I_row += stride*input_col_stride;
    }

}


template <uint32_t _W_ob, uint32_t _C_ob>
inline void initial_conv_kernel_combined(
    uint32_t first,
     uint32_t _C_ib, uint32_t step,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O)
{
    // printf("first: %d\n", first);
    DEF_TILE_C(_W_ob, _C_ob);
    if (first)
    {
        ZERO_TILE_C(_W_ob, _C_ob);
    }
    else
    {
        LOAD_TILE_C(O, _W_ob, _C_ob);
    }

    // int updates = 0;
    // uint32_t step = stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int filter_offset_h = n * W_f * _C_ib * _C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
            int input_stencil_w = m * _C_ib + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib / UNROLL; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                int p_cur = ii;

                FMA_TILE_C(step, a, b, p_cur, _W_ob, _C_ob);
            }
        }
    }

    STORE_TILE_C(O, _W_ob, _C_ob);
}

template <uint32_t _W_ob, uint32_t _C_ob>
inline void initial_conv_kernel_end_combined(
    uint32_t first,
    uint32_t _C_ib, uint32_t step,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t W_last,
    uint32_t W_padding)
{
    DEF_TILE_C_END(_W_ob, _C_ob);
    if (first)
    {
        ZERO_END_C(W_last, _C_ob);
    }
    else
    {
        LOAD_LAST_C(O, _W_ob, _C_ob, W_last);
    }

    // int updates = 0;
    // uint32_t step = stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int filter_offset_h = n * W_f * _C_ib * _C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
            int input_stencil_w = m * _C_ib + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib / UNROLL; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                int p_cur = ii;
                FMA_END_C(step, a, b, p_cur, _W_ob, _C_ob, W_last);
            }
        }
    }

    STORE_END_C(O, _W_ob, _C_ob, W_last);
}

template <uint32_t _W_ob, uint32_t _C_ob>
inline void initial_conv_kernel_padding_left_combined(
    uint32_t first,
    uint32_t stride,
    uint32_t _C_ib, uint32_t step,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t W_padding)
{
    DEF_TILE_C_END(_W_ob, _C_ob);
    // left padding elements
    float *O_ptr = O;
    float *I_ptr = I;

    int W_i_valid = W_f - W_padding;

    for (uint32_t k_p = 0; k_p < W_padding; k_p += stride)
    {
        if (first)
        {
            ZERO_END_C(1, _C_ob);
        }
        else
        {
            LOAD_LAST_C(O_ptr, _W_ob, _C_ob, 1);
        }

        // COMPUTE_W_PADDING(0, H_f, 0, W_i_valid, W_f, _C_ib, 1);
        COMPUTE_W_PADDING(0, H_f, W_i_valid, W_f, W_f, _C_ib, 1);

        STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

        W_i_valid -= stride;
        O_ptr += _C_ob;
        I_ptr += stride * _C_ib;
    }
}

template <uint32_t _W_ob, uint32_t _C_ob>
inline void initial_conv_kernel_padding_right_combined(
    uint32_t first,
    uint32_t stride,
    uint32_t _C_ib, uint32_t step,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t W_last,
    uint32_t W_padding)
{
    // printf("%d %d \n", W_last, W_padding);
    DEF_TILE_C_END(_W_ob, _C_ob);
    if (first)
    {
        ZERO_END_C(W_last, _C_ob);
    }
    else
    {
        LOAD_LAST_C(O, _W_ob, _C_ob, W_last);
    }

    // int updates = 0;
    // uint32_t step = stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int filter_offset_h = n * W_f * _C_ib * _C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
            int input_stencil_w = m * _C_ib + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib / UNROLL; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                int p_cur = ii;
                FMA_END_C(step, a, b, p_cur, _W_ob, _C_ob, W_last);
            }
        }
    }

    STORE_END_C(O, _W_ob, _C_ob, W_last);

    //right padding elements


    float * O_ptr = O + W_last * C_ob;
    float *I_ptr = I + (W_last * stride) * _C_ib;

    int W_i_valid = W_f - W_padding;

    for (uint32_t k_p = 0; k_p < W_padding; k_p += stride)
    {
        if (first)
        {
            ZERO_END_C(1, _C_ob);
        }
        else
        {
            LOAD_LAST_C(O_ptr, _W_ob, _C_ob, 1);
        }

        COMPUTE_W_PADDING(0, H_f, 0, W_i_valid, W_f, _C_ib, 1);

        STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

        W_i_valid -= stride;
        O_ptr += _C_ob;
        I_ptr += stride * _C_ib;
    }
}


template <uint32_t _W_ob, uint32_t _C_ob>
inline void initial_conv_kernel_padding_bottom_combined(
    uint32_t first,
    uint32_t stride,
    uint32_t _C_ib, uint32_t step,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t H_b_padding,
    // uint32_t W_pad_row,
    uint32_t W_o,
    uint32_t W_last,
    uint32_t W_f_padding,
    uint32_t W_b_padding,
    uint32_t W_pad_row,
    uint32_t W_total)
{
    // printf("calling bottom padding element computation \n");
    DEF_TILE_C_END(_W_ob, _C_ob);

    uint32_t H_i_valid = H_f - H_b_padding; // Which filter rows are multiplied with non-zeros
    uint32_t W_i_valid = W_f_padding; // Which filter elements are multiplied with non-zeros

    // printf("W_i_valid:%d \t H_i_valid:%d \n", W_i_valid, H_i_valid);
    float *I_row = I;
    // Iterate over each padded output row
    uint32_t l = 0;
    for (uint32_t l_p = 0; l_p < H_b_padding; l_p += stride)
    {
        // printf("row : %d\n", l_p + 1);
        float *O_row = O + l * (W_total * _C_ob);
        float *O_ptr = O_row;
        float *I_ptr = I_row;

        // padded row elements
        // uint32_t k = 0;
        for (uint32_t k_p = 0; k_p < W_f_padding; k_p += stride)
        {
            // printf("\t col : %d\n", k_p + 1);

            // O_ptr = O_row + k * _C_ob;
            if (first)
            {
                ZERO_END_C(1, _C_ob);
            }
            else
            {
                LOAD_LAST_C(O_ptr, _W_ob, _C_ob, 1);
            }

            COMPUTE_W_PADDING(0, H_i_valid, W_i_valid, W_f, W_f, _C_ib, 1);

            STORE_END_C(O_ptr, _W_ob, _C_ob, 1);
            // printf("Output after padding %.2f \n", O_ptr[0]);
            W_i_valid -= stride;
            O_ptr += _C_ob;
            // k++;
        }

        // int k_w_output = W_f_padding / stride + (W_f_padding % stride != 0);
        // printf("calc v sim %d %d %d  %d \n", k, k_w_output, k == k_w_output, k_w_output + );


        // full row elements
        I_ptr = I_row + W_pad_row * _C_ib;
        // float *I_ptr = I_row + W_pad_row * _C_ib;
        {
            // float * O_ptr = O + k * _C_ob;
            DEF_TILE_C(_W_ob, _C_ob);

            for (uint32_t k = 0; k < W_o; k += _W_ob)
            {
                if (first)
                {
                    ZERO_TILE_C(_W_ob, _C_ob);
                }
                else
                {
                    LOAD_TILE_C(O_ptr, _W_ob, _C_ob);
                }

                for (uint32_t n = 0; n < H_i_valid; n++)
                {

                    int filter_offset_h = n * W_f * _C_ib * _C_ob;
                    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

                    for (uint32_t m = 0; m < W_f; m++)
                    {

                        int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
                        int input_stencil_w = m * _C_ib + input_stencil_h;

                        float *b = F + filter_offset_w;
                        float *a = I_ptr + input_stencil_w;
                        for (uint32_t ii = 0; ii < _C_ib / UNROLL; ii++)
                        {

                            // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                            int p_cur = ii;

                            FMA_TILE_C(step, a, b, p_cur, _W_ob, _C_ob);
                        }
                    }
                }

                STORE_TILE_C(O_ptr, _W_ob, _C_ob);

                O_ptr += _W_ob * _C_ob;
                I_ptr += stride * _W_ob * _C_ib;
            }
        }


        // clean up tile elements
        //  O_ptr = O_row + (k_w_output + W_o)*_C_ob;
        if (first)
        {
            ZERO_END_C(W_last, _C_ob);
        }
        else
        {
            LOAD_LAST_C(O_ptr, _W_ob, _C_ob, W_last);
        }

        COMPUTE_W_PADDING(0, H_i_valid, 0, W_f, W_f, _C_ib, W_last);

        STORE_END_C(O_ptr, _W_ob, _C_ob, W_last);



        // back padded row elements
        W_i_valid = W_f - W_b_padding;

        I_ptr += (W_last*stride)* _C_ib;
        // printf("I ptr index : %d\n", (I_ptr - I_row)/_C_ib);
        O_ptr += (W_last)*_C_ob;
        // printf("O ptr index : %d\n", (O_ptr - O) / _C_ob);
        for (uint32_t k_p = 0; k_p < W_b_padding; k_p += stride)
        {
            // float *O_ptr = O_row + k * _C_ob;
            if (first)
            {
                ZERO_END_C(1, _C_ob);
            }
            else
            {
                LOAD_LAST_C(O_ptr, _W_ob, _C_ob, 1);
            }
            COMPUTE_W_PADDING(0, H_i_valid, 0, W_i_valid, W_f, _C_ib, 1);
            STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

            W_i_valid -= stride;
            O_ptr += _C_ob;
            I_ptr += stride * _C_ib;
        }

        H_i_valid -= stride;
        l++;
        I_row += stride * input_col_stride;
    }
}














template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void conv_kernel_padding_top_combined(
    uint32_t first,
    uint32_t stride,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t H_f_padding,
    // uint32_t W_pad_row,
    uint32_t W_o,
    uint32_t W_last,
    uint32_t W_f_padding,
    uint32_t W_b_padding,
    uint32_t W_pad_row,
    uint32_t W_total)
{
    // printf("calling top padding element computation \n");
    DEF_TILE_C_END(_W_ob, _C_ob);

    uint32_t H_i_valid = H_f_padding; // Which filter rows are multiplied with non-zeros
    uint32_t W_i_valid = W_f_padding; // Which filter elements are multiplied with non-zeros

    float *I_row = I;
    // Iterate over each padded output row
    uint32_t l = 0;
    for (uint32_t l_p = 0; l_p < H_f_padding; l_p += stride)
    {
        // printf("row : %d\n", l_p + 1);
        float *O_row = O + l * (W_total * _C_ob);
        float *O_ptr = O_row;
        float *I_ptr = I_row;

        // padded row elements
        // uint32_t k = 0;
        for (uint32_t k_p = 0; k_p < W_f_padding; k_p += stride)
        {

            if (first)
            {
                ZERO_END_C(1, _C_ob);
            }
            else
            {
                LOAD_LAST_C(O_ptr, _W_ob, _C_ob, 1);
            }

            COMPUTE_W_PADDING(H_i_valid, H_f, W_i_valid, W_f, W_f, _C_ib, 1);

            STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

            W_i_valid -= stride;
            O_ptr += _C_ob;
            // k++;
        }

        // full row elements
        I_ptr = I_row + W_pad_row * _C_ib;
        {
            DEF_TILE_C(_W_ob, _C_ob);

            for (uint32_t k = 0; k < W_o; k += _W_ob)
            {
                if (first)
                {
                    ZERO_TILE_C(_W_ob, _C_ob);
                }
                else
                {
                    LOAD_TILE_C(O_ptr, _W_ob, _C_ob);
                }
                for (uint32_t n = H_i_valid; n < H_f; n++)
                {

                    int filter_offset_h = n * W_f * _C_ib * _C_ob;
                    int input_stencil_h = /*input_col_offset +*/ (n - H_i_valid) * input_col_stride /*+ input_row_offset*/;

                    for (uint32_t m = 0; m < W_f; m++)
                    {

                        int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
                        int input_stencil_w = m * _C_ib + input_stencil_h;

                        float *b = F + filter_offset_w;
                        float *a = I_ptr + input_stencil_w;
                        for (uint32_t ii = 0; ii < _C_ib / UNROLL; ii++)
                        {

                            // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                            int p_cur = ii;

                            FMA_TILE_C(step, a, b, p_cur, _W_ob, _C_ob);
                        }
                    }
                }

                STORE_TILE_C(O_ptr, _W_ob, _C_ob);

                O_ptr += _W_ob * _C_ob;
                I_ptr += stride * _W_ob * _C_ib;
            }
        }

        // clean up tile elements
        //  O_ptr = O_row + (k + W_o)*_C_ob;
        I_ptr = I_row + (W_pad_row + stride * W_o) * _C_ib;
        if (first)
        {
            ZERO_END_C(W_last, _C_ob);
        }
        else
        {
            LOAD_LAST_C(O_ptr, _W_ob, _C_ob, W_last);
        }

        COMPUTE_W_PADDING(H_i_valid, H_f, 0, W_f, W_f, _C_ib, W_last);

        STORE_END_C(O_ptr, _W_ob, _C_ob, W_last);

        // back padded row elements
        O_ptr += W_last * _C_ob;
        I_ptr += (W_last * stride) * _C_ib;
        W_i_valid = W_f - W_b_padding;

        for (uint32_t k_p = 0; k_p < W_b_padding; k_p += stride)
        {
            if (first)
            {
                ZERO_END_C(1, _C_ob);
            }
            else
            {
                LOAD_LAST_C(O_ptr, _W_ob, _C_ob, 1);
            }

            COMPUTE_W_PADDING(H_i_valid, H_f, 0, W_i_valid, W_f, _C_ib, 1);

            STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

            W_i_valid -= stride;
            O_ptr += _C_ob;
            I_ptr += stride * _C_ib;
        }

        H_i_valid -= stride;
        l++;
        I_row += stride * input_col_stride;
    }
}

template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void conv_kernel_padding_left_combined(
    uint32_t first,
    uint32_t stride,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t W_padding)
{
    DEF_TILE_C_END(_W_ob, _C_ob);
    // left padding elements
    float *O_ptr = O;
    float *I_ptr = I;

    int W_i_valid = W_padding;

    for (uint32_t k_p = 0; k_p < W_padding; k_p += stride)
    {
        if (first)
        {
            ZERO_END_C(1, _C_ob);
        }
        else
        {
            LOAD_LAST_C(O_ptr, _W_ob, _C_ob, 1);
        }

        COMPUTE_W_PADDING(0, H_f, W_i_valid, W_f, W_f, _C_ib, 1);

        STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

        W_i_valid -= stride;
        O_ptr += _C_ob;
        I_ptr += stride * _C_ib;
    }
}

    template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
    inline void conv_kernel_combined(
        uint32_t first,
        uint32_t H_f, uint32_t W_f,
        uint32_t input_col_stride,
        float *I,
        float *F,
        float *O)
{
    // printf("first: %d\n", first);
    DEF_TILE_C(_W_ob, _C_ob);
    if(first)
    {
        ZERO_TILE_C(_W_ob, _C_ob);
    }
    else
    {
        LOAD_TILE_C(O, _W_ob, _C_ob);
    }


    // int updates = 0;
    // uint32_t step = stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int filter_offset_h = n * W_f * _C_ib * _C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
            int input_stencil_w = m * _C_ib + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib / UNROLL; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                int p_cur = ii;

                FMA_TILE_C(step, a, b, p_cur, _W_ob, _C_ob);
            }
        }
    }

    STORE_TILE_C(O, _W_ob, _C_ob);
}

template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void conv_kernel_end_combined(
    uint32_t first,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t W_last,
    uint32_t W_padding)
{
    DEF_TILE_C_END(_W_ob, _C_ob);
    if(first)
    {
        ZERO_END_C(W_last, _C_ob);
    }
    else
    {
        LOAD_LAST_C(O, _W_ob, _C_ob, W_last);
    }

    // int updates = 0;
    // uint32_t step = stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int filter_offset_h = n * W_f * _C_ib * _C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
            int input_stencil_w = m * _C_ib + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib / UNROLL; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                int p_cur = ii;
                FMA_END_C(step, a, b, p_cur, _W_ob, _C_ob, W_last);
            }
        }
    }

    STORE_END_C(O, _W_ob, _C_ob, W_last);
}

template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void conv_kernel_padding_right_combined(
    uint32_t first,
    uint32_t stride,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t W_last,
    uint32_t W_padding)
{
    // printf("%d %d \n", W_last, W_padding);
    DEF_TILE_C_END(_W_ob, _C_ob);
    if (first)
    {
        ZERO_END_C(W_last, _C_ob);
    }
    else
    {
        LOAD_LAST_C(O, _W_ob, _C_ob, W_last);
    }

    // int updates = 0;
    // uint32_t step = stride*_C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int filter_offset_h = n * W_f * _C_ib * _C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
            int input_stencil_w = m * _C_ib + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < _C_ib / UNROLL; ii++)
            {

                // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                int p_cur = ii;
                FMA_END_C(step, a, b, p_cur, _W_ob, _C_ob, W_last);
            }
        }
    }

    STORE_END_C(O, _W_ob, _C_ob, W_last);

    // right padding elements

    float *O_ptr = O + W_last * C_ob;
    float *I_ptr = I + (W_last * stride) * _C_ib;

    int W_i_valid = W_f - W_padding;

    for (uint32_t k_p = 0; k_p < W_padding; k_p += stride)
    {
        if (first)
        {
            ZERO_END_C(1, _C_ob);
        }
        else
        {
            LOAD_LAST_C(O_ptr, _W_ob, _C_ob, 1);
        }

        COMPUTE_W_PADDING(0, H_f, 0, W_i_valid, W_f, _C_ib, 1);

        STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

        W_i_valid -= stride;
        O_ptr += _C_ob;
        I_ptr += stride * _C_ib;
    }
}

// template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
// inline void conv_kernel_padding_end_combined(
//     uint32_t first,
//     uint32_t H_f, uint32_t W_f,
//     uint32_t input_col_stride,
//     float *I,
//     float *F,
//     float *O,
//     uint32_t H_padding,
//     uint32_t W_o,
//     uint32_t W_last,
//     uint32_t W_padding)
// {
//     DEF_TILE_C_END(_W_ob, _C_ob);
//     if (first)
//     {
//         ZERO_END_C(W_last, _C_ob);
//     }
//     else
//     {
//         LOAD_LAST_C(O, _W_ob, _C_ob, W_last);
//     }

//     // int updates = 0;
//     // uint32_t step = stride*_C_ob;
//     // int count = 0;
//     for (uint32_t n = 0; n < H_f; n++)
//     {

//         int filter_offset_h = n * W_f * _C_ib * _C_ob;
//         int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

//         for (uint32_t m = 0; m < W_f; m++)
//         {

//             int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
//             int input_stencil_w = m * _C_ib + input_stencil_h;

//             float *b = F + filter_offset_w;
//             float *a = I + input_stencil_w;
//             for (uint32_t ii = 0; ii < _C_ib / UNROLL; ii++)
//             {

//                 // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

//                 int p_cur = ii;
//                 FMA_END_C(step, a, b, p_cur, _W_ob, _C_ob, W_last);
//             }
//         }
//     }

//     STORE_END_C(O, _W_ob, _C_ob, W_last);
// }


template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void conv_kernel_padding_bottom_combined(
    uint32_t first,
    uint32_t stride,
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t H_b_padding,
    // uint32_t W_pad_row,
    uint32_t W_o,
    uint32_t W_last,
    uint32_t W_f_padding,
    uint32_t W_b_padding,
    uint32_t W_pad_row,
    uint32_t W_total)
{
    // printf("calling bottom padding element computation \n");
    DEF_TILE_C_END(_W_ob, _C_ob);

    uint32_t H_i_valid = H_f - H_b_padding; // Which filter rows are multiplied with non-zeros
    uint32_t W_i_valid = W_f_padding;       // Which filter elements are multiplied with non-zeros

    // printf("W_i_valid:%d \t H_i_valid:%d \n", W_i_valid, H_i_valid);
    float *I_row = I;
    // Iterate over each padded output row
    uint32_t l = 0;
    for (uint32_t l_p = 0; l_p < H_b_padding; l_p += stride)
    {
        // printf("row : %d\n", l_p + 1);
        float *O_row = O + l * (W_total * _C_ob);
        float *O_ptr = O_row;
        float *I_ptr = I_row;

        // padded row elements
        // uint32_t k = 0;
        for (uint32_t k_p = 0; k_p < W_f_padding; k_p += stride)
        {
            // printf("\t col : %d\n", k_p + 1);

            // O_ptr = O_row + k * _C_ob;
            if (first)
            {
                ZERO_END_C(1, _C_ob);
            }
            else
            {
                LOAD_LAST_C(O_ptr, _W_ob, _C_ob, 1);
            }

            COMPUTE_W_PADDING(0, H_i_valid, W_i_valid, W_f, W_f, _C_ib, 1);

            STORE_END_C(O_ptr, _W_ob, _C_ob, 1);
            // printf("Output after padding %.2f \n", O_ptr[0]);
            W_i_valid -= stride;
            O_ptr += _C_ob;
            // k++;
        }

        // int k_w_output = W_f_padding / stride + (W_f_padding % stride != 0);
        // printf("calc v sim %d %d %d  %d \n", k, k_w_output, k == k_w_output, k_w_output + );

        // full row elements
        I_ptr = I_row + W_pad_row * _C_ib;
        // float *I_ptr = I_row + W_pad_row * _C_ib;
        {
            // float * O_ptr = O + k * _C_ob;
            DEF_TILE_C(_W_ob, _C_ob);

            for (uint32_t k = 0; k < W_o; k += _W_ob)
            {
                if (first)
                {
                    ZERO_TILE_C(_W_ob, _C_ob);
                }
                else
                {
                    LOAD_TILE_C(O_ptr, _W_ob, _C_ob);
                }

                for (uint32_t n = 0; n < H_i_valid; n++)
                {

                    int filter_offset_h = n * W_f * _C_ib * _C_ob;
                    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

                    for (uint32_t m = 0; m < W_f; m++)
                    {

                        int filter_offset_w = m * _C_ib * _C_ob + filter_offset_h;
                        int input_stencil_w = m * _C_ib + input_stencil_h;

                        float *b = F + filter_offset_w;
                        float *a = I_ptr + input_stencil_w;
                        for (uint32_t ii = 0; ii < _C_ib / UNROLL; ii++)
                        {

                            // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                            int p_cur = ii;

                            FMA_TILE_C(step, a, b, p_cur, _W_ob, _C_ob);
                        }
                    }
                }

                STORE_TILE_C(O_ptr, _W_ob, _C_ob);

                O_ptr += _W_ob * _C_ob;
                I_ptr += stride * _W_ob * _C_ib;
            }
        }

        // clean up tile elements
        //  O_ptr = O_row + (k_w_output + W_o)*_C_ob;
        if (first)
        {
            ZERO_END_C(W_last, _C_ob);
        }
        else
        {
            LOAD_LAST_C(O_ptr, _W_ob, _C_ob, W_last);
        }

        COMPUTE_W_PADDING(0, H_i_valid, 0, W_f, W_f, _C_ib, W_last);

        STORE_END_C(O_ptr, _W_ob, _C_ob, W_last);

        // back padded row elements
        W_i_valid = W_f - W_b_padding;

        I_ptr += (W_last * stride) * _C_ib;
        // printf("I ptr index : %d\n", (I_ptr - I_row)/_C_ib);
        O_ptr += (W_last)*_C_ob;
        // printf("O ptr index : %d\n", (O_ptr - O) / _C_ob);
        for (uint32_t k_p = 0; k_p < W_b_padding; k_p += stride)
        {
            // float *O_ptr = O_row + k * _C_ob;
            if (first)
            {
                ZERO_END_C(1, _C_ob);
            }
            else
            {
                LOAD_LAST_C(O_ptr, _W_ob, _C_ob, 1);
            }
            COMPUTE_W_PADDING(0, H_i_valid, 0, W_i_valid, W_f, _C_ib, 1);
            STORE_END_C(O_ptr, _W_ob, _C_ob, 1);

            W_i_valid -= stride;
            O_ptr += _C_ob;
            I_ptr += stride * _C_ib;
        }

        H_i_valid -= stride;
        l++;
        I_row += stride * input_col_stride;
    }
}
