// Header File For different Versions of Fusing Pooling with a Convolution
//scalar version




#include "scalar_intrinsics.h"

// void print256_float(__m256 var)
// {
//     float val[8];
//     memcpy(val, &var, sizeof(val));
//     printf("Numerical: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n",
//            val[0], val[1], val[2], val[3], val[4], val[5],
//            val[6], val[7]);
// }

template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step, uint32_t H_f, uint32_t W_f>
inline void dw_kernel(
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O)
{


    ZERO_TILE_C(_W_ob, _C_ob);

    int updates = 0;
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

                int p_cur = ii;

                DW_TILE_C(step, a, b, _W_ob, _C_ob);
            }
        }
    }

    STORE_TILE_C(O, _W_ob, _C_ob);
}

template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step, uint32_t H_f, uint32_t W_f>
inline void dw_kernel_end(
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t W_last)
{

    ZERO_END_C(W_last, _C_ob);

    int updates = 0;
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

                int p_cur = ii;

                DW_END_C(step, a, b, W_last, _C_ob);
                // printf("%d %d %d %.2f %.2f %.2f\n", n, m, ii, a[0], b[0], c_tile[0]);
            }
        }
    }

    STORE_END_C(O, _W_ob, _C_ob, W_last);
}

// //Sub Stencil used pooling kernels
// template <uint32_t step, uint32_t pool_stride, uint32_t pool_H_f, uint32_t pool_W_f>
// void row_dw_kernel(
//     float *I,
//     uint32_t O_row,
//     uint32_t O_pool_col,
//     uint32_t pool_col_stride,
//     float * F_dw,
//     float *O_pool,
//     uint32_t H_o
//     )
// {
//     //write to as many rows as required
//     if(O_row %(pool_stride) == 0 && (O_row + pool_H_f - 1) < H_o)
//     {
//         float *O_pool_ptr = O_pool +
//                                 ((O_row) / pool_stride) * pool_col_stride +
//                                      (O_pool_col) * _C_ob;

//         float *b = F_dw;
//         LOAD_TILE_C_strided_DW(I, step, _W_ob, _C_ob);
//         MUL_TILE_C(b, _W_ob, _C_ob);
//         b+=_C_ob;
//         for (uint32_t m = 1; m < pool_W_f; m++)
//         {
//             int input_stencil_w = m * _C_ob;
//             float *a = I + input_stencil_w;
//             DW_TILE_C(step, a, b, _W_ob, _C_ob);
//             b+=_C_ob;
//         }
//         STORE_TILE_C(O_pool_ptr, _W_ob, _C_ob);
//     }
//     for(uint32_t n_p = 1; n_p < pool_H_f; n_p++)
//     {
//         if ((O_row - n_p) % pool_stride == 0 && (int)(O_row - n_p) >= 0 && (O_row + pool_H_f - (n_p + 1)) < H_o)
//         {
//             //Load the partial updates
//             float * O_pool_ptr = O_pool +
//                                     ((O_row - n_p) / pool_stride) * pool_col_stride +
//                                         (O_pool_col) * _C_ob;
//             float * b = F_dw + n_p*(pool_W_f)*_C_ob;
//             LOAD_TILE_C_DW(O_pool_ptr, _W_ob, _C_ob);
//             for (uint32_t m = 0; m < pool_W_f; m++)
//             {
//                 int input_stencil_w = m * _C_ob;
//                 float *a = I + input_stencil_w;
//                 DW_TILE_C(step, a, b, _W_ob, _C_ob);
//                 b += _C_ob;
//             }
//             STORE_TILE_C(O_pool_ptr, _W_ob, _C_ob);
//         }
//     }
// }

// template <uint32_t step, uint32_t pool_stride, uint32_t pool_H_f, uint32_t pool_W_f>
// void row_dw_kernel_end(
//     float *I,
//     uint32_t O_row,
//     uint32_t O_pool_col,
//     uint32_t pool_col_stride,
//     float *F_dw,
//     float *O_pool,
//     uint32_t H_o,
//     uint32_t W_last)
// {
//     // printf("%d last \n", W_last);
//         //write to as many rows as required
//         if (O_row % (pool_stride) == 0 && (O_row + pool_H_f - 1) < H_o)
//     {
//         float *O_pool_ptr = O_pool +
//                             ((O_row) / pool_stride) * pool_col_stride +
//                             (O_pool_col)*_C_ob;

//         float *b = F_dw;
//         LOAD_LAST_C_strided(I, step, _W_ob, _C_ob, W_last);
//         // printf("%.2f %.2f \n", c_tile[0], I[0]);
//         MUL_END_C(b, W_last, _C_ob);
//         // printf("%.2f %.2f \n", c_tile[0], I[0]);

//             b += _C_ob;
//         for (uint32_t m = 1; m < pool_W_f; m++)
//         {
//             int input_stencil_w = m * _C_ob;
//             float *a = I + input_stencil_w;
//             DW_END_C(step, a, b, W_last, _C_ob);
//             // printf("%.2f %.2f \n", c_tile[0], a[0]);

//             b += _C_ob;
//         }
//         STORE_END_C_POOL(O_pool_ptr, _W_ob, _C_ob, W_last);
//     }
//     for (uint32_t n_p = 1; n_p < pool_H_f; n_p++)
//     {
//         if ((O_row - n_p) % pool_stride == 0 && (int)(O_row - n_p) >= 0 && (O_row + pool_H_f - (n_p + 1)) < H_o)
//         {
//             //Load the partial updates
//             float * O_pool_ptr = O_pool +
//                             ((O_row - n_p) / pool_stride) * pool_col_stride +
//                             (O_pool_col)*_C_ob;
//             float *b = F_dw + n_p * (pool_W_f)*_C_ob;
//             LOAD_LAST_C_POOL(O_pool_ptr, _W_ob, _C_ob, W_last);
//             for (uint32_t m = 0; m < pool_W_f; m++)
//             {
//                 int input_stencil_w = m * _C_ob;
//                 float *a = I + input_stencil_w;
//                 DW_END_C(step, a, b, W_last, _C_ob);
//                 // printf("%.2f %.2f \n", c_tile[0], a[0]);

//                 b += _C_ob;
//             }
//             STORE_END_C_POOL(O_pool_ptr, _W_ob, _C_ob, W_last);
//             // printf("%.2f \n", O_pool_ptr[0]);
//         }
//     }
    
// }

// // work on the Convolution Register Tile size: _W_ob x _C_ob
// template <uint32_t step, uint32_t H_f, uint32_t W_f, uint32_t pool_stride, uint32_t pool_H_f, uint32_t pool_W_f>
// void fused_conv_dw_kernel(
//     uint32_t input_col_stride,
//     float *I,
//     float *F,
//     float *O,
//     uint32_t O_row,
//     uint32_t O_col,
//     uint32_t pool_col_stride,
//     float * F_dw,
//     float *O_pool,
//     uint32_t H_o,
//     uint32_t W_o_full)
// {

//     LOAD_TILE_C(O, _W_ob, _C_ob);

//     int updates = 0;
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
//             for (uint32_t ii = 0; ii < _C_ib; ii++)
//             {

//                 // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

//                 int p_cur = ii;

//                 FMA_TILE_C(step, a, b, p_cur, _W_ob, _C_ob);
//             }
//         }
//     }
//     //buffer the outputs locally
//     STORE_TILE_INTER(_W_ob, _C_ob);
//     //Fused pooling
//     DW_TILE_IP(pool_col_stride, _W_ob, _C_ob, pool_stride, pool_H_f, pool_W_f, F_dw, O_row, O_col, O_pool, H_o, W_o_full);
// }

// template <uint32_t step, uint32_t H_f, uint32_t W_f, uint32_t pool_stride, uint32_t pool_H_f, uint32_t pool_W_f>
// inline void fused_conv_dw_kernel_end(
//     uint32_t input_col_stride,
//     float *I,
//     float *F,
//     float *O,
//     uint32_t W_last,
//     uint32_t O_row,
//     uint32_t O_col,
//     uint32_t pool_col_stride,
//     float * F_dw,
//     float *O_pool,
//     uint32_t H_o,
//     uint32_t W_o_full)
// {
//     LOAD_LAST_C(O, _W_ob, _C_ob, W_last);

//     int updates = 0;
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
//             for (uint32_t ii = 0; ii < _C_ib; ii++)
//             {

//                 // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

//                 int p_cur = ii;

//                 FMA_END_C(step, a, b, p_cur, _W_ob, _C_ob, W_last);
//             }
//         }
//     }
//     // STORE_END_C(O, _W_ob, _C_ob, W_last);

//     DW_END_IP(pool_col_stride, W_last, _C_ob, pool_stride, pool_H_f, pool_W_f, F_dw, O_row, O_col, O_pool, H_o, W_o_full);
// }
