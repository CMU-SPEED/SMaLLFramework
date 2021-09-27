// Header File For different Versions of Fusing Pooling with a Convolution
//scalar version



#include "scalar_intrinsics.h"


template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void pool_kernel(
                            uint32_t input_col_stride,
                            float * I,
                            float * O){


    LOAD_TILE_C_strided(I, step, W_ob_pool, C_ob);
    // printf("%d %.2f\n", (W_ob_pool - 1), c_tile[(W_ob_pool - 1) * C_ob]);

    int updates = 0;
    // uint32_t step = stride*C_ob;
    // int count = 0;

    for(uint32_t m = 1; m < W_f; m++){

        int input_stencil_w = m*C_ib;

        float *a = I + input_stencil_w;
        // printf("%d %d %.2f step %d\t", 0, m, a[(W_ob_pool - 1)*step], step);

        MAX_TILE_C(step, a, W_ob_pool, C_ob);
        // printf("%d %.2f\n", (W_ob_pool - 1), c_tile[(W_ob_pool - 1) * C_ob]);
    }
    for(uint32_t n = 1; n < H_f; n++){

        int input_stencil_h = n * input_col_stride;

        for(uint32_t m = 0; m < W_f; m++){

            int input_stencil_w = m*C_ib + input_stencil_h;

            float *a = I + input_stencil_w;
            // printf("%d %d %.2f \t", n, m, a[(W_ob_pool - 1) * step]);

            MAX_TILE_C(step, a, W_ob_pool, C_ob);
            // printf("%d %.2f\n", (W_ob_pool - 1) , c_tile[(W_ob_pool - 1) * C_ob]);
        }
    }

    // printf("%f\n", I[0]);
    STORE_TILE_C_POOL(O, W_ob_pool, C_ob);

}

template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void pool_kernel_end(
                            uint32_t input_col_stride,
                            float * I,
                            float * O,
                            uint32_t W_last
                           ){


    LOAD_LAST_C_strided(I, step, W_ob_pool, C_ob, W_last);

    int updates = 0;
  // uint32_t step = C_ob;//stride*C_ob;
  // int count = 0;
    for(uint32_t m = 1; m < W_f; m++){
        int input_stencil_w = m*C_ob;
        float *a = I + input_stencil_w;
        MAX_END_C(step, a, W_last, C_ob);
    }
        for(uint32_t n = 1; n < H_f; n++){


        int input_stencil_h =  n * input_col_stride;

        for(uint32_t m = 0; m < W_f; m++){
            int input_stencil_w = m*C_ob + input_stencil_h;
            float *a = I + input_stencil_w;

            MAX_END_C(step, a, W_last, C_ob);
            
        }
    }

    STORE_END_C_POOL(O, W_ob_pool, C_ob, W_last);

}

//Sub-Stencil fused pooling kernels
template <uint32_t step, uint32_t pool_stride, uint32_t pool_H_f, uint32_t pool_W_f>
void row_pool_kernel(
    float *I,
    uint32_t O_row,
    uint32_t O_pool_col,
    uint32_t pool_col_stride,
    float *O_pool,
    uint32_t H_o
    )
{
    //write to as many rows as required
    if(O_row %(pool_stride) == 0 && (O_row + pool_H_f - 1) < H_o)
    {
        float *O_pool_ptr = O_pool + 
                                ((O_row) / pool_stride) * pool_col_stride +
                                     (O_pool_col) * C_ob;

        LOAD_TILE_C_strided(I, step, W_ob_pool, C_ob);
        for (uint32_t m = 1; m < pool_W_f; m++)
        {
            int input_stencil_w = m * C_ob;
            float *a = I + input_stencil_w;
            MAX_TILE_C(step, a, W_ob_pool, C_ob);
        }
        STORE_TILE_C_POOL(O_pool_ptr, W_ob_pool, C_ob);
    }
    for(uint32_t n_p = 1; n_p < pool_H_f; n_p++)
    {
        if ((O_row - n_p) % pool_stride == 0 && (int)(O_row - n_p) >= 0 && (O_row + pool_H_f - (n_p + 1)) < H_o)
        {
            //Load the partial updates
            float *O_pool_ptr = O_pool + 
                                    ((O_row - n_p) / pool_stride) * pool_col_stride +
                                        (O_pool_col) * C_ob;
            LOAD_TILE_C_POOL(O_pool_ptr, W_ob_pool, C_ob); 
            for (uint32_t m = 0; m < pool_W_f; m++)
            {   
                int input_stencil_w = m * C_ob;
                float *a = I + input_stencil_w;
                MAX_TILE_C(step, a, W_ob_pool, C_ob);
            }
            STORE_TILE_C_POOL(O_pool_ptr, W_ob_pool, C_ob);
        }
    }
}

template <uint32_t step, uint32_t pool_stride, uint32_t pool_H_f, uint32_t pool_W_f>
void row_pool_kernel_end(
    float *I,
    uint32_t O_row,
    uint32_t O_pool_col,
    uint32_t pool_col_stride,
    float *O_pool,
    uint32_t H_o,
    uint32_t W_last)
{
    //write to as many rows as required
    if (O_row % (pool_stride) == 0 && (O_row + pool_H_f - 1) < H_o)
    {
        float *O_pool_ptr = O_pool +
                            ((O_row) / pool_stride) * pool_col_stride +
                            (O_pool_col)*C_ob;

        LOAD_LAST_C_strided(I, step, W_ob_pool, C_ob, W_last);
        for (uint32_t m = 1; m < pool_W_f; m++)
        {
            int input_stencil_w = m * C_ob;
            float *a = I + input_stencil_w;
            MAX_END_C(step, a, W_last, C_ob);
        }
        STORE_END_C(O_pool_ptr, W_ob_pool, C_ob, W_last);
    }
    for (uint32_t n_p = 1; n_p < pool_H_f; n_p++)
    {
        if ((O_row - n_p) % pool_stride == 0 && (int)(O_row - n_p) >= 0 && (O_row + pool_H_f - (n_p + 1)) < H_o)
        {
            //Load the partial updates
            float *O_pool_ptr = O_pool +
                               ((O_row - n_p) / pool_stride) * pool_col_stride +
                               (O_pool_col)*C_ob;
            LOAD_LAST_C_POOL(O_pool_ptr, W_ob_pool, C_ob, W_last);
            for (uint32_t m = 0; m < pool_W_f; m++)
            {
                int input_stencil_w = m * C_ob;
                float *a = I + input_stencil_w;
                MAX_END_C(step, a, W_last, C_ob);
            }
            STORE_END_C_POOL(O_pool_ptr, W_ob_pool, C_ob, W_last);
        }
    }
}

// work on the Convolution Register Tile size: W_ob x C_ob
template <uint32_t step, uint32_t H_f, uint32_t W_f, uint32_t pool_stride, uint32_t pool_H_f, uint32_t pool_W_f>
void fused_conv_pool_kernel(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O,
                            uint32_t O_row,
                            uint32_t O_col,
                            uint32_t pool_col_stride,
                            float * O_pool,
                            uint32_t H_o,
                            uint32_t W_o_full
                            ){

    LOAD_TILE_C(O, W_ob, C_ob);

    int updates = 0;
  // uint32_t step = stride*C_ob;
  // int count = 0;
    for(uint32_t n = 0; n < H_f; n++){

        int filter_offset_h = n*W_f*C_ib*C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for(uint32_t m = 0; m < W_f; m++){

            int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
            int input_stencil_w = m*C_ib + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for(uint32_t ii = 0 ; ii < C_ib; ii++){

                // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                int p_cur = ii ;

                FMA_TILE_C(step, a, b, p_cur, W_ob, C_ob);

            }
        }
    }
    // STORE_TILE_C(O, W_ob, C_ob);
    //Fused pooling
    MAX_TILE_IP(pool_col_stride, W_ob, C_ob, pool_stride, pool_H_f, pool_W_f, O_row, O_col, O_pool, H_o, W_o_full) ;
}



template <uint32_t step, uint32_t H_f, uint32_t W_f, uint32_t pool_stride, uint32_t pool_H_f, uint32_t pool_W_f>
inline void fused_conv_pool_kernel_end(
                            uint32_t input_col_stride,
                            float * I,
                            float * F,
                            float * O,
                            uint32_t W_last,
                            uint32_t O_row,
                            uint32_t O_col,
                            uint32_t pool_col_stride,
                            float * O_pool,
                            uint32_t H_o,
                            uint32_t W_o_full
                            ){
    LOAD_LAST_C(O, W_ob, C_ob, W_last);

    int updates = 0;
    // uint32_t step = stride*C_ob;
    // int count = 0;
    for(uint32_t n = 0; n < H_f; n++){

        int filter_offset_h = n*W_f*C_ib*C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for(uint32_t m = 0; m < W_f; m++){

            int filter_offset_w = m*C_ib*C_ob + filter_offset_h;
            int input_stencil_w = m*C_ib + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for(uint32_t ii = 0 ; ii < C_ib; ii++){

            // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

            int p_cur = ii ;

            FMA_END_C(step, a, b, p_cur, W_ob, C_ob, W_last);
            }
        }
    }
    // STORE_END_C(O, W_ob, C_ob, W_last);

    MAX_END_IP(pool_col_stride, W_last, C_ob, pool_stride, pool_H_f, pool_W_f, O_row, O_col, O_pool, H_o, W_o_full) ;


}
