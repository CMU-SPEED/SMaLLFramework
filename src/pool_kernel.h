// Header File For different Versions of Fusing Pooling with a Convolution
//scalar version
#define POOL_UNROLL 8

#define POOL_KERNEL 3
#define POOL_STRIDE 2

#define GPOOL_W_ob 6

#define W_ob_pool 3

#include "scalar_intrinsics.h"


template <uint32_t step, uint32_t H_f, uint32_t W_f>
inline void pool_kernel(
                            uint32_t input_col_stride,
                            float * I,
                            float * O){


    LOAD_TILE_C_strided(I, step, W_ob_pool, C_ob);

    int updates = 0;
    // uint32_t step = stride*C_ob;
    // int count = 0;

    for(uint32_t m = 1; m < W_f; m++){

        int input_stencil_w = m*C_ib;

        float *a = I + input_stencil_w;

        MAX_TILE_C(step, a, W_ob_pool, C_ob);

        
    }
    for(uint32_t n = 0; n < H_f; n++){

        int input_stencil_h = n * input_col_stride;

        for(uint32_t m = 0; m < W_f; m++){

            int input_stencil_w = m*C_ib + input_stencil_h;

            float *a = I + input_stencil_w;
        
            MAX_TILE_C(step, a, W_ob_pool, C_ob);

            
        }
    }


    STORE_TILE_C(O, W_ob_pool, C_ob);

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

        int input_stencil_w = m*C_ib;
        float *a = I + input_stencil_w;

        MAX_END_C(step, a, W_ob_pool, C_ob, W_last);


    }
        for(uint32_t n = 0; n < H_f; n++){


        int input_stencil_h =  n * input_col_stride;

        for(uint32_t m = 0; m < W_f; m++){

            int input_stencil_w = m*C_ib + input_stencil_h;
            float *a = I + input_stencil_w;

            MAX_END_C(step, a, W_ob_pool, C_ob, W_last);

            
        }
    }

    STORE_END_C(O, W_ob_pool, C_ob, W_last);

}