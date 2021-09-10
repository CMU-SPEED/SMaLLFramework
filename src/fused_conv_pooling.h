#include "pool_kernel.h"



template <uint32_t stride, uint32_t H_f, uint32_t W_f>
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
uint32_t W_o_pool_full , H_o_pool;
op_dim(W_o, POOL_STRIDE, POOL_KERNEL, W_o_pool_full);
op_dim(H_o, POOL_STRIDE, POOL_KERNEL, H_o_pool);
uint32_t W_o_pool = (W_o_pool_full/W_ob_pool)*W_ob_pool;
uint32_t W_pool_last = W_o_pool_full - W_o_pool;
  // printf("\n %d %d to %d %d\n",H_i, W_i,H_o, W_o);
  // H_o -= (H_o%2==0);
uint32_t offset = 0;
#if PARALLEL==1
#pragma omp parallel for
#endif
for(uint32_t j = 0; j < C; j+=C_ob)
{
		float * I_block_ptr = I + (j/C_ob)*H_o*W_o*C_ob;
		float * O_block_ptr = O + (j/C_ob)*H_o_pool*W_o_pool_full*C_ob;
		float * I_row_ptr = I_block_ptr;
		float * O_row_ptr = O_block_ptr;
		uint32_t input_row_stride = W_o*C_ob;
		for(uint32_t l = 0; l < H_o_pool; l++){
			float * I_ptr = I_row_ptr;
			float * O_ptr = O_row_ptr;
			for(uint32_t k = 0; k < W_o_pool; k+= W_ob_pool){
				pool_kernel<stride*C_ob, H_f, W_f>(input_row_stride,I_ptr, O_ptr);
				I_ptr += stride*W_ob_pool*C_ob;
				O_ptr += W_ob_pool*C_ob;
			}
			pool_kernel_end<stride*C_ob, H_f, W_f>(input_row_stride,I_ptr, O_ptr, W_pool_last);
			I_row_ptr += stride*W_o*C_ob;
			O_row_ptr += W_o_pool_full*C_ob;
		}
	}
}



