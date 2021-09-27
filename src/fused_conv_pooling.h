#include "pool_kernel.h"



template <uint32_t pool_stride, uint32_t pool_H_f, uint32_t pool_W_f>
void pooling(
    uint32_t C,
    uint32_t H_o,
    uint32_t W_o_full,
    float * I,
    float * O
)
{
	uint32_t W_o_pool_full , H_o_pool;
	op_dim(W_o_full, pool_stride, pool_W_f, W_o_pool_full);
	op_dim(H_o, pool_stride, pool_H_f, H_o_pool);
	uint32_t W_o_pool = (W_o_pool_full/W_ob_pool)*W_ob_pool;
	uint32_t W_pool_last = W_o_pool_full - W_o_pool;
	// printf("\n %d %d to %d %d\n",H_o, W_o_full,H_o_pool, W_o_pool_full);
	// printf("s: %d k: %d", pool_stride, pool_H_f);
	// printf("%d %d \n ", W_o_pool, W_pool_last);
	// H_o -= (H_o%2==0);
	uint32_t offset = 0;
	#if PARALLEL==1
	#pragma omp parallel for
	#endif
	for(uint32_t j = 0; j < C; j+=C_ob)
	{
		float * I_block_ptr = I + (j/C_ob) * H_o * W_o_full * C_ob;
		float * O_block_ptr = O + (j/C_ob) * H_o_pool * W_o_pool_full * C_ob;
		float * I_row_ptr = I_block_ptr;
		float * O_row_ptr = O_block_ptr;
		uint32_t input_row_pool_stride = W_o_full * C_ob;
		for(uint32_t l = 0; l < H_o_pool; l++)
		{
			float * I_ptr = I_row_ptr;
			float * O_ptr = O_row_ptr;
			for(uint32_t k = 0; k < W_o_pool; k+= W_ob_pool)
			{
				// printf("%.2f\t index %d %dx%dx%d \n", I_ptr[0], (I_ptr - I), pool_stride, W_ob_pool, C_ob);
				pool_kernel<pool_stride * C_ob, pool_H_f, pool_W_f>(input_row_pool_stride,I_ptr, O_ptr);
				I_ptr += pool_stride * W_ob_pool * C_ob;
				// printf("%.2f index %d \n", I_ptr[0], (I_ptr - I));
				O_ptr += W_ob_pool * C_ob;
			}
			pool_kernel_end<pool_stride * C_ob, pool_H_f, pool_W_f>(input_row_pool_stride, I_ptr, O_ptr, W_pool_last);
			I_row_ptr += pool_stride * W_o_full * C_ob;
			O_row_ptr += W_o_pool_full * C_ob;
		}
	}
}

template <uint32_t stride, uint32_t H_f, uint32_t W_f, uint32_t pool_stride, uint32_t pool_H_f, uint32_t pool_W_f>
void channel_block_fused_pooling(
	uint32_t C_i,
	uint32_t C_o,
	// uint32_t H_f,
	// uint32_t W_f,
	uint32_t H_i,
	uint32_t W_i,
	// uint32_t stride,
	float *I,
	float *F,
	float *O_buffers,
	float *O)
{
	uint32_t H_o = 0;
	op_dim(H_i, stride, H_f, H_o);
	uint32_t W_o_full = 0;
	op_dim(W_i, stride, W_f, W_o_full);
	uint32_t W_o = (W_o_full / W_ob) * W_ob;
	uint32_t W_last = W_o_full % W_ob;
	uint32_t W_o_pool_full, H_o_pool;
	op_dim(W_o_full, pool_stride, pool_W_f, W_o_pool_full);
	op_dim(H_o, pool_stride, pool_H_f, H_o_pool);
	uint32_t W_o_pool = (W_o_pool_full / W_ob_pool) * W_ob_pool;
	uint32_t W_pool_last = W_o_pool_full - W_o_pool;
	// printf("%d %d \n ", W_o, W_last);

#if PARALLEL == 1
#pragma omp parallel for
#endif
	for (uint32_t j = 0; j < C_o; j += C_ob)
	{
#if PARALLEL == 1
		int tid = omp_get_thread_num();
		float *O_buffer = O_buffers + (tid) * (H_o * W_o_full * (C_ob));
#else
		float *O_buffer = O_buffers;
#endif
		uint32_t filter_o_c_block = (j / C_ob) * (C_i / C_ib) * H_f * W_f * C_ib * C_ob;

		//First Input Channel Block
		// These are all 0
		uint32_t input_block_offset = (0 / C_ib) * H_i * W_i * C_ib;
		uint32_t filter_i_c_block = (0 / C_ib) * H_f * W_f * C_ib * C_ob + filter_o_c_block;

		float *filter_block_ptr = F + filter_i_c_block;

		for (uint32_t l = 0; l < H_o; l++)
		{

			uint32_t col_offset = l * W_o_full * C_ob;
			uint32_t input_col_offset = (l * stride) * W_i * C_ob + input_block_offset;

			uint32_t input_row_offset = 0;
			float *I_ptr = I + input_col_offset;

			float *O_ptr = O_buffer + col_offset;
			for (uint32_t k = 0; k < W_o; k += W_ob)
			{

				conv_kernel_start<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr);

				I_ptr += stride * W_ob * C_ob;
				O_ptr += W_ob * C_ob;
			}

			conv_kernel_start_end<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
		}

		//Second - Last Channel Block
		for (uint32_t i = C_ib; i < (C_i); i += C_ib)
		{

			uint32_t input_block_offset = (i / C_ib) * H_i * W_i * C_ib;
			uint32_t filter_i_c_block = (i / C_ib) * H_f * W_f * C_ib * C_ob + filter_o_c_block;
			float *filter_block_ptr = F + filter_i_c_block;

			for (uint32_t l = 0; l < H_o; l++)
			{

				uint32_t col_offset = l * W_o_full * C_ob;
				uint32_t input_col_offset = (l * stride) * W_i * C_ob + input_block_offset;

				float *I_ptr = I + input_col_offset;
				float *O_ptr = O_buffer + col_offset;

				for (uint32_t k = 0; k < W_o; k += W_ob)
				{

					// uint32_t input_row_offset = (k * stride) * C_ob;
					// float *I_ptr = I + input_row_offset + input_col_offset;

					conv_kernel<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr);

					I_ptr += stride * W_ob * C_ob;
					O_ptr += W_ob * C_ob;
				}
				conv_kernel_end<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
			}
		}
		//Fused Pooling
		float *I_block_ptr = O_buffer;
		float *O_block_ptr = O + (j / C_ob) * H_o_pool * W_o_pool_full * C_ob;
		float *I_row_ptr = I_block_ptr;
		float *O_row_ptr = O_block_ptr;
		uint32_t input_row_pool_stride = W_o_full * C_ob;
		for(uint32_t l = 0; l < H_o_pool; l++)
		{
			float *I_ptr = I_row_ptr;
			float *O_ptr = O_row_ptr;
			for(uint32_t k = 0; k < W_o_pool; k += W_ob_pool)
			{
				// printf("%.2f\t index %d %dx%dx%d \n", I_ptr[0], (I_ptr - O_buffer), pool_stride ,  W_ob_pool , C_ob);
				pool_kernel<pool_stride * C_ob, pool_H_f, pool_W_f>(input_row_pool_stride, I_ptr, O_ptr);
				I_ptr += pool_stride * W_ob_pool * C_ob;
				// printf("%.2f index %d  \n ", I_ptr[0], (I_ptr - O_buffer));
				O_ptr += W_ob_pool * C_ob;
			}
			pool_kernel_end<pool_stride * C_ob, pool_H_f, pool_W_f>(input_row_pool_stride, I_ptr, O_ptr, W_pool_last);
			I_row_ptr += pool_stride * W_o_full * C_ob;
			O_row_ptr += W_o_pool_full * C_ob;
		}
	}
}

template <uint32_t stride, uint32_t H_f, uint32_t W_f, uint32_t pool_stride, uint32_t pool_H_f, uint32_t pool_W_f>
void row_full_fused_pooling(
	uint32_t C_i,
	uint32_t C_o,
	// uint32_t H_f,
	// uint32_t W_f,
	uint32_t H_i,
	uint32_t W_i,
	// uint32_t stride,
	float *I,
	float *F,
	float *O_buffers,
	float *O)
{
	uint32_t H_o = 0;
	op_dim(H_i, stride, H_f, H_o);
	uint32_t W_o_full = 0;
	op_dim(W_i, stride, W_f, W_o_full);

	uint32_t W_o = (W_o_full / W_ob) * W_ob;
	uint32_t W_last = W_o_full % W_ob;
	uint32_t W_o_pool_full, H_o_pool;
	op_dim(W_o_full, pool_stride, pool_W_f, W_o_pool_full);
	op_dim(H_o, pool_stride, pool_H_f, H_o_pool);
	uint32_t W_o_pool = (W_o_pool_full / W_ob_pool) * W_ob_pool;
	uint32_t W_pool_last = W_o_pool_full - W_o_pool;

#if PARALLEL == 1
#pragma omp parallel for
#endif
	for (uint32_t j = 0; j < C_o; j += C_ob)
	{
#if PARALLEL == 1
		int tid = omp_get_thread_num();
		float *O_buffer = O_buffers + (tid) * (H_o * W_o_full * (C_ob));
#else
		float *O_buffer = O_buffers;
#endif
		uint32_t filter_o_c_block = (j / C_ob) * (C_i / C_ib) * H_f * W_f * C_ib * C_ob;

		//First Input Channel Block
		// These are all 0
		uint32_t input_block_offset = (0 / C_ib) * H_i * W_i * C_ib;
		uint32_t filter_i_c_block = (0 / C_ib) * H_f * W_f * C_ib * C_ob + filter_o_c_block;

		float *filter_block_ptr = F + filter_i_c_block;

		for (uint32_t l = 0; l < H_o; l++)
		{

			uint32_t col_offset = l * W_o_full * C_ob;
			uint32_t input_col_offset = (l * stride) * W_i * C_ob + input_block_offset;

			uint32_t input_row_offset = 0;
			float *I_ptr = I + input_col_offset;

			float *O_ptr = O_buffer + col_offset;
			for (uint32_t k = 0; k < W_o; k += W_ob)
			{

				conv_kernel_start<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr);

				I_ptr += stride * W_ob * C_ob;
				O_ptr += W_ob * C_ob;
			}

			conv_kernel_start_end<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
		}

		//Second - Penultimate Channel Block
		for (uint32_t i = C_ib; i < (C_i - C_ib); i += C_ib)
		{
			uint32_t input_block_offset = (i / C_ib) * H_i * W_i * C_ib;
			uint32_t filter_i_c_block = (i / C_ib) * H_f * W_f * C_ib * C_ob + filter_o_c_block;
			float *filter_block_ptr = F + filter_i_c_block;
			for (uint32_t l = 0; l < H_o; l++)
			{
				uint32_t col_offset = l * W_o_full * C_ob;
				uint32_t input_col_offset = (l * stride) * W_i * C_ob + input_block_offset;
				float *I_ptr = I + input_col_offset;
				float *O_ptr = O_buffer + col_offset;
				for (uint32_t k = 0; k < W_o; k += W_ob)
				{
					// uint32_t input_row_offset = (k * stride) * C_ob;
					// float *I_ptr = I + input_row_offset + input_col_offset;
					conv_kernel<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr);
					I_ptr += stride * W_ob * C_ob;
					O_ptr += W_ob * C_ob;
				}
				conv_kernel_end<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
			}
		}

		//Fused Pooling
		// Last Input Channel Block
		float *pool_I_block_ptr = O_buffer;
		
		float *O_block_ptr = O + (j / C_ob) * H_o_pool * W_o_pool_full * C_ob;
		float *pool_I_row_ptr = pool_I_block_ptr;
		float *O_row_ptr = O_block_ptr;
		uint32_t input_row_pool_stride = W_o_full * C_ob;

		//Compute First H_f Convolution Rows

		uint32_t l_conv = 0; //to keep track of which conv rows need to be computed
		for (uint32_t l = 0; l < H_o_pool; l++)
		{
			//last row required for lth pooling row
			uint32_t conv_u_bound = (l * pool_stride + pool_W_f);
			//Compute new convolution rows
			uint32_t input_block_offset = ((C_i - C_ib) / C_ib) * H_i * W_i * C_ib;
			uint32_t filter_i_c_block = ((C_i - C_ob) / C_ib) * H_f * W_f * C_ib * C_ob + filter_o_c_block;
			float *filter_block_ptr = F + filter_i_c_block;
			for (; l_conv < conv_u_bound; l_conv++)
			{
				uint32_t col_offset = l_conv * W_o_full * C_ob;
				uint32_t input_col_offset = (l_conv * stride) * W_i * C_ob + input_block_offset;
				float *I_ptr = I + input_col_offset;
				float *O_ptr = O_buffer + col_offset;
				for (uint32_t k = 0; k < W_o; k += W_ob)
				{
					// uint32_t input_row_offset = (k * stride) * C_ob;
					// float *I_ptr = I + input_row_offset + input_col_offset;
					conv_kernel<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr);
					I_ptr += stride * W_ob * C_ob;
					O_ptr += W_ob * C_ob;
				}
				conv_kernel_end<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
			}
			

			float *I_ptr = pool_I_row_ptr;
			float *O_ptr = O_row_ptr;
			for (uint32_t k = 0; k < W_o_pool; k += W_ob_pool)
			{

				pool_kernel<pool_stride * C_ob, pool_H_f, pool_W_f>(input_row_pool_stride, I_ptr, O_ptr);
				I_ptr += pool_stride * W_ob_pool * C_ob;
				O_ptr += W_ob_pool * C_ob;
			}
			pool_kernel_end<pool_stride * C_ob, pool_H_f, pool_W_f>(input_row_pool_stride, I_ptr, O_ptr, W_pool_last);
			pool_I_row_ptr += pool_stride * W_o_full * C_ob;
			O_row_ptr += W_o_pool_full * C_ob;
		}
	}
}

template <uint32_t stride, uint32_t H_f, uint32_t W_f, uint32_t pool_stride, uint32_t pool_H_f, uint32_t pool_W_f>
void row_partial_fused_pooling(
	uint32_t C_i,
	uint32_t C_o,
	// uint32_t H_f,
	// uint32_t W_f,
	uint32_t H_i,
	uint32_t W_i,
	// uint32_t stride,
	float *I,
	float *F,
	float *O_buffers,
	float *O)
{
	uint32_t H_o = 0;
	op_dim(H_i, stride, H_f, H_o);
	uint32_t W_o_full = 0;
	op_dim(W_i, stride, W_f, W_o_full);

	uint32_t W_o = (W_o_full / W_ob) * W_ob;
	uint32_t W_last = W_o_full % W_ob;
	uint32_t W_o_pool_full, H_o_pool;
	op_dim(W_o_full, pool_stride, pool_W_f, W_o_pool_full);
	op_dim(H_o, pool_stride, pool_H_f, H_o_pool);
	uint32_t W_o_pool = (W_o_pool_full / W_ob_pool) * W_ob_pool;
	uint32_t W_pool_last = W_o_pool_full - W_o_pool;

#if PARALLEL == 1
#pragma omp parallel for
#endif
	for (uint32_t j = 0; j < C_o; j += C_ob)
	{
#if PARALLEL == 1
		int tid = omp_get_thread_num();
		float *O_buffer = O_buffers + (tid) * (H_o * W_o_full * (C_ob));
#else
		float *O_buffer = O_buffers;
#endif
		uint32_t filter_o_c_block = (j / C_ob) * (C_i / C_ib) * H_f * W_f * C_ib * C_ob;

		//First Input Channel Block
		// These are all 0
		uint32_t input_block_offset = (0 / C_ib) * H_i * W_i * C_ib;
		uint32_t filter_i_c_block = (0 / C_ib) * H_f * W_f * C_ib * C_ob + filter_o_c_block;

		float *filter_block_ptr = F + filter_i_c_block;

		for (uint32_t l = 0; l < H_o; l++)
		{

			uint32_t col_offset = l * W_o_full * C_ob;
			uint32_t input_col_offset = (l * stride) * W_i * C_ob + input_block_offset;

			uint32_t input_row_offset = 0;
			float *I_ptr = I + input_col_offset;

			float *O_ptr = O_buffer + col_offset;
			for (uint32_t k = 0; k < W_o; k += W_ob)
			{

				conv_kernel_start<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr);

				I_ptr += stride * W_ob * C_ob;
				O_ptr += W_ob * C_ob;
			}

			conv_kernel_start_end<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
		}

		//Second - Penultimate Channel Block
		for (uint32_t i = C_ib; i < (C_i - C_ib); i += C_ib)
		{
			uint32_t input_block_offset = (i / C_ib) * H_i * W_i * C_ib;
			uint32_t filter_i_c_block = (i / C_ib) * H_f * W_f * C_ib * C_ob + filter_o_c_block;
			float *filter_block_ptr = F + filter_i_c_block;
			for (uint32_t l = 0; l < H_o; l++)
			{
				uint32_t col_offset = l * W_o_full * C_ob;
				uint32_t input_col_offset = (l * stride) * W_i * C_ob + input_block_offset;
				float *I_ptr = I + input_col_offset;
				float *O_ptr = O_buffer + col_offset;
				for (uint32_t k = 0; k < W_o; k += W_ob)
				{
					conv_kernel<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr);
					I_ptr += stride * W_ob * C_ob;
					O_ptr += W_ob * C_ob;
				}
				conv_kernel_end<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
			}
		}

		//Fused Pooling
		// Last Input Channel Block
		input_block_offset = ((C_i - C_ib) / C_ib) * H_i * W_i * C_ib;
		filter_i_c_block = ((C_i - C_ib) / C_ib) * H_f * W_f * C_ib * C_ob + filter_o_c_block;
		filter_block_ptr = F + filter_i_c_block;

		float *O_pool_block = O + (j / C_ob) * H_o_pool * W_o_pool_full * C_ob;
		uint32_t pool_col_stride = W_o_pool_full * C_ob;
		float *O_pool_ptr = O_pool_block;
		for (uint32_t l = 0; l < H_o; l++)
		{
			uint32_t col_offset = l * W_o_full * C_ob;
			uint32_t input_col_offset = (l * stride) * W_i * C_ob + input_block_offset;
			float *I_ptr = I + input_col_offset;
			float *O_ptr = O_buffer + col_offset;
			for (uint32_t k = 0; k < W_o; k += W_ob)
			{
				uint32_t input_row_offset = (k * stride) * C_ob;
				float *I_ptr = I + input_row_offset + input_col_offset;
				conv_kernel<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr);
				I_ptr += stride * W_ob * C_ob;
				O_ptr += W_ob * C_ob;
			}

			float *O_conv_ptr = O_buffer + col_offset;
			conv_kernel_end<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
			for (uint32_t k = 0; k < W_o_pool; k += W_ob_pool)
			{
				row_pool_kernel<pool_stride * C_ob, pool_stride, pool_H_f, pool_W_f>( O_conv_ptr, l, k, pool_col_stride, O_pool_ptr, H_o);
				O_conv_ptr += pool_stride*W_ob*C_ob;
			}
			row_pool_kernel_end<pool_stride * C_ob, pool_stride, pool_H_f, pool_W_f>(O_conv_ptr, l, W_o_pool, pool_col_stride, O_pool_ptr, H_o, W_last);
		}
	
	}
}


	template <uint32_t stride, uint32_t H_f, uint32_t W_f, uint32_t pool_stride, uint32_t pool_H_f, uint32_t pool_W_f>
	void pixel_block_fused_pooling(
		uint32_t C_i,
		uint32_t C_o,
		// uint32_t H_f,
		// uint32_t W_f,
		uint32_t H_i,
		uint32_t W_i,
		// uint32_t stride,
		float *I,
		float *F,
		float *O_buffers,
		float *O)
	{
		uint32_t H_o = 0;
		op_dim(H_i, stride, H_f, H_o);
		uint32_t W_o_full = 0;
		op_dim(W_i, stride, W_f, W_o_full);

		uint32_t W_o = (W_o_full / W_ob) * W_ob;
		uint32_t W_last = W_o_full % W_ob;
		uint32_t W_o_pool_full, H_o_pool;
		op_dim(W_o_full, pool_stride, pool_W_f, W_o_pool_full);
		op_dim(H_o, pool_stride, pool_H_f, H_o_pool);
		uint32_t W_o_pool = (W_o_pool_full / W_ob_pool) * W_ob_pool;
		uint32_t W_pool_last = W_o_pool_full - W_o_pool;

#if PARALLEL == 1
#pragma omp parallel for
#endif
		for (uint32_t j = 0; j < C_o; j += C_ob)
		{
#if PARALLEL == 1
			int tid = omp_get_thread_num();
			float *O_buffer = O_buffers + (tid) * (H_o * W_o_full * (C_ob));
#else
			float *O_buffer = O_buffers;
#endif
			uint32_t filter_o_c_block = (j / C_ob) * (C_i / C_ib) * H_f * W_f * C_ib * C_ob;

			//First Input Channel Block
			// These are all 0
			uint32_t input_block_offset = (0 / C_ib) * H_i * W_i * C_ib;
			uint32_t filter_i_c_block = (0 / C_ib) * H_f * W_f * C_ib * C_ob + filter_o_c_block;

			float *filter_block_ptr = F + filter_i_c_block;

			for (uint32_t l = 0; l < H_o; l++)
			{

				uint32_t col_offset = l * W_o_full * C_ob;
				uint32_t input_col_offset = (l * stride) * W_i * C_ob + input_block_offset;

				uint32_t input_row_offset = 0;
				float *I_ptr = I + input_col_offset;

				float *O_ptr = O_buffer + col_offset;
				for (uint32_t k = 0; k < W_o; k += W_ob)
				{

					conv_kernel_start<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr);

					I_ptr += stride * W_ob * C_ob;
					O_ptr += W_ob * C_ob;
				}

				conv_kernel_start_end<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
			}

			//Second - Penultimate Input Channel Block
			for (uint32_t i = C_ib; i < (C_i - C_ib); i += C_ib)
			{

				uint32_t input_block_offset = (i / C_ib) * H_i * W_i * C_ib;
				uint32_t filter_i_c_block = (i / C_ib) * H_f * W_f * C_ib * C_ob + filter_o_c_block;
				float *filter_block_ptr = F + filter_i_c_block;

				for (uint32_t l = 0; l < H_o; l++)
				{

					uint32_t col_offset = l * W_o_full * C_ob;
					uint32_t input_col_offset = (l * stride) * W_i * C_ob + input_block_offset;

					float *I_ptr = I + input_col_offset;
					float *O_ptr = O_buffer + col_offset;

					for (uint32_t k = 0; k < W_o; k += W_ob)
					{

						// uint32_t input_row_offset = (k * stride) * C_ob;
						// float *I_ptr = I + input_row_offset + input_col_offset;

						conv_kernel<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr);

						I_ptr += stride * W_ob * C_ob;
						O_ptr += W_ob * C_ob;
					}
					conv_kernel_end<stride * C_ob, H_f, W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr, W_last);
				}
			}

			//Last Input Channel Block
			input_block_offset = ((C_i - C_ib) / C_ib) * H_i * W_i * C_ib;
			filter_i_c_block = ((C_i - C_ib) / C_ib) * H_f * W_f * C_ib * C_ob + filter_o_c_block;
			filter_block_ptr = F + filter_i_c_block;

			float *O_pool_block = O + (j / C_ob) * H_o_pool * W_o_pool_full * C_ob;
			uint32_t pool_col_stride = W_o_pool_full * C_ob;
			float *O_pool_ptr = O_pool_block;
			for (uint32_t l = 0; l < H_o; l++)
			{
				uint32_t col_offset = l * W_o_full * C_ob;
				uint32_t input_col_offset = (l * stride) * W_i * C_ob + input_block_offset;
				float *I_ptr = I + input_col_offset;
				float *O_ptr = O_buffer + col_offset;

				for (uint32_t k = 0; k < W_o; k += W_ob)
				{
					fused_conv_pool_kernel<stride * C_ob, H_f, W_f, pool_stride, pool_H_f, pool_W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr, l, k, pool_col_stride, O_pool_ptr, H_o, W_o_full);
					I_ptr += stride * W_ob * C_ob;
					O_ptr += W_ob * C_ob;
				}
				fused_conv_pool_kernel_end<stride * C_ob, H_f, W_f, pool_stride, pool_H_f, pool_W_f>(W_i * C_ib, I_ptr, filter_block_ptr, O_ptr, W_last, l, W_o, pool_col_stride, O_pool_ptr, H_o, W_o_full);
			}
		}
	}
