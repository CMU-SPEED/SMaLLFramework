// header file for implementations of depthwise convolution and fusion
// a la MobileNet

// Header File For different Versions of Fusing Pooling with a Convolution
#define POOL_UNROLL 8

#define DW_KERNEL 3
// to stay consistent with the pooling example
#define DW_STRIDE 2

#defin W_o_dw 3
template <uint32_t stride, uint32_t H_f, uint32_t W_f>
void inline dw_microkernel(uint32_t row_in,float * I, float * F, float * O)
{
  __m256 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  //mul filter 0,1 and input 0,1
  r0 = _mm256_load_ps(F + 1*C_ob);
  r1 = _mm256_load_ps(F + 1*C_ob + SIMD);

  r2 = _mm256_load_ps(I + 1*C_ob);
  r3 = _mm256_load_ps(I + 1*C_ob + SIMD);

  r4 = _mm256_load_ps(I + 1*stride + 1*C_ob);
  r5 = _mm256_load_ps(I + 1*stride + 1*C_ob + SIMD);

  r6 = _mm256_load_ps(I + 2*stride + 1*C_ob);
  r7 = _mm256_load_ps(I + 2*stride + 1*C_ob + SIMD);

  r2 = _mm256_mul_ps(r2, r0);
  r3 = _mm256_mul_ps(r3, r1);

  r4 = _mm256_mul_ps(r4, r0);
  r5 = _mm256_mul_ps(r5, r1);

  r6 = _mm256_mul_ps(r6, r0);
  r7 = _mm256_mul_ps(r7, r1);

  r0 = _mm256_load_ps(F + 0*C_ob);
  r1 = _mm256_load_ps(F + 0*C_ob + SIMD);

  r8 = _mm256_load_ps(I + 0*C_ob);
  r9 = _mm256_load_ps(I + 0*C_ob + SIMD);

  r10 = _mm256_load_ps(I + 1*stride + 0*C_ob);
  r11 = _mm256_load_ps(I + 1*stride + 0*C_ob + SIMD);

  r12 = _mm256_load_ps(I + 2*stride + 0*C_ob);
  r13 = _mm256_load_ps(I + 2*stride + 0*C_ob + SIMD);

  r2 = _mm256_fmadd_ps(r8, r0, r2);
  r3 = _mm256_fmadd_ps(r9, r1, r3);

  r4 = _mm256_fmadd_ps(r10, r0, r4);
  r5 = _mm256_fmadd_ps(r11, r1, r5);

  r6 = _mm256_fmadd_ps(r12, r0, r6);
  r7 = _mm256_fmadd_ps(r13, r1, r7);

  r0 = _mm256_load_ps(F + 2*C_ob);
  r1 = _mm256_load_ps(F + 2*C_ob + SIMD);

  r14 = _mm256_load_ps(I + 2*stride + 2*C_ob);
  r15 = _mm256_load_ps(I + 2*stride + 2*C_ob + SIMD);

  r2 = _mm256_fmadd_ps(r10, r0, r2);
  r3 = _mm256_fmadd_ps(r11, r1, r3);

  r4 = _mm256_fmadd_ps(r12, r0, r4);
  r5 = _mm256_fmadd_ps(r13, r1, r5);

  r6 = _mm256_fmadd_ps(r14, r0, r6);
  r7 = _mm256_fmadd_ps(r15, r1, r7);

  //compute this tile in SIMD
  float * F_ptr = F + W_f * C_ob;
  float * I_ptr = I + row_in * C_ob;
  for(uint32_t n = 1; n < H_f; n++)
  {
    #pragma GCC unroll
    for(uint32_t m = 0; m < W_f; m++)
    {
      r0 = _mm256_load_ps(F_ptr + m*C_ob);
      r1 = _mm256_load_ps(F_ptr + m*C_ob + SIMD);

      r8 = _mm256_load_ps(I_ptr + m*C_ob);
      r9 = _mm256_load_ps(I_ptr + m*C_ob + SIMD);

      r10 = _mm256_load_ps(I_ptr + 1*stride + 0*C_ob);
      r11 = _mm256_load_ps(I_ptr + 1*stride + 0*C_ob + SIMD);

      r12 = _mm256_load_ps(I_ptr + 2*stride + 0*C_ob);
      r13 = _mm256_load_ps(I_ptr + 2*stride + 0*C_ob + SIMD);

      r2 = _mm256_fmadd_ps(r8, r0, r2);
      r3 = _mm256_fmadd_ps(r9, r1, r3);

      r4 = _mm256_fmadd_ps(r10, r0, r4);
      r5 = _mm256_fmadd_ps(r11, r1, r5);

      r6 = _mm256_fmadd_ps(r12, r0, r6);
      r7 = _mm256_fmadd_ps(r13, r1, r7);
    }
    I_ptr += row_in * C_ob;
    F_ptr += W_f * C_ob;
  }

  //store Tile
  _mm256_store_ps(O, r2);
  _mm256_store_ps(O + SIMD, r3);

  _mm256_store_ps(O + 1*C_ob , r4);
  _mm256_store_ps(O + 1*C_ob + SIMD, r5);

  _mm256_store_ps(O + 2*C_ob , r4);
  _mm256_store_ps(O + 2*C_ob + SIMD, r5);

}
//unfused depthwise convolution
template <uint32_t stride, uint32_t H_f, uint32_t W_f>
void dwise_convolution(
  uint32_t C_i,
  uint32_t C_o,
  uint32_t H_i,
  uint32_t W_i,
  float * I,
  float * F,
  float * O
)
{
  uint32_t H_o, W_o, W_o_int;
  op_dim(H_i, stride,H_f,H_o);
  op_dim(W_i, stride,W_f,W_o);
  W_o_int = (W_o_int/W_o_dw)*(W_o_dw);
  printf("%u %u\n", W_o, W_o_int);
  float *in_ptr = I,
        *out_ptr = O,
        *filter_ptr = F;
  for(uint32_t j = 0; j < C_o; j++)
  {
    for(uint32_t i = 0; i < C_i; i+= C_ib)
    {
      float * O_ptr_row = O + i * W_o * H_o * C_ob;
      float * I_ptr_row = I + i * W_i * H_i * C_ob;
      float * F_ptr = F + i * W_f * H_f * C_ob
      for(uint32_t l = 0; l < H_o; l++)
      {
        float * O_ptr_col = O_ptr_row;
        for(uint32_t k = 0; k < W_o_int; k+= W_o_dw)
        {

          dw_microkernel(I_ptr_col, F_ptr, O_ptr_col);
          O_ptr_col += W_o_dw * C_ob;
          I_ptr_col += stride * W_o_dw * C_ob;
        }
        //clean up tile

        O_ptr_row += W_o*C_ob;
        I_ptr_row += stride*W_i*C_ob;

      }
    }
  }
}
