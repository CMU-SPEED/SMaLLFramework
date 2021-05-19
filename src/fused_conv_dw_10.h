// header file for implementations of depthwise convolution and fusion
// a la MobileNet

// Header File For different Versions of Fusing Pooling with a Convolution
#define POOL_UNROLL 8

#define DW_KERNEL 3
// to stay consistent with the pooling example
#define DW_STRIDE 2

#defin W_o_dw 5

// largest microkernel
template <uint32_t stride, uint32_t H_f, uint32_t W_f>
void inline dw_microkernel(uint32_t row_in,float * I, float * F, float * O)
{
  __m256 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, f0, f1, f2, f3, i0, i1;


  //compute this tile in SIMD
  float * F_ptr = F;
  float * I_ptr = I;
  //multiply the first set
  //middle element
  {
    f0 = _mm256_load_ps(F_ptr + 1*C_ob);
    f1 = _mm256_load_ps(F_ptr + 1*C_ob + SIMD);

    i0 = _mm256_load_ps(I_ptr + 1*C_ob);
    i1 = _mm256_load_ps(I_ptr + 1*C_ob + SIMD);

    c0 = _mm256_mul_ps(f0,i0);
    c1 = _mm256_mul_ps(f1, i1);

    I_ptr += stride*C_ob;

    i0 = _mm256_load_ps(I_ptr);
    i1 = _mm256_load_ps(I_ptr + SIMD);

    c2 = _mm256_mul_ps(f0,i0);
    c3 = _mm256_mul_ps(f1, i1);

    I_ptr += stride*C_ob;

    i0 = _mm256_load_ps(I_ptr);
    i1 = _mm256_load_ps(I_ptr + SIMD);

    c4 = _mm256_mul_ps(f0,i0);
    c5 = _mm256_mul_ps(f1, i1);

    I_ptr += stride*C_ob;

    i0 = _mm256_load_ps(I_ptr);
    i1 = _mm256_load_ps(I_ptr + SIMD);

    c6 = _mm256_mul_ps(f0,i0);
    c7 = _mm256_mul_ps(f1, i1);

    I_ptr += stride*C_ob;

    i0 = _mm256_load_ps(I_ptr);
    i1 = _mm256_load_ps(I_ptr + SIMD);

    c8 = _mm256_mul_ps(f0,i0);
    c9 = _mm256_mul_ps(f1, i1);

  }
  //first and third
  {
    I_ptr = I;
    f0 = _mm256_load_ps(F_ptr + 0*C_ob);
    f1 = _mm256_load_ps(F_ptr + 0*C_ob + SIMD);
    f2 = _mm256_load_ps(F_ptr + 2*C_ob);
    f3 = _mm256_load_ps(F_ptr + 2*C_ob + SIMD);

    // first element 2 updates
    i0 = _mm256_load_ps(I_ptr + 0*C_ob);
    i1 = _mm256_load_ps(I_ptr + 0*C_ob + SIMD);

    c0 = _mm256_fmadd_ps(f0, i0, c0);
    c1 = _mm256_fmadd_ps(f1, i1, c1);

    i0 = _mm256_load_ps(I_ptr + 2*C_ob);
    i1 = _mm256_load_ps(I_ptr + 2*C_ob + SIMD);

    c0 = _mm256_fmadd_ps(f2, i0, c0);
    c1 = _mm256_fmadd_ps(f3, i1, c1);


    //second element
    // reuse the inputs from second element
    c2 = _mm256_fmadd_ps(f0, i0, c2);
    c3 = _mm256_fmadd_ps(f1, i1, c3);
    // reset pointer to load the third input element for each output
    I_ptr += stride*C_ob + 2*C_ob;

    i0 = _mm256_load_ps(I_ptr);
    i1 = _mm256_load_ps(I_ptr + SIMD);

    c2 = _mm256_fmadd_ps(f2, i0, c2);
    c3 = _mm256_fmadd_ps(f3, i1, c3);

    //Third element
    // reuse inputs
    c4 = _mm256_fmadd_ps(f0, i0, c4);
    c5 = _mm256_fmadd_ps(f1, i1, c5);

    I_ptr += stride*C_ob;

    i0 = _mm256_load_ps(I_ptr);
    i1 = _mm256_load_ps(I_ptr + SIMD);

    c4 = _mm256_fmadd_ps(f2, i0, c4);
    c5 = _mm256_fmadd_ps(f3, i1, c5);

    //Fourth element
    // reuse inputs
    c6 = _mm256_fmadd_ps(f0, i0, c6);
    c7 = _mm256_fmadd_ps(f1, i1, c7);

    I_ptr += stride*C_ob;

    i0 = _mm256_load_ps(I_ptr);
    i1 = _mm256_load_ps(I_ptr + SIMD);

    c6 = _mm256_fmadd_ps(f2, i0, c6);
    c7 = _mm256_fmadd_ps(f3, i1, c7);

    //Fifth element
    // reuse inputs
    c6 = _mm256_fmadd_ps(f0, i0, c6);
    c7 = _mm256_fmadd_ps(f1, i1, c7);

    I_ptr += stride*C_ob;

    i0 = _mm256_load_ps(I_ptr);
    i1 = _mm256_load_ps(I_ptr + SIMD);

    c8 = _mm256_fmadd_ps(f2, i0, c8);
    c9 = _mm256_fmadd_ps(f3, i1, c9);

  }

  for(uint32_t n = 1; n < H_f; n++)
  {
    F_ptr = F +  n*  W_f *C_ob;
    I_ptr = I +  n*row_in*C_ob;
    //middle element
    {
      f0 = _mm256_load_ps(F_ptr + 1*C_ob);
      f1 = _mm256_load_ps(F_ptr + 1*C_ob + SIMD);

      i0 = _mm256_load_ps(I_ptr + 1*C_ob);
      i1 = _mm256_load_ps(I_ptr + 1*C_ob + SIMD);

      c0 = _mm256_fmadd_ps(f0, i0, c0);
      c1 = _mm256_fmadd_ps(f1, i1, c1);

      I_ptr += stride*C_ob;

      i0 = _mm256_load_ps(I_ptr);
      i1 = _mm256_load_ps(I_ptr + SIMD);

      c2 = _mm256_fmadd_ps(f0, i0, c2);
      c3 = _mm256_fmadd_ps(f1, i1, c3);

      I_ptr += stride*C_ob;

      i0 = _mm256_load_ps(I_ptr);
      i1 = _mm256_load_ps(I_ptr + SIMD);

      c4 = _mm256_fmadd_ps(f0, i0, c4);
      c5 = _mm256_fmadd_ps(f1, i1, c5);

      I_ptr += stride*C_ob;

      i0 = _mm256_load_ps(I_ptr);
      i1 = _mm256_load_ps(I_ptr + SIMD);

      c6 = _mm256_fmadd_ps(f0, i0, c6);
      c7 = _mm256_fmadd_ps(f1, i1, c7);

      I_ptr += stride*C_ob;

      i0 = _mm256_load_ps(I_ptr);
      i1 = _mm256_load_ps(I_ptr + SIMD);

      c8 = _mm256_fmadd_ps(f0, i0, c8);
      c9 = _mm256_fmadd_ps(f1, i1, c9);

    }
    //first and third
    {
      I_ptr = I + n * row_in * C_ob;
      f0 = _mm256_load_ps(F_ptr + 0*C_ob);
      f1 = _mm256_load_ps(F_ptr + 0*C_ob + SIMD);
      f2 = _mm256_load_ps(F_ptr + 2*C_ob);
      f3 = _mm256_load_ps(F_ptr + 2*C_ob + SIMD);

      // first element 2 updates
      i0 = _mm256_load_ps(I_ptr + 0*C_ob);
      i1 = _mm256_load_ps(I_ptr + 0*C_ob + SIMD);

      c0 = _mm256_fmadd_ps(f0, i0, c0);
      c1 = _mm256_fmadd_ps(f1, i1, c1);

      i0 = _mm256_load_ps(I_ptr + 2*C_ob);
      i1 = _mm256_load_ps(I_ptr + 2*C_ob + SIMD);

      c0 = _mm256_fmadd_ps(f2, i0, c0);
      c1 = _mm256_fmadd_ps(f3, i1, c1);


      //second element
      // reuse the inputs from second element
      c2 = _mm256_fmadd_ps(f0, i0, c2);
      c3 = _mm256_fmadd_ps(f1, i1, c3);
      // reset pointer to load the third input element for each output
      I_ptr += stride*C_ob + 2*C_ob;

      i0 = _mm256_load_ps(I_ptr);
      i1 = _mm256_load_ps(I_ptr + SIMD);

      c2 = _mm256_fmadd_ps(f2, i0, c2);
      c3 = _mm256_fmadd_ps(f3, i1, c3);

      //Third element
      // reuse inputs
      c4 = _mm256_fmadd_ps(f0, i0, c4);
      c5 = _mm256_fmadd_ps(f1, i1, c5);

      I_ptr += stride*C_ob;

      i0 = _mm256_load_ps(I_ptr);
      i1 = _mm256_load_ps(I_ptr + SIMD);

      c4 = _mm256_fmadd_ps(f2, i0, c4);
      c5 = _mm256_fmadd_ps(f3, i1, c5);

      //Fourth element
      // reuse inputs
      c6 = _mm256_fmadd_ps(f0, i0, c6);
      c7 = _mm256_fmadd_ps(f1, i1, c7);

      I_ptr += stride*C_ob;

      i0 = _mm256_load_ps(I_ptr);
      i1 = _mm256_load_ps(I_ptr + SIMD);

      c6 = _mm256_fmadd_ps(f2, i0, c6);
      c7 = _mm256_fmadd_ps(f3, i1, c7);

      //Fifth element
      // reuse inputs
      c6 = _mm256_fmadd_ps(f0, i0, c6);
      c7 = _mm256_fmadd_ps(f1, i1, c7);

      I_ptr += stride*C_ob;

      i0 = _mm256_load_ps(I_ptr);
      i1 = _mm256_load_ps(I_ptr + SIMD);

      c8 = _mm256_fmadd_ps(f2, i0, c8);
      c9 = _mm256_fmadd_ps(f3, i1, c9);

    }

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
