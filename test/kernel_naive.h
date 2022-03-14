
// #include <immintrin.h>



// activations kernels
// pooling kernels
template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void activation_kernel(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *O)
{
  ZERO_TILE_C( _W_ob, _C_ob);

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

      float *a = I + input_stencil_w;
      for (uint32_t ii = 0; ii < _C_ib; ii++)
      {

        // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

        int p_cur = ii;

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
    uint32_t W_last)
{

  ZERO_END_C(_W_ob, _C_ob);

  int updates = 0;
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

        int p_cur = ii;
        MAX_END_C(step, a, W_last, _C_ob);
      }
    }
  }

  STORE_END_C(O, _W_ob, _C_ob, W_last);
}

//pooling kernels
template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void pool_kernel(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *O)
{

  LOAD_TILE_C_strided(I, step, _W_ob, _C_ob);

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

      float *a = I + input_stencil_w;
      for (uint32_t ii = 0; ii < _C_ib; ii++)
      {

        // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

        int p_cur = ii;

        MAX_TILE_C(step, a,_W_ob, _C_ob);
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
    uint32_t W_last)
{

  LOAD_LAST_C_strided(I, step, _W_ob, _C_ob, W_last);

  int updates = 0;
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

        int p_cur = ii;
        MAX_END_C(step, a, W_last, _C_ob);
      }
    }
  }

  STORE_END_C(O, _W_ob, _C_ob, W_last);
}

//depthwise kernels
template <int32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void dw_kernel(
    uint32_t H_f, uint32_t W_f,
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
      }
    }
  }

  STORE_END_C(O, _W_ob, _C_ob, W_last);
  // printf("%.5f %.5f  %.5f  %.5f %.5f %.5f %.5f %.5f \n", O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7]);
}



//convolution kernels
template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void conv_kernel(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O)
{

  LOAD_TILE_C(O, _W_ob, _C_ob);

  int updates = 0;
  // uint32_t step = stride*_C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*_C_ib*_C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*_C_ib*_C_ob + filter_offset_h;
      int input_stencil_w = m*_C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;
      for(uint32_t ii = 0 ; ii < _C_ib/UNROLL; ii++){

        // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

        int p_cur = ii ;

        FMA_TILE_C(step, a, b, p_cur, _W_ob, _C_ob);

      }
    }
  }


STORE_TILE_C(O, _W_ob, _C_ob);
}

template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void conv_kernel_start(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O)
{

  ZERO_TILE_C(_W_ob, _C_ob);

  int updates = 0;
  // uint32_t step = _C_ob;//stride*_C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*_C_ib*_C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*_C_ib*_C_ob + filter_offset_h;
      int input_stencil_w = m*_C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;
      for(uint32_t ii = 0 ; ii < _C_ib/UNROLL; ii++){

        // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

        int p_cur = ii;

        FMA_TILE_C(step, a, b, p_cur, _W_ob, _C_ob);

      }
    }
  }


STORE_TILE_C(O,_W_ob, _C_ob);
}

// cleanup convolution kernels
template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void conv_kernel_start_end(
    uint32_t H_f, uint32_t W_f,
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
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*_C_ib*_C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*_C_ib*_C_ob + filter_offset_h;
      int input_stencil_w = m*_C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;
      for(uint32_t ii = 0 ; ii < _C_ib/UNROLL; ii++){

        // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

        int p_cur = ii;

        FMA_END_C(step, a, b, p_cur, _W_ob, _C_ob, W_last);
        // printf("%d %d %d %.2f %.2f %.2f\n", n, m, ii, a[0], b[0], c_tile[0]);
      }
    }
  }

  STORE_END_C(O, _W_ob, _C_ob, W_last);
}

template <uint32_t _W_ob, uint32_t _C_ob, uint32_t _C_ib, uint32_t step>
inline void conv_kernel_end(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t W_last)
{

  LOAD_LAST_C(O, _W_ob, _C_ob, W_last);
  int updates = 0;
  // uint32_t step = stride*_C_ob;
  // int count = 0;
  for(uint32_t n = 0; n < H_f; n++){

    int filter_offset_h = n*W_f*_C_ib*_C_ob;
    int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

    for(uint32_t m = 0; m < W_f; m++){

      int filter_offset_w = m*_C_ib*_C_ob + filter_offset_h;
      int input_stencil_w = m*_C_ib + input_stencil_h;

      float *b = F + filter_offset_w;
      float *a = I + input_stencil_w;
      for(uint32_t ii = 0 ; ii < _C_ib/UNROLL; ii++){

        // kernel_conv(_W_ob,_C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

        int p_cur = ii ;
        FMA_END_C(step, a, b, p_cur, _W_ob, _C_ob, W_last);
      }
    }
  }

  STORE_END_C(O, _W_ob, _C_ob, W_last);
}