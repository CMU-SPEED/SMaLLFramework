// header file for different versions of fusing a 1x1 convolution with a 3x3
// minimum 32 input channels


//kernels
// no hints
//assumes stride = 1
inline void fused_microkernel(
                         float * I,
                         uint32_t C_o_1x1,
                         float * F_1x1,
                         uint32_t out_panel_size,
                         float * O_1x1
){
  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  // filter format for the 1x1 : C_i/C_ib x C_o/C_ob x C_ib x C_ob
  float * O = O_1x1;
  for(uint32_t j = 0; j < C_o_1x1; j+= C_ob){
    float *b = F_1x1 + (j/C_ob) * C_ob * C_ob;
    float *a = I;
    c0 = _mm256_load_ps(O + (0 * C_ob));
     c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);
     c2 = _mm256_load_ps(O + (1 * C_ob));
     c3 = _mm256_load_ps(O + (1 * C_ob) + SIMD);
     c4 = _mm256_load_ps(O + (2 * C_ob));
     c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);
     c6 = _mm256_load_ps(O + (3 * C_ob));
     c7 = _mm256_load_ps(O + (3 * C_ob) + SIMD);
     c8 = _mm256_load_ps(O + (4 * C_ob));
     c9 = _mm256_load_ps(O + (4 * C_ob) + SIMD);
     c10 = _mm256_load_ps(O + (5 * C_ob));
     c11 = _mm256_load_ps(O + (5 * C_ob) + SIMD);
    // Reuse this tile as much as possible
    for(uint32_t ii = 0 ; ii < C_ob; ii++){
      b0 = _mm256_load_ps(b + (ii * C_ob));
      b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
      a_reg = _mm256_broadcast_ss(a + (ii));
      uint32_t p_cur = ii + C_ob;
      c0 = _mm256_fmadd_ps(a_reg, b0, c0);
      c1 = _mm256_fmadd_ps(a_reg, b1, c1);
      a_reg = _mm256_broadcast_ss(a + (p_cur));
      p_cur += C_ob;
      c2 = _mm256_fmadd_ps(a_reg, b0, c2);
      c3 = _mm256_fmadd_ps(a_reg, b1, c3);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c4 = _mm256_fmadd_ps(a_reg, b0, c4);
      c5 = _mm256_fmadd_ps(a_reg, b1, c5);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c6 = _mm256_fmadd_ps(a_reg, b0, c6);
      c7 = _mm256_fmadd_ps(a_reg, b1, c7);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c8 = _mm256_fmadd_ps(a_reg, b0, c8);
      c9 = _mm256_fmadd_ps(a_reg, b1, c9);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c10 = _mm256_fmadd_ps(a_reg, b0, c10);
      c11 = _mm256_fmadd_ps(a_reg, b1, c11);
      // count++;
    }
    // Store it out
    _mm256_store_ps(O + (0 * C_ob), c0);
    _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
    _mm256_store_ps(O + (1 * C_ob), c2);
    _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
    _mm256_store_ps(O + (2 * C_ob), c4);
    _mm256_store_ps(O + (2 * C_ob + SIMD), c5);
    _mm256_store_ps(O + (3 * C_ob), c6);
    _mm256_store_ps(O + (3 * C_ob + SIMD), c7);
    _mm256_store_ps(O + (4 * C_ob), c8);
    _mm256_store_ps(O + (4 * C_ob + SIMD), c9);
    _mm256_store_ps(O + (5 * C_ob), c10);
    _mm256_store_ps(O + (5 * C_ob + SIMD), c11);

    O = O + out_panel_size;
  }
}
//assumes stride = 1
inline void fused_microkernel_start(
                         float * I,
                         uint32_t C_o_1x1,
                         float * F_1x1,
                         uint32_t out_panel_size,
                         float * O_1x1
){


  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;
  // filter format for the 1x1 : C_i/C_ib x C_o/C_ob x C_ib x C_ob
  float * O = O_1x1;
  for(uint32_t j = 0; j < C_o_1x1; j+= C_ob){
    // float * F_ptr = F + j * C_ob * C_ob;
    float *b = F_1x1 + (j/C_ob) * C_ob * C_ob;
    float *a = I;
    c0 = _mm256_setzero_ps();
     c1 = _mm256_setzero_ps();
     c2 = _mm256_setzero_ps();
     c3 = _mm256_setzero_ps();
     c4 = _mm256_setzero_ps();
     c5 = _mm256_setzero_ps();
     c6 = _mm256_setzero_ps();
     c7 = _mm256_setzero_ps();
     c8 = _mm256_setzero_ps();
     c9 = _mm256_setzero_ps();
     c10 = _mm256_setzero_ps();
     c11 = _mm256_setzero_ps();
    // Reuse this tile as much as possible
    for(uint32_t ii = 0 ; ii < C_ob; ii++){
      b0 = _mm256_load_ps(b + (ii * C_ob));
      b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
      a_reg = _mm256_broadcast_ss(a + (ii));
      uint32_t p_cur = ii + C_ob;
      c0 = _mm256_fmadd_ps(a_reg, b0, c0);
      c1 = _mm256_fmadd_ps(a_reg, b1, c1);
      a_reg = _mm256_broadcast_ss(a + (p_cur));
      p_cur += C_ob;
      c2 = _mm256_fmadd_ps(a_reg, b0, c2);
      c3 = _mm256_fmadd_ps(a_reg, b1, c3);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c4 = _mm256_fmadd_ps(a_reg, b0, c4);
      c5 = _mm256_fmadd_ps(a_reg, b1, c5);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c6 = _mm256_fmadd_ps(a_reg, b0, c6);
      c7 = _mm256_fmadd_ps(a_reg, b1, c7);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c8 = _mm256_fmadd_ps(a_reg, b0, c8);
      c9 = _mm256_fmadd_ps(a_reg, b1, c9);
      a_reg = _mm256_broadcast_ss(a +  (p_cur));
      p_cur += C_ob;
      c10 = _mm256_fmadd_ps(a_reg, b0, c10);
      c11 = _mm256_fmadd_ps(a_reg, b1, c11);
    }
    // Store it out
    _mm256_store_ps(O + (0 * C_ob), c0);
    _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
    _mm256_store_ps(O + (1 * C_ob), c2);
    _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
    _mm256_store_ps(O + (2 * C_ob), c4);
    _mm256_store_ps(O + (2 * C_ob + SIMD), c5);
    _mm256_store_ps(O + (3 * C_ob), c6);
    _mm256_store_ps(O + (3 * C_ob + SIMD), c7);
    _mm256_store_ps(O + (4 * C_ob), c8);
    _mm256_store_ps(O + (4 * C_ob + SIMD), c9);
    _mm256_store_ps(O + (5 * C_ob), c10);
    _mm256_store_ps(O + (5 * C_ob + SIMD), c11);

    O = O + out_panel_size;
  }
}

//Fuse at the h loop
// Re-introduce the blocking on 1x1 output channels

inline void fused_H_microkernel(
                            float * I,
                            float * F,
                            float * O)
{

  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  c0 = _mm256_load_ps(O + (0 * C_ob));
   c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);
   c2 = _mm256_load_ps(O + (1 * C_ob));
   c3 = _mm256_load_ps(O + (1 * C_ob) + SIMD);
   c4 = _mm256_load_ps(O + (2 * C_ob));
   c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);
   c6 = _mm256_load_ps(O + (3 * C_ob));
   c7 = _mm256_load_ps(O + (3 * C_ob) + SIMD);
   c8 = _mm256_load_ps(O + (4 * C_ob));
   c9 = _mm256_load_ps(O + (4 * C_ob) + SIMD);
   c10 = _mm256_load_ps(O + (5 * C_ob));
   c11 = _mm256_load_ps(O + (5 * C_ob) + SIMD);
  int updates = 0;
  uint32_t step = C_ob;


      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        float *b = F ;
        float *a = I ;
        int p_cur = ii ;
        b0 = _mm256_load_ps(b + (ii * C_ob));
        b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        a_reg = _mm256_broadcast_ss(a + (p_cur));
        p_cur += step;
        c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        a_reg = _mm256_broadcast_ss(a +  (p_cur));
        p_cur += step;
        c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        c11 = _mm256_fmadd_ps(a_reg, b1, c11);
      }


  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
  _mm256_store_ps(O + (2 * C_ob), c4);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c5);
  _mm256_store_ps(O + (3 * C_ob), c6);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c7);
  _mm256_store_ps(O + (4 * C_ob), c8);
  _mm256_store_ps(O + (4 * C_ob + SIMD), c9);
  _mm256_store_ps(O + (5 * C_ob), c10);
  _mm256_store_ps(O + (5 * C_ob + SIMD), c11);

}


inline void fused_H_microkernel_start(
                            float * I,
                            float * F,
                            float * O)
{
  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  float *b = F;
  float *a = I;
  c0 = _mm256_setzero_ps();
   c1 = _mm256_setzero_ps();
   c2 = _mm256_setzero_ps();
   c3 = _mm256_setzero_ps();
   c4 = _mm256_setzero_ps();
   c5 = _mm256_setzero_ps();
   c6 = _mm256_setzero_ps();
   c7 = _mm256_setzero_ps();
   c8 = _mm256_setzero_ps();
   c9 = _mm256_setzero_ps();
   c10 = _mm256_setzero_ps();
   c11 = _mm256_setzero_ps();
  // Reuse this tile as much as possible
  for(uint32_t ii = 0 ; ii < C_ob; ii++){
    b0 = _mm256_load_ps(b + (ii * C_ob));
    b1 = _mm256_load_ps(b + (ii * C_ob + SIMD));
    a_reg = _mm256_broadcast_ss(a + (ii));
    uint32_t p_cur = ii + C_ob;
    c0 = _mm256_fmadd_ps(a_reg, b0, c0);
    c1 = _mm256_fmadd_ps(a_reg, b1, c1);
    a_reg = _mm256_broadcast_ss(a + (p_cur));
    p_cur += C_ob;
    c2 = _mm256_fmadd_ps(a_reg, b0, c2);
    c3 = _mm256_fmadd_ps(a_reg, b1, c3);
    a_reg = _mm256_broadcast_ss(a +  (p_cur));
    p_cur += C_ob;
    c4 = _mm256_fmadd_ps(a_reg, b0, c4);
    c5 = _mm256_fmadd_ps(a_reg, b1, c5);
    a_reg = _mm256_broadcast_ss(a +  (p_cur));
    p_cur += C_ob;
    c6 = _mm256_fmadd_ps(a_reg, b0, c6);
    c7 = _mm256_fmadd_ps(a_reg, b1, c7);
    a_reg = _mm256_broadcast_ss(a +  (p_cur));
    p_cur += C_ob;
    c8 = _mm256_fmadd_ps(a_reg, b0, c8);
    c9 = _mm256_fmadd_ps(a_reg, b1, c9);
    a_reg = _mm256_broadcast_ss(a +  (p_cur));
    p_cur += C_ob;
    c10 = _mm256_fmadd_ps(a_reg, b0, c10);
    c11 = _mm256_fmadd_ps(a_reg, b1, c11);
  }
  // Store it out
  _mm256_store_ps(O + (0 * C_ob), c0);
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);
  _mm256_store_ps(O + (1 * C_ob), c2);
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);
  _mm256_store_ps(O + (2 * C_ob), c4);
  _mm256_store_ps(O + (2 * C_ob + SIMD), c5);
  _mm256_store_ps(O + (3 * C_ob), c6);
  _mm256_store_ps(O + (3 * C_ob + SIMD), c7);
  _mm256_store_ps(O + (4 * C_ob), c8);
  _mm256_store_ps(O + (4 * C_ob + SIMD), c9);
  _mm256_store_ps(O + (5 * C_ob), c10);
  _mm256_store_ps(O + (5 * C_ob + SIMD), c11);

}



// C_i >= 32
template <uint32_t stride, uint32_t H_f, uint32_t W_f>
inline void fused_direct_convolution(
  uint32_t C_i,
  uint32_t C_o,
  uint32_t C_o_1x1,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
  float * I,
  float * F,
  float * F_1x1,
  float * O_buffer,
  float * O_1x1
){

  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);

  // buffer is reused for different blocks

  // First output block; don't load the 1x1 output

    // uint32_t block_offset = (j/C_ob)*H_o*W_o*C_ob;
    // uint32_t filter_o_c_block = (0/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;

    //uint32_t filer_1x1_block = (0/C_ob)*C_ob*C_o_1x1;
    // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob /*+ filter_o_c_block*/;
    float *filter_block_ptr = F + filter_i_c_block;

    for(uint32_t l = 0; l < H_o; l++){

        uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

        for(uint32_t k = 0; k < W_o; k += W_ob){

          uint32_t input_row_offset = (k * stride)*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;

          conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

      }
    }

    for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib){
      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob /*+ filter_o_c_block*/;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;
            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

        }
      }
    }

    //for(uint32_t i = C_ib; i < C_i; i += C_ib) (last iteration)

    input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
    filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob /*+ filter_o_c_block*/;
    filter_block_ptr = F + filter_i_c_block;
    for(uint32_t l = 0; l < H_o; l++){

        uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

        uint32_t o_1x1_row_offset = l * W_o * C_ob;
        for(uint32_t k = 0; k < W_o; k += W_ob){

          uint32_t input_row_offset = (k * stride)*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;

          uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
          conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
          fused_microkernel_start(O_buffer + col_offset + k *C_ob,
                                  C_o_1x1,
                                  F_1x1,
                                  H_o*W_o*C_ob,
                                  O_1x1 +
                                  o_1x1_col_offset);
      }

    }
    //}


  float * F_1x1_ptr;
  for(uint32_t j = C_ob; j < C_o; j += C_ob){
    F_1x1_ptr = F_1x1 + (j/C_ob)*(C_o_1x1/C_ob)*(C_ob)*C_ob;
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;


    // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float *filter_block_ptr = F + filter_i_c_block;

    for(uint32_t l = 0; l < H_o; l++){

        uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

        for(uint32_t k = 0; k < W_o; k += W_ob){

          uint32_t input_row_offset = (k * stride)*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;

          conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

      }
    }

    for(uint32_t i = C_ib; i < C_i-C_ib; i += C_ib){
      uint32_t input_block_offset = (i/C_ib)*H_i*W_i*C_ib;
      uint32_t filter_i_c_block = (i/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
      float *filter_block_ptr = F + filter_i_c_block;

      for(uint32_t l = 0; l < H_o; l++){

          uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
          uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){

            uint32_t input_row_offset = (k * stride)*C_ob;
            float * I_ptr = I + input_row_offset + input_col_offset;
            conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);

        }
      }
    }

    //for(uint32_t i = C_ib; i < C_i; i += C_ib) (last iteration)

    input_block_offset = ((C_i - C_ib)/C_ib)*H_i*W_i*C_ib;
    filter_i_c_block = ((C_i - C_ib)/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
    filter_block_ptr = F + filter_i_c_block;
    for(uint32_t l = 0; l < H_o; l++){

        uint32_t col_offset = l*W_o*C_ob /*+ block_offset*/;
        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;

        uint32_t o_1x1_row_offset = l * W_o * C_ob;
        for(uint32_t k = 0; k < W_o; k += W_ob){

          uint32_t input_row_offset = (k * stride)*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;

          uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
          conv_microkernel<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + col_offset + k *C_ob);
          fused_microkernel(O_buffer + col_offset + k *C_ob,
                                  C_o_1x1,
                                  F_1x1_ptr,
                                  H_o*W_o*C_ob,
                                  O_1x1 +
                                  o_1x1_col_offset);
      }
    }
  }
}

template <uint32_t stride, uint32_t H_f, uint32_t W_f>
inline void fused_direct_convolution_c16(
  uint32_t C_i,
  uint32_t C_o,
  uint32_t C_o_1x1,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
  float * I,
  float * F,
  float * F_1x1,
  float * O_buffer,
  float * O_1x1
){

  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);

  // buffer is reused for different blocks

  // First output block; don't load the 1x1 output

    // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob /*+ filter_o_c_block*/;
    float *filter_block_ptr = F + filter_i_c_block;

    for(uint32_t l = 0; l < H_o; l++){


        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
        uint32_t o_1x1_row_offset = l * W_o * C_ob;
        for(uint32_t k = 0; k < W_o; k += W_ob){

          uint32_t input_row_offset = (k * stride)*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;
          uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
          conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer);
          fused_microkernel_start(O_buffer,
                                  C_o_1x1,
                                  F_1x1,
                                  H_o*W_o*C_ob,
                                  O_1x1 +
                                  o_1x1_col_offset);
      }
    }

  float * F_1x1_ptr;
  for(uint32_t j = C_ob; j < C_o; j += C_ob){
    F_1x1_ptr = F_1x1 + (j/C_ob)*(C_o_1x1/C_ob)*(C_ob)*C_ob;
    uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;

    // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;

    float *filter_block_ptr = F + filter_i_c_block;

    for(uint32_t l = 0; l < H_o; l++){

        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
        uint32_t o_1x1_row_offset = l * W_o * C_ob;
        for(uint32_t k = 0; k < W_o; k += W_ob){

          uint32_t input_row_offset = (k * stride)*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;
          uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
          conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer);
          fused_microkernel(O_buffer,
                                  C_o_1x1,
                                  F_1x1_ptr,
                                  H_o*W_o*C_ob,
                                  O_1x1 +
                                  o_1x1_col_offset);
      }
    }

  }
}


template <uint32_t stride, uint32_t H_f, uint32_t W_f>
inline void fused_H_direct_convolution_c16(
  uint32_t C_i,
  uint32_t C_o,
  uint32_t C_o_1x1,
  // uint32_t H_f,
  // uint32_t W_f,
  uint32_t H_i,
  uint32_t W_i,
  // uint32_t stride,
  float * I,
  float * F,
  float * F_1x1,
  float * O_buffer,
  float * O_1x1
){

  uint32_t H_o = 0;
  op_dim(H_i, stride,H_f,H_o);
  uint32_t W_o = 0;
  op_dim(W_i, stride,W_f,W_o);

  // buffer is reused for different blocks

  // First output block; don't load the 1x1 output
  {
    // These are all 0
    uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
    uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob /*+ filter_o_c_block*/;
    float *filter_block_ptr = F + filter_i_c_block;
    float * I_ptr = I;
    for(uint32_t l = 0; l < H_o; l++){


        uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
        for(uint32_t k = 0; k < W_o; k += W_ob){
          I_ptr += k*stride*C_ob;
          uint32_t input_row_offset = k * stride*C_ob;
          float * I_ptr = I + input_row_offset + input_col_offset;
          conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer + k *C_ob);

        }
        uint32_t o_1x1_row_offset = l * W_o * C_ob;
        // reuse the filter 1x1 ptr before storing
        // maximum reuse says to put channels loops outside of this
        // first output channel block of 1x1
        {
          float * O_1x1_ptr = O_1x1 + o_1x1_row_offset;

          for(uint32_t k = 0; k < W_o; k += W_ob){
            if(k > 3){
              printf("%d index", k*C_ob);
            }
            O_1x1_ptr += k * C_ob;
            // conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer);
            fused_H_microkernel_start(
                                    O_buffer + k*C_ob,
                                    F_1x1,
                                    O_1x1_ptr);
          }
        }// end first output channel block of 1x1

        // 2nd to last block of 1x1 outputs
        for(uint32_t j_ = C_ob; j_ < C_o_1x1; j_++){
          // printf("%d channel block\n", j_/C_ob);fflush(0);
          float *F_1x1_ptr = F_1x1 + j_*C_ob*C_ob;
          float * O_1x1_ptr = O_1x1 + j_*(W_o*C_ob)+ o_1x1_row_offset;
          for(uint32_t k = 0; k < W_o; k += W_ob){
            O_1x1_ptr += k * C_ob  ;
            // conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer);
            fused_H_microkernel_start(
                                O_buffer + k*(C_ob),
                                F_1x1_ptr,
                                O_1x1_ptr);
          // printf("%d channel block\n", j_/C_ob);fflush(0);
          }
        }// end 2nd to last block of 1x1 outputs

      }

  }// End first output block of 3x3

  // float * F_1x1_ptr;
  // for(uint32_t j = C_ob; j < C_o; j += C_ob){
  //   F_1x1_ptr = F_1x1 + (j/C_ob)*(C_o_1x1/C_ob)*(C_ob)*C_ob;
  //   uint32_t filter_o_c_block = (j/C_ob)*(C_i/C_ib)*H_f*W_f*C_ib*C_ob;
  //
  //   // These are all 0
  //   uint32_t input_block_offset = (0/C_ib)*H_i*W_i*C_ib;
  //   uint32_t filter_i_c_block = (0/C_ib)*H_f*W_f*C_ib*C_ob + filter_o_c_block;
  //
  //   float *filter_block_ptr = F + filter_i_c_block;
  //
  //   for(uint32_t l = 0; l < H_o; l++){
  //
  //       uint32_t input_col_offset = (l * stride)*W_i*C_ob + input_block_offset;
  //       uint32_t o_1x1_row_offset = l * W_o * C_ob;
  //       for(uint32_t k = 0; k < W_o; k += W_ob){
  //
  //         uint32_t input_row_offset = (k * stride)*C_ob;
  //         float * I_ptr = I + input_row_offset + input_col_offset;
  //         uint32_t o_1x1_col_offset = k * C_ob + o_1x1_row_offset;
  //         conv_microkernel_start<stride*C_ob, H_f, W_f>(W_i*C_ib, I_ptr, filter_block_ptr, O_buffer);
  //         fused_microkernel(O_buffer,
  //                                 C_o_1x1,
  //                                 F_1x1_ptr,
  //                                 H_o*W_o*C_ob,
  //                                 O_1x1 +
  //                                 o_1x1_col_offset);
  //     }
  //   }
  //
  // }
}
