
// assumes stride 1
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
    LOAD_12_C(O);
    // Reuse this tile as much as possible
    for(uint32_t ii = 0 ; ii < C_ob; ii++){
      uint32_t p_cur = ii;
      FMA_12_C(C_ob, a, b, p_cur);
      // count++;
    }
    // Store it out
    STORE_12_C(O);


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
    ZERO_12_C();

    // Reuse this tile as much as possible
    for(uint32_t ii = 0 ; ii < C_ob; ii++){
      uint32_t p_cur = ii;
      FMA_12_C(C_ob, a, b, p_cur);
    }
    // Store it out
    STORE_12_C(O);

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

  LOAD_12_C(O);

  int updates = 0;
  uint32_t step = C_ob;

  float *b = F ;
  float *a = I ;
      for(uint32_t ii = 0 ; ii < C_ib; ii++){

        // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);
        int p_cur = ii ;
        FMA_12_C(C_ob, a, b, p_cur);
        // b0 = _mm256_load_ps(b + (p_cur * C_ob));
        // b1 = _mm256_load_ps(b + (p_cur * C_ob + SIMD));
        // a_reg = _mm256_broadcast_ss(a + (p_cur));
        // p_cur += step;
        // c0 = _mm256_fmadd_ps(a_reg, b0, c0);
        // c1 = _mm256_fmadd_ps(a_reg, b1, c1);
        // a_reg = _mm256_broadcast_ss(a + (p_cur));
        // p_cur += step;
        // c2 = _mm256_fmadd_ps(a_reg, b0, c2);
        // c3 = _mm256_fmadd_ps(a_reg, b1, c3);
        // a_reg = _mm256_broadcast_ss(a +  (p_cur));
        // p_cur += step;
        // c4 = _mm256_fmadd_ps(a_reg, b0, c4);
        // c5 = _mm256_fmadd_ps(a_reg, b1, c5);
        // a_reg = _mm256_broadcast_ss(a +  (p_cur));
        // p_cur += step;
        // c6 = _mm256_fmadd_ps(a_reg, b0, c6);
        // c7 = _mm256_fmadd_ps(a_reg, b1, c7);
        // a_reg = _mm256_broadcast_ss(a +  (p_cur));
        // p_cur += step;
        // c8 = _mm256_fmadd_ps(a_reg, b0, c8);
        // c9 = _mm256_fmadd_ps(a_reg, b1, c9);
        // a_reg = _mm256_broadcast_ss(a +  (p_cur));
        // p_cur += step;
        // c10 = _mm256_fmadd_ps(a_reg, b0, c10);
        // c11 = _mm256_fmadd_ps(a_reg, b1, c11);
      }

      STORE_12_C(O);


}


inline void fused_H_microkernel_start(
                            float * I,
                            float * F,
                            float * O)
{
  __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

  float *b = F;
  float *a = I;
  ZERO_12_C();

  // Reuse this tile as much as possible
  for(uint32_t ii = 0 ; ii < C_ob; ii++){
    uint32_t p_cur = 1;
    FMA_12_C(C_ob, a, b, p_cur);
    // b0 = _mm256_load_ps(b + (p_cur * C_ob));
    // b1 = _mm256_load_ps(b + (p_cur * C_ob + SIMD));
    // a_reg = _mm256_broadcast_ss(a + (p_cur));
    // uint32_t p_cur = ii + C_ob;
    // c0 = _mm256_fmadd_ps(a_reg, b0, c0);
    // c1 = _mm256_fmadd_ps(a_reg, b1, c1);
    // a_reg = _mm256_broadcast_ss(a + (p_cur));
    // p_cur += C_ob;
    // c2 = _mm256_fmadd_ps(a_reg, b0, c2);
    // c3 = _mm256_fmadd_ps(a_reg, b1, c3);
    // a_reg = _mm256_broadcast_ss(a +  (p_cur));
    // p_cur += C_ob;
    // c4 = _mm256_fmadd_ps(a_reg, b0, c4);
    // c5 = _mm256_fmadd_ps(a_reg, b1, c5);
    // a_reg = _mm256_broadcast_ss(a +  (p_cur));
    // p_cur += C_ob;
    // c6 = _mm256_fmadd_ps(a_reg, b0, c6);
    // c7 = _mm256_fmadd_ps(a_reg, b1, c7);
    // a_reg = _mm256_broadcast_ss(a +  (p_cur));
    // p_cur += C_ob;
    // c8 = _mm256_fmadd_ps(a_reg, b0, c8);
    // c9 = _mm256_fmadd_ps(a_reg, b1, c9);
    // a_reg = _mm256_broadcast_ss(a +  (p_cur));
    // p_cur += C_ob;
    // c10 = _mm256_fmadd_ps(a_reg, b0, c10);
    // c11 = _mm256_fmadd_ps(a_reg, b1, c11);
  }
  STORE_12_C(O);// Store it out


}
