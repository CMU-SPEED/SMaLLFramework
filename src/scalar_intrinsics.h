//scalar versions of all the microkernels for platform portability

// Initializations
#define ZERO_4_C() \
   o2 = _mm256_setzero_ps();\
   o3 = _mm256_setzero_ps();\
   o4 = _mm256_setzero_ps();\
   o5 = _mm256_setzero_ps();\

#define ZERO_2_C() \
  o0 = _mm256_setzero_ps();\
  o1 = _mm256_setzero_ps();\

#define ZERO_12_C() \
  float c_tile[W_ob*C_ob] = {0};



// Loads
#define LOAD_2_C(O, c0, c1) \
  c0 = _mm256_load_ps(O + (0 * C_ob));\
  c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);\

#define LOAD_12_C(O)\
  float c_tile[W_ob*C_ob];\
  for (int kk = 0; kk < W_ob; kk++) {\
      for (int jj = 0; jj < C_ob; jj++) {\
        c_tile[kk*C_ob + jj] = O[kk*C_ob + jj];\
      }\
  }\

#define LOAD_14_C(O) \
  float c_tile[POOL_W_ib*C_ob];\
  for (int kk = 0; kk < POOL_W_ib; kk++) {\
      for (int jj = 0; jj < C_ob; jj++) {\
        c_tile[kk*C_ob + jj] = O[kk*C_ob + jj];\
      }\
  }\


#define LOAD_10_C(O)\
   c0 = _mm256_load_ps(O + (0 * C_ob));\
   c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);\
   c2 = _mm256_load_ps(O + (1 * C_ob));\
   c3 = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
   c4 = _mm256_load_ps(O + (2 * C_ob));\
   c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);\
   c6 = _mm256_load_ps(O + (3 * C_ob));\
   c7 = _mm256_load_ps(O + (3 * C_ob) + SIMD);\
   c8 = _mm256_load_ps(O + (4 * C_ob));\
   c9 = _mm256_load_ps(O + (4 * C_ob) + SIMD);\


#define LOAD_6_C(O, step)\
  c0 = _mm256_load_ps(O + (0 * step));\
  c1 = _mm256_load_ps(O + (0 * step) + SIMD);\
  c2 = _mm256_load_ps(O + (1 * step));\
  c3 = _mm256_load_ps(O + (1 * step) + SIMD);\
  c4 = _mm256_load_ps(O + (2 * step));\
  c5 = _mm256_load_ps(O + (2 * step) + SIMD);\

#define LOAD_4_C(O, step)\
   c0 = _mm256_load_ps(O + (0 * step));\
   c1 = _mm256_load_ps(O + (0 * step) + SIMD);\
   c2 = _mm256_load_ps(O + (1 * step));\
   c3 = _mm256_load_ps(O + (1 * step) + SIMD);\

#define LOAD_W(W, f00, f01)\
  f00 = _mm256_load_ps(W);\
  f01 = _mm256_load_ps(W + SIMD);\




//Stores
#define STORE_2_C(O, c0, c1)\
 _mm256_store_ps(O + (0 * C_ob), c0);\
 _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);\

#define STORE_10_DW(O, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9)\
  _mm256_store_ps(O + (0 * C_ob), c0);\
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);\
  _mm256_store_ps(O + (1 * C_ob), c2);\
  _mm256_store_ps(O + (1 * C_ob) + SIMD, c3);\
  _mm256_store_ps(O + (2 * C_ob), c4);\
  _mm256_store_ps(O + (2 * C_ob) + SIMD, c5);\
  _mm256_store_ps(O + (3 * C_ob), c6);\
  _mm256_store_ps(O + (3 * C_ob) + SIMD, c7);\
  _mm256_store_ps(O + (4 * C_ob), c8);\
  _mm256_store_ps(O + (4 * C_ob) + SIMD, c9);\

#define STORE_12_C(O) \
for (int kk = 0; kk < W_ob; kk++) {\
    for (int jj = 0; jj < C_ob; jj++) {\
        O[kk*C_ob + jj] = c_tile[kk*C_ob + jj];\
    }\
}\

#define STORE_10_C(O) \
  _mm256_store_ps(O + (0 * C_ob), c0);\
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);\
  _mm256_store_ps(O + (1 * C_ob), c2);\
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);\
  _mm256_store_ps(O + (2 * C_ob), c4);\
  _mm256_store_ps(O + (2 * C_ob + SIMD), c5);\
  _mm256_store_ps(O + (3 * C_ob), c6);\
  _mm256_store_ps(O + (3 * C_ob + SIMD), c7);\
  _mm256_store_ps(O + (4 * C_ob), c8);\
  _mm256_store_ps(O + (4 * C_ob + SIMD), c9);\

#define STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11) \
  _mm256_store_ps(O + (0 * C_ob), c0);\
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);\
  _mm256_store_ps(O + (1 * C_ob), c2);\
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);\
  _mm256_store_ps(O + (2 * C_ob), c6);\
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);\
  _mm256_store_ps(O + (3 * C_ob), c10);\
  _mm256_store_ps(O + (3 * C_ob + SIMD), c11);\

#define STORE_6_C(O, store_indices) \
  _Pragma("GCC ivdep")\
  for(uint32_t kk_p = 0; kk_p < W_ob; kk_p++){\
    float * pixel = store_indices[kk_p];\
    _Pragma("omp simd")\
    for(uint32_t jj = 0; jj < C_ob; jj++){\
      O[kk_p*C_ob + jj] = *(pixel + jj);\
    }\
  }\


#define STORE_4_C(O, c2, c3, c6, c7) \
  _mm256_store_ps(O + (0 * C_ob), c2);\
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);\
  _mm256_store_ps(O + (1 * C_ob), c6);\
  _mm256_store_ps(O + (1 * C_ob + SIMD), c7);\




//Depthwise Computations
#define DW_START(F, I)\
  float * I_ptr = I;\
  f0 = _mm256_load_ps(F);\
  f1 = _mm256_load_ps(F + SIMD);\
                /**/      \
  i0 = _mm256_load_ps(I_ptr);\
  i1 = _mm256_load_ps(I_ptr + SIMD);\
  /**/\
  c0 = _mm256_mul_ps(f0,i0);\
  c1 = _mm256_mul_ps(f1, i1);\
  /**/\
  I_ptr += DW_STRIDE*C_ob;\
 /**/                \
  i0 = _mm256_load_ps(I_ptr);\
  i1 = _mm256_load_ps(I_ptr + SIMD);\
  /**/\
  c2 = _mm256_mul_ps(f0,i0);\
  c3 = _mm256_mul_ps(f1, i1);\
  /**/\
  I_ptr += DW_STRIDE*C_ob;\
  /**/                \
  i0 = _mm256_load_ps(I_ptr);\
  i1 = _mm256_load_ps(I_ptr + SIMD);\
  /**/\
  c4 = _mm256_mul_ps(f0,i0);\
  c5 = _mm256_mul_ps(f1, i1);\
  /**/\
  I_ptr += DW_STRIDE*C_ob;\
  /**/                \
  i0 = _mm256_load_ps(I_ptr);\
  i1 = _mm256_load_ps(I_ptr + SIMD);\
  /**/\
  c6 = _mm256_mul_ps(f0,i0);\
  c7 = _mm256_mul_ps(f1, i1);\
  /**/\
  I_ptr += DW_STRIDE*C_ob;\
  /**/                \
  i0 = _mm256_load_ps(I_ptr);\
  i1 = _mm256_load_ps(I_ptr + SIMD);\
  /**/\
  c8 = _mm256_mul_ps(f0,i0);\
  c9 = _mm256_mul_ps(f1, i1);\

#define DW_FMA(F, I)\
  float * I_ptr = I;\
  f0 = _mm256_load_ps(F);\
  f1 = _mm256_load_ps(F + SIMD);\
                /**/      \
  i0 = _mm256_load_ps(I_ptr);\
  i1 = _mm256_load_ps(I_ptr + SIMD);\
  /**/\
  c0 = _mm256_fmadd_ps(f0, i0, c0);\
  c1 = _mm256_fmadd_ps(f1, i1, c1);\
  /**/\
  I_ptr += DW_STRIDE*C_ob;\
 /**/                \
  i0 = _mm256_load_ps(I_ptr);\
  i1 = _mm256_load_ps(I_ptr + SIMD);\
  /**/\
  c2 = _mm256_fmadd_ps(f0, i0, c2);\
  c3 = _mm256_fmadd_ps(f1, i1, c3);\
  /**/\
  I_ptr += DW_STRIDE*C_ob;\
  /**/                \
  i0 = _mm256_load_ps(I_ptr);\
  i1 = _mm256_load_ps(I_ptr + SIMD);\
  /**/\
  c4 = _mm256_fmadd_ps(f0, i0, c4);\
  c5 = _mm256_fmadd_ps(f1, i1, c5);\
  /**/\
  I_ptr += DW_STRIDE*C_ob;\
  /**/                \
  i0 = _mm256_load_ps(I_ptr);\
  i1 = _mm256_load_ps(I_ptr + SIMD);\
  /**/\
  c6 = _mm256_fmadd_ps(f0, i0, c6);\
  c7 = _mm256_fmadd_ps(f1, i1, c7);\
  /**/\
  I_ptr += DW_STRIDE*C_ob;\
  /**/                \
  i0 = _mm256_load_ps(I_ptr);\
  i1 = _mm256_load_ps(I_ptr + SIMD);\
  /**/\
  c8 = _mm256_fmadd_ps(f0, i0, c8);\
  c9 = _mm256_fmadd_ps(f1, i1, c9);\

#define DW_FMA_2(W) \
  LOAD_W(W, f00, f01);\
  c10 = _mm256_mul_ps(c0, f00); c11 = _mm256_mul_ps(c1, f01)\
  c12 = _mm256_mul_ps(c4, f00); c13 = _mm256_mul_ps(c5, f01);\
  LOAD_W(W + C_ob, f00, f01);\
  c10 = _mm256_fmadd_ps(c2, f00, c10); c11 = _mm256_fmadd_ps(f01, c3, c11);\
  c12 = _mm256_fmadd_ps(c6, f00, c12); c13 = _mm256_fmadd_ps(f01, c7, c13);\
  LOAD_W(W + 2 * C_ob, f00, f01);\
  c10 = _mm256_fmadd_ps(c6, f00, c10); c11 = _mm256_fmadd_ps(f01, c7, c11);\
  c12 = _mm256_fmadd_ps(c8, f00, c12); c13 = _mm256_fmadd_ps(f01, c8, c13);\



//Convolution Computation
//(Strided GEMM)
// Pointer to C defined in the outer scope
#define FMA_12_C(step, a, b, p_cur)  \
 {\
   float *c_pixel;\
   _Pragma("GCC ivdep")\
   for(int kk = 0; kk < W_ob; kk++)\
   {\
     float a_val = *(a + p_cur + kk*step);\
     c_pixel = c_tile + kk*C_ob;\
     _Pragma("omp simd")\
     for(int jj = 0; jj < C_ob; jj++) \
     {\
       float b_val = *(b + p_cur + jj);\
       *(c_pixel+jj) += a_val*b_val;\
     }\
   }\
 }




#define FMA_10_C(step, a, b, p_cur) \
  b0 = _mm256_load_ps(b + (p_cur * C_ob));\
  b1 = _mm256_load_ps(b + (p_cur * C_ob + SIMD));\
  a_reg = _mm256_broadcast_ss(a + (p_cur));\
  p_cur += step;\
  c0 = _mm256_fmadd_ps(a_reg, b0, c0);\
  c1 = _mm256_fmadd_ps(a_reg, b1, c1);\
  a_reg = _mm256_broadcast_ss(a + (p_cur));\
  p_cur += step;\
  c2 = _mm256_fmadd_ps(a_reg, b0, c2);\
  c3 = _mm256_fmadd_ps(a_reg, b1, c3);\
  a_reg = _mm256_broadcast_ss(a +  (p_cur));\
  p_cur += step;\
  c4 = _mm256_fmadd_ps(a_reg, b0, c4);\
  c5 = _mm256_fmadd_ps(a_reg, b1, c5);\
  a_reg = _mm256_broadcast_ss(a +  (p_cur));\
  p_cur += step;\
  c6 = _mm256_fmadd_ps(a_reg, b0, c6);\
  c7 = _mm256_fmadd_ps(a_reg, b1, c7);\
  a_reg = _mm256_broadcast_ss(a +  (p_cur));\
  p_cur += step;\
  c8 = _mm256_fmadd_ps(a_reg, b0, c8);\
  c9 = _mm256_fmadd_ps(a_reg, b1, c9);\
  a_reg = _mm256_broadcast_ss(a +  (p_cur));\



//Max Pooling Computations
#define MAX_3_row(O, step)\
  c6 = _mm256_load_ps(O + (0 * step));\
  c7 = _mm256_load_ps(O + (0 * step) + SIMD);\
  c8 = _mm256_load_ps(O + (1 * step));\
  c9 = _mm256_load_ps(O + (1 * step) + SIMD);\
  c10 = _mm256_load_ps(O + (2 * step));\
  c11 = _mm256_load_ps(O + (2 * step) + SIMD);\
  c0 = _mm256_max_ps(c6, c0);\
  c1 = _mm256_max_ps(c7, c1);\
  c2 = _mm256_max_ps(c8, c2);\
  c3 = _mm256_max_ps(c9, c3);\
  c4 = _mm256_max_ps(c10, c4);\
  c5 = _mm256_max_ps(c11, c5);\

#define MAX_2_row(O, step)\
  c6 = _mm256_load_ps(O + (0 * step));\
  c7 = _mm256_load_ps(O + (0 * step) + SIMD);\
  c8 = _mm256_load_ps(O + (1 * step));\
  c9 = _mm256_load_ps(O + (1 * step) + SIMD);\
  c0 = _mm256_max_ps(c6, c0);\
  c1 = _mm256_max_ps(c7, c1);\
  c2 = _mm256_max_ps(c8, c2);\
  c3 = _mm256_max_ps(c9, c3);\

//not doing partial Updates
//reduce stores
#define MAX_3() \
  c2 = _mm256_max_ps(c2,c0);\
  c3 = _mm256_max_ps(c3,c1);\
  c6 = _mm256_max_ps(c6,c4);\
  c7 = _mm256_max_ps(c7,c5);\
  c10 = _mm256_max_ps(c10,c8);\
  c11 = _mm256_max_ps(c11,c9);\
  /**/\
  c2 = _mm256_max_ps(c2,c4);\
  c3 = _mm256_max_ps(c3,c5);\
  c6 = _mm256_max_ps(c6,c8);\
  c7 = _mm256_max_ps(c7,c9);\
  c10 = _mm256_max_ps(c10,b0);\
  c11 = _mm256_max_ps(c11,b1);\

#define MAX_2()\
c2 = _mm256_max_ps(c2,c0);\
c3 = _mm256_max_ps(c3,c1);\
c6 = _mm256_max_ps(c6,c4);\
c7 = _mm256_max_ps(c7,c5);\
/**/\
c2 = _mm256_max_ps(c2,c4);\
c3 = _mm256_max_ps(c3,c5);\
c6 = _mm256_max_ps(c6,c8);\
c7 = _mm256_max_ps(c7,c9);\

#define ACCUM_3(O)\
 c0 = _mm256_load_ps(O + (0 * C_ob));\
 c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);\
 c4 = _mm256_load_ps(O + (1 * C_ob));\
 c5 = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
 c8 = _mm256_load_ps(O + (2 * C_ob));\
 c9 = _mm256_load_ps(O + (2 * C_ob) + SIMD);\
 /*Accumulate with updates from previous row*/\
 c2 = _mm256_max_ps(c2,c0);\
 c3 = _mm256_max_ps(c3,c1);\
 c6 = _mm256_max_ps(c6,c4);\
 c7 = _mm256_max_ps(c7,c5);\
 c10 = _mm256_max_ps(c10,c8);\
 c11 = _mm256_max_ps(c11,c9);\

#define ACCUM_2(O)\
 c0 = _mm256_load_ps(O + (0 * C_ob));\
 c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);\
 c4 = _mm256_load_ps(O + (1 * C_ob));\
 c5 = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
 /*Accumulate with updates from previous row*/\
 c2 = _mm256_max_ps(c2,c0);\
 c3 = _mm256_max_ps(c3,c1);\
 c6 = _mm256_max_ps(c6,c4);\
 c7 = _mm256_max_ps(c7,c5);\

#define MAX_START() \
 /*Local Max */\
 _Pragma("GCC ivdep")\
 for(uint32_t kk_p = 0; kk_p < POOL_W_ob - 1; kk_p++){\
   float * p_pixel = c_tile + kk_p*POOL_STRIDE + C_ob;
   float * c_left = c_tile + kk_p*POOL_STRIDE;
   float * c_right = c_tile + kk_p*POOL_STRIDE + 2*C_ob;
   _Pragma("omp simd")\
   for(uint32_t jj = 0; jj < C_ob; jj++){\
     *(p_pixel + jj) = (*(p_pixel + jj) > *(c_left + jj)? *(p_pixel + jj) : *(c_left + jj));
     *(p_pixel + jj) = (*(p_pixel + jj) > *(c_right + jj)? *(p_pixel + jj) : *(c_right + jj));

   }\

 float * p_pixel = c_tile + (POOL_W_ob - 1)*POOL_STRIDE + C_ob;
 float * c_left = c_tile + (POOL_W_ob - 1)*POOL_STRIDE;

 _Pragma("omp simd")\
 for(uint32_t jj = 0; jj < C_ob; jj++){\
   *(p_pixel + jj) = (*(p_pixel + jj) > *(c_left + jj)? *(p_pixel + jj) : *(c_left + jj));


   store_indices[kk_p] = p_pixel;
 }\


#define MAX(O) \
 /*Load Updates from previous tile*/\
 b0 = _mm256_load_ps(O);\
 b1 = _mm256_load_ps(O + SIMD);\
 /**/\
 c2 = _mm256_max_ps(c2,c0);\
 c3 = _mm256_max_ps(c3,c1);\
 c6 = _mm256_max_ps(c6,c4);\
 c7 = _mm256_max_ps(c7,c5);\
 c10 = _mm256_max_ps(c10,c8);\
 c11 = _mm256_max_ps(c11,c9);\
 /**/\
 c2 = _mm256_max_ps(c2,c4);\
 c3 = _mm256_max_ps(c3,c5);\
 c6 = _mm256_max_ps(c6,c8);\
 c7 = _mm256_max_ps(c7,c9);\
 /*update previous tile*/\
 c0 = _mm256_max_ps(c0, b0);\
 c1 = _mm256_max_ps(c1,b1);\

#define MAX_END(O) \
 /*load the partial update from the previous tile*/\
 b0 = _mm256_load_ps(O);\
 b1 = _mm256_load_ps(O + SIMD);\
 /**/\
 c2 = _mm256_max_ps(c2,c0);\
 c3 = _mm256_max_ps(c3,c1);\
 c6 = _mm256_max_ps(c6,c4);\
 c7 = _mm256_max_ps(c7,c5);\
 /**/\
 c2 = _mm256_max_ps(c2,c4);\
 c3 = _mm256_max_ps(c3,c5);\
 c6 = _mm256_max_ps(c6,c8);\
 c7 = _mm256_max_ps(c7,c9);\
 /*Accumulate with previous tile*/\
 c0 = _mm256_max_ps(c0, b0);\
 c1 = _mm256_max_ps(c1,b1);\

#define ACCUM_START(O) \
 /*Load Previous*/\
 b0 = _mm256_load_ps(O + (0 * C_ob));\
 b1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);\
 a_reg = _mm256_load_ps(O + (1 * C_ob));\
 __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
 /*accumulate with previous rows*/\
 c2 = _mm256_max_ps(c2, b0);\
 c3 = _mm256_max_ps(c3, b1);\
 /**/\
 c6 = _mm256_max_ps(c6, a_reg);\
 c7 = _mm256_max_ps(c7, temp);\

#define ACCUM(O) \
 /*Load partial outputs from previous row*/\
 a_reg = _mm256_load_ps(O + (1 * C_ob));\
 __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
 c4 = _mm256_load_ps(O + (2 * C_ob));\
 c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);\
 c8 = _mm256_load_ps(O + (3 * C_ob));\
 c9 = _mm256_load_ps(O + (3 * C_ob) + SIMD);\
 b0 = _mm256_load_ps(O);\
 b1 = _mm256_load_ps(O + SIMD);\
 /*accumulate with previous row*/\
 c2 = _mm256_max_ps(c2, a_reg);\
 c3 = _mm256_max_ps(c3, temp);\
 c6 = _mm256_max_ps(c6,c4);\
 c7 = _mm256_max_ps(c7,c5);\
 c10 = _mm256_max_ps(c10,c8);\
 c11 = _mm256_max_ps(c11,c9);\
 c0 = _mm256_max_ps(c0, b0);\
 c1 = _mm256_max_ps(c1, b1);\

#define ACCUM_END(O) \
   /*load partial updates from previous row*/\
   a_reg = _mm256_load_ps(O);\
   __m256 temp = _mm256_load_ps(O + SIMD);\
   c4 = _mm256_load_ps(O + (1 * C_ob));\
   c8 = _mm256_load_ps(O + (2 * C_ob));\
   c5 = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
   c9 = _mm256_load_ps(O + (2 * C_ob) + SIMD);\
   /**/\
   c2 = _mm256_max_ps(c2, c4);\
   c3 = _mm256_max_ps(c3, c5);\
   /**/\
   c6 = _mm256_max_ps(c6, c8);\
   c7 = _mm256_max_ps(c7, c9);\
   /**/\
   c0 = _mm256_max_ps(c0,a_reg);\
   c1 = _mm256_max_ps(c1, temp);\

#define ACCUM_MAX_START(O) \
   /*Load Updates from Previous Row */\
 b0 = _mm256_load_ps(O + (0 * C_ob));\
 b1 = _mm256_load_ps(O + (1 * C_ob));\
 a_reg = _mm256_load_ps(O + (2 * C_ob));\
 __m256 temp = _mm256_load_ps(O + (0 * C_ob) + SIMD);\
 /* */\
 c2 = _mm256_max_ps(c2,c0);\
 c3 = _mm256_max_ps(c3,c1);\
 c6 = _mm256_max_ps(c6,c4);\
 c7 = _mm256_max_ps(c7,c5);\
 c10 = _mm256_max_ps(c10,c8);\
 c11 = _mm256_max_ps(c11,c9);\
 /*Load Updates from Previous Row */\
 c0 = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
 c1 = _mm256_load_ps(O + (2 * C_ob) + SIMD);\
 /* */\
 c2 = _mm256_max_ps(c2,c4);\
 c3 = _mm256_max_ps(c3,c5);\
 c6 = _mm256_max_ps(c6,c8);\
 c7 = _mm256_max_ps(c7,c9);\
 /**/\
 /*accumulate with previous rows*/\
 c2 = _mm256_max_ps(c2, b0);\
 c3 = _mm256_max_ps(c3, temp);\
 /**/\
 c6 = _mm256_max_ps(c6, b1);\
 c7 = _mm256_max_ps(c7, c0);\
 /**/\
 c10 = _mm256_max_ps(c10, a_reg);\
 c11 = _mm256_max_ps(c11, c1);\

#define ACCUM_MAX(O) \
 /*Load Previous Tile Updates*/\
 b0 = _mm256_load_ps(O);\
 b1 = _mm256_load_ps(O + SIMD);\
 a_reg = _mm256_load_ps(O + (1 * C_ob));\
 __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
 /*Local Max*/\
 c2 = _mm256_max_ps(c2,c0);\
 c3 = _mm256_max_ps(c3,c1);\
 c6 = _mm256_max_ps(c6,c4);\
 c7 = _mm256_max_ps(c7,c5);\
 c10 = _mm256_max_ps(c10,c8);\
 c11 = _mm256_max_ps(c11,c9);\
 /**/\
 c2 = _mm256_max_ps(c2,c4);\
 c3 = _mm256_max_ps(c3,c5);\
 c6 = _mm256_max_ps(c6,c8);\
 c7 = _mm256_max_ps(c7,c9);\
 /*Load Previous Row Updates*/\
 c4 = _mm256_load_ps(O + (2 * C_ob));\
 c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);\
 c8 = _mm256_load_ps(O + (3 * C_ob));\
 c9 = _mm256_load_ps(O + (3 * C_ob) + SIMD);\
 /* accumulate with previous tile*/\
 c0 = _mm256_max_ps(c0, b0);\
 c1 = _mm256_max_ps(c1,b1);\
 /*accumulate with previous row*/\
 c2 = _mm256_max_ps(c2, a_reg);\
 c3 = _mm256_max_ps(c3, temp);\
 c6 = _mm256_max_ps(c6,c4);\
 c7 = _mm256_max_ps(c7,c5);\
 c10 = _mm256_max_ps(c10,c8);\
 c11 = _mm256_max_ps(c11,c9);\

#define ACCUM_MAX_END(O) \
 /*Load Updates from previous tile*/\
 b0 = _mm256_load_ps(O);\
 b1 = _mm256_load_ps(O + SIMD);\
 /*Local Max*/\
 c2 = _mm256_max_ps(c2,c0);\
 c3 = _mm256_max_ps(c3,c1);\
 c6 = _mm256_max_ps(c6,c4);\
 c7 = _mm256_max_ps(c7,c5);\
 /**/\
 c2 = _mm256_max_ps(c2,c4);\
 c3 = _mm256_max_ps(c3,c5);\
 c6 = _mm256_max_ps(c6,c8);\
 c7 = _mm256_max_ps(c7,c9);\
 /*Accumulate with previous tile*/\
 c0 = _mm256_max_ps(c0, b0);\
 c1 = _mm256_max_ps(c1,b1);\
 /*load updates from previous row*/\
 b0 = _mm256_load_ps(O + (1 * C_ob));\
 b1 = _mm256_load_ps(O + (2 * C_ob));\
 a_reg = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
 __m256 temp = _mm256_load_ps(O + (2 * C_ob) + SIMD);\
 /*Accumulate with previous row*/\
 c2 = _mm256_max_ps(c2, b0);\
 c3 = _mm256_max_ps(c3, a_reg);\
 c6 = _mm256_max_ps(c6, b1);\
 c7 = _mm256_max_ps(c7, temp);




// Average Pooling Computations
#define AVG_3()\
  /*Initialize a variable to multiply with 1/9*/\
  a_reg = _mm256_broadcast_ss(&weight);\
  /*Multiply with the reused*/\
  c0 = _mm256_mul_ps(c0, a_reg);\
  c1 = _mm256_mul_ps(c1, a_reg);\
  c4 = _mm256_mul_ps(c4, a_reg);\
  c5 = _mm256_mul_ps(c5, a_reg);\
  c8 = _mm256_mul_ps(c8, a_reg);\
  c9 = _mm256_mul_ps(c9, a_reg);\
  b0 = _mm256_mul_ps(b0, a_reg);\
  b1 = _mm256_mul_ps(b1, a_reg);\
  /**/\
  c2 = _mm256_fmadd_ps(a_reg,c2,c0);\
  c3 = _mm256_fmadd_ps(a_reg,c3,c1);\
  c6 = _mm256_fmadd_ps(a_reg,c6,c4);\
  c7 = _mm256_fmadd_ps(a_reg,c7,c5);\
  c10 = _mm256_fmadd_ps(a_reg,c10,c8);\
  c11 = _mm256_fmadd_ps(a_reg,c11,c9);\
  /**/\
  c2 = _mm256_add_ps(c2,c4);\
  c3 = _mm256_add_ps(c3,c5);\
  c6 = _mm256_add_ps(c6,c8);\
  c7 = _mm256_add_ps(c7,c9);\
  c10 = _mm256_add_ps(c10,b0);\
  c11 = _mm256_add_ps(c11,b1);\

#define AVG_2()\
  /*Initialize a variable to multiply with 1/9*/ \
  a_reg = _mm256_broadcast_ss(&weight);\
  /*Multiply with the reused*/\
  c0 = _mm256_mul_ps(c0, a_reg);\
  c1 = _mm256_mul_ps(c1, a_reg);\
  c4 = _mm256_mul_ps(c4, a_reg);\
  c5 = _mm256_mul_ps(c5, a_reg);\
  c8 = _mm256_mul_ps(c8, a_reg);\
  c9 = _mm256_mul_ps(c9, a_reg);\
  /**/\
  c2 = _mm256_fmadd_ps(a_reg,c2,c0);\
  c3 = _mm256_fmadd_ps(a_reg,c3,c1);\
  c6 = _mm256_fmadd_ps(a_reg,c6,c4);\
  c7 = _mm256_fmadd_ps(a_reg,c7,c5);\
  /**/\
  c2 = _mm256_add_ps(c2,c4);\
  c3 = _mm256_add_ps(c3,c5);\
  c6 = _mm256_add_ps(c6,c8);\
  c7 = _mm256_add_ps(c7,c9);\

#define ACCUM_AVG_3(O)\
  c0 = _mm256_load_ps(O + (0 * C_ob));\
  c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);\
  c4 = _mm256_load_ps(O + (1 * C_ob));\
  c5 = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
  c8 = _mm256_load_ps(O + (2 * C_ob));\
  c9 = _mm256_load_ps(O + (2 * C_ob) + SIMD);\
  /*Accumulate with updates from previous row*/\
  c2 = _mm256_add_ps(c2,c0);\
  c3 = _mm256_add_ps(c3,c1);\
  c6 = _mm256_add_ps(c6,c4);\
  c7 = _mm256_add_ps(c7,c5);\
  c10 = _mm256_add_ps(c10,c8);\
  c11 = _mm256_add_ps(c11,c9);\

#define ACCUM_AVG_2(O)\
 c0 = _mm256_load_ps(O + (0 * C_ob));\
 c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);\
 c4 = _mm256_load_ps(O + (1 * C_ob));\
 c5 = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
 /*Accumulate with updates from previous row*/\
 c2 = _mm256_add_ps(c2,c0);\
 c3 = _mm256_add_ps(c3,c1);\
 c6 = _mm256_add_ps(c6,c4);\
 c7 = _mm256_add_ps(c7,c5);\

#define FMA_START() \
  /*Local Max */\
  b0 = _mm256_broadcast_ss(&weight);\
  /*multiply with 1/9*/\
  c0 = _mm256_mul_ps(c0, b0);    c1 = _mm256_mul_ps(c1, b0);\
  c4 = _mm256_mul_ps(c4, b0);    c5 = _mm256_mul_ps(c5, b0);\
  c8 = _mm256_mul_ps(c8, b0);    c9 = _mm256_mul_ps(c9, b0);\
  /*FMA the scaled*/\
  c2 = _mm256_fmadd_ps(c2,b0, c0);    c3 = _mm256_fmadd_ps(c3,c1, b0);\
  c6 = _mm256_fmadd_ps(c6,c4, b0);    c7 = _mm256_fmadd_ps(c7,c5, b0);\
  c10 = _mm256_fmadd_ps(c10,c8, b0);  c11 = _mm256_fmadd_ps(c11,c9,b0);\
  /*Add the second Update*/\
  c2 = _mm256_add_ps(c2,c4);    c3 = _mm256_add_ps(c3,c5);\
  c6 = _mm256_add_ps(c6,c8);    c7 = _mm256_add_ps(c7,c9);\


#define FMA(O) \
  /*Load Updates from previous tile*/\
  b0 = _mm256_load_ps(O);\
  b1 = _mm256_load_ps(O + SIMD);\
  a_reg = _mm256_broadcast_ss(&weight);\
  /*multiply with 1/9*/\
  c0 = _mm256_mul_ps(c0, a_reg);    c1 = _mm256_mul_ps(c1, a_reg);\
  c4 = _mm256_mul_ps(c4, a_reg);    c5 = _mm256_mul_ps(c5, a_reg);\
  c8 = _mm256_mul_ps(c8, a_reg);    c9 = _mm256_mul_ps(c9, a_reg);\
  /**/\
  c2 = _mm256_fmadd_ps(c2,a_reg, c0);    c3 = _mm256_fmadd_ps(c3,a_reg, c1);\
  c6 = _mm256_fmadd_ps(c6,a_reg, c4);     c7 = _mm256_fmadd_ps(c7,a_reg, c5);\
  c10 = _mm256_fmadd_ps(c10,a_reg, c8);    c11 = _mm256_fmadd_ps(c11,a_reg, c9);\
  /**/\
  c2 = _mm256_add_ps(c2,c4);    c3 = _mm256_add_ps(c3,c5);\
  c6 = _mm256_add_ps(c6,c8);    c7 = _mm256_add_ps(c7,c9);\
  /*update previous tile*/\
  c0 = _mm256_add_ps(c0, b0);     c1 = _mm256_add_ps(c1,b1);\


#define FMA_END(O) \
  /*load the partial update from the previous tile*/\
  b0 = _mm256_load_ps(O);\
  b1 = _mm256_load_ps(O + SIMD);\
  a_reg = _mm256_broadcast_ss(&weight);\
  /*multiply with 1/9*/\
  c0 = _mm256_mul_ps(c0, a_reg);    c1 = _mm256_mul_ps(c1, a_reg);\
  c4 = _mm256_mul_ps(c4, a_reg);    c5 = _mm256_mul_ps(c5, a_reg);\
  c8 = _mm256_mul_ps(c8, a_reg);    c9 = _mm256_mul_ps(c9, a_reg);\
  /**/\
  c2 = _mm256_fmadd_ps(c2,a_reg, c0);    c3 = _mm256_fmadd_ps(c3,a_reg, c1);\
  c6 = _mm256_fmadd_ps(c6,a_reg, c4);     c7 = _mm256_fmadd_ps(c7,a_reg, c5);\
  c10 = _mm256_fmadd_ps(c10,a_reg, c8);    c11 = _mm256_fmadd_ps(c11,a_reg, c9);\
  /**/\
  c2 = _mm256_add_ps(c2,c4);    c3 = _mm256_add_ps(c3,c5);\
  c6 = _mm256_add_ps(c6,c8);    c7 = _mm256_add_ps(c7,c9);\
  /*Accumulate with previous tile*/\
  c0 = _mm256_add_ps(c0, b0);     c1 = _mm256_add_ps(c1,b1);\


#define ACCUM_AVG_START(O) \
  /*Load Previous*/\
  b0 = _mm256_load_ps(O + (0 * C_ob));\
  b1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);\
  a_reg = _mm256_load_ps(O + (1 * C_ob));\
  __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
  /*accumulate with previous rows*/\
  c2 = _mm256_add_ps(c2, b0);    c3 = _mm256_add_ps(c3, b1);\
  /**/\
  c6 = _mm256_add_ps(c6, a_reg); c7 = _mm256_add_ps(c7, temp);\


#define ACCUM_AVG(O) \
  /*Load partial outputs from previous row*/\
  a_reg = _mm256_load_ps(O + (1 * C_ob));\
  __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
  c4 = _mm256_load_ps(O + (2 * C_ob));\
  c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);\
  c8 = _mm256_load_ps(O + (3 * C_ob));\
  c9 = _mm256_load_ps(O + (3 * C_ob) + SIMD);\
  b0 = _mm256_load_ps(O);\
  b1 = _mm256_load_ps(O + SIMD);\
  /*accumulate with previous row*/\
  c2 = _mm256_add_ps(c2, a_reg);  c3 = _mm256_add_ps(c3, temp);\
  c6 = _mm256_add_ps(c6,c4);      c7 = _mm256_add_ps(c7,c5);\
  c0 = _mm256_add_ps(c0, b0);     c1 = _mm256_add_ps(c1, b1);\



#define ACCUM_AVG_END(O) \
  /*load partial updates from previous row*/\
  b0 = _mm256_broadcast_ss(&weight);\
  a_reg = _mm256_load_ps(O);\
  __m256 temp = _mm256_load_ps(O + SIMD);\
  c4 = _mm256_load_ps(O + (1 * C_ob));\
  c8 = _mm256_load_ps(O + (2 * C_ob));\
  c5 = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
  c9 = _mm256_load_ps(O + (2 * C_ob) + SIMD);\
  /**/\
  c2 = _mm256_fmadd_ps(c2, c4, b0);\
  c3 = _mm256_fmadd_ps(c3, c5, b0);\
  /**/\
  c6 = _mm256_fmadd_ps(c6, c8, b0);\
  c7 = _mm256_fmadd_ps(c7, c9, b0);\
  /**/\
  c0 = _mm256_fmadd_ps(c0,a_reg, b0);\
  c1 = _mm256_fmadd_ps(c1, temp, b0);\


#define ACCUM_FMA_START(O) \
    /*Load Updates from Previous Row */\
  b1 = _mm256_load_ps(O + (1 * C_ob));\
  a_reg = _mm256_load_ps(O + (2 * C_ob));\
  __m256 temp = _mm256_load_ps(O + (0 * C_ob) + SIMD);\
  /*Multiply with 1/9*/\
  c0 = _mm256_mul_ps(c0, b0);    c1 = _mm256_mul_ps(c1, b0);\
  c4 = _mm256_mul_ps(c4, b0);    c5 = _mm256_mul_ps(c5, b0);\
  c8 = _mm256_mul_ps(c8, b0);    c9 = _mm256_mul_ps(c9, a_reg);\
  /* */\
  c2 = _mm256_fmadd_ps(c2,b0, c0);    c3 = _mm256_fmadd_ps(c3,b0, c1);\
  c6 = _mm256_fmadd_ps(c6,b0, c4);     c7 = _mm256_fmadd_ps(c7,b0, c5);\
  c10 = _mm256_fmadd_ps(c10,b0, c8);    c11 = _mm256_fmadd_ps(c11,b0, c9);\
  b0 = _mm256_load_ps(O + (0 * C_ob));\
  /*Load Updates from Previous Row */\
  c0 = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
  c1 = _mm256_load_ps(O + (2 * C_ob) + SIMD);\
  /* */\
  c2 = _mm256_add_ps(c2,c4);    c3 = _mm256_add_ps(c3,c5);\
  c6 = _mm256_add_ps(c6,c8);    c7 = _mm256_add_ps(c7,c9);\
  /**/\
  /*accumulate with previous rows*/\
  c2 = _mm256_add_ps(c2, b0);    c3 = _mm256_add_ps(c3, temp);\
  /**/\
  c6 = _mm256_add_ps(c6, b1);    c7 = _mm256_add_ps(c7, c0);\
  /**/\
  c10 = _mm256_add_ps(c10, a_reg);    c11 = _mm256_add_ps(c11, c1);\


#define ACCUM_FMA(O) \
  /*Load Previous Tile Updates*/\
  b0 = _mm256_broadcast_ss(&weight);\
  b0 = _mm256_load_ps(O);\
  b1 = _mm256_load_ps(O + SIMD);\
  a_reg = _mm256_load_ps(O + (1 * C_ob));\
  __m256 temp = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
  /*Multiply with 1/9*/\
  c0 = _mm256_mul_ps(c0, b0);    c1 = _mm256_mul_ps(c1, b0);\
  c4 = _mm256_mul_ps(c4, b0);    c5 = _mm256_mul_ps(c5, b0);\
  c8 = _mm256_mul_ps(c8, b0);    c9 = _mm256_mul_ps(c9, a_reg);\
  /* */\
  c2 = _mm256_fmadd_ps(c2,b0, c0);    c3 = _mm256_fmadd_ps(c3,b0, c1);\
  c6 = _mm256_fmadd_ps(c6,b0, c4);     c7 = _mm256_fmadd_ps(c7,b0, c5);\
  c10 = _mm256_fmadd_ps(c10,b0, c8);    c11 = _mm256_fmadd_ps(c11,b0, c9);\
  /**/\
  b0 = _mm256_load_ps(O + (0 * C_ob));\
  /* */\
  c2 = _mm256_add_ps(c2,c4);    c3 = _mm256_add_ps(c3,c5);\
  c6 = _mm256_add_ps(c6,c8);    c7 = _mm256_add_ps(c7,c9);\
  /*Load Previous Row Updates*/\
  b0 = _mm256_load_ps(O + (0 * C_ob));\
  c4 = _mm256_load_ps(O + (2 * C_ob));\
  c5 = _mm256_load_ps(O + (2 * C_ob) + SIMD);\
  c8 = _mm256_load_ps(O + (3 * C_ob));\
  c9 = _mm256_load_ps(O + (3 * C_ob) + SIMD);\
  /* accumulate with previous tile*/\
  c0 = _mm256_add_ps(c0, b0);\
  c1 = _mm256_add_ps(c1,b1);\
  /*accumulate with previous row*/\
  c2 = _mm256_add_ps(c2, a_reg);\
  c3 = _mm256_add_ps(c3, temp);\
  c6 = _mm256_add_ps(c6,c4);\
  c7 = _mm256_add_ps(c7,c5);\
  c10 = _mm256_add_ps(c10,c8);\
  c11 = _mm256_add_ps(c11,c9);\


#define ACCUM_FMA_END(O) \
  /*Load Updates from previous tile*/\
  a_reg = _mm256_broadcast_ss(&weight);\
  b0 = _mm256_load_ps(O);\
  b1 = _mm256_load_ps(O + SIMD);\
  /*Multiply with 1/9*/\
  c0 = _mm256_mul_ps(c0, b0);    c1 = _mm256_mul_ps(c1, b0);\
  c4 = _mm256_mul_ps(c4, b0);    c5 = _mm256_mul_ps(c5, b0);\
  c8 = _mm256_mul_ps(c8, b0);    c9 = _mm256_mul_ps(c9, a_reg);\
  /* */\
  c2 = _mm256_fmadd_ps(c2,b0, c0);    c3 = _mm256_fmadd_ps(c3,b0, c1);\
  c6 = _mm256_fmadd_ps(c6,b0, c4);     c7 = _mm256_fmadd_ps(c7,b0, c5);\
  c10 = _mm256_fmadd_ps(c10,b0, c8);    c11 = _mm256_fmadd_ps(c11,b0, c9);\
  /**/\
  c2 = _mm256_add_ps(c2,c4);    c3 = _mm256_add_ps(c3,c5);\
  c6 = _mm256_add_ps(c6,c8);    c7 = _mm256_add_ps(c7,c9);\
  /*Accumulate with previous tile*/\
  c0 = _mm256_add_ps(c0, b0);\
  c1 = _mm256_add_ps(c1,b1);\
  /*load updates from previous row*/\
  b0 = _mm256_load_ps(O + (1 * C_ob));\
  b1 = _mm256_load_ps(O + (2 * C_ob));\
  a_reg = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
  __m256 temp = _mm256_load_ps(O + (2 * C_ob) + SIMD);\
  /*Accumulate with previous row*/\
  c2 = _mm256_add_ps(c2, b0);\
  c3 = _mm256_add_ps(c3, a_reg);\
  c6 = _mm256_add_ps(c6, b1);\
  c7 = _mm256_add_ps(c7, temp);



//Global Pooling computations
//  For the end of a spatial normalization (layerNorm, Global Pooling)
//  Parallel Reductions
#define DIVIDE(O, divisor)\
    o0 = _mm256_load_ps(O + (0 * C_ob));\
    o1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);\
    o2 = _mm256_broadcast_ss(&divisor);\
    o0 = _mm256_mul_ps(o0, o2);\
    o1 = _mm256_mul_ps(o1, o2);\
    _mm256_store_ps(O + (0 * C_ob), o0);\
    _mm256_store_ps(O + (0 * C_ob) + SIMD, o1);\

#define ADD_12_C()\
   o0 = _mm256_add_ps(c0, o0);\
   o1 = _mm256_add_ps(c1, o1);\
   o2 = _mm256_add_ps(c2, o2);\
   o3 = _mm256_add_ps(c3, o3);\
   o4 = _mm256_add_ps(c4, o4);\
   o5 = _mm256_add_ps(c5, o5);\
   o0 = _mm256_add_ps(c6, o0);\
   o1 = _mm256_add_ps(c7, o1);\
   o2 = _mm256_add_ps(c8, o2);\
   o3 = _mm256_add_ps(c9, o3);\
   o4 = _mm256_add_ps(c10, o4);\
   o5 = _mm256_add_ps(c11, o5);\

#define REDUCE_1_C()\
   o0 = _mm256_add_ps(o0, o2);\
   o1 = _mm256_add_ps(o1, o3);\

#define REDUCE_2_C()\
  o0 = _mm256_add_ps(o0, o2);\
  o1 = _mm256_add_ps(o1, o3);\
  o0 = _mm256_add_ps(o0, o4);\
  o1 = _mm256_add_ps(o1, o5);
