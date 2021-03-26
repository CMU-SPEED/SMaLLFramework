
#define LOAD_12_C(O) \
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
   c10 = _mm256_load_ps(O + (5 * C_ob));\
   c11 = _mm256_load_ps(O + (5 * C_ob) + SIMD);\


#define LOAD_14_C(O) \
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
  c10 = _mm256_load_ps(O + (5 * C_ob));\
  c11 = _mm256_load_ps(O + (5 * C_ob) + SIMD);\
  b0 = _mm256_load_ps(O + (6 * C_ob));\
  b1 = _mm256_load_ps(O + (6 * C_ob) + SIMD);\


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
 c9 = _mm256_load_ps(O + (4 * C_ob) + SIMD);

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
    c3 = _mm256_max_ps(c9, c3);

#define DW_FMA_2(W) \
  LOAD_W(W, f00, f01);\
  c10 = _mm256_mul_ps(c0, f00); c11 = _mm256_mul_ps(c1, f01)\
  c12 = _mm256_mul_ps(c4, f00); c13 = _mm256_mul_ps(c5, f01);\
  LOAD_W(W + C_ob, f00, f01);\
  c10 = _mm256_fmadd_ps(c2, f00, c10); c11 = _mm256_fmadd_ps(f01, c3, c11);\
  c12 = _mm256_fmadd_ps(c6, f00, c12); c13 = _mm256_fmadd_ps(f01, c7, c13);\
  LOAD_W(W + 2 * C_ob, f00, f01);\
  c10 = _mm256_fmadd_ps(c6, f00, c10); c11 = _mm256_fmadd_ps(f01, c7, c11);\
  c12 = _mm256_fmadd_ps(c8, f00, c12); c13 = _mm256_fmadd_ps(f01, c8, c13);




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


#define MAX_2() \
c2 = _mm256_max_ps(c2,c0);\
c3 = _mm256_max_ps(c3,c1);\
c6 = _mm256_max_ps(c6,c4);\
c7 = _mm256_max_ps(c7,c5);\
/**/\
c2 = _mm256_max_ps(c2,c4);\
c3 = _mm256_max_ps(c3,c5);\
c6 = _mm256_max_ps(c6,c8);\
c7 = _mm256_max_ps(c7,c9);\

#define AVG_2() \
/*Initialize a variable to multiply with 1/9*/\
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
 c11 = _mm256_max_ps(c11,c9);

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
  c11 = _mm256_add_ps(c11,c9);

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


#define ACCUM_AVG_2(O)\
 c0 = _mm256_load_ps(O + (0 * C_ob));\
 c1 = _mm256_load_ps(O + (0 * C_ob) + SIMD);\
 c4 = _mm256_load_ps(O + (1 * C_ob));\
 c5 = _mm256_load_ps(O + (1 * C_ob) + SIMD);\
 /*Accumulate with updates from previous row*/\
 c2 = _mm256_add_ps(c2,c0);\
 c3 = _mm256_add_ps(c3,c1);\
 c6 = _mm256_add_ps(c6,c4);\
 c7 = _mm256_add_ps(c7,c5);

#define ZERO_12_C() \
  c0 = _mm256_setzero_ps();\
   c1 = _mm256_setzero_ps();\
   c2 = _mm256_setzero_ps();\
   c3 = _mm256_setzero_ps();\
   c4 = _mm256_setzero_ps();\
   c5 = _mm256_setzero_ps();\
   c6 = _mm256_setzero_ps();\
   c7 = _mm256_setzero_ps();\
   c8 = _mm256_setzero_ps();\
   c9 = _mm256_setzero_ps();\
   c10 = _mm256_setzero_ps();\
   c11 = _mm256_setzero_ps();\


#define FMA_12_C(step, a, b, p_cur)  \
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
  p_cur += step;\
  c10 = _mm256_fmadd_ps(a_reg, b0, c10);\
  c11 = _mm256_fmadd_ps(a_reg, b1, c11);\


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


#define STORE_12_C(O) \
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
  _mm256_store_ps(O + (5 * C_ob), c10);\
  _mm256_store_ps(O + (5 * C_ob + SIMD), c11);\


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


#define STORE_6_C(O, c2, c3, c6, c7, c10, c11) \
  _mm256_store_ps(O + (0 * C_ob), c2);\
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);\
  _mm256_store_ps(O + (1 * C_ob), c6);\
  _mm256_store_ps(O + (1 * C_ob + SIMD), c7);\
  _mm256_store_ps(O + (2 * C_ob), c10);\
  _mm256_store_ps(O + (2 * C_ob + SIMD), c11);\


#define STORE_4_C(O, c2, c3, c6, c7) \
  _mm256_store_ps(O + (0 * C_ob), c2);\
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);\
  _mm256_store_ps(O + (1 * C_ob), c6);\
  _mm256_store_ps(O + (1 * C_ob + SIMD), c7);\


#define MAX_START() \
  /*Local Max */\
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
