#define LOAD_12_C(O){\
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
}

#define ZERO_12_C(){\
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
}

#define FMA_12_C(step, a, b, p_cur){\
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
}

#define FMA_10_C(step, a, b, p_cur){\
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
}

#define STORE_12_C(O){\
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
}

#define STORE_10_C(O){\
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
}

#define STORE_8_C(O, c0, c1, c2, c3, c6, c7, c10, c11){\
  _mm256_store_ps(O + (0 * C_ob), c0);\
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c1);\
  _mm256_store_ps(O + (1 * C_ob), c2);\
  _mm256_store_ps(O + (1 * C_ob + SIMD), c3);\
  _mm256_store_ps(O + (2 * C_ob), c6);\
  _mm256_store_ps(O + (2 * C_ob + SIMD), c7);\
  _mm256_store_ps(O + (3 * C_ob), c10);\
  _mm256_store_ps(O + (3 * C_ob + SIMD), c11);\
}

#define STORE_6_C(O, c2, c3, c6, c7, c10, c11){\
  _mm256_store_ps(O + (0 * C_ob), c2);\
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);\
  _mm256_store_ps(O + (1 * C_ob), c6);\
  _mm256_store_ps(O + (1 * C_ob + SIMD), c7);\
  _mm256_store_ps(O + (2 * C_ob), c10);\
  _mm256_store_ps(O + (2 * C_ob + SIMD), c11);\
}

#define STORE_4_C(O, c2, c3, c6, c7){\
  _mm256_store_ps(O + (0 * C_ob), c2);\
  _mm256_store_ps(O + (0 * C_ob) + SIMD, c3);\
  _mm256_store_ps(O + (1 * C_ob), c6);\
  _mm256_store_ps(O + (1 * C_ob + SIMD), c7);\
}

#define MAX_START(){\
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
}

#define MAX(O){\
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
}

#define MAX_END(O){\
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
}

#define ACCUM_START(O){\
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
}

#define ACCUM(O){\
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
}

#define ACCUM_END(O){\
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
}

#define ACCUM_MAX_START(O){\
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
}

#define ACCUM_MAX(O){\
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
}

#define ACCUM_MAX_END(O){\
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
  c7 = _mm256_max_ps(c7, temp);\
}
