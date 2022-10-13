#include <arm_neon.h>
#ifdef DEF_TILE_C
#undef DEF_TILE_C
#endif

#define DEF_TILE_C(W_ob, C_ob)\
float c_tile[W_ob * C_ob];\
float32x4_t c_0_0;\
float32x4_t c_0_1;\
float32x4_t c_0_2;\
float32x4_t c_0_3;\
float32x4_t c_1_0;\
float32x4_t c_1_1;\
float32x4_t c_1_2;\
float32x4_t c_1_3;\
float32x4_t c_2_0;\
float32x4_t c_2_1;\
float32x4_t c_2_2;\
float32x4_t c_2_3;\
float32x4_t c_3_0;\
float32x4_t c_3_1;\
float32x4_t c_3_2;\
float32x4_t c_3_3;\
float32x4_t c_4_0;\
float32x4_t c_4_1;\
float32x4_t c_4_2;\
float32x4_t c_4_3;\
float32x4_t c_5_0;\
float32x4_t c_5_1;\
float32x4_t c_5_2;\
float32x4_t c_5_3;\
    float32x4_t B0, B1,            \
        A0, C00, C01, C02, C03,    \
        A1, C10, C11, C12, C13,    \
        A2, C20, C21, C22, C23,    \
        A3, C30, C31, C32, C33,    \
        A4, C40, C41, C42, C43,    \
        A5, C50, C51, C52, C53;    \



#ifdef ZERO_TILE_C
#undef ZERO_TILE_C
#endif

#define ZERO_TILE_C(W_ob, C_ob) \
    float32x4_t B0, B1,         \
        A0, C00, C01, C02, C03, \
        A1, C10, C11, C12, C13, \
        A2, C20, C21, C22, C23, \
        A3, C30, C31, C32, C33, \
        A4, C40, C41, C42, C43, \
        A5, C50, C51, C52, C53; \
    C00 = vmovq_n_f32(0);       \
    C01 = vmovq_n_f32(0);       \
    C02 = vmovq_n_f32(0);       \
    C03 = vmovq_n_f32(0);       \
    C10 = vmovq_n_f32(0);       \
    C11 = vmovq_n_f32(0);       \
    C12 = vmovq_n_f32(0);       \
    C13 = vmovq_n_f32(0);       \
    C20 = vmovq_n_f32(0);       \
    C21 = vmovq_n_f32(0);       \
    C22 = vmovq_n_f32(0);       \
    C23 = vmovq_n_f32(0);       \
    C30 = vmovq_n_f32(0);       \
    C31 = vmovq_n_f32(0);       \
    C32 = vmovq_n_f32(0);       \
    C33 = vmovq_n_f32(0);       \
    C40 = vmovq_n_f32(0);       \
    C41 = vmovq_n_f32(0);       \
    C42 = vmovq_n_f32(0);       \
    C43 = vmovq_n_f32(0);       \
    C50 = vmovq_n_f32(0);       \
    C51 = vmovq_n_f32(0);       \
    C52 = vmovq_n_f32(0);       \
    C53 = vmovq_n_f32(0);



// #define ZERO_TILE_C(W_ob, C_ob)\
// c_0_0 = vdupq_n_f32(0);\
// c_0_1 = vdupq_n_f32(0);\
// c_0_2 = vdupq_n_f32(0);\
// c_0_3 = vdupq_n_f32(0);\
// c_1_0 = vdupq_n_f32(0);\
// c_1_1 = vdupq_n_f32(0);\
// c_1_2 = vdupq_n_f32(0);\
// c_1_3 = vdupq_n_f32(0);\
// c_2_0 = vdupq_n_f32(0);\
// c_2_1 = vdupq_n_f32(0);\
// c_2_2 = vdupq_n_f32(0);\
// c_2_3 = vdupq_n_f32(0);\
// c_3_0 = vdupq_n_f32(0);\
// c_3_1 = vdupq_n_f32(0);\
// c_3_2 = vdupq_n_f32(0);\
// c_3_3 = vdupq_n_f32(0);\
// c_4_0 = vdupq_n_f32(0);\
// c_4_1 = vdupq_n_f32(0);\
// c_4_2 = vdupq_n_f32(0);\
// c_4_3 = vdupq_n_f32(0);\
// c_5_0 = vdupq_n_f32(0);\
// c_5_1 = vdupq_n_f32(0);\
// c_5_2 = vdupq_n_f32(0);\
// c_5_3 = vdupq_n_f32(0);\

#ifdef LOAD_TILE_C
#undef LOAD_TILE_C
#endif

#define LOAD_TILE_C(O, W_ob, C_ob) \
    float32x4_t B0, B1,            \
        A0, C00, C01, C02, C03,    \
        A1, C10, C11, C12, C13,    \
        A2, C20, C21, C22, C23,    \
        A3, C30, C31, C32, C33,    \
        A4, C40, C41, C42, C43,    \
        A5, C50, C51, C52, C53;    \
    C00 = vld1q_f32(&Crow(0, 0));  \
    C01 = vld1q_f32(&Crow(0, 4));  \
    C02 = vld1q_f32(&Crow(0, 8));  \
    C03 = vld1q_f32(&Crow(0, 12)); \
    C10 = vld1q_f32(&Crow(1, 0));  \
    C11 = vld1q_f32(&Crow(1, 4));  \
    C12 = vld1q_f32(&Crow(1, 8));  \
    C13 = vld1q_f32(&Crow(1, 12)); \
    C20 = vld1q_f32(&Crow(2, 0));  \
    C21 = vld1q_f32(&Crow(2, 4));  \
    C22 = vld1q_f32(&Crow(2, 8));  \
    C23 = vld1q_f32(&Crow(2, 12)); \
    C30 = vld1q_f32(&Crow(3, 0));  \
    C31 = vld1q_f32(&Crow(3, 4));  \
    C32 = vld1q_f32(&Crow(3, 8));  \
    C33 = vld1q_f32(&Crow(3, 12)); \
    C40 = vld1q_f32(&Crow(4, 0));  \
    C41 = vld1q_f32(&Crow(4, 4));  \
    C42 = vld1q_f32(&Crow(4, 8));  \
    C43 = vld1q_f32(&Crow(4, 12)); \
    C50 = vld1q_f32(&Crow(5, 0));  \
    C51 = vld1q_f32(&Crow(5, 4));  \
    C52 = vld1q_f32(&Crow(5, 8));  \
    C53 = vld1q_f32(&Crow(5, 12));\


// #define LOAD_TILE_C(O, W_ob, C_ob)\
// c_0_0 = vld1q_f32(O + 0 * C_ob + 0 * SIMD);\
// c_0_1 = vld1q_f32(O + 0 * C_ob + 1 * SIMD);\
// c_0_2 = vld1q_f32(O + 0 * C_ob + 2 * SIMD);\
// c_0_3 = vld1q_f32(O + 0 * C_ob + 3 * SIMD);\
// c_1_0 = vld1q_f32(O + 1 * C_ob + 0 * SIMD);\
// c_1_1 = vld1q_f32(O + 1 * C_ob + 1 * SIMD);\
// c_1_2 = vld1q_f32(O + 1 * C_ob + 2 * SIMD);\
// c_1_3 = vld1q_f32(O + 1 * C_ob + 3 * SIMD);\
// c_2_0 = vld1q_f32(O + 2 * C_ob + 0 * SIMD);\
// c_2_1 = vld1q_f32(O + 2 * C_ob + 1 * SIMD);\
// c_2_2 = vld1q_f32(O + 2 * C_ob + 2 * SIMD);\
// c_2_3 = vld1q_f32(O + 2 * C_ob + 3 * SIMD);\
// c_3_0 = vld1q_f32(O + 3 * C_ob + 0 * SIMD);\
// c_3_1 = vld1q_f32(O + 3 * C_ob + 1 * SIMD);\
// c_3_2 = vld1q_f32(O + 3 * C_ob + 2 * SIMD);\
// c_3_3 = vld1q_f32(O + 3 * C_ob + 3 * SIMD);\
// c_4_0 = vld1q_f32(O + 4 * C_ob + 0 * SIMD);\
// c_4_1 = vld1q_f32(O + 4 * C_ob + 1 * SIMD);\
// c_4_2 = vld1q_f32(O + 4 * C_ob + 2 * SIMD);\
// c_4_3 = vld1q_f32(O + 4 * C_ob + 3 * SIMD);\
// c_5_0 = vld1q_f32(O + 5 * C_ob + 0 * SIMD);\
// c_5_1 = vld1q_f32(O + 5 * C_ob + 1 * SIMD);\
// c_5_2 = vld1q_f32(O + 5 * C_ob + 2 * SIMD);\
// c_5_3 = vld1q_f32(O + 5 * C_ob + 3 * SIMD);\

#ifdef LOAD_TILE_C_strided
#undef LOAD_TILE_C_strided
#endif

#define LOAD_TILE_C_strided(O, step, W_ob, C_ob)\
c_0_0 = vld1q_f32(O + 0 * step + 0 * SIMD);\
c_0_1 = vld1q_f32(O + 0 * step + 1 * SIMD);\
c_0_2 = vld1q_f32(O + 0 * step + 2 * SIMD);\
c_0_3 = vld1q_f32(O + 0 * step + 3 * SIMD);\
c_1_0 = vld1q_f32(O + 1 * step + 0 * SIMD);\
c_1_1 = vld1q_f32(O + 1 * step + 1 * SIMD);\
c_1_2 = vld1q_f32(O + 1 * step + 2 * SIMD);\
c_1_3 = vld1q_f32(O + 1 * step + 3 * SIMD);\
c_2_0 = vld1q_f32(O + 2 * step + 0 * SIMD);\
c_2_1 = vld1q_f32(O + 2 * step + 1 * SIMD);\
c_2_2 = vld1q_f32(O + 2 * step + 2 * SIMD);\
c_2_3 = vld1q_f32(O + 2 * step + 3 * SIMD);\
c_3_0 = vld1q_f32(O + 3 * step + 0 * SIMD);\
c_3_1 = vld1q_f32(O + 3 * step + 1 * SIMD);\
c_3_2 = vld1q_f32(O + 3 * step + 2 * SIMD);\
c_3_3 = vld1q_f32(O + 3 * step + 3 * SIMD);\
c_4_0 = vld1q_f32(O + 4 * step + 0 * SIMD);\
c_4_1 = vld1q_f32(O + 4 * step + 1 * SIMD);\
c_4_2 = vld1q_f32(O + 4 * step + 2 * SIMD);\
c_4_3 = vld1q_f32(O + 4 * step + 3 * SIMD);\
c_5_0 = vld1q_f32(O + 5 * step + 0 * SIMD);\
c_5_1 = vld1q_f32(O + 5 * step + 1 * SIMD);\
c_5_2 = vld1q_f32(O + 5 * step + 2 * SIMD);\
c_5_3 = vld1q_f32(O + 5 * step + 3 * SIMD);\

#ifdef STORE_TILE_C
#undef STORE_TILE_C
#endif

#define STORE_TILE_C(O, W_ob, C_ob) \
    vst1q_f32(&Crow(0, 0), C00);    \
    vst1q_f32(&Crow(0, 4), C01);    \
    vst1q_f32(&Crow(0, 8), C02);    \
    vst1q_f32(&Crow(0, 12), C03);   \
    vst1q_f32(&Crow(1, 0), C10);    \
    vst1q_f32(&Crow(1, 4), C11);    \
    vst1q_f32(&Crow(1, 8), C12);    \
    vst1q_f32(&Crow(1, 12), C13);   \
    vst1q_f32(&Crow(2, 0), C20);    \
    vst1q_f32(&Crow(2, 4), C21);    \
    vst1q_f32(&Crow(2, 8), C22);    \
    vst1q_f32(&Crow(2, 12), C23);   \
    vst1q_f32(&Crow(3, 0), C30);    \
    vst1q_f32(&Crow(3, 4), C31);    \
    vst1q_f32(&Crow(3, 8), C32);    \
    vst1q_f32(&Crow(3, 12), C33);   \
    vst1q_f32(&Crow(4, 0), C40);    \
    vst1q_f32(&Crow(4, 4), C41);    \
    vst1q_f32(&Crow(4, 8), C42);    \
    vst1q_f32(&Crow(4, 12), C43);   \
    vst1q_f32(&Crow(5, 0), C50);    \
    vst1q_f32(&Crow(5, 4), C51);    \
    vst1q_f32(&Crow(5, 8), C52);    \
    vst1q_f32(&Crow(5, 12), C53);   \


// #define STORE_TILE_C(O, W_ob, C_ob)\
// vst1q_f32(O + 0 * C_ob + 0 * SIMD, c_0_0);\
// vst1q_f32(O + 0 * C_ob + 1 * SIMD, c_0_1);\
// vst1q_f32(O + 0 * C_ob + 2 * SIMD, c_0_2);\
// vst1q_f32(O + 0 * C_ob + 3 * SIMD, c_0_3);\
// vst1q_f32(O + 1 * C_ob + 0 * SIMD, c_1_0);\
// vst1q_f32(O + 1 * C_ob + 1 * SIMD, c_1_1);\
// vst1q_f32(O + 1 * C_ob + 2 * SIMD, c_1_2);\
// vst1q_f32(O + 1 * C_ob + 3 * SIMD, c_1_3);\
// vst1q_f32(O + 2 * C_ob + 0 * SIMD, c_2_0);\
// vst1q_f32(O + 2 * C_ob + 1 * SIMD, c_2_1);\
// vst1q_f32(O + 2 * C_ob + 2 * SIMD, c_2_2);\
// vst1q_f32(O + 2 * C_ob + 3 * SIMD, c_2_3);\
// vst1q_f32(O + 3 * C_ob + 0 * SIMD, c_3_0);\
// vst1q_f32(O + 3 * C_ob + 1 * SIMD, c_3_1);\
// vst1q_f32(O + 3 * C_ob + 2 * SIMD, c_3_2);\
// vst1q_f32(O + 3 * C_ob + 3 * SIMD, c_3_3);\
// vst1q_f32(O + 4 * C_ob + 0 * SIMD, c_4_0);\
// vst1q_f32(O + 4 * C_ob + 1 * SIMD, c_4_1);\
// vst1q_f32(O + 4 * C_ob + 2 * SIMD, c_4_2);\
// vst1q_f32(O + 4 * C_ob + 3 * SIMD, c_4_3);\
// vst1q_f32(O + 5 * C_ob + 0 * SIMD, c_5_0);\
// vst1q_f32(O + 5 * C_ob + 1 * SIMD, c_5_1);\
// vst1q_f32(O + 5 * C_ob + 2 * SIMD, c_5_2);\
// vst1q_f32(O + 5 * C_ob + 3 * SIMD, c_5_3);\

#ifdef CONV_TILE_C
#undef CONV_TILE_C
#endif

#define fma_reg_broadcast(c_register, b_register, a_register, offset)                            \
    {                                                                                            \
        __asm__ volatile(                                                                        \
            "fmla %[c_reg].4s, %[b_reg].4s, " #a_register ".s[" #offset "]                 \n\t" \
            : [c_reg] "+w"(c_register)                                                           \
            : [b_reg] "w"(b_register));                                                          \
    }
#define Arow(a1, a2) Ar[(a1) * (step) + (a2)]
#define Brow(a1, a2) Br[(a1) * (C_ob) + (a2)]
#define Crow(a1, a2) O[(a1) * (C_ob) + (a2)]
#define Ctrow(a1, a2) Ctmp[(a1) * (C_ob) + (a2)]
#define Atrow(a1, a2, step) Atmp[(a1) * (step) + (a2)]
#define Atrow_no_stride(a1, a2) Atmp[(a2) * (ldAt) + (a1)]

#define Btrow(a1, a2) Btmp[(a1) * (ldBt) + (a2)]

#define Ctref(a1, a2) Ctmp[(a2) * (ldCt) + (a1)]
#define Atref(a1, a2) Atmp[(a2) * (ldAt) + (a1)]


#define CONV_TILE_C(step, a, b, W_ob, C_ob)\
    {                                             \
        float *Atmp = a + 0;              \
        float *Bptr = b + 0 * C_ob;  \
        A0 = vld1q_f32(&Atrow(0, ii, step));            \
        A1 = vld1q_f32(&Atrow(1, ii, step));            \
        A2 = vld1q_f32(&Atrow(2, ii, step));            \
        A3 = vld1q_f32(&Atrow(3, ii, step));            \
        A4 = vld1q_f32(&Atrow(4, ii, step));            \
        A5 = vld1q_f32(&Atrow(5, ii, step));            \
                                                  \
        B0 = vld1q_f32(Bptr + 0 * SIMD);          \
        B1 = vld1q_f32(Bptr + 1 * SIMD);          \
                                                  \
        fma_reg_broadcast(C00, B0, v0, 0);        \
        fma_reg_broadcast(C01, B1, v0, 0);        \
        fma_reg_broadcast(C10, B0, v1, 0);        \
        fma_reg_broadcast(C11, B1, v1, 0);        \
        fma_reg_broadcast(C20, B0, v2, 0);        \
        fma_reg_broadcast(C21, B1, v2, 0);        \
        fma_reg_broadcast(C30, B0, v3, 0);        \
        fma_reg_broadcast(C31, B1, v3, 0);        \
        fma_reg_broadcast(C40, B0, v4, 0);        \
        fma_reg_broadcast(C41, B1, v4, 0);        \
        fma_reg_broadcast(C50, B0, v5, 0);        \
        fma_reg_broadcast(C51, B1, v5, 0);        \
                                                  \
        B0 = vld1q_f32(Bptr + 2 * SIMD);          \
        B1 = vld1q_f32(Bptr + 3 * SIMD);          \
                                                  \
        fma_reg_broadcast(C02, B0, v0, 0);        \
        fma_reg_broadcast(C03, B1, v0, 0);        \
        fma_reg_broadcast(C12, B0, v1, 0);        \
        fma_reg_broadcast(C13, B1, v1, 0);        \
        fma_reg_broadcast(C22, B0, v2, 0);        \
        fma_reg_broadcast(C23, B1, v2, 0);        \
        fma_reg_broadcast(C32, B0, v3, 0);        \
        fma_reg_broadcast(C33, B1, v3, 0);        \
        fma_reg_broadcast(C42, B0, v4, 0);        \
        fma_reg_broadcast(C43, B1, v4, 0);        \
        fma_reg_broadcast(C52, B0, v5, 0);        \
        fma_reg_broadcast(C53, B1, v5, 0);        \
                                                  \
        Bptr += C_ob;                             \
        B0 = vld1q_f32(Bptr + 0 * SIMD);          \
        B1 = vld1q_f32(Bptr + 1 * SIMD);          \
                                                  \
        fma_reg_broadcast(C00, B0, v0, 1);        \
        fma_reg_broadcast(C01, B1, v0, 1);        \
        fma_reg_broadcast(C10, B0, v1, 1);        \
        fma_reg_broadcast(C11, B1, v1, 1);        \
        fma_reg_broadcast(C20, B0, v2, 1);        \
        fma_reg_broadcast(C21, B1, v2, 1);        \
        fma_reg_broadcast(C30, B0, v3, 1);        \
        fma_reg_broadcast(C31, B1, v3, 1);        \
        fma_reg_broadcast(C40, B0, v4, 1);        \
        fma_reg_broadcast(C41, B1, v4, 1);        \
        fma_reg_broadcast(C50, B0, v5, 1);        \
        fma_reg_broadcast(C51, B1, v5, 1);        \
                                                  \
        B0 = vld1q_f32(Bptr + 2 * SIMD);          \
        B1 = vld1q_f32(Bptr + 3 * SIMD);          \
                                                  \
        fma_reg_broadcast(C02, B0, v0, 1);        \
        fma_reg_broadcast(C03, B1, v0, 1);        \
        fma_reg_broadcast(C12, B0, v1, 1);        \
        fma_reg_broadcast(C13, B1, v1, 1);        \
        fma_reg_broadcast(C22, B0, v2, 1);        \
        fma_reg_broadcast(C23, B1, v2, 1);        \
        fma_reg_broadcast(C32, B0, v3, 1);        \
        fma_reg_broadcast(C33, B1, v3, 1);        \
        fma_reg_broadcast(C42, B0, v4, 1);        \
        fma_reg_broadcast(C43, B1, v4, 1);        \
        fma_reg_broadcast(C52, B0, v5, 1);        \
        fma_reg_broadcast(C53, B1, v5, 1);        \
                                                  \
        Bptr += C_ob;                             \
        B0 = vld1q_f32(Bptr + 0 * SIMD);          \
        B1 = vld1q_f32(Bptr + 1 * SIMD);          \
                                                  \
        fma_reg_broadcast(C00, B0, v0, 2);        \
        fma_reg_broadcast(C01, B1, v0, 2);        \
        fma_reg_broadcast(C10, B0, v1, 2);        \
        fma_reg_broadcast(C11, B1, v1, 2);        \
        fma_reg_broadcast(C20, B0, v2, 2);        \
        fma_reg_broadcast(C21, B1, v2, 2);        \
        fma_reg_broadcast(C30, B0, v3, 2);        \
        fma_reg_broadcast(C31, B1, v3, 2);        \
        fma_reg_broadcast(C40, B0, v4, 2);        \
        fma_reg_broadcast(C41, B1, v4, 2);        \
        fma_reg_broadcast(C50, B0, v5, 2);        \
        fma_reg_broadcast(C51, B1, v5, 2);        \
                                                  \
        B0 = vld1q_f32(Bptr + 2 * SIMD);          \
        B1 = vld1q_f32(Bptr + 3 * SIMD);          \
                                                  \
        fma_reg_broadcast(C02, B0, v0, 2);        \
        fma_reg_broadcast(C03, B1, v0, 2);        \
        fma_reg_broadcast(C12, B0, v1, 2);        \
        fma_reg_broadcast(C13, B1, v1, 2);        \
        fma_reg_broadcast(C22, B0, v2, 2);        \
        fma_reg_broadcast(C23, B1, v2, 2);        \
        fma_reg_broadcast(C32, B0, v3, 2);        \
        fma_reg_broadcast(C33, B1, v3, 2);        \
        fma_reg_broadcast(C42, B0, v4, 2);        \
        fma_reg_broadcast(C43, B1, v4, 2);        \
        fma_reg_broadcast(C52, B0, v5, 2);        \
        fma_reg_broadcast(C53, B1, v5, 2);        \
                                                  \
        Bptr += C_ob;                             \
        B0 = vld1q_f32(Bptr + 0 * SIMD);          \
        B1 = vld1q_f32(Bptr + 1 * SIMD);          \
                                                  \
        fma_reg_broadcast(C00, B0, v0, 3);        \
        fma_reg_broadcast(C01, B1, v0, 3);        \
        fma_reg_broadcast(C10, B0, v1, 3);        \
        fma_reg_broadcast(C11, B1, v1, 3);        \
        fma_reg_broadcast(C20, B0, v2, 3);        \
        fma_reg_broadcast(C21, B1, v2, 3);        \
        fma_reg_broadcast(C30, B0, v3, 3);        \
        fma_reg_broadcast(C31, B1, v3, 3);        \
        fma_reg_broadcast(C40, B0, v4, 3);        \
        fma_reg_broadcast(C41, B1, v4, 3);        \
        fma_reg_broadcast(C50, B0, v5, 3);        \
        fma_reg_broadcast(C51, B1, v5, 3);        \
                                                  \
        B0 = vld1q_f32(Bptr + 2 * SIMD);          \
        B1 = vld1q_f32(Bptr + 3 * SIMD);          \
                                                  \
        fma_reg_broadcast(C02, B0, v0, 3);        \
        fma_reg_broadcast(C03, B1, v0, 3);        \
        fma_reg_broadcast(C12, B0, v1, 3);        \
        fma_reg_broadcast(C13, B1, v1, 3);        \
        fma_reg_broadcast(C22, B0, v2, 3);        \
        fma_reg_broadcast(C23, B1, v2, 3);        \
        fma_reg_broadcast(C32, B0, v3, 3);        \
        fma_reg_broadcast(C33, B1, v3, 3);        \
        fma_reg_broadcast(C42, B0, v4, 3);        \
        fma_reg_broadcast(C43, B1, v4, 3);        \
        fma_reg_broadcast(C52, B0, v5, 3);        \
        fma_reg_broadcast(C53, B1, v5, 3);        \
                                                  \
        Bptr += C_ob;                             \
    }



// #define CONV_TILE_C(step, a, b, W_ob, C_ob)\
// float32x4_t a_0;\
// float32x4_t a_1;\
// float32x4_t a_2;\
// float32x4_t a_3;\
// float32x4_t a_4;\
// float32x4_t a_5;\
// float32x4_t b_0;\
// float32x4_t b_1;\
// a_0 = vld1q_f32(a + 0 * step + 0 * SIMD);\
// b_0 = vld1q_f32(b + (0 * SIMD + 0)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 0);\
// b_1 = vld1q_f32(b + (0 * SIMD + 0)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 0);\
// a_1 = vld1q_f32(a + 1 * step + 0 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 0);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 0);\
// a_2 = vld1q_f32(a + 2 * step + 0 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 0);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 0);\
// a_3 = vld1q_f32(a + 3 * step + 0 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 0);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 0);\
// a_4 = vld1q_f32(a + 4 * step + 0 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 0);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 0);\
// a_5 = vld1q_f32(a + 5 * step + 0 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 0);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 0);\
// a_0 = vld1q_f32(a + 0 * step + 0 * SIMD);\
// b_0 = vld1q_f32(b + (0 * SIMD + 1)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 1);\
// b_1 = vld1q_f32(b + (0 * SIMD + 1)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 1);\
// a_1 = vld1q_f32(a + 1 * step + 0 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 1);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 1);\
// a_2 = vld1q_f32(a + 2 * step + 0 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 1);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 1);\
// a_3 = vld1q_f32(a + 3 * step + 0 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 1);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 1);\
// a_4 = vld1q_f32(a + 4 * step + 0 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 1);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 1);\
// a_5 = vld1q_f32(a + 5 * step + 0 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 1);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 1);\
// a_0 = vld1q_f32(a + 0 * step + 0 * SIMD);\
// b_0 = vld1q_f32(b + (0 * SIMD + 2)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 2);\
// b_1 = vld1q_f32(b + (0 * SIMD + 2)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 2);\
// a_1 = vld1q_f32(a + 1 * step + 0 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 2);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 2);\
// a_2 = vld1q_f32(a + 2 * step + 0 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 2);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 2);\
// a_3 = vld1q_f32(a + 3 * step + 0 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 2);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 2);\
// a_4 = vld1q_f32(a + 4 * step + 0 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 2);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 2);\
// a_5 = vld1q_f32(a + 5 * step + 0 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 2);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 2);\
// a_0 = vld1q_f32(a + 0 * step + 0 * SIMD);\
// b_0 = vld1q_f32(b + (0 * SIMD + 3)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 3);\
// b_1 = vld1q_f32(b + (0 * SIMD + 3)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 3);\
// a_1 = vld1q_f32(a + 1 * step + 0 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 3);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 3);\
// a_2 = vld1q_f32(a + 2 * step + 0 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 3);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 3);\
// a_3 = vld1q_f32(a + 3 * step + 0 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 3);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 3);\
// a_4 = vld1q_f32(a + 4 * step + 0 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 3);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 3);\
// a_5 = vld1q_f32(a + 5 * step + 0 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 3);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 3);\
// b_0 = vld1q_f32(b + (0 * SIMD + 0)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 0);\
// b_1 = vld1q_f32(b + (0 * SIMD + 0)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 0);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 0);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 0);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 0);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 0);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 0);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 0);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 0);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 0);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 0);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 0);\
// b_0 = vld1q_f32(b + (0 * SIMD + 1)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 1);\
// b_1 = vld1q_f32(b + (0 * SIMD + 1)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 1);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 1);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 1);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 1);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 1);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 1);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 1);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 1);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 1);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 1);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 1);\
// b_0 = vld1q_f32(b + (0 * SIMD + 2)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 2);\
// b_1 = vld1q_f32(b + (0 * SIMD + 2)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 2);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 2);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 2);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 2);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 2);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 2);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 2);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 2);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 2);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 2);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 2);\
// b_0 = vld1q_f32(b + (0 * SIMD + 3)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 3);\
// b_1 = vld1q_f32(b + (0 * SIMD + 3)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 3);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 3);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 3);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 3);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 3);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 3);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 3);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 3);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 3);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 3);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 3);\
// a_0 = vld1q_f32(a + 0 * step + 1 * SIMD);\
// b_0 = vld1q_f32(b + (1 * SIMD + 0)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 0);\
// b_1 = vld1q_f32(b + (1 * SIMD + 0)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 0);\
// a_1 = vld1q_f32(a + 1 * step + 1 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 0);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 0);\
// a_2 = vld1q_f32(a + 2 * step + 1 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 0);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 0);\
// a_3 = vld1q_f32(a + 3 * step + 1 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 0);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 0);\
// a_4 = vld1q_f32(a + 4 * step + 1 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 0);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 0);\
// a_5 = vld1q_f32(a + 5 * step + 1 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 0);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 0);\
// a_0 = vld1q_f32(a + 0 * step + 1 * SIMD);\
// b_0 = vld1q_f32(b + (1 * SIMD + 1)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 1);\
// b_1 = vld1q_f32(b + (1 * SIMD + 1)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 1);\
// a_1 = vld1q_f32(a + 1 * step + 1 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 1);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 1);\
// a_2 = vld1q_f32(a + 2 * step + 1 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 1);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 1);\
// a_3 = vld1q_f32(a + 3 * step + 1 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 1);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 1);\
// a_4 = vld1q_f32(a + 4 * step + 1 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 1);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 1);\
// a_5 = vld1q_f32(a + 5 * step + 1 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 1);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 1);\
// a_0 = vld1q_f32(a + 0 * step + 1 * SIMD);\
// b_0 = vld1q_f32(b + (1 * SIMD + 2)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 2);\
// b_1 = vld1q_f32(b + (1 * SIMD + 2)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 2);\
// a_1 = vld1q_f32(a + 1 * step + 1 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 2);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 2);\
// a_2 = vld1q_f32(a + 2 * step + 1 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 2);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 2);\
// a_3 = vld1q_f32(a + 3 * step + 1 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 2);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 2);\
// a_4 = vld1q_f32(a + 4 * step + 1 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 2);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 2);\
// a_5 = vld1q_f32(a + 5 * step + 1 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 2);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 2);\
// a_0 = vld1q_f32(a + 0 * step + 1 * SIMD);\
// b_0 = vld1q_f32(b + (1 * SIMD + 3)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 3);\
// b_1 = vld1q_f32(b + (1 * SIMD + 3)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 3);\
// a_1 = vld1q_f32(a + 1 * step + 1 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 3);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 3);\
// a_2 = vld1q_f32(a + 2 * step + 1 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 3);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 3);\
// a_3 = vld1q_f32(a + 3 * step + 1 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 3);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 3);\
// a_4 = vld1q_f32(a + 4 * step + 1 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 3);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 3);\
// a_5 = vld1q_f32(a + 5 * step + 1 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 3);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 3);\
// b_0 = vld1q_f32(b + (1 * SIMD + 0)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 0);\
// b_1 = vld1q_f32(b + (1 * SIMD + 0)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 0);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 0);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 0);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 0);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 0);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 0);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 0);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 0);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 0);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 0);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 0);\
// b_0 = vld1q_f32(b + (1 * SIMD + 1)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 1);\
// b_1 = vld1q_f32(b + (1 * SIMD + 1)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 1);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 1);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 1);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 1);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 1);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 1);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 1);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 1);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 1);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 1);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 1);\
// b_0 = vld1q_f32(b + (1 * SIMD + 2)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 2);\
// b_1 = vld1q_f32(b + (1 * SIMD + 2)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 2);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 2);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 2);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 2);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 2);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 2);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 2);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 2);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 2);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 2);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 2);\
// b_0 = vld1q_f32(b + (1 * SIMD + 3)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 3);\
// b_1 = vld1q_f32(b + (1 * SIMD + 3)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 3);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 3);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 3);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 3);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 3);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 3);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 3);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 3);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 3);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 3);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 3);\
// a_0 = vld1q_f32(a + 0 * step + 2 * SIMD);\
// b_0 = vld1q_f32(b + (2 * SIMD + 0)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 0);\
// b_1 = vld1q_f32(b + (2 * SIMD + 0)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 0);\
// a_1 = vld1q_f32(a + 1 * step + 2 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 0);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 0);\
// a_2 = vld1q_f32(a + 2 * step + 2 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 0);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 0);\
// a_3 = vld1q_f32(a + 3 * step + 2 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 0);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 0);\
// a_4 = vld1q_f32(a + 4 * step + 2 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 0);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 0);\
// a_5 = vld1q_f32(a + 5 * step + 2 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 0);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 0);\
// a_0 = vld1q_f32(a + 0 * step + 2 * SIMD);\
// b_0 = vld1q_f32(b + (2 * SIMD + 1)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 1);\
// b_1 = vld1q_f32(b + (2 * SIMD + 1)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 1);\
// a_1 = vld1q_f32(a + 1 * step + 2 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 1);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 1);\
// a_2 = vld1q_f32(a + 2 * step + 2 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 1);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 1);\
// a_3 = vld1q_f32(a + 3 * step + 2 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 1);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 1);\
// a_4 = vld1q_f32(a + 4 * step + 2 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 1);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 1);\
// a_5 = vld1q_f32(a + 5 * step + 2 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 1);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 1);\
// a_0 = vld1q_f32(a + 0 * step + 2 * SIMD);\
// b_0 = vld1q_f32(b + (2 * SIMD + 2)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 2);\
// b_1 = vld1q_f32(b + (2 * SIMD + 2)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 2);\
// a_1 = vld1q_f32(a + 1 * step + 2 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 2);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 2);\
// a_2 = vld1q_f32(a + 2 * step + 2 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 2);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 2);\
// a_3 = vld1q_f32(a + 3 * step + 2 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 2);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 2);\
// a_4 = vld1q_f32(a + 4 * step + 2 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 2);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 2);\
// a_5 = vld1q_f32(a + 5 * step + 2 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 2);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 2);\
// a_0 = vld1q_f32(a + 0 * step + 2 * SIMD);\
// b_0 = vld1q_f32(b + (2 * SIMD + 3)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 3);\
// b_1 = vld1q_f32(b + (2 * SIMD + 3)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 3);\
// a_1 = vld1q_f32(a + 1 * step + 2 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 3);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 3);\
// a_2 = vld1q_f32(a + 2 * step + 2 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 3);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 3);\
// a_3 = vld1q_f32(a + 3 * step + 2 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 3);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 3);\
// a_4 = vld1q_f32(a + 4 * step + 2 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 3);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 3);\
// a_5 = vld1q_f32(a + 5 * step + 2 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 3);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 3);\
// b_0 = vld1q_f32(b + (2 * SIMD + 0)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 0);\
// b_1 = vld1q_f32(b + (2 * SIMD + 0)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 0);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 0);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 0);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 0);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 0);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 0);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 0);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 0);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 0);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 0);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 0);\
// b_0 = vld1q_f32(b + (2 * SIMD + 1)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 1);\
// b_1 = vld1q_f32(b + (2 * SIMD + 1)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 1);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 1);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 1);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 1);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 1);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 1);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 1);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 1);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 1);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 1);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 1);\
// b_0 = vld1q_f32(b + (2 * SIMD + 2)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 2);\
// b_1 = vld1q_f32(b + (2 * SIMD + 2)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 2);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 2);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 2);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 2);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 2);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 2);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 2);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 2);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 2);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 2);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 2);\
// b_0 = vld1q_f32(b + (2 * SIMD + 3)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 3);\
// b_1 = vld1q_f32(b + (2 * SIMD + 3)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 3);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 3);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 3);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 3);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 3);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 3);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 3);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 3);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 3);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 3);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 3);\
// a_0 = vld1q_f32(a + 0 * step + 3 * SIMD);\
// b_0 = vld1q_f32(b + (3 * SIMD + 0)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 0);\
// b_1 = vld1q_f32(b + (3 * SIMD + 0)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 0);\
// a_1 = vld1q_f32(a + 1 * step + 3 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 0);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 0);\
// a_2 = vld1q_f32(a + 2 * step + 3 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 0);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 0);\
// a_3 = vld1q_f32(a + 3 * step + 3 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 0);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 0);\
// a_4 = vld1q_f32(a + 4 * step + 3 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 0);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 0);\
// a_5 = vld1q_f32(a + 5 * step + 3 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 0);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 0);\
// a_0 = vld1q_f32(a + 0 * step + 3 * SIMD);\
// b_0 = vld1q_f32(b + (3 * SIMD + 1)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 1);\
// b_1 = vld1q_f32(b + (3 * SIMD + 1)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 1);\
// a_1 = vld1q_f32(a + 1 * step + 3 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 1);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 1);\
// a_2 = vld1q_f32(a + 2 * step + 3 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 1);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 1);\
// a_3 = vld1q_f32(a + 3 * step + 3 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 1);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 1);\
// a_4 = vld1q_f32(a + 4 * step + 3 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 1);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 1);\
// a_5 = vld1q_f32(a + 5 * step + 3 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 1);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 1);\
// a_0 = vld1q_f32(a + 0 * step + 3 * SIMD);\
// b_0 = vld1q_f32(b + (3 * SIMD + 2)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 2);\
// b_1 = vld1q_f32(b + (3 * SIMD + 2)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 2);\
// a_1 = vld1q_f32(a + 1 * step + 3 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 2);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 2);\
// a_2 = vld1q_f32(a + 2 * step + 3 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 2);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 2);\
// a_3 = vld1q_f32(a + 3 * step + 3 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 2);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 2);\
// a_4 = vld1q_f32(a + 4 * step + 3 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 2);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 2);\
// a_5 = vld1q_f32(a + 5 * step + 3 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 2);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 2);\
// a_0 = vld1q_f32(a + 0 * step + 3 * SIMD);\
// b_0 = vld1q_f32(b + (3 * SIMD + 3)*C_ob + (0 * 2 + 0)*SIMD);\
// c_0_0 = vfmaq_laneq_f32(c_0_0, b_0, a_0, 3);\
// b_1 = vld1q_f32(b + (3 * SIMD + 3)*C_ob + (0 * 2 + 1)*SIMD);\
// c_0_1 = vfmaq_laneq_f32(c_0_1, b_1, a_0, 3);\
// a_1 = vld1q_f32(a + 1 * step + 3 * SIMD);\
// c_1_0 = vfmaq_laneq_f32(c_1_0, b_0, a_1, 3);\
// c_1_1 = vfmaq_laneq_f32(c_1_1, b_1, a_1, 3);\
// a_2 = vld1q_f32(a + 2 * step + 3 * SIMD);\
// c_2_0 = vfmaq_laneq_f32(c_2_0, b_0, a_2, 3);\
// c_2_1 = vfmaq_laneq_f32(c_2_1, b_1, a_2, 3);\
// a_3 = vld1q_f32(a + 3 * step + 3 * SIMD);\
// c_3_0 = vfmaq_laneq_f32(c_3_0, b_0, a_3, 3);\
// c_3_1 = vfmaq_laneq_f32(c_3_1, b_1, a_3, 3);\
// a_4 = vld1q_f32(a + 4 * step + 3 * SIMD);\
// c_4_0 = vfmaq_laneq_f32(c_4_0, b_0, a_4, 3);\
// c_4_1 = vfmaq_laneq_f32(c_4_1, b_1, a_4, 3);\
// a_5 = vld1q_f32(a + 5 * step + 3 * SIMD);\
// c_5_0 = vfmaq_laneq_f32(c_5_0, b_0, a_5, 3);\
// c_5_1 = vfmaq_laneq_f32(c_5_1, b_1, a_5, 3);\
// b_0 = vld1q_f32(b + (3 * SIMD + 0)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 0);\
// b_1 = vld1q_f32(b + (3 * SIMD + 0)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 0);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 0);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 0);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 0);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 0);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 0);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 0);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 0);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 0);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 0);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 0);\
// b_0 = vld1q_f32(b + (3 * SIMD + 1)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 1);\
// b_1 = vld1q_f32(b + (3 * SIMD + 1)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 1);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 1);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 1);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 1);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 1);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 1);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 1);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 1);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 1);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 1);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 1);\
// b_0 = vld1q_f32(b + (3 * SIMD + 2)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 2);\
// b_1 = vld1q_f32(b + (3 * SIMD + 2)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 2);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 2);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 2);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 2);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 2);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 2);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 2);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 2);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 2);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 2);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 2);\
// b_0 = vld1q_f32(b + (3 * SIMD + 3)*C_ob + (1 * 2 + 0)*SIMD);\
// c_0_2 = vfmaq_laneq_f32(c_0_2, b_0, a_0, 3);\
// b_1 = vld1q_f32(b + (3 * SIMD + 3)*C_ob + (1 * 2 + 1)*SIMD);\
// c_0_3 = vfmaq_laneq_f32(c_0_3, b_1, a_0, 3);\
// c_1_2 = vfmaq_laneq_f32(c_1_2, b_0, a_1, 3);\
// c_1_3 = vfmaq_laneq_f32(c_1_3, b_1, a_1, 3);\
// c_2_2 = vfmaq_laneq_f32(c_2_2, b_0, a_2, 3);\
// c_2_3 = vfmaq_laneq_f32(c_2_3, b_1, a_2, 3);\
// c_3_2 = vfmaq_laneq_f32(c_3_2, b_0, a_3, 3);\
// c_3_3 = vfmaq_laneq_f32(c_3_3, b_1, a_3, 3);\
// c_4_2 = vfmaq_laneq_f32(c_4_2, b_0, a_4, 3);\
// c_4_3 = vfmaq_laneq_f32(c_4_3, b_1, a_4, 3);\
// c_5_2 = vfmaq_laneq_f32(c_5_2, b_0, a_5, 3);\
// c_5_3 = vfmaq_laneq_f32(c_5_3, b_1, a_5, 3);\

#ifdef MAX_TILE_C
#undef MAX_TILE_C
#endif

#define MAX_TILE_C(step, a, W_ob, C_ob)\
float32x4_t av; \
av = vld1q_f32(a + 0 * step + 0 * SIMD);\
c_0_0 = vmaxq_f32(c_0_0, av);\
av = vld1q_f32(a + 0 * step + 1 * SIMD);\
c_0_1 = vmaxq_f32(c_0_1, av);\
av = vld1q_f32(a + 0 * step + 2 * SIMD);\
c_0_2 = vmaxq_f32(c_0_2, av);\
av = vld1q_f32(a + 0 * step + 3 * SIMD);\
c_0_3 = vmaxq_f32(c_0_3, av);\
av = vld1q_f32(a + 1 * step + 0 * SIMD);\
c_1_0 = vmaxq_f32(c_1_0, av);\
av = vld1q_f32(a + 1 * step + 1 * SIMD);\
c_1_1 = vmaxq_f32(c_1_1, av);\
av = vld1q_f32(a + 1 * step + 2 * SIMD);\
c_1_2 = vmaxq_f32(c_1_2, av);\
av = vld1q_f32(a + 1 * step + 3 * SIMD);\
c_1_3 = vmaxq_f32(c_1_3, av);\
av = vld1q_f32(a + 2 * step + 0 * SIMD);\
c_2_0 = vmaxq_f32(c_2_0, av);\
av = vld1q_f32(a + 2 * step + 1 * SIMD);\
c_2_1 = vmaxq_f32(c_2_1, av);\
av = vld1q_f32(a + 2 * step + 2 * SIMD);\
c_2_2 = vmaxq_f32(c_2_2, av);\
av = vld1q_f32(a + 2 * step + 3 * SIMD);\
c_2_3 = vmaxq_f32(c_2_3, av);\
av = vld1q_f32(a + 3 * step + 0 * SIMD);\
c_3_0 = vmaxq_f32(c_3_0, av);\
av = vld1q_f32(a + 3 * step + 1 * SIMD);\
c_3_1 = vmaxq_f32(c_3_1, av);\
av = vld1q_f32(a + 3 * step + 2 * SIMD);\
c_3_2 = vmaxq_f32(c_3_2, av);\
av = vld1q_f32(a + 3 * step + 3 * SIMD);\
c_3_3 = vmaxq_f32(c_3_3, av);\
av = vld1q_f32(a + 4 * step + 0 * SIMD);\
c_4_0 = vmaxq_f32(c_4_0, av);\
av = vld1q_f32(a + 4 * step + 1 * SIMD);\
c_4_1 = vmaxq_f32(c_4_1, av);\
av = vld1q_f32(a + 4 * step + 2 * SIMD);\
c_4_2 = vmaxq_f32(c_4_2, av);\
av = vld1q_f32(a + 4 * step + 3 * SIMD);\
c_4_3 = vmaxq_f32(c_4_3, av);\
av = vld1q_f32(a + 5 * step + 0 * SIMD);\
c_5_0 = vmaxq_f32(c_5_0, av);\
av = vld1q_f32(a + 5 * step + 1 * SIMD);\
c_5_1 = vmaxq_f32(c_5_1, av);\
av = vld1q_f32(a + 5 * step + 2 * SIMD);\
c_5_2 = vmaxq_f32(c_5_2, av);\
av = vld1q_f32(a + 5 * step + 3 * SIMD);\
c_5_3 = vmaxq_f32(c_5_3, av);\

#ifdef DW_TILE_C
#undef DW_TILE_C
#endif

#define DW_TILE_C(step, a, b, W_ob, C_ob)\
float32x4_t av; \
float32x4_t b_0 = vld1q_f32(b + 0*SIMD);\
float32x4_t b_1 = vld1q_f32(b + 1*SIMD);\
float32x4_t b_2 = vld1q_f32(b + 2*SIMD);\
float32x4_t b_3 = vld1q_f32(b + 3*SIMD);\
av = vld1q_f32(a + 0 * step + 0 * SIMD);\
c_0_0 = vfmaq_f32(c_0_0, av, b_0);\
av = vld1q_f32(a + 0 * step + 1 * SIMD);\
c_0_1 = vfmaq_f32(c_0_1, av, b_1);\
av = vld1q_f32(a + 0 * step + 2 * SIMD);\
c_0_2 = vfmaq_f32(c_0_2, av, b_2);\
av = vld1q_f32(a + 0 * step + 3 * SIMD);\
c_0_3 = vfmaq_f32(c_0_3, av, b_3);\
av = vld1q_f32(a + 1 * step + 0 * SIMD);\
c_1_0 = vfmaq_f32(c_1_0, av, b_0);\
av = vld1q_f32(a + 1 * step + 1 * SIMD);\
c_1_1 = vfmaq_f32(c_1_1, av, b_1);\
av = vld1q_f32(a + 1 * step + 2 * SIMD);\
c_1_2 = vfmaq_f32(c_1_2, av, b_2);\
av = vld1q_f32(a + 1 * step + 3 * SIMD);\
c_1_3 = vfmaq_f32(c_1_3, av, b_3);\
av = vld1q_f32(a + 2 * step + 0 * SIMD);\
c_2_0 = vfmaq_f32(c_2_0, av, b_0);\
av = vld1q_f32(a + 2 * step + 1 * SIMD);\
c_2_1 = vfmaq_f32(c_2_1, av, b_1);\
av = vld1q_f32(a + 2 * step + 2 * SIMD);\
c_2_2 = vfmaq_f32(c_2_2, av, b_2);\
av = vld1q_f32(a + 2 * step + 3 * SIMD);\
c_2_3 = vfmaq_f32(c_2_3, av, b_3);\
av = vld1q_f32(a + 3 * step + 0 * SIMD);\
c_3_0 = vfmaq_f32(c_3_0, av, b_0);\
av = vld1q_f32(a + 3 * step + 1 * SIMD);\
c_3_1 = vfmaq_f32(c_3_1, av, b_1);\
av = vld1q_f32(a + 3 * step + 2 * SIMD);\
c_3_2 = vfmaq_f32(c_3_2, av, b_2);\
av = vld1q_f32(a + 3 * step + 3 * SIMD);\
c_3_3 = vfmaq_f32(c_3_3, av, b_3);\
av = vld1q_f32(a + 4 * step + 0 * SIMD);\
c_4_0 = vfmaq_f32(c_4_0, av, b_0);\
av = vld1q_f32(a + 4 * step + 1 * SIMD);\
c_4_1 = vfmaq_f32(c_4_1, av, b_1);\
av = vld1q_f32(a + 4 * step + 2 * SIMD);\
c_4_2 = vfmaq_f32(c_4_2, av, b_2);\
av = vld1q_f32(a + 4 * step + 3 * SIMD);\
c_4_3 = vfmaq_f32(c_4_3, av, b_3);\
av = vld1q_f32(a + 5 * step + 0 * SIMD);\
c_5_0 = vfmaq_f32(c_5_0, av, b_0);\
av = vld1q_f32(a + 5 * step + 1 * SIMD);\
c_5_1 = vfmaq_f32(c_5_1, av, b_1);\
av = vld1q_f32(a + 5 * step + 2 * SIMD);\
c_5_2 = vfmaq_f32(c_5_2, av, b_2);\
av = vld1q_f32(a + 5 * step + 3 * SIMD);\
c_5_3 = vfmaq_f32(c_5_3, av, b_3);\

