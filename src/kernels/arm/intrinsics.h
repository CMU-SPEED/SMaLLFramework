#include <arm_neon.h>
void print(float32x4_t vec)
{
    float a[4];
    vst1q_f32((float32_t *)a, vec);
    for (int i = 0; i < 4; ++i)
        printf("%.4f \t", a[i]);
    printf("\n");
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

#define fma_reg_broadcast(c_register, b_register, a_register, offset)                            \
    {                                                                                            \
        __asm__ volatile(                                                                        \
            "fmla %[c_reg].4s, %[b_reg].4s, " #a_register ".s[" #offset "]                 \n\t" \
            : [c_reg] "+w"(c_register)                                                           \
            : [b_reg] "w"(b_register));                                                          \
    }

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



// Convolution Computation
//(Strided GEMM)
//  Pouint32_ter to C defined in the outer scope
#define FMA_TILE_C(step, a, b, p_cur, W_ob, C_ob) \
    {                                             \
        float *Atmp = a + p_cur;              \
        float *Bptr = b + p_cur * C_ob;  \
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


// scalar versions of all the microkernels for platform portability

// Architecture specific tiling params

// __m256 a_reg,b0,b1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13;

// Initializations

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

#define ZERO_END_C(W_ob, C_ob) \
    float c_tile[W_ob * C_ob] = {0};

// Loads


#define LOAD_LAST_C(O, W_ob, C_ob, W_last)              \
    float c_tile[W_ob * C_ob];                          \
    for (uint32_t kk = 0; kk < W_last; kk++)            \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
        }                                               \
    }

// Pooling Loads

#define LOAD_TILE_C_POOL(O, W_ob, C_ob)                 \
    float c_tile[W_ob * C_ob];                          \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
        }                                               \
    }

#define LOAD_LAST_C_POOL(O, W_ob, C_ob, W_last)         \
    float c_tile[W_ob * C_ob];                          \
    for (uint32_t kk = 0; kk < W_last; kk++)            \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
        }                                               \
    }

#define LOAD_TILE_C_DW(O, W_ob, C_ob)                   \
    float c_tile[W_ob * C_ob];                          \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
        }                                               \
    }

// strided loads
#define LOAD_TILE_C_strided(O, step, _W_ob, _C_ob)       \
float32x4_t B0, B1,            \
    A0, C00, C01, C02, C03,    \
    A1, C10, C11, C12, C13,    \
    A2, C20, C21, C22, C23,    \
    A3, C30, C31, C32, C33,    \
    A4, C40, C41, C42, C43,    \
    A5, C50, C51, C52, C53;    \
    float *Atmp = O;\
C00 = vld1q_f32(&Atrow(0, 0, step));  \
C01 = vld1q_f32(&Atrow(0, 4, step));  \
C02 = vld1q_f32(&Atrow(0, 8, step));  \
C03 = vld1q_f32(&Atrow(0, 12, step)); \
C10 = vld1q_f32(&Atrow(1, 0, step));  \
C11 = vld1q_f32(&Atrow(1, 4, step));  \
C12 = vld1q_f32(&Atrow(1, 8, step));  \
C13 = vld1q_f32(&Atrow(1, 12, step)); \
C20 = vld1q_f32(&Atrow(2, 0, step));  \
C21 = vld1q_f32(&Atrow(2, 4, step));  \
C22 = vld1q_f32(&Atrow(2, 8, step));  \
C23 = vld1q_f32(&Atrow(2, 12, step)); \
C30 = vld1q_f32(&Atrow(3, 0, step));  \
C31 = vld1q_f32(&Atrow(3, 4, step));  \
C32 = vld1q_f32(&Atrow(3, 8, step));  \
C33 = vld1q_f32(&Atrow(3, 12, step)); \
C40 = vld1q_f32(&Atrow(4, 0, step));  \
C41 = vld1q_f32(&Atrow(4, 4, step));  \
C42 = vld1q_f32(&Atrow(4, 8, step));  \
C43 = vld1q_f32(&Atrow(4, 12, step)); \
C50 = vld1q_f32(&Atrow(5, 0, step));  \
C51 = vld1q_f32(&Atrow(5, 4, step));  \
C52 = vld1q_f32(&Atrow(5, 8, step));  \
C53 = vld1q_f32(&Atrow(5, 12, step)); \


#define LOAD_LAST_C_strided(O, step, W_ob, C_ob, W_last) \
    float c_tile[W_ob * C_ob];                           \
    for (uint32_t kk = 0; kk < W_last; kk++)             \
    {                                                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)           \
        {                                                \
            c_tile[kk * C_ob + jj] = O[kk * step + jj];  \
        }                                                \
    }

#define LOAD_TILE_C_strided_DW(O, step, _W_ob, _C_ob)    \
    float c_tile[_W_ob * _C_ob];                         \
    for (uint32_t kk = 0; kk < _W_ob; kk++)              \
    {                                                    \
        for (uint32_t jj = 0; jj < _C_ob; jj++)          \
        {                                                \
            c_tile[kk * _C_ob + jj] = O[kk * step + jj]; \
        }                                                \
    }
// Stores



#define STORE_END_C(O, W_ob, C_ob, W_last)              \
    for (uint32_t kk = 0; kk < W_last; kk++)            \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
        }                                               \
    }

#define STORE_TILE_C_POOL(O, W_ob_pool, C_ob)           \
    for (uint32_t kk = 0; kk < W_ob_pool; kk++)         \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
        }                                               \
    }

#define STORE_END_C_POOL(O, W_ob_pool, C_ob, W_last)    \
    for (uint32_t kk = 0; kk < W_last; kk++)            \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
        }                                               \
    }

#define STORE_TILE_INTER(W_ob, C_ob) \
    void *do_nothing = NULL;

#define FMA_END_C(step, a, b, p_cur, W_ob, C_ob, W_last) \
    float *c_pixel;                                      \
    float *a_channel = a + p_cur;                        \
    for (uint32_t kk = 0; kk < W_last; kk++)             \
    {                                                    \
        float a_val = *(a_channel);                      \
        c_pixel = c_tile + kk * C_ob;                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)           \
        {                                                \
            float b_val = *(b + p_cur * C_ob + jj);      \
            *(c_pixel + jj) += a_val * b_val;            \
        }                                                \
        a_channel += step;                               \
    }

// Pooling
//   Max pooling

#define MAX_TILE_C(step, a, W_ob, C_ob) \
float * Atmp = a;                           \
A0 = vld1q_f32(&Atrow(0, 0, step));         \
A1 = vld1q_f32(&Atrow(0, 4, step));    \
A2 = vld1q_f32(&Atrow(0, 8, step));         \
A3 = vld1q_f32(&Atrow(0, 12, step));    \
/*Compute row 0*/\
C00 = vmaxq_f32(A0, C00);                  \
B0 = vld1q_f32(&Atrow(1, 0, step));\
C01 = vmaxq_f32(A1, C01);                  \
B1 = vld1q_f32(&Atrow(1, 4, step));\
C02 = vmaxq_f32(A2, C02);                  \
A0 = vld1q_f32(&Atrow(1, 8, step));  \
C03 = vmaxq_f32(A3, C03);                  \
A1 = vld1q_f32(&Atrow(1, 12, step));         \
/*Compute row 1*/\
C10 = vmaxq_f32(B0, C10);                  \
A2 = vld1q_f32(&Atrow(2, 0, step));\
C11 = vmaxq_f32(B1, C11);                  \
A3 = vld1q_f32(&Atrow(2, 4, step));\
C12 = vmaxq_f32(A0, C12);                  \
B0 = vld1q_f32(&Atrow(2, 8, step));  \
C13 = vmaxq_f32(A1, C13);                  \
B1 = vld1q_f32(&Atrow(2, 12, step));   \
/*Compute row 2*/\
C20 = vmaxq_f32(A2, C20);                  \
A0 = vld1q_f32(&Atrow(3, 0, step));\
C21 = vmaxq_f32(A3, C21);                  \
A1 = vld1q_f32(&Atrow(3, 4, step));\
C22 = vmaxq_f32(B0, C22);                  \
A2 = vld1q_f32(&Atrow(3, 8, step));  \
C23 = vmaxq_f32(B1, C23);                  \
A3 = vld1q_f32(&Atrow(3, 12, step));   \
/*Compute row 3*/\
C30 = vmaxq_f32(A0, C30);                  \
B0 = vld1q_f32(&Atrow(4, 0, step));\
C31 = vmaxq_f32(A1, C31);                  \
B1 = vld1q_f32(&Atrow(4, 4, step));\
C32 = vmaxq_f32(A2, C32);                  \
A0 = vld1q_f32(&Atrow(4, 8, step));  \
C33 = vmaxq_f32(A3, C33);                  \
A1 = vld1q_f32(&Atrow(4, 12, step));         \
/*Compute row 4*/\
C40 = vmaxq_f32(B0, C40);                  \
A2 = vld1q_f32(&Atrow(5, 0, step));\
C41 = vmaxq_f32(B1, C41);                  \
A3 = vld1q_f32(&Atrow(5, 4, step));\
C42 = vmaxq_f32(A0, C42);                  \
B0 = vld1q_f32(&Atrow(5, 8, step));  \
C43 = vmaxq_f32(A1, C43);                  \
B1 = vld1q_f32(&Atrow(5, 12, step));   \
/*Compute row 1*/\
C50 = vmaxq_f32(A2, C50);                  \
C51 = vmaxq_f32(A3, C51);                  \
C52 = vmaxq_f32(B0, C52);                  \
C53 = vmaxq_f32(B1, C53);


#define MAX_END_C(step, a, W_last, C_ob)                                                \
    float *c_pixel = c_tile;                                                            \
    float *a_pixel = a;                                                                 \
    for (uint32_t kk = 0; kk < W_last; kk++)                                            \
    {                                                                                   \
        float *c_channel = c_pixel;                                                     \
        float *a_channel = a_pixel;                                                     \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                          \
        {                                                                               \
            *(c_channel) = (*(a_channel) > *(c_channel)) ? *(a_channel) : *(c_channel); \
            c_channel++;                                                                \
            a_channel++;                                                                \
        }                                                                               \
        a_pixel += step;                                                                \
        c_pixel += C_ob;                                                                \
    }

#define MAX_TILE_IP(pool_col_stride, W_ob, C_ob, pool_stride, pool_H_f, pool_W_f, O_row, O_col, O_pool, H_o, W_o_full)            \
    float *c_pixel = c_tile;                                                                                                      \
    for (uint32_t kk = 0; kk < W_ob; kk++)                                                                                        \
    {                                                                                                                             \
        if (O_row % pool_stride == 0 && (O_row + pool_H_f - 1) < H_o)                                                             \
        {                                                                                                                         \
            float *p_row = O_pool + ((O_row) / pool_stride) * pool_col_stride;                                                    \
            if (O_col % pool_stride == 0 && (O_col + pool_W_f - 1) < W_o_full)                                                    \
            {                                                                                                                     \
                float *p_pixel = p_row + ((O_col) / pool_stride) * C_ob;                                                          \
                float *p_channel = p_pixel;                                                                                       \
                float *c_channel = c_pixel;                                                                                       \
                for (uint32_t jj = 0; jj < C_ob; jj++)                                                                            \
                {                                                                                                                 \
                    *(p_channel) = *(c_channel);                                                                                  \
                    p_channel++;                                                                                                  \
                    c_channel++;                                                                                                  \
                }                                                                                                                 \
            }                                                                                                                     \
            for (uint32_t m_p = 1; m_p < pool_W_f; m_p++)                                                                         \
            {                                                                                                                     \
                if ((O_col - m_p) % pool_stride == 0 && (int)(O_col - m_p) >= 0 && (O_col + pool_W_f - (m_p + 1)) < W_o_full)     \
                {                                                                                                                 \
                    float *p_pixel = p_row + ((O_col - m_p) / pool_stride) * C_ob;                                                \
                    float *p_channel = p_pixel;                                                                                   \
                    float *c_channel = c_pixel;                                                                                   \
                    for (uint32_t jj = 0; jj < C_ob; jj++)                                                                        \
                    {                                                                                                             \
                        *(p_channel) = (*(c_channel) > *(p_channel)) ? *(c_channel) : *(p_channel);                               \
                        p_channel++;                                                                                              \
                        c_channel++;                                                                                              \
                    }                                                                                                             \
                }                                                                                                                 \
            }                                                                                                                     \
        }                                                                                                                         \
        for (uint32_t n_p = 1; n_p < pool_H_f; n_p++)                                                                             \
        {                                                                                                                         \
            if ((O_row - n_p) % pool_stride == 0 && (int)(O_row - n_p) >= 0 && (O_row + pool_H_f - (n_p + 1)) < H_o)              \
            {                                                                                                                     \
                float *p_row = O_pool + ((O_row - n_p) / pool_stride) * pool_col_stride;                                          \
                for (uint32_t m_p = 0; m_p < pool_W_f; m_p++)                                                                     \
                {                                                                                                                 \
                    if ((O_col - m_p) % pool_stride == 0 && (int)(O_col - m_p) >= 0 && (O_col + pool_W_f - (m_p + 1)) < W_o_full) \
                    {                                                                                                             \
                        float *p_pixel = p_row + ((O_col - m_p) / pool_stride) * C_ob;                                            \
                        float *p_channel = p_pixel;                                                                               \
                        float *c_channel = c_pixel;                                                                               \
                        for (uint32_t jj = 0; jj < C_ob; jj++)                                                                    \
                        {                                                                                                         \
                            *(p_channel) = (*(c_channel) > *(p_channel)) ? *(c_channel) : *(p_channel);                           \
                            p_channel++;                                                                                          \
                            c_channel++;                                                                                          \
                        }                                                                                                         \
                    }                                                                                                             \
                }                                                                                                                 \
            }                                                                                                                     \
        }                                                                                                                         \
        c_pixel += C_ob;                                                                                                          \
        O_col++;                                                                                                                  \
    }

#define MAX_END_IP(pool_col_stride, W_last, C_ob, pool_stride, pool_H_f, pool_W_f, O_row, O_col, O_pool, H_o, W_o_full)                       \
    float *c_pixel = c_tile;                                                                                                                  \
    uint32_t O_col_cur = O_col;                                                                                                               \
    for (uint32_t kk = 0; kk < W_last; kk++)                                                                                                  \
    {                                                                                                                                         \
        c_pixel = c_tile + kk * C_ob;                                                                                                         \
        if (O_row % pool_stride == 0 && (O_row + pool_H_f - 1) < H_o)                                                                         \
        {                                                                                                                                     \
            float *p_row = O_pool + ((O_row) / pool_stride) * pool_col_stride;                                                                \
            if (O_col_cur % pool_stride == 0 && (O_col_cur + pool_W_f - 1) < W_o_full)                                                        \
            {                                                                                                                                 \
                float *p_pixel = p_row + ((O_col_cur) / pool_stride) * C_ob;                                                                  \
                float *p_channel = p_pixel;                                                                                                   \
                float *c_channel = c_pixel;                                                                                                   \
                for (uint32_t jj = 0; jj < C_ob; jj++)                                                                                        \
                {                                                                                                                             \
                    *(p_channel) = *(c_channel);                                                                                              \
                                                                                                                                              \
                    p_channel++;                                                                                                              \
                    c_channel++;                                                                                                              \
                }                                                                                                                             \
            }                                                                                                                                 \
            for (uint32_t m_p = 1; m_p < pool_W_f; m_p++)                                                                                     \
            {                                                                                                                                 \
                if ((O_col_cur - m_p) % pool_stride == 0 && (int)(O_col_cur - m_p) >= 0 && (O_col_cur + pool_W_f - (m_p + 1)) < W_o_full)     \
                {                                                                                                                             \
                    float *p_pixel = p_row + ((O_col_cur - m_p) / pool_stride) * C_ob;                                                        \
                    float *p_channel = p_pixel;                                                                                               \
                    float *c_channel = c_pixel;                                                                                               \
                    for (uint32_t jj = 0; jj < C_ob; jj++)                                                                                    \
                    {                                                                                                                         \
                        *(p_channel) = (*(c_channel) > *(p_channel)) ? *(c_channel) : *(p_channel);                                           \
                        p_channel++;                                                                                                          \
                        c_channel++;                                                                                                          \
                    }                                                                                                                         \
                }                                                                                                                             \
            }                                                                                                                                 \
        }                                                                                                                                     \
        for (uint32_t n_p = 1; n_p < pool_H_f; n_p++)                                                                                         \
        {                                                                                                                                     \
            if ((O_row - n_p) % pool_stride == 0 && (int)(O_row - n_p) >= 0 && (O_row + pool_H_f - (n_p + 1)) < H_o)                          \
            {                                                                                                                                 \
                float *p_row = O_pool + ((O_row - n_p) / pool_stride) * pool_col_stride;                                                      \
                for (uint32_t m_p = 0; m_p < pool_W_f; m_p++)                                                                                 \
                {                                                                                                                             \
                    if ((O_col_cur - m_p) % pool_stride == 0 && (int)(O_col_cur - m_p) >= 0 && (O_col_cur + pool_W_f - (m_p + 1)) < W_o_full) \
                    {                                                                                                                         \
                        float *p_pixel = p_row + ((O_col_cur - m_p) / pool_stride) * C_ob;                                                    \
                        float *p_channel = p_pixel;                                                                                           \
                        float *c_channel = c_pixel;                                                                                           \
                        for (uint32_t jj = 0; jj < C_ob; jj++)                                                                                \
                        {                                                                                                                     \
                            *(p_channel) = (*(c_channel) > *(p_channel)) ? *(c_channel) : *(p_channel);                                       \
                            p_channel++;                                                                                                      \
                            c_channel++;                                                                                                      \
                        }                                                                                                                     \
                    }                                                                                                                         \
                }                                                                                                                             \
            }                                                                                                                                 \
        }                                                                                                                                     \
        O_col_cur++;                                                                                                                          \
    }

// DW Convolution
#define MUL_TILE_C(b, W_ob, C_ob)              \
    float *c_pixel = c_tile;                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)     \
    {                                          \
        float *c_channel = c_pixel;            \
        float *b_channel = b;                  \
        for (uint32_t jj = 0; jj < C_ob; jj++) \
        {                                      \
            *(c_channel) *= *(b_channel);      \
            c_channel++;                       \
            b_channel++;                       \
        }                                      \
        c_pixel += C_ob;                       \
    }

#define MUL_END_C(b, W_ob, C_ob)               \
    float *c_pixel = c_tile;                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)     \
    {                                          \
        float *c_channel = c_pixel;            \
        float *b_channel = b;                  \
        for (uint32_t jj = 0; jj < C_ob; jj++) \
        {                                      \
            *(c_channel) *= *(b_channel);      \
            c_channel++;                       \
            b_channel++;                       \
        }                                      \
        c_pixel += C_ob;                       \
    }

#define DW_TILE_C(step, a, b, W_ob, C_ob)                      \
B0 = vld1q_f32(b + 0 * SIMD);                                   \
B1 = vld1q_f32(b + 1 * SIMD);                            \
A4 = vld1q_f32(b + 2 * SIMD);\
A5 = vld1q_f32(b + 3 * SIMD);\
A0 = vld1q_f32(a + (0 * step) + 0 * SIMD);                     \
A1 = vld1q_f32(a + (0 * step) + 1 * SIMD);              \
A2 = vld1q_f32(a + (0 * step) + 2 * SIMD);                     \
A3 = vld1q_f32(a + (0 * step) + 3 * SIMD);              \
\
C00 = vfmaq_f32(C00, A0, B0);                         \
A0 = vld1q_f32(a + (1 * step) + 0 * SIMD);  \
C01 = vfmaq_f32(C01, A1, B1);                         \
A1 = vld1q_f32(a + (1 * step) + 1 * SIMD);  \
C02 = vfmaq_f32(C02, A2, A4);                         \
A2 = vld1q_f32(a + (1 * step) + 2 * SIMD);  \
C03 = vfmaq_f32(C03, A3, A5);                         \
A3 = vld1q_f32(a + (1 * step) + 3 * SIMD);  \
\
C10 = vfmaq_f32(C10, A0, B0);                         \
A0 = vld1q_f32(a + (2 * step) + 0 * SIMD);  \
C11 = vfmaq_f32(C11, A1, B1);                         \
A1 = vld1q_f32(a + (2 * step) + 1 * SIMD);  \
C12 = vfmaq_f32(C12, A2, A4);                         \
A2 = vld1q_f32(a + (2 * step) + 2 * SIMD);  \
C13 = vfmaq_f32(C13, A3, A5);                         \
A3 = vld1q_f32(a + (2 * step) + 3 * SIMD);  \
\
C20 = vfmaq_f32(C20, A0, B0);                         \
A0 = vld1q_f32(a + (3 * step) + 0 * SIMD);  \
C21 = vfmaq_f32(C21, A1, B1);                         \
A1 = vld1q_f32(a + (3 * step) + 1 * SIMD);  \
C22 = vfmaq_f32(C22, A2, A4);                         \
A2 = vld1q_f32(a + (3 * step) + 2 * SIMD);  \
C23 = vfmaq_f32(C23, A3, A5);                         \
A3 = vld1q_f32(a + (3 * step) + 3 * SIMD);  \
\
C30 = vfmaq_f32(C30, A0, B0);                         \
A0 = vld1q_f32(a + (4 * step) + 0 * SIMD);  \
C31 = vfmaq_f32(C31, A1, B1);                         \
A1 = vld1q_f32(a + (4 * step) + 1 * SIMD);  \
C32 = vfmaq_f32(C32, A2, A4);                         \
A2 = vld1q_f32(a + (4 * step) + 2 * SIMD);  \
C33 = vfmaq_f32(C33, A3, A5);                         \
A3 = vld1q_f32(a + (4 * step) + 3 * SIMD);  \
\
C40 = vfmaq_f32(C40, A0, B0);                         \
A0 = vld1q_f32(a + (5 * step) + 0 * SIMD);  \
C41 = vfmaq_f32(C41, A1, B1);                         \
A1 = vld1q_f32(a + (5 * step) + 1 * SIMD);  \
C42 = vfmaq_f32(C42, A2, A4);                         \
A2 = vld1q_f32(a + (5 * step) + 2 * SIMD);  \
C43 = vfmaq_f32(C43, A3, A5);                         \
A3 = vld1q_f32(a + (5 * step) + 3 * SIMD);  \
\
C50 = vfmaq_f32(C50, A0, B0);                         \
C51 = vfmaq_f32(C51, A1, B1);                         \
C52 = vfmaq_f32(C52, A2, A4);                         \
C53 = vfmaq_f32(C53, A3, A5);                         \



#define DW_END_C(step, a, b, W_ob, C_ob)                       \
    {                                                          \
        float *c_pixel = c_tile;                               \
        float *a_pixel = a;                                    \
        for (uint32_t kk = 0; kk < W_ob; kk++)                 \
        {                                                      \
            float *c_channel = c_pixel;                        \
            float *a_channel = a_pixel;                        \
            float *b_channel = b;                              \
            for (uint32_t jj = 0; jj < C_ob; jj++)             \
            {                                                  \
                *(c_channel) += (*(a_channel) * *(b_channel)); \
                c_channel++;                                   \
                b_channel++;                                   \
                a_channel++;                                   \
            }                                                  \
            a_pixel += step;                                   \
            c_pixel += C_ob;                                   \
        }                                                      \
    }

#define DW_TILE_IP(pool_col_stride, W_ob, C_ob, pool_stride, pool_H_f, pool_W_f, F, O_row, O_col, O_pool, H_o, W_o_full)          \
    float *c_pixel = c_tile;                                                                                                      \
    for (uint32_t kk = 0; kk < W_ob; kk++)                                                                                        \
    {                                                                                                                             \
        if (O_row % pool_stride == 0 && (O_row + pool_H_f - 1) < H_o)                                                             \
        {                                                                                                                         \
            float *p_row = O_pool + ((O_row) / pool_stride) * pool_col_stride;                                                    \
            if (O_col % pool_stride == 0 && (O_col + pool_W_f - 1) < W_o_full)                                                    \
            {                                                                                                                     \
                float *p_pixel = p_row + ((O_col) / pool_stride) * C_ob;                                                          \
                float *p_channel = p_pixel;                                                                                       \
                float *c_channel = c_pixel;                                                                                       \
                float *b = F;                                                                                                     \
                float *b_channel = b;                                                                                             \
                for (uint32_t jj = 0; jj < C_ob; jj++)                                                                            \
                {                                                                                                                 \
                    *(p_channel) = *(c_channel) * *(b_channel);                                                                   \
                    p_channel++;                                                                                                  \
                    c_channel++;                                                                                                  \
                    b_channel++;                                                                                                  \
                }                                                                                                                 \
            }                                                                                                                     \
            for (uint32_t m_p = 1; m_p < pool_W_f; m_p++)                                                                         \
            {                                                                                                                     \
                if ((O_col - m_p) % pool_stride == 0 && (int)(O_col - m_p) >= 0 && (O_col + pool_W_f - (m_p + 1)) < W_o_full)     \
                {                                                                                                                 \
                    float *p_pixel = p_row + ((O_col - m_p) / pool_stride) * C_ob;                                                \
                    float *p_channel = p_pixel;                                                                                   \
                    float *c_channel = c_pixel;                                                                                   \
                    float *b = F + m_p * C_ob;                                                                                    \
                    float *b_channel = b;                                                                                         \
                    for (uint32_t jj = 0; jj < C_ob; jj++)                                                                        \
                    {                                                                                                             \
                        *(p_channel) += *(c_channel) * *(b_channel);                                                              \
                        p_channel++;                                                                                              \
                        c_channel++;                                                                                              \
                        b_channel++;                                                                                              \
                    }                                                                                                             \
                }                                                                                                                 \
            }                                                                                                                     \
        }                                                                                                                         \
        for (uint32_t n_p = 1; n_p < pool_H_f; n_p++)                                                                             \
        {                                                                                                                         \
            if ((O_row - n_p) % pool_stride == 0 && (int)(O_row - n_p) >= 0 && (O_row + pool_H_f - (n_p + 1)) < H_o)              \
            {                                                                                                                     \
                float *p_row = O_pool + ((O_row - n_p) / pool_stride) * pool_col_stride;                                          \
                for (uint32_t m_p = 0; m_p < pool_W_f; m_p++)                                                                     \
                {                                                                                                                 \
                    if ((O_col - m_p) % pool_stride == 0 && (int)(O_col - m_p) >= 0 && (O_col + pool_W_f - (m_p + 1)) < W_o_full) \
                    {                                                                                                             \
                        float *p_pixel = p_row + ((O_col - m_p) / pool_stride) * C_ob;                                            \
                        float *p_channel = p_pixel;                                                                               \
                        float *c_channel = c_pixel;                                                                               \
                        float *b = F + n_p * pool_W_f * C_ob + m_p * C_ob;                                                        \
                        float *b_channel = b;                                                                                     \
                        for (uint32_t jj = 0; jj < C_ob; jj++)                                                                    \
                        {                                                                                                         \
                            *(p_channel) += *(c_channel) * *(b_channel);                                                          \
                            p_channel++;                                                                                          \
                            c_channel++;                                                                                          \
                            b_channel++;                                                                                          \
                        }                                                                                                         \
                    }                                                                                                             \
                }                                                                                                                 \
            }                                                                                                                     \
        }                                                                                                                         \
        c_pixel += C_ob;                                                                                                          \
        O_col++;                                                                                                                  \
    }

#define DW_END_IP(pool_col_stride, W_last, C_ob, pool_stride, pool_H_f, pool_W_f, F, O_row, O_col, O_pool, H_o, W_o_full)             \
    float *c_pixel = c_tile;                                                                                                          \
    uint32_t O_col_cur = O_col;                                                                                                       \
    for (uint32_t kk = 0; kk < W_last; kk++)                                                                                          \
    {                                                                                                                                 \
        {                                                                                                                             \
            if (O_row % pool_stride == 0 && (O_row + pool_H_f - 1) < H_o)                                                             \
            {                                                                                                                         \
                float *p_row = O_pool + ((O_row) / pool_stride) * pool_col_stride;                                                    \
                if (O_col % pool_stride == 0 && (O_col + pool_W_f - 1) < W_o_full)                                                    \
                {                                                                                                                     \
                    float *p_pixel = p_row + ((O_col) / pool_stride) * C_ob;                                                          \
                    float *p_channel = p_pixel;                                                                                       \
                    float *c_channel = c_pixel;                                                                                       \
                    float *b = F;                                                                                                     \
                    float *b_channel = b;                                                                                             \
                    for (uint32_t jj = 0; jj < C_ob; jj++)                                                                            \
                    {                                                                                                                 \
                        *(p_channel) = *(c_channel) * *(b_channel);                                                                   \
                        p_channel++;                                                                                                  \
                        c_channel++;                                                                                                  \
                        b_channel++;                                                                                                  \
                    }                                                                                                                 \
                }                                                                                                                     \
                for (uint32_t m_p = 1; m_p < pool_W_f; m_p++)                                                                         \
                {                                                                                                                     \
                    if ((O_col - m_p) % pool_stride == 0 && (int)(O_col - m_p) >= 0 && (O_col + pool_W_f - (m_p + 1)) < W_o_full)     \
                    {                                                                                                                 \
                        float *p_pixel = p_row + ((O_col - m_p) / pool_stride) * C_ob;                                                \
                        float *p_channel = p_pixel;                                                                                   \
                        float *c_channel = c_pixel;                                                                                   \
                        float *b = F + m_p * C_ob;                                                                                    \
                        float *b_channel = b;                                                                                         \
                        for (uint32_t jj = 0; jj < C_ob; jj++)                                                                        \
                        {                                                                                                             \
                            *(p_channel) += *(c_channel) * *(b_channel);                                                              \
                            p_channel++;                                                                                              \
                            c_channel++;                                                                                              \
                            b_channel++;                                                                                              \
                        }                                                                                                             \
                    }                                                                                                                 \
                }                                                                                                                     \
            }                                                                                                                         \
            for (uint32_t n_p = 1; n_p < pool_H_f; n_p++)                                                                             \
            {                                                                                                                         \
                if ((O_row - n_p) % pool_stride == 0 && (int)(O_row - n_p) >= 0 && (O_row + pool_H_f - (n_p + 1)) < H_o)              \
                {                                                                                                                     \
                    float *p_row = O_pool + ((O_row - n_p) / pool_stride) * pool_col_stride;                                          \
                    for (uint32_t m_p = 0; m_p < pool_W_f; m_p++)                                                                     \
                    {                                                                                                                 \
                        if ((O_col - m_p) % pool_stride == 0 && (int)(O_col - m_p) >= 0 && (O_col + pool_W_f - (m_p + 1)) < W_o_full) \
                        {                                                                                                             \
                            float *p_pixel = p_row + ((O_col - m_p) / pool_stride) * C_ob;                                            \
                            float *p_channel = p_pixel;                                                                               \
                            float *c_channel = c_pixel;                                                                               \
                            float *b = F + n_p * pool_W_f * C_ob + m_p * C_ob;                                                        \
                            float *b_channel = b;                                                                                     \
                            for (uint32_t jj = 0; jj < C_ob; jj++)                                                                    \
                            {                                                                                                         \
                                *(p_channel) += *(c_channel) * *(b_channel);                                                          \
                                p_channel++;                                                                                          \
                                c_channel++;                                                                                          \
                                b_channel++;                                                                                          \
                            }                                                                                                         \
                        }                                                                                                             \
                    }                                                                                                                 \
                }                                                                                                                     \
            }                                                                                                                         \
            c_pixel += C_ob;                                                                                                          \
            O_col++;                                                                                                                  \
        }                                                                                                                             \
    }

#define ADD_TILE_C_G(I, W_ob_g, C_ob)          \
    float *i_pixel = I;                        \
    float *c_pixel = c_tile;                   \
    for (uint32_t mm = 0; mm < W_ob_g; mm++)   \
    {                                          \
        float *c_channel = c_pixel;            \
        float *i_channel = i_pixel;            \
        for (uint32_t kk = 0; kk < C_ob; kk++) \
        {                                      \
            *c_channel += *i_channel;          \
            c_channel++;                       \
            i_channel++;                       \
        }                                      \
        c_pixel += C_ob;                       \
        i_pixel += C_ob;                       \
    }

#define ADD_LAST_C_G(I, W_last, C_ob)          \
    float *i_pixel = I;                        \
    float *c_pixel = c_tile;                   \
    for (uint32_t mm = 0; mm < W_last; mm++)   \
    {                                          \
        float *c_channel = c_pixel;            \
        float *i_channel = i_pixel;            \
        for (uint32_t kk = 0; kk < C_ob; kk++) \
        {                                      \
            *c_channel += *i_channel;          \
            c_channel++;                       \
            i_channel++;                       \
        }                                      \
        c_pixel += C_ob;                       \
        i_pixel += C_ob;                       \
    }

#define REDUCE_div_C(O, d, W_ob_g, C_ob)           \
    {                                              \
        float *c_pixel = c_tile;                   \
        float *O_channel = O;                      \
        float *c_channel = c_pixel;                \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)   \
        {                                          \
            float *O_channel = O;                  \
            float *c_channel = c_pixel;            \
            for (uint32_t kk = 0; kk < C_ob; kk++) \
            {                                      \
                *O_channel += *c_channel;          \
                O_channel++;                       \
                c_channel++;                       \
            }                                      \
            c_pixel += C_ob;                       \
        }                                          \
        O_channel = O;                             \
        for (uint32_t kk = 0; kk < C_ob; kk++)     \
        {                                          \
            *O_channel *= d;                       \
            O_channel++;                           \
        }                                          \
    }

#define REDUCE_C(O, W_ob_g, C_ob)                  \
    {                                              \
        float *c_pixel = c_tile;                   \
        float *O_channel = O;                      \
        float *c_channel = c_pixel;                \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)   \
        {                                          \
            float *O_channel = O;                  \
            float *c_channel = c_pixel;            \
            for (uint32_t kk = 0; kk < C_ob; kk++) \
            {                                      \
                *O_channel += *c_channel;          \
                O_channel++;                       \
                c_channel++;                       \
            }                                      \
            c_pixel += C_ob;                       \
        }                                          \
    }

#define REDUCE_C_last(O, W_last, C_ob)             \
    {                                              \
        float *c_pixel = c_tile;                   \
        float *O_channel = O;                      \
        float *c_channel = c_pixel;                \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)   \
        {                                          \
            float *O_channel = O;                  \
            float *c_channel = c_pixel;            \
            for (uint32_t kk = 0; kk < C_ob; kk++) \
            {                                      \
                *O_channel += *c_channel;          \
                O_channel++;                       \
                c_channel++;                       \
            }                                      \
            c_pixel += C_ob;                       \
        }                                          \
    }
