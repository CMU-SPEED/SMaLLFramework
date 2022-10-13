
// #include <immintrin.h>

// #ifdef "zen2_intrinsics.h"
//     #undef
// #include "zen2_intrinsics.h"
#include "scalar.h"
#include <arm_neon.h>
#define SIMD 4
// activations kernels
// pooling kernels

#define Arow(a1, a2) Ar[(a1) * (ldA) + (a2)]
#define Brow(a1, a2) Br[(a1) * (ldB) + (a2)]
#define Crow(a1, a2) C[(a1) * (ldC) + (a2)]
#define Ctrow(a1, a2) Ctmp[(a1) * (ldCt) + (a2)]
#define Atrow(a1, a2) Atmp[(a1) * (ldAt) + (a2)]
#define Atrow_no_stride(a1, a2) Atmp[(a2) * (ldAt) + (a1)]

#define Btrow(a1, a2) Btmp[(a1) * (ldBt) + (a2)]

#define Ctref(a1, a2) Ctmp[(a2) * (ldCt) + (a1)]
#define Atref(a1, a2) Atmp[(a2) * (ldAt) + (a1)]

void print(float32x4_t vec)
{
    float a[4];
    vst1q_f32((float32_t *)a, vec);
    for (int i = 0; i < 4; ++i)
        printf("%.4f \t", a[i]);
    printf("\n");
}

// microkernel

#define load_a(a_ptr)                              \
    {                                              \
        __asm__ volatile(                          \
            "ldr x0,%[aaddr] \n\t"                 \
            "ldr q0, [x0, #0]  \n\t"               \
            "ldr q1, [x0, #64]  \n\t"              \
            "ldr q2, [x0, #128]  \n\t"             \
            "ldr q3, [x0, #192]  \n\t"             \
            "ldr q4, [x0, #256]  \n\t"             \
            "ldr q5, [x0, #320]  \n\t" :\
:[aaddr] "m"(a_ptr)                                \
            : "x0",                                \
              "q0", "q1", "q2", "q3", "q4", "q5"); \
    }







// convolution kernels

#define conv_kernel_enrique_6x16(                                                                       \
    first,                                                                                              \
    W_ob, C_ob, C_ib, step,                                                                             \
    H_f, W_f,                                                                                           \
    input_col_stride,                                                                                   \
    I,                                                                                                  \
    F,                                                                                                  \
    O)                                                                                                  \
    {                                                                                                   \
        float *Ar = I;                                                                                  \
        int i, j, k, baseB = 0, ldCt = C_ob, Amr, Bnr, ldA = step, ldAt = C_ob, ldC = C_ob, ldB = C_ob; \
        float32x4_t B0, B1,                                                                             \
            A0, C00, C01, C02, C03,                                                                     \
            A1, C10, C11, C12, C13,                                                                     \
            A2, C20, C21, C22, C23,                                                                     \
            A3, C30, C31, C32, C33,                                                                     \
            A4, C40, C41, C42, C43,                                                                     \
            A5, C50, C51, C52, C53;                                                                     \
        float zero = 0.0, one = 1.0, *Atmp, *C;                                                         \
        C = O;                                                                                          \
        if (first == 0)                                                                                 \
        {                                                                                               \
            C00 = vmovq_n_f32(0);                                                                       \
            C01 = vmovq_n_f32(0);                                                                       \
            C02 = vmovq_n_f32(0);                                                                       \
            C03 = vmovq_n_f32(0);                                                                       \
            C10 = vmovq_n_f32(0);                                                                       \
            C11 = vmovq_n_f32(0);                                                                       \
            C12 = vmovq_n_f32(0);                                                                       \
            C13 = vmovq_n_f32(0);                                                                       \
            C20 = vmovq_n_f32(0);                                                                       \
            C21 = vmovq_n_f32(0);                                                                       \
            C22 = vmovq_n_f32(0);                                                                       \
            C23 = vmovq_n_f32(0);                                                                       \
            C30 = vmovq_n_f32(0);                                                                       \
            C31 = vmovq_n_f32(0);                                                                       \
            C32 = vmovq_n_f32(0);                                                                       \
            C33 = vmovq_n_f32(0);                                                                       \
            C40 = vmovq_n_f32(0);                                                                       \
            C41 = vmovq_n_f32(0);                                                                       \
            C42 = vmovq_n_f32(0);                                                                       \
            C43 = vmovq_n_f32(0);                                                                       \
            C50 = vmovq_n_f32(0);                                                                       \
            C51 = vmovq_n_f32(0);                                                                       \
            C52 = vmovq_n_f32(0);                                                                       \
            C53 = vmovq_n_f32(0);                                                                       \
        }                                                                                               \
        else                                                                                            \
        {                                                                                               \
            C00 = vld1q_f32(&Crow(0, 0));                                                               \
            C01 = vld1q_f32(&Crow(0, 4));                                                               \
            C02 = vld1q_f32(&Crow(0, 8));                                                               \
            C03 = vld1q_f32(&Crow(0, 12));                                                              \
            C10 = vld1q_f32(&Crow(1, 0));                                                               \
            C11 = vld1q_f32(&Crow(1, 4));                                                               \
            C12 = vld1q_f32(&Crow(1, 8));                                                               \
            C13 = vld1q_f32(&Crow(1, 12));                                                              \
            C20 = vld1q_f32(&Crow(2, 0));                                                               \
            C21 = vld1q_f32(&Crow(2, 4));                                                               \
            C22 = vld1q_f32(&Crow(2, 8));                                                               \
            C23 = vld1q_f32(&Crow(2, 12));                                                              \
            C30 = vld1q_f32(&Crow(3, 0));                                                               \
            C31 = vld1q_f32(&Crow(3, 4));                                                               \
            C32 = vld1q_f32(&Crow(3, 8));                                                               \
            C33 = vld1q_f32(&Crow(3, 12));                                                              \
            C40 = vld1q_f32(&Crow(4, 0));                                                               \
            C41 = vld1q_f32(&Crow(4, 4));                                                               \
            C42 = vld1q_f32(&Crow(4, 8));                                                               \
            C43 = vld1q_f32(&Crow(4, 12));                                                              \
            C50 = vld1q_f32(&Crow(5, 0));                                                               \
            C51 = vld1q_f32(&Crow(5, 4));                                                               \
            C52 = vld1q_f32(&Crow(5, 8));                                                               \
            C53 = vld1q_f32(&Crow(5, 12));                                                              \
        }                                                                                               \
        int updates = 0;                                                                                \
        float *b = F;                                                                                   \
        for (uint32_t n = 0; n < H_f; n++)                                                              \
        {                                                                                               \
            int filter_offset_h = n * W_f * C_ib * C_ob;                                                \
            int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;   \
            for (uint32_t m = 0; m < W_f; m++)                                                          \
            {                                                                                           \
                int filter_offset_w = m * C_ib * C_ob + filter_offset_h;                                \
                int input_stencil_w = m * C_ib + input_stencil_h;                                       \
                float *Bptr = b;                                                                        \
                float *a = I + input_stencil_w;                                                         \
                Atmp = a;                                                                               \
                for (uint32_t ii = 0; ii < C_ib; ii += SIMD)                                            \
                {                                                                                       \
                    COMPUTE_KERNEL_6x16_NOPACK_UNROLL_4(C_ob, ii);                                      \
                }                                                                                       \
                b += C_ib * C_ob;                                                                       \
            }                                                                                           \
        }                                                                                               \
                                                                                                        \
        vst1q_f32(&Crow(0, 0), C00);                                                                    \
        vst1q_f32(&Crow(0, 4), C01);                                                                    \
        vst1q_f32(&Crow(0, 8), C02);                                                                    \
        vst1q_f32(&Crow(0, 12), C03);                                                                   \
        vst1q_f32(&Crow(1, 0), C10);                                                                    \
        vst1q_f32(&Crow(1, 4), C11);                                                                    \
        vst1q_f32(&Crow(1, 8), C12);                                                                    \
        vst1q_f32(&Crow(1, 12), C13);                                                                   \
        vst1q_f32(&Crow(2, 0), C20);                                                                    \
        vst1q_f32(&Crow(2, 4), C21);                                                                    \
        vst1q_f32(&Crow(2, 8), C22);                                                                    \
        vst1q_f32(&Crow(2, 12), C23);                                                                   \
        vst1q_f32(&Crow(3, 0), C30);                                                                    \
        vst1q_f32(&Crow(3, 4), C31);                                                                    \
        vst1q_f32(&Crow(3, 8), C32);                                                                    \
        vst1q_f32(&Crow(3, 12), C33);                                                                   \
        vst1q_f32(&Crow(4, 0), C40);                                                                    \
        vst1q_f32(&Crow(4, 4), C41);                                                                    \
        vst1q_f32(&Crow(4, 8), C42);                                                                    \
        vst1q_f32(&Crow(4, 12), C43);                                                                   \
        vst1q_f32(&Crow(5, 0), C50);                                                                    \
        vst1q_f32(&Crow(5, 4), C51);                                                                    \
        vst1q_f32(&Crow(5, 8), C52);                                                                    \
        vst1q_f32(&Crow(5, 12), C53);                                                                   \
    }

#define conv_kernel_enrique(                                                                          \
    first,                                                                                            \
    W_ob, C_ob, C_ib, step,                                                                           \
    H_f, W_f,                                                                                         \
    input_col_stride,                                                                                 \
    I,                                                                                                \
    F,                                                                                                \
    O)                                                                                                \
    {                                                                                                 \
        float *Ar = I;                                                                                \
        constexpr int NR = C_ob;                                                                      \
        constexpr int MR = W_ob;                                                                      \
        int i, j, k, ldCt = NR, Amr, Bnr, ldA = NR, ldAt = W_ob, ldC = C_ob, ldB = C_ob;              \
        float32x4_t C00, C01, C02,                                                                    \
            C10, C11, C12,                                                                            \
            C20, C21, C22,                                                                            \
            C30, C31, C32,                                                                            \
            C40, C41, C42,                                                                            \
            C50, C51, C52,                                                                            \
            C60, C61, C62,                                                                            \
            A0, A1, A2, A3, A4, A5, A6, B0, B1, B2;                                                   \
        float zero = 0.0, one = 1.0, Ctmp[MR * NR], *Atmp, *C;                                        \
        C = O;                                                                                        \
        if (first == 0)                                                                               \
        {                                                                                             \
            C00 = vmovq_n_f32(0);                                                                     \
            C01 = vmovq_n_f32(0);                                                                     \
            C02 = vmovq_n_f32(0);                                                                     \
            C10 = vmovq_n_f32(0);                                                                     \
            C11 = vmovq_n_f32(0);                                                                     \
            C12 = vmovq_n_f32(0);                                                                     \
            C20 = vmovq_n_f32(0);                                                                     \
            C21 = vmovq_n_f32(0);                                                                     \
            C22 = vmovq_n_f32(0);                                                                     \
            C30 = vmovq_n_f32(0);                                                                     \
            C31 = vmovq_n_f32(0);                                                                     \
            C32 = vmovq_n_f32(0);                                                                     \
            C40 = vmovq_n_f32(0);                                                                     \
            C41 = vmovq_n_f32(0);                                                                     \
            C42 = vmovq_n_f32(0);                                                                     \
            C50 = vmovq_n_f32(0);                                                                     \
            C51 = vmovq_n_f32(0);                                                                     \
            C52 = vmovq_n_f32(0);                                                                     \
            C60 = vmovq_n_f32(0);                                                                     \
            C61 = vmovq_n_f32(0);                                                                     \
            C62 = vmovq_n_f32(0);                                                                     \
        }                                                                                             \
        else                                                                                          \
        {                                                                                             \
            C00 = vld1q_f32(&Crow(0, 0));                                                             \
            C01 = vld1q_f32(&Crow(0, 4));                                                             \
            C02 = vld1q_f32(&Crow(0, 8));                                                             \
            C10 = vld1q_f32(&Crow(1, 0));                                                             \
            C11 = vld1q_f32(&Crow(1, 4));                                                             \
            C12 = vld1q_f32(&Crow(1, 8));                                                             \
            C20 = vld1q_f32(&Crow(2, 0));                                                             \
            C21 = vld1q_f32(&Crow(2, 4));                                                             \
            C22 = vld1q_f32(&Crow(2, 8));                                                             \
            C30 = vld1q_f32(&Crow(3, 0));                                                             \
            C31 = vld1q_f32(&Crow(3, 4));                                                             \
            C32 = vld1q_f32(&Crow(3, 8));                                                             \
            C40 = vld1q_f32(&Crow(4, 0));                                                             \
            C41 = vld1q_f32(&Crow(4, 4));                                                             \
            C42 = vld1q_f32(&Crow(4, 8));                                                             \
            C50 = vld1q_f32(&Crow(5, 0));                                                             \
            C51 = vld1q_f32(&Crow(5, 4));                                                             \
            C52 = vld1q_f32(&Crow(5, 8));                                                             \
            C60 = vld1q_f32(&Crow(6, 0));                                                             \
            C61 = vld1q_f32(&Crow(6, 4));                                                             \
            C62 = vld1q_f32(&Crow(6, 8));                                                             \
        }                                                                                             \
        int updates = 0;                                                                              \
        for (uint32_t n = 0; n < H_f; n++)                                                            \
        {                                                                                             \
            int filter_offset_h = n * W_f * C_ib * C_ob;                                              \
            int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/; \
            for (uint32_t m = 0; m < W_f; m++)                                                        \
            {                                                                                         \
                int filter_offset_w = m * C_ib * C_ob + filter_offset_h;                              \
                int input_stencil_w = m * C_ib + input_stencil_h;                                     \
                float *b = F + filter_offset_w;                                                       \
                float *Bptr = b;                                                                      \
                float *a = I + input_stencil_w;                                                       \
                Atmp = a;                                                                             \
                for (uint32_t ii = 0; ii < C_ib; ii += SIMD)                                          \
                {                                                                                     \
                    int p_cur = ii;                                                                   \
                    baseB = ii * C_ob;                                                                \
                    A0 = vld1q_f32(&Atrow(0, ii));                                                    \
                    A1 = vld1q_f32(&Atrow(1, ii));                                                    \
                    A2 = vld1q_f32(&Atrow(2, ii));                                                    \
                    A3 = vld1q_f32(&Atrow(3, ii));                                                    \
                    A4 = vld1q_f32(&Atrow(4, ii));                                                    \
                    A5 = vld1q_f32(&Atrow(5, ii));                                                    \
                    A6 = vld1q_f32(&Atrow(6, ii));                                                    \
                                                                                                      \
                    B0 = vld1q_f32(Bptr + baseB);                                                     \
                    B1 = vld1q_f32(&Bptr[baseB + 4]);                                                 \
                    B2 = vld1q_f32(&Bptr[baseB + 8]);                                                 \
                                                                                                      \
                    C00 = vfmaq_laneq_f32(C00, B0, A0, 0);                                            \
                    C01 = vfmaq_laneq_f32(C01, B1, A0, 0);                                            \
                    C02 = vfmaq_laneq_f32(C02, B2, A0, 0);                                            \
                    C10 = vfmaq_laneq_f32(C10, B0, A1, 0);                                            \
                    C11 = vfmaq_laneq_f32(C11, B1, A1, 0);                                            \
                    C12 = vfmaq_laneq_f32(C12, B2, A1, 0);                                            \
                    C20 = vfmaq_laneq_f32(C20, B0, A2, 0);                                            \
                    C21 = vfmaq_laneq_f32(C21, B1, A2, 0);                                            \
                    C22 = vfmaq_laneq_f32(C22, B2, A2, 0);                                            \
                    C30 = vfmaq_laneq_f32(C30, B0, A3, 0);                                            \
                    C31 = vfmaq_laneq_f32(C31, B1, A3, 0);                                            \
                    C32 = vfmaq_laneq_f32(C32, B2, A3, 0);                                            \
                    C40 = vfmaq_laneq_f32(C40, B0, A4, 0);                                            \
                    C41 = vfmaq_laneq_f32(C41, B1, A4, 0);                                            \
                    C42 = vfmaq_laneq_f32(C42, B2, A4, 0);                                            \
                    C50 = vfmaq_laneq_f32(C50, B0, A5, 0);                                            \
                    C51 = vfmaq_laneq_f32(C51, B1, A5, 0);                                            \
                    C52 = vfmaq_laneq_f32(C52, B2, A5, 0);                                            \
                    C60 = vfmaq_laneq_f32(C60, B0, A6, 0);                                            \
                    C61 = vfmaq_laneq_f32(C61, B1, A6, 0);                                            \
                    C62 = vfmaq_laneq_f32(C62, B2, A6, 0);                                            \
                                                                                                      \
                    B0 = vld1q_f32(&Bptr[baseB + ldB]);                                               \
                    B1 = vld1q_f32(&Bptr[baseB + ldB + 4]);                                           \
                    B2 = vld1q_f32(&Bptr[baseB + ldB + 8]);                                           \
                                                                                                      \
                    C00 = vfmaq_laneq_f32(C00, B0, A0, 1);                                            \
                    C01 = vfmaq_laneq_f32(C01, B1, A0, 1);                                            \
                    C02 = vfmaq_laneq_f32(C02, B2, A0, 1);                                            \
                    C10 = vfmaq_laneq_f32(C10, B0, A1, 1);                                            \
                    C11 = vfmaq_laneq_f32(C11, B1, A1, 1);                                            \
                    C12 = vfmaq_laneq_f32(C12, B2, A1, 1);                                            \
                    C20 = vfmaq_laneq_f32(C20, B0, A2, 1);                                            \
                    C21 = vfmaq_laneq_f32(C21, B1, A2, 1);                                            \
                    C22 = vfmaq_laneq_f32(C22, B2, A2, 1);                                            \
                    C30 = vfmaq_laneq_f32(C30, B0, A3, 1);                                            \
                    C31 = vfmaq_laneq_f32(C31, B1, A3, 1);                                            \
                    C32 = vfmaq_laneq_f32(C32, B2, A3, 1);                                            \
                    C40 = vfmaq_laneq_f32(C40, B0, A4, 1);                                            \
                    C41 = vfmaq_laneq_f32(C41, B1, A4, 1);                                            \
                    C42 = vfmaq_laneq_f32(C42, B2, A4, 1);                                            \
                    C50 = vfmaq_laneq_f32(C50, B0, A5, 1);                                            \
                    C51 = vfmaq_laneq_f32(C51, B1, A5, 1);                                            \
                    C52 = vfmaq_laneq_f32(C52, B2, A5, 1);                                            \
                    C60 = vfmaq_laneq_f32(C60, B0, A6, 1);                                            \
                    C61 = vfmaq_laneq_f32(C61, B1, A6, 1);                                            \
                    C62 = vfmaq_laneq_f32(C62, B2, A6, 1);                                            \
                                                                                                      \
                    B0 = vld1q_f32(&Bptr[baseB + 2 * ldB]);                                           \
                    B1 = vld1q_f32(&Bptr[baseB + 2 * ldB + 4]);                                       \
                    B2 = vld1q_f32(&Bptr[baseB + 2 * ldB + 8]);                                       \
                                                                                                      \
                    C00 = vfmaq_laneq_f32(C00, B0, A0, 2);                                            \
                    C01 = vfmaq_laneq_f32(C01, B1, A0, 2);                                            \
                    C02 = vfmaq_laneq_f32(C02, B2, A0, 2);                                            \
                    C10 = vfmaq_laneq_f32(C10, B0, A1, 2);                                            \
                    C11 = vfmaq_laneq_f32(C11, B1, A1, 2);                                            \
                    C12 = vfmaq_laneq_f32(C12, B2, A1, 2);                                            \
                    C20 = vfmaq_laneq_f32(C20, B0, A2, 2);                                            \
                    C21 = vfmaq_laneq_f32(C21, B1, A2, 2);                                            \
                    C22 = vfmaq_laneq_f32(C22, B2, A2, 2);                                            \
                    C30 = vfmaq_laneq_f32(C30, B0, A3, 2);                                            \
                    C31 = vfmaq_laneq_f32(C31, B1, A3, 2);                                            \
                    C32 = vfmaq_laneq_f32(C32, B2, A3, 2);                                            \
                    C40 = vfmaq_laneq_f32(C40, B0, A4, 2);                                            \
                    C41 = vfmaq_laneq_f32(C41, B1, A4, 2);                                            \
                    C42 = vfmaq_laneq_f32(C42, B2, A4, 2);                                            \
                    C50 = vfmaq_laneq_f32(C50, B0, A5, 2);                                            \
                    C51 = vfmaq_laneq_f32(C51, B1, A5, 2);                                            \
                    C52 = vfmaq_laneq_f32(C52, B2, A5, 2);                                            \
                    C60 = vfmaq_laneq_f32(C60, B0, A6, 2);                                            \
                    C61 = vfmaq_laneq_f32(C61, B1, A6, 2);                                            \
                    C62 = vfmaq_laneq_f32(C62, B2, A6, 2);                                            \
                                                                                                      \
                    B0 = vld1q_f32(&Bptr[baseB + 3 * ldB]);                                           \
                    B1 = vld1q_f32(&Bptr[baseB + 3 * ldB + 4]);                                       \
                    B2 = vld1q_f32(&Bptr[baseB + 3 * ldB + 8]);                                       \
                                                                                                      \
                    C00 = vfmaq_laneq_f32(C00, B0, A0, 3);                                            \
                    C01 = vfmaq_laneq_f32(C01, B1, A0, 3);                                            \
                    C02 = vfmaq_laneq_f32(C02, B2, A0, 3);                                            \
                    C10 = vfmaq_laneq_f32(C10, B0, A1, 3);                                            \
                    C11 = vfmaq_laneq_f32(C11, B1, A1, 3);                                            \
                    C12 = vfmaq_laneq_f32(C12, B2, A1, 3);                                            \
                    C20 = vfmaq_laneq_f32(C20, B0, A2, 3);                                            \
                    C21 = vfmaq_laneq_f32(C21, B1, A2, 3);                                            \
                    C22 = vfmaq_laneq_f32(C22, B2, A2, 3);                                            \
                    C30 = vfmaq_laneq_f32(C30, B0, A3, 3);                                            \
                    C31 = vfmaq_laneq_f32(C31, B1, A3, 3);                                            \
                    C32 = vfmaq_laneq_f32(C32, B2, A3, 3);                                            \
                    C40 = vfmaq_laneq_f32(C40, B0, A4, 3);                                            \
                    C41 = vfmaq_laneq_f32(C41, B1, A4, 3);                                            \
                    C42 = vfmaq_laneq_f32(C42, B2, A4, 3);                                            \
                    C50 = vfmaq_laneq_f32(C50, B0, A5, 3);                                            \
                    C51 = vfmaq_laneq_f32(C51, B1, A5, 3);                                            \
                    C52 = vfmaq_laneq_f32(C52, B2, A5, 3);                                            \
                    C60 = vfmaq_laneq_f32(C60, B0, A6, 3);                                            \
                    C61 = vfmaq_laneq_f32(C61, B1, A6, 3);                                            \
                    C62 = vfmaq_laneq_f32(C62, B2, A6, 3);                                            \
                }                                                                                     \
            }                                                                                         \
        }                                                                                             \
                                                                                                      \
        vst1q_f32(&Crow(0, 0), C00);                                                                  \
        vst1q_f32(&Crow(0, 4), C01);                                                                  \
        vst1q_f32(&Crow(0, 8), C02);                                                                  \
        vst1q_f32(&Crow(1, 0), C10);                                                                  \
        vst1q_f32(&Crow(1, 4), C11);                                                                  \
        vst1q_f32(&Crow(1, 8), C12);                                                                  \
        vst1q_f32(&Crow(2, 0), C20);                                                                  \
        vst1q_f32(&Crow(2, 4), C21);                                                                  \
        vst1q_f32(&Crow(2, 8), C22);                                                                  \
        vst1q_f32(&Crow(3, 0), C30);                                                                  \
        vst1q_f32(&Crow(3, 4), C31);                                                                  \
        vst1q_f32(&Crow(3, 8), C32);                                                                  \
        vst1q_f32(&Crow(4, 0), C40);                                                                  \
        vst1q_f32(&Crow(4, 4), C41);                                                                  \
        vst1q_f32(&Crow(4, 8), C42);                                                                  \
        vst1q_f32(&Crow(5, 0), C50);                                                                  \
        vst1q_f32(&Crow(5, 4), C51);                                                                  \
        vst1q_f32(&Crow(5, 8), C52);                                                                  \
        vst1q_f32(&Crow(6, 0), C60);                                                                  \
        vst1q_f32(&Crow(6, 4), C61);                                                                  \
        vst1q_f32(&Crow(6, 8), C62);                                                                  \
    }

#define conv_kernel_enrique_interleaved(                                                              \
    first,                                                                                            \
    W_ob, C_ob, C_ib, step,                                                                           \
    H_f, W_f,                                                                                         \
    input_col_stride,                                                                                 \
    I,                                                                                                \
    F,                                                                                                \
    O)                                                                                                \
    {                                                                                                 \
        float *Ar = I;                                                                                \
        constexpr int NR = C_ob;                                                                      \
        constexpr int MR = W_ob;                                                                      \
        int i, j, k, baseB = 0, ldCt = NR, Amr, Bnr, ldA = NR, ldAt = C_ib, ldC = C_ob, ldB = C_ob;   \
        float32x4_t C00, C01, C02,                                                                    \
            C10, C11, C12,                                                                            \
            C20, C21, C22,                                                                            \
            C30, C31, C32,                                                                            \
            C40, C41, C42,                                                                            \
            C50, C51, C52,                                                                            \
            C60, C61, C62,                                                                            \
            A0, A1, A2, A3, A4, A5, A6, B0, B1, B2;                                                   \
        float zero = 0.0, one = 1.0, Ctmp[MR * NR], *Atmp, *C;                                        \
        C = O;                                                                                        \
        if (first == 0)                                                                               \
        {                                                                                             \
            C00 = vmovq_n_f32(0);                                                                     \
            C01 = vmovq_n_f32(0);                                                                     \
            C02 = vmovq_n_f32(0);                                                                     \
            C10 = vmovq_n_f32(0);                                                                     \
            C11 = vmovq_n_f32(0);                                                                     \
            C12 = vmovq_n_f32(0);                                                                     \
            C20 = vmovq_n_f32(0);                                                                     \
            C21 = vmovq_n_f32(0);                                                                     \
            C22 = vmovq_n_f32(0);                                                                     \
            C30 = vmovq_n_f32(0);                                                                     \
            C31 = vmovq_n_f32(0);                                                                     \
            C32 = vmovq_n_f32(0);                                                                     \
            C40 = vmovq_n_f32(0);                                                                     \
            C41 = vmovq_n_f32(0);                                                                     \
            C42 = vmovq_n_f32(0);                                                                     \
            C50 = vmovq_n_f32(0);                                                                     \
            C51 = vmovq_n_f32(0);                                                                     \
            C52 = vmovq_n_f32(0);                                                                     \
            C60 = vmovq_n_f32(0);                                                                     \
            C61 = vmovq_n_f32(0);                                                                     \
            C62 = vmovq_n_f32(0);                                                                     \
        }                                                                                             \
        else                                                                                          \
        {                                                                                             \
            C00 = vld1q_f32(&Crow(0, 0));                                                             \
            C01 = vld1q_f32(&Crow(0, 4));                                                             \
            C02 = vld1q_f32(&Crow(0, 8));                                                             \
            C10 = vld1q_f32(&Crow(1, 0));                                                             \
            C11 = vld1q_f32(&Crow(1, 4));                                                             \
            C12 = vld1q_f32(&Crow(1, 8));                                                             \
            C20 = vld1q_f32(&Crow(2, 0));                                                             \
            C21 = vld1q_f32(&Crow(2, 4));                                                             \
            C22 = vld1q_f32(&Crow(2, 8));                                                             \
            C30 = vld1q_f32(&Crow(3, 0));                                                             \
            C31 = vld1q_f32(&Crow(3, 4));                                                             \
            C32 = vld1q_f32(&Crow(3, 8));                                                             \
            C40 = vld1q_f32(&Crow(4, 0));                                                             \
            C41 = vld1q_f32(&Crow(4, 4));                                                             \
            C42 = vld1q_f32(&Crow(4, 8));                                                             \
            C50 = vld1q_f32(&Crow(5, 0));                                                             \
            C51 = vld1q_f32(&Crow(5, 4));                                                             \
            C52 = vld1q_f32(&Crow(5, 8));                                                             \
            C60 = vld1q_f32(&Crow(6, 0));                                                             \
            C61 = vld1q_f32(&Crow(6, 4));                                                             \
            C62 = vld1q_f32(&Crow(6, 8));                                                             \
        }                                                                                             \
        int updates = 0;                                                                              \
        for (uint32_t n = 0; n < H_f; n++)                                                            \
        {                                                                                             \
            int filter_offset_h = n * W_f * C_ib * C_ob;                                              \
            int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/; \
            for (uint32_t m = 0; m < W_f; m++)                                                        \
            {                                                                                         \
                int filter_offset_w = m * C_ib * C_ob + filter_offset_h;                              \
                int input_stencil_w = m * C_ib + input_stencil_h;                                     \
                float *b = F + filter_offset_w;                                                       \
                float *Bptr = b;                                                                      \
                float *a = I + input_stencil_w;                                                       \
                Atmp = a;                                                                             \
                for (uint32_t ii = 0; ii < C_ib; ii += SIMD)                                          \
                {                                                                                     \
                    int p_cur = ii;                                                                   \
                    baseB = ii * C_ob;                                                                \
                    A0 = vld1q_dup_f32(&Atrow(0, ii));                                                \
                    A1 = vld1q_dup_f32(&Atrow(1, ii));                                                \
                    A2 = vld1q_dup_f32(&Atrow(2, ii));                                                \
                    A3 = vld1q_dup_f32(&Atrow(3, ii));                                                \
                    A4 = vld1q_dup_f32(&Atrow(4, ii));                                                \
                    A5 = vld1q_dup_f32(&Atrow(5, ii));                                                \
                    A6 = vld1q_dup_f32(&Atrow(6, ii));                                                \
                                                                                                      \
                    B0 = vld1q_f32(Bptr + baseB);                                                     \
                    B1 = vld1q_f32(&Bptr[baseB + 4]);                                                 \
                    B2 = vld1q_f32(&Bptr[baseB + 8]);                                                 \
                                                                                                      \
                    C00 = vfmaq_f32(C00, B0, A0);                                                     \
                    C01 = vfmaq_f32(C01, B1, A0);                                                     \
                    C02 = vfmaq_f32(C02, B2, A0);                                                     \
                    A0 = vld1q_dup_f32(&Atrow(0, ii + 1));                                            \
                    C10 = vfmaq_f32(C10, B0, A1);                                                     \
                    C11 = vfmaq_f32(C11, B1, A1);                                                     \
                    C12 = vfmaq_f32(C12, B2, A1);                                                     \
                    A1 = vld1q_dup_f32(&Atrow(1, ii + 1));                                            \
                    C20 = vfmaq_f32(C20, B0, A2);                                                     \
                    C21 = vfmaq_f32(C21, B1, A2);                                                     \
                    C22 = vfmaq_f32(C22, B2, A2);                                                     \
                    A2 = vld1q_dup_f32(&Atrow(2, ii + 1));                                            \
                    C30 = vfmaq_f32(C30, B0, A3);                                                     \
                    C31 = vfmaq_f32(C31, B1, A3);                                                     \
                    C32 = vfmaq_f32(C32, B2, A3);                                                     \
                    A3 = vld1q_dup_f32(&Atrow(3, ii + 1));                                            \
                    C40 = vfmaq_f32(C40, B0, A4);                                                     \
                    C41 = vfmaq_f32(C41, B1, A4);                                                     \
                    C42 = vfmaq_f32(C42, B2, A4);                                                     \
                    A4 = vld1q_dup_f32(&Atrow(4, ii + 1));                                            \
                    C50 = vfmaq_f32(C50, B0, A5);                                                     \
                    C51 = vfmaq_f32(C51, B1, A5);                                                     \
                    C52 = vfmaq_f32(C52, B2, A5);                                                     \
                    A5 = vld1q_dup_f32(&Atrow(5, ii + 1));                                            \
                    C60 = vfmaq_f32(C60, B0, A6);                                                     \
                    C61 = vfmaq_f32(C61, B1, A6);                                                     \
                    C62 = vfmaq_f32(C62, B2, A6);                                                     \
                    A6 = vld1q_dup_f32(&Atrow(6, ii + 1));                                            \
                                                                                                      \
                    B0 = vld1q_f32(&Bptr[baseB + ldB]);                                               \
                    B1 = vld1q_f32(&Bptr[baseB + ldB + 4]);                                           \
                    B2 = vld1q_f32(&Bptr[baseB + ldB + 8]);                                           \
                                                                                                      \
                    C00 = vfmaq_f32(C00, B0, A0);                                                     \
                    C01 = vfmaq_f32(C01, B1, A0);                                                     \
                    C02 = vfmaq_f32(C02, B2, A0);                                                     \
                    A0 = vld1q_dup_f32(&Atrow(0, ii + 2));                                            \
                    C10 = vfmaq_f32(C10, B0, A1);                                                     \
                    C11 = vfmaq_f32(C11, B1, A1);                                                     \
                    C12 = vfmaq_f32(C12, B2, A1);                                                     \
                    A1 = vld1q_dup_f32(&Atrow(1, ii + 2));                                            \
                    C20 = vfmaq_f32(C20, B0, A2);                                                     \
                    C21 = vfmaq_f32(C21, B1, A2);                                                     \
                    C22 = vfmaq_f32(C22, B2, A2);                                                     \
                    A2 = vld1q_dup_f32(&Atrow(2, ii + 2));                                            \
                    C30 = vfmaq_f32(C30, B0, A3);                                                     \
                    C31 = vfmaq_f32(C31, B1, A3);                                                     \
                    C32 = vfmaq_f32(C32, B2, A3);                                                     \
                    A3 = vld1q_dup_f32(&Atrow(3, ii + 2));                                            \
                    C40 = vfmaq_f32(C40, B0, A4);                                                     \
                    C41 = vfmaq_f32(C41, B1, A4);                                                     \
                    C42 = vfmaq_f32(C42, B2, A4);                                                     \
                    A4 = vld1q_dup_f32(&Atrow(4, ii + 2));                                            \
                    C50 = vfmaq_f32(C50, B0, A5);                                                     \
                    C51 = vfmaq_f32(C51, B1, A5);                                                     \
                    C52 = vfmaq_f32(C52, B2, A5);                                                     \
                    A5 = vld1q_dup_f32(&Atrow(5, ii + 2));                                            \
                    C60 = vfmaq_f32(C60, B0, A6);                                                     \
                    C61 = vfmaq_f32(C61, B1, A6);                                                     \
                    C62 = vfmaq_f32(C62, B2, A6);                                                     \
                    A6 = vld1q_dup_f32(&Atrow(6, ii + 2));                                            \
                                                                                                      \
                    B0 = vld1q_f32(&Bptr[baseB + 2 * ldB]);                                           \
                    B1 = vld1q_f32(&Bptr[baseB + 2 * ldB + 4]);                                       \
                    B2 = vld1q_f32(&Bptr[baseB + 2 * ldB + 8]);                                       \
                                                                                                      \
                    C00 = vfmaq_f32(C00, B0, A0);                                                     \
                    C01 = vfmaq_f32(C01, B1, A0);                                                     \
                    C02 = vfmaq_f32(C02, B2, A0);                                                     \
                    A0 = vld1q_dup_f32(&Atrow(0, ii + 3));                                            \
                    C10 = vfmaq_f32(C10, B0, A1);                                                     \
                    C11 = vfmaq_f32(C11, B1, A1);                                                     \
                    C12 = vfmaq_f32(C12, B2, A1);                                                     \
                    A1 = vld1q_dup_f32(&Atrow(1, ii + 3));                                            \
                    C20 = vfmaq_f32(C20, B0, A2);                                                     \
                    C21 = vfmaq_f32(C21, B1, A2);                                                     \
                    C22 = vfmaq_f32(C22, B2, A2);                                                     \
                    A2 = vld1q_dup_f32(&Atrow(2, ii + 3));                                            \
                    C30 = vfmaq_f32(C30, B0, A3);                                                     \
                    C31 = vfmaq_f32(C31, B1, A3);                                                     \
                    C32 = vfmaq_f32(C32, B2, A3);                                                     \
                    A3 = vld1q_dup_f32(&Atrow(3, ii + 3));                                            \
                    C40 = vfmaq_f32(C40, B0, A4);                                                     \
                    C41 = vfmaq_f32(C41, B1, A4);                                                     \
                    C42 = vfmaq_f32(C42, B2, A4);                                                     \
                    A4 = vld1q_dup_f32(&Atrow(4, ii + 3));                                            \
                    C50 = vfmaq_f32(C50, B0, A5);                                                     \
                    C51 = vfmaq_f32(C51, B1, A5);                                                     \
                    C52 = vfmaq_f32(C52, B2, A5);                                                     \
                    A5 = vld1q_dup_f32(&Atrow(5, ii + 3));                                            \
                    C60 = vfmaq_f32(C60, B0, A6);                                                     \
                    C61 = vfmaq_f32(C61, B1, A6);                                                     \
                    C62 = vfmaq_f32(C62, B2, A6);                                                     \
                    A6 = vld1q_dup_f32(&Atrow(6, ii + 3));                                            \
                                                                                                      \
                    B0 = vld1q_f32(&Bptr[baseB + 3 * ldB]);                                           \
                    B1 = vld1q_f32(&Bptr[baseB + 3 * ldB + 4]);                                       \
                    B2 = vld1q_f32(&Bptr[baseB + 3 * ldB + 8]);                                       \
                                                                                                      \
                    C00 = vfmaq_f32(C00, B0, A0);                                                     \
                    C01 = vfmaq_f32(C01, B1, A0);                                                     \
                    C02 = vfmaq_f32(C02, B2, A0);                                                     \
                    C10 = vfmaq_f32(C10, B0, A1);                                                     \
                    C11 = vfmaq_f32(C11, B1, A1);                                                     \
                    C12 = vfmaq_f32(C12, B2, A1);                                                     \
                    C20 = vfmaq_f32(C20, B0, A2);                                                     \
                    C21 = vfmaq_f32(C21, B1, A2);                                                     \
                    C22 = vfmaq_f32(C22, B2, A2);                                                     \
                    C30 = vfmaq_f32(C30, B0, A3);                                                     \
                    C31 = vfmaq_f32(C31, B1, A3);                                                     \
                    C32 = vfmaq_f32(C32, B2, A3);                                                     \
                    C40 = vfmaq_f32(C40, B0, A4);                                                     \
                    C41 = vfmaq_f32(C41, B1, A4);                                                     \
                    C42 = vfmaq_f32(C42, B2, A4);                                                     \
                    C50 = vfmaq_f32(C50, B0, A5);                                                     \
                    C51 = vfmaq_f32(C51, B1, A5);                                                     \
                    C52 = vfmaq_f32(C52, B2, A5);                                                     \
                    C60 = vfmaq_f32(C60, B0, A6);                                                     \
                    C61 = vfmaq_f32(C61, B1, A6);                                                     \
                    C62 = vfmaq_f32(C62, B2, A6);                                                     \
                }                                                                                     \
            }                                                                                         \
        }                                                                                             \
                                                                                                      \
        vst1q_f32(&Crow(0, 0), C00);                                                                  \
        vst1q_f32(&Crow(0, 4), C01);                                                                  \
        vst1q_f32(&Crow(0, 8), C02);                                                                  \
        vst1q_f32(&Crow(1, 0), C10);                                                                  \
        vst1q_f32(&Crow(1, 4), C11);                                                                  \
        vst1q_f32(&Crow(1, 8), C12);                                                                  \
        vst1q_f32(&Crow(2, 0), C20);                                                                  \
        vst1q_f32(&Crow(2, 4), C21);                                                                  \
        vst1q_f32(&Crow(2, 8), C22);                                                                  \
        vst1q_f32(&Crow(3, 0), C30);                                                                  \
        vst1q_f32(&Crow(3, 4), C31);                                                                  \
        vst1q_f32(&Crow(3, 8), C32);                                                                  \
        vst1q_f32(&Crow(4, 0), C40);                                                                  \
        vst1q_f32(&Crow(4, 4), C41);                                                                  \
        vst1q_f32(&Crow(4, 8), C42);                                                                  \
        vst1q_f32(&Crow(5, 0), C50);                                                                  \
        vst1q_f32(&Crow(5, 4), C51);                                                                  \
        vst1q_f32(&Crow(5, 8), C52);                                                                  \
        vst1q_f32(&Crow(6, 0), C60);                                                                  \
        vst1q_f32(&Crow(6, 4), C61);                                                                  \
        vst1q_f32(&Crow(6, 8), C62);                                                                  \
    }

#define conv_kernel_enrique_gemm(                                                                   \
    first,                                                                                          \
    W_ob, C_ob, C_ib, step,                                                                         \
    H_f, W_f,                                                                                       \
    input_col_stride,                                                                               \
    I,                                                                                              \
    F,                                                                                              \
    O)                                                                                              \
    {                                                                                               \
        float *Ar = I;                                                                              \
        constexpr int NR = C_ob;                                                                    \
        constexpr int MR = W_ob;                                                                    \
        int i, j, k, baseB = 0, ldCt = NR, Amr, Bnr, ldA = NR, ldAt = W_ob, ldC = C_ob, ldB = C_ob; \
        float32x4_t C00, C01, C02,                                                                  \
            C10, C11, C12,                                                                          \
            C20, C21, C22,                                                                          \
            C30, C31, C32,                                                                          \
            C40, C41, C42,                                                                          \
            C50, C51, C52,                                                                          \
            C60, C61, C62,                                                                          \
            A0, A1, A2, A3, A4, A5, A6, B0, B1, B2;                                                 \
        float zero = 0.0, one = 1.0, Ctmp[MR * NR], *Atmp, *C;                                      \
        C = O;                                                                                      \
        if (first == 0)                                                                             \
        {                                                                                           \
            C00 = vmovq_n_f32(0);                                                                   \
            C01 = vmovq_n_f32(0);                                                                   \
            C02 = vmovq_n_f32(0);                                                                   \
            C10 = vmovq_n_f32(0);                                                                   \
            C11 = vmovq_n_f32(0);                                                                   \
            C12 = vmovq_n_f32(0);                                                                   \
            C20 = vmovq_n_f32(0);                                                                   \
            C21 = vmovq_n_f32(0);                                                                   \
            C22 = vmovq_n_f32(0);                                                                   \
            C30 = vmovq_n_f32(0);                                                                   \
            C31 = vmovq_n_f32(0);                                                                   \
            C32 = vmovq_n_f32(0);                                                                   \
            C40 = vmovq_n_f32(0);                                                                   \
            C41 = vmovq_n_f32(0);                                                                   \
            C42 = vmovq_n_f32(0);                                                                   \
            C50 = vmovq_n_f32(0);                                                                   \
            C51 = vmovq_n_f32(0);                                                                   \
            C52 = vmovq_n_f32(0);                                                                   \
            C60 = vmovq_n_f32(0);                                                                   \
            C61 = vmovq_n_f32(0);                                                                   \
            C62 = vmovq_n_f32(0);                                                                   \
        }                                                                                           \
        else                                                                                        \
        {                                                                                           \
            C00 = vld1q_f32(&Crow(0, 0));                                                           \
            C01 = vld1q_f32(&Crow(0, 4));                                                           \
            C02 = vld1q_f32(&Crow(0, 8));                                                           \
            C10 = vld1q_f32(&Crow(1, 0));                                                           \
            C11 = vld1q_f32(&Crow(1, 4));                                                           \
            C12 = vld1q_f32(&Crow(1, 8));                                                           \
            C20 = vld1q_f32(&Crow(2, 0));                                                           \
            C21 = vld1q_f32(&Crow(2, 4));                                                           \
            C22 = vld1q_f32(&Crow(2, 8));                                                           \
            C30 = vld1q_f32(&Crow(3, 0));                                                           \
            C31 = vld1q_f32(&Crow(3, 4));                                                           \
            C32 = vld1q_f32(&Crow(3, 8));                                                           \
            C40 = vld1q_f32(&Crow(4, 0));                                                           \
            C41 = vld1q_f32(&Crow(4, 4));                                                           \
            C42 = vld1q_f32(&Crow(4, 8));                                                           \
            C50 = vld1q_f32(&Crow(5, 0));                                                           \
            C51 = vld1q_f32(&Crow(5, 4));                                                           \
            C52 = vld1q_f32(&Crow(5, 8));                                                           \
            C60 = vld1q_f32(&Crow(6, 0));                                                           \
            C61 = vld1q_f32(&Crow(6, 4));                                                           \
            C62 = vld1q_f32(&Crow(6, 8));                                                           \
        }                                                                                           \
        int updates = 0;                                                                            \
        for (uint32_t ii = 0; ii < H_f * W_f * C_ib; ii += SIMD)                                    \
        {                                                                                           \
            int filter_offset_w = ii * C_ob;                                                        \
            int input_stencil_w = ii;                                                               \
            float *b = F + filter_offset_w;                                                         \
            float *Bptr = b;                                                                        \
            float *a = I + input_stencil_w;                                                         \
            Atmp = a;                                                                               \
            int p_cur = ii;                                                                         \
            baseB = ii * C_ob;                                                                      \
            A0 = vld1q_dup_f32(&Atrow_no_stride(0, ii));                                            \
            A1 = vld1q_dup_f32(&Atrow_no_stride(1, ii));                                            \
            A2 = vld1q_dup_f32(&Atrow_no_stride(2, ii));                                            \
            A3 = vld1q_dup_f32(&Atrow_no_stride(3, ii));                                            \
            A4 = vld1q_dup_f32(&Atrow_no_stride(4, ii));                                            \
            A5 = vld1q_dup_f32(&Atrow_no_stride(5, ii));                                            \
            A6 = vld1q_dup_f32(&Atrow_no_stride(6, ii));                                            \
                                                                                                    \
            B0 = vld1q_f32(Bptr + baseB);                                                           \
            B1 = vld1q_f32(&Bptr[baseB + 4]);                                                       \
            B2 = vld1q_f32(&Bptr[baseB + 8]);                                                       \
                                                                                                    \
            C00 = vfmaq_f32(C00, B0, A0);                                                           \
            C01 = vfmaq_f32(C01, B1, A0);                                                           \
            C02 = vfmaq_f32(C02, B2, A0);                                                           \
            A0 = vld1q_dup_f32(&Atrow_no_stride(0, ii + 1));                                        \
            C10 = vfmaq_f32(C10, B0, A1);                                                           \
            C11 = vfmaq_f32(C11, B1, A1);                                                           \
            C12 = vfmaq_f32(C12, B2, A1);                                                           \
            A1 = vld1q_dup_f32(&Atrow_no_stride(1, ii + 1));                                        \
            C20 = vfmaq_f32(C20, B0, A2);                                                           \
            C21 = vfmaq_f32(C21, B1, A2);                                                           \
            C22 = vfmaq_f32(C22, B2, A2);                                                           \
            A2 = vld1q_dup_f32(&Atrow_no_stride(2, ii + 1));                                        \
            C30 = vfmaq_f32(C30, B0, A3);                                                           \
            C31 = vfmaq_f32(C31, B1, A3);                                                           \
            C32 = vfmaq_f32(C32, B2, A3);                                                           \
            A3 = vld1q_dup_f32(&Atrow_no_stride(3, ii + 1));                                        \
            C40 = vfmaq_f32(C40, B0, A4);                                                           \
            C41 = vfmaq_f32(C41, B1, A4);                                                           \
            C42 = vfmaq_f32(C42, B2, A4);                                                           \
            A4 = vld1q_dup_f32(&Atrow_no_stride(4, ii + 1));                                        \
            C50 = vfmaq_f32(C50, B0, A5);                                                           \
            C51 = vfmaq_f32(C51, B1, A5);                                                           \
            C52 = vfmaq_f32(C52, B2, A5);                                                           \
            A5 = vld1q_dup_f32(&Atrow_no_stride(5, ii + 1));                                        \
            C60 = vfmaq_f32(C60, B0, A6);                                                           \
            C61 = vfmaq_f32(C61, B1, A6);                                                           \
            C62 = vfmaq_f32(C62, B2, A6);                                                           \
            A6 = vld1q_dup_f32(&Atrow_no_stride(6, ii + 1));                                        \
                                                                                                    \
            B0 = vld1q_f32(&Bptr[baseB + ldB]);                                                     \
            B1 = vld1q_f32(&Bptr[baseB + ldB + 4]);                                                 \
            B2 = vld1q_f32(&Bptr[baseB + ldB + 8]);                                                 \
                                                                                                    \
            C00 = vfmaq_f32(C00, B0, A0);                                                           \
            C01 = vfmaq_f32(C01, B1, A0);                                                           \
            C02 = vfmaq_f32(C02, B2, A0);                                                           \
            A0 = vld1q_dup_f32(&Atrow_no_stride(0, ii + 2));                                        \
            C10 = vfmaq_f32(C10, B0, A1);                                                           \
            C11 = vfmaq_f32(C11, B1, A1);                                                           \
            C12 = vfmaq_f32(C12, B2, A1);                                                           \
            A1 = vld1q_dup_f32(&Atrow_no_stride(1, ii + 2));                                        \
            C20 = vfmaq_f32(C20, B0, A2);                                                           \
            C21 = vfmaq_f32(C21, B1, A2);                                                           \
            C22 = vfmaq_f32(C22, B2, A2);                                                           \
            A2 = vld1q_dup_f32(&Atrow_no_stride(2, ii + 2));                                        \
            C30 = vfmaq_f32(C30, B0, A3);                                                           \
            C31 = vfmaq_f32(C31, B1, A3);                                                           \
            C32 = vfmaq_f32(C32, B2, A3);                                                           \
            A3 = vld1q_dup_f32(&Atrow_no_stride(3, ii + 2));                                        \
            C40 = vfmaq_f32(C40, B0, A4);                                                           \
            C41 = vfmaq_f32(C41, B1, A4);                                                           \
            C42 = vfmaq_f32(C42, B2, A4);                                                           \
            A4 = vld1q_dup_f32(&Atrow_no_stride(4, ii + 2));                                        \
            C50 = vfmaq_f32(C50, B0, A5);                                                           \
            C51 = vfmaq_f32(C51, B1, A5);                                                           \
            C52 = vfmaq_f32(C52, B2, A5);                                                           \
            A5 = vld1q_dup_f32(&Atrow_no_stride(5, ii + 2));                                        \
            C60 = vfmaq_f32(C60, B0, A6);                                                           \
            C61 = vfmaq_f32(C61, B1, A6);                                                           \
            C62 = vfmaq_f32(C62, B2, A6);                                                           \
            A6 = vld1q_dup_f32(&Atrow_no_stride(6, ii + 2));                                        \
                                                                                                    \
            B0 = vld1q_f32(&Bptr[baseB + 2 * ldB]);                                                 \
            B1 = vld1q_f32(&Bptr[baseB + 2 * ldB + 4]);                                             \
            B2 = vld1q_f32(&Bptr[baseB + 2 * ldB + 8]);                                             \
                                                                                                    \
            C00 = vfmaq_f32(C00, B0, A0);                                                           \
            C01 = vfmaq_f32(C01, B1, A0);                                                           \
            C02 = vfmaq_f32(C02, B2, A0);                                                           \
            A0 = vld1q_dup_f32(&Atrow_no_stride(0, ii + 3));                                        \
            C10 = vfmaq_f32(C10, B0, A1);                                                           \
            C11 = vfmaq_f32(C11, B1, A1);                                                           \
            C12 = vfmaq_f32(C12, B2, A1);                                                           \
            A1 = vld1q_dup_f32(&Atrow_no_stride(1, ii + 3));                                        \
            C20 = vfmaq_f32(C20, B0, A2);                                                           \
            C21 = vfmaq_f32(C21, B1, A2);                                                           \
            C22 = vfmaq_f32(C22, B2, A2);                                                           \
            A2 = vld1q_dup_f32(&Atrow_no_stride(2, ii + 3));                                        \
            C30 = vfmaq_f32(C30, B0, A3);                                                           \
            C31 = vfmaq_f32(C31, B1, A3);                                                           \
            C32 = vfmaq_f32(C32, B2, A3);                                                           \
            A3 = vld1q_dup_f32(&Atrow_no_stride(3, ii + 3));                                        \
            C40 = vfmaq_f32(C40, B0, A4);                                                           \
            C41 = vfmaq_f32(C41, B1, A4);                                                           \
            C42 = vfmaq_f32(C42, B2, A4);                                                           \
            A4 = vld1q_dup_f32(&Atrow_no_stride(4, ii + 3));                                        \
            C50 = vfmaq_f32(C50, B0, A5);                                                           \
            C51 = vfmaq_f32(C51, B1, A5);                                                           \
            C52 = vfmaq_f32(C52, B2, A5);                                                           \
            A5 = vld1q_dup_f32(&Atrow_no_stride(5, ii + 3));                                        \
            C60 = vfmaq_f32(C60, B0, A6);                                                           \
            C61 = vfmaq_f32(C61, B1, A6);                                                           \
            C62 = vfmaq_f32(C62, B2, A6);                                                           \
            A6 = vld1q_dup_f32(&Atrow_no_stride(6, ii + 3));                                        \
                                                                                                    \
            B0 = vld1q_f32(&Bptr[baseB + 3 * ldB]);                                                 \
            B1 = vld1q_f32(&Bptr[baseB + 3 * ldB + 4]);                                             \
            B2 = vld1q_f32(&Bptr[baseB + 3 * ldB + 8]);                                             \
                                                                                                    \
            C00 = vfmaq_f32(C00, B0, A0);                                                           \
            C01 = vfmaq_f32(C01, B1, A0);                                                           \
            C02 = vfmaq_f32(C02, B2, A0);                                                           \
            C10 = vfmaq_f32(C10, B0, A1);                                                           \
            C11 = vfmaq_f32(C11, B1, A1);                                                           \
            C12 = vfmaq_f32(C12, B2, A1);                                                           \
            C20 = vfmaq_f32(C20, B0, A2);                                                           \
            C21 = vfmaq_f32(C21, B1, A2);                                                           \
            C22 = vfmaq_f32(C22, B2, A2);                                                           \
            C30 = vfmaq_f32(C30, B0, A3);                                                           \
            C31 = vfmaq_f32(C31, B1, A3);                                                           \
            C32 = vfmaq_f32(C32, B2, A3);                                                           \
            C40 = vfmaq_f32(C40, B0, A4);                                                           \
            C41 = vfmaq_f32(C41, B1, A4);                                                           \
            C42 = vfmaq_f32(C42, B2, A4);                                                           \
            C50 = vfmaq_f32(C50, B0, A5);                                                           \
            C51 = vfmaq_f32(C51, B1, A5);                                                           \
            C52 = vfmaq_f32(C52, B2, A5);                                                           \
            C60 = vfmaq_f32(C60, B0, A6);                                                           \
            C61 = vfmaq_laneq_f32(C61, B1, A6, 3);                                                  \
            C62 = vfmaq_laneq_f32(C62, B2, A6, 3);                                                  \
        }                                                                                           \
                                                                                                    \
        vst1q_f32(&Crow(0, 0), C00);                                                                \
        vst1q_f32(&Crow(0, 4), C01);                                                                \
        vst1q_f32(&Crow(0, 8), C02);                                                                \
        vst1q_f32(&Crow(1, 0), C10);                                                                \
        vst1q_f32(&Crow(1, 4), C11);                                                                \
        vst1q_f32(&Crow(1, 8), C12);                                                                \
        vst1q_f32(&Crow(2, 0), C20);                                                                \
        vst1q_f32(&Crow(2, 4), C21);                                                                \
        vst1q_f32(&Crow(2, 8), C22);                                                                \
        vst1q_f32(&Crow(3, 0), C30);                                                                \
        vst1q_f32(&Crow(3, 4), C31);                                                                \
        vst1q_f32(&Crow(3, 8), C32);                                                                \
        vst1q_f32(&Crow(4, 0), C40);                                                                \
        vst1q_f32(&Crow(4, 4), C41);                                                                \
        vst1q_f32(&Crow(4, 8), C42);                                                                \
        vst1q_f32(&Crow(5, 0), C50);                                                                \
        vst1q_f32(&Crow(5, 4), C51);                                                                \
        vst1q_f32(&Crow(5, 8), C52);                                                                \
        vst1q_f32(&Crow(6, 0), C60);                                                                \
        vst1q_f32(&Crow(6, 4), C61);                                                                \
        vst1q_f32(&Crow(6, 8), C62);                                                                \
    }

template <uint32_t W_ob, uint32_t C_ob, uint32_t C_ib, uint32_t step>
inline void conv_kernel_start_enrique(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O)
{
    float *Ar = I;
    // printf("SIMD kernel from enrique\n");
    constexpr int NR = C_ob;
    constexpr int MR = W_ob;
    int i, j, k, baseB = 0, ldCt = NR, Amr, Bnr, ldA = NR, ldAt = C_ib, ldC = C_ob, ldB = C_ob;
    // ZERO_TILE_C(W_ob, C_ob);

    float32x4_t C00, C01, C02,
        C10, C11, C12,
        C20, C21, C22,
        C30, C31, C32,
        C40, C41, C42,
        C50, C51, C52,
        C60, C61, C62,
        A0, A1, A2, A3, A4, A5, A6, B0, B1, B2;
    float zero = 0.0, one = 1.0, Ctmp[MR * NR], *Atmp, *C;

    C = O;
    C00 = vmovq_n_f32(0);
    C01 = vmovq_n_f32(0);
    C02 = vmovq_n_f32(0);
    C10 = vmovq_n_f32(0);
    C11 = vmovq_n_f32(0);
    C12 = vmovq_n_f32(0);
    C20 = vmovq_n_f32(0);
    C21 = vmovq_n_f32(0);
    C22 = vmovq_n_f32(0);
    C30 = vmovq_n_f32(0);
    C31 = vmovq_n_f32(0);
    C32 = vmovq_n_f32(0);
    C40 = vmovq_n_f32(0);
    C41 = vmovq_n_f32(0);
    C42 = vmovq_n_f32(0);
    C50 = vmovq_n_f32(0);
    C51 = vmovq_n_f32(0);
    C52 = vmovq_n_f32(0);
    C60 = vmovq_n_f32(0);
    C61 = vmovq_n_f32(0);
    C62 = vmovq_n_f32(0);
    // End ZERO Tile

    int updates = 0;
    // uint32_t step = C_ob;//stride*C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int filter_offset_h = n * W_f * C_ib * C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int filter_offset_w = m * C_ib * C_ob + filter_offset_h;
            int input_stencil_w = m * C_ib + input_stencil_h;

            float *b = F + filter_offset_w;
            float *Bptr = b;
            float *a = I + input_stencil_w;
            Atmp = a;
            for (uint32_t ii = 0; ii < C_ib; ii += SIMD)
            {

                // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                int p_cur = ii;
                baseB = ii * C_ob;
                // printf(" filter ip channel col_offset %d\n", baseB);
                // FMA_TILE_C(step, a, b, p_cur, W_ob, C_ob);
                // multiply with step to support other strides
                A0 = vld1q_f32(&Atrow(0, ii));
                A1 = vld1q_f32(&Atrow(1, ii));
                A2 = vld1q_f32(&Atrow(2, ii));
                A3 = vld1q_f32(&Atrow(3, ii));
                A4 = vld1q_f32(&Atrow(4, ii));
                A5 = vld1q_f32(&Atrow(5, ii));
                A6 = vld1q_f32(&Atrow(6, ii)); // printf(" loaded A \n");
                // printf("%.2f %.2f %.2f %.2f ", Atrow(0, ii),  Atrow(0, ii+1)  Atrow(0, ii));

                B0 = vld1q_f32(Bptr + baseB);     // printf(" filter ip channel col_offset %d\n", baseB);
                B1 = vld1q_f32(&Bptr[baseB + 4]); // printf(" filter ip channel col_offset %d\n", baseB+4);
                B2 = vld1q_f32(&Bptr[baseB + 8]); // printf(" filter ip channel col_offset %d\n", baseB+8);

                C00 = vfmaq_laneq_f32(C00, B0, A0, 0);
                C01 = vfmaq_laneq_f32(C01, B1, A0, 0);
                C02 = vfmaq_laneq_f32(C02, B2, A0, 0);
                C10 = vfmaq_laneq_f32(C10, B0, A1, 0);
                C11 = vfmaq_laneq_f32(C11, B1, A1, 0);
                C12 = vfmaq_laneq_f32(C12, B2, A1, 0);
                C20 = vfmaq_laneq_f32(C20, B0, A2, 0);
                C21 = vfmaq_laneq_f32(C21, B1, A2, 0);
                C22 = vfmaq_laneq_f32(C22, B2, A2, 0);
                C30 = vfmaq_laneq_f32(C30, B0, A3, 0);
                C31 = vfmaq_laneq_f32(C31, B1, A3, 0);
                C32 = vfmaq_laneq_f32(C32, B2, A3, 0);
                C40 = vfmaq_laneq_f32(C40, B0, A4, 0);
                C41 = vfmaq_laneq_f32(C41, B1, A4, 0);
                C42 = vfmaq_laneq_f32(C42, B2, A4, 0);
                C50 = vfmaq_laneq_f32(C50, B0, A5, 0);
                C51 = vfmaq_laneq_f32(C51, B1, A5, 0);
                C52 = vfmaq_laneq_f32(C52, B2, A5, 0);
                C60 = vfmaq_laneq_f32(C60, B0, A6, 0);
                C61 = vfmaq_laneq_f32(C61, B1, A6, 0);
                C62 = vfmaq_laneq_f32(C62, B2, A6, 0);

                B0 = vld1q_f32(&Bptr[baseB + ldB]);
                B1 = vld1q_f32(&Bptr[baseB + ldB + 4]);
                B2 = vld1q_f32(&Bptr[baseB + ldB + 8]);

                C00 = vfmaq_laneq_f32(C00, B0, A0, 1);
                C01 = vfmaq_laneq_f32(C01, B1, A0, 1);
                C02 = vfmaq_laneq_f32(C02, B2, A0, 1);
                C10 = vfmaq_laneq_f32(C10, B0, A1, 1);
                C11 = vfmaq_laneq_f32(C11, B1, A1, 1);
                C12 = vfmaq_laneq_f32(C12, B2, A1, 1);
                C20 = vfmaq_laneq_f32(C20, B0, A2, 1);
                C21 = vfmaq_laneq_f32(C21, B1, A2, 1);
                C22 = vfmaq_laneq_f32(C22, B2, A2, 1);
                C30 = vfmaq_laneq_f32(C30, B0, A3, 1);
                C31 = vfmaq_laneq_f32(C31, B1, A3, 1);
                C32 = vfmaq_laneq_f32(C32, B2, A3, 1);
                C40 = vfmaq_laneq_f32(C40, B0, A4, 1);
                C41 = vfmaq_laneq_f32(C41, B1, A4, 1);
                C42 = vfmaq_laneq_f32(C42, B2, A4, 1);
                C50 = vfmaq_laneq_f32(C50, B0, A5, 1);
                C51 = vfmaq_laneq_f32(C51, B1, A5, 1);
                C52 = vfmaq_laneq_f32(C52, B2, A5, 1);
                C60 = vfmaq_laneq_f32(C60, B0, A6, 1);
                C61 = vfmaq_laneq_f32(C61, B1, A6, 1);
                C62 = vfmaq_laneq_f32(C62, B2, A6, 1);

                B0 = vld1q_f32(&Bptr[baseB + 2 * ldB]);
                B1 = vld1q_f32(&Bptr[baseB + 2 * ldB + 4]);
                B2 = vld1q_f32(&Bptr[baseB + 2 * ldB + 8]);

                C00 = vfmaq_laneq_f32(C00, B0, A0, 2);
                C01 = vfmaq_laneq_f32(C01, B1, A0, 2);
                C02 = vfmaq_laneq_f32(C02, B2, A0, 2);
                C10 = vfmaq_laneq_f32(C10, B0, A1, 2);
                C11 = vfmaq_laneq_f32(C11, B1, A1, 2);
                C12 = vfmaq_laneq_f32(C12, B2, A1, 2);
                C20 = vfmaq_laneq_f32(C20, B0, A2, 2);
                C21 = vfmaq_laneq_f32(C21, B1, A2, 2);
                C22 = vfmaq_laneq_f32(C22, B2, A2, 2);
                C30 = vfmaq_laneq_f32(C30, B0, A3, 2);
                C31 = vfmaq_laneq_f32(C31, B1, A3, 2);
                C32 = vfmaq_laneq_f32(C32, B2, A3, 2);
                C40 = vfmaq_laneq_f32(C40, B0, A4, 2);
                C41 = vfmaq_laneq_f32(C41, B1, A4, 2);
                C42 = vfmaq_laneq_f32(C42, B2, A4, 2);
                C50 = vfmaq_laneq_f32(C50, B0, A5, 2);
                C51 = vfmaq_laneq_f32(C51, B1, A5, 2);
                C52 = vfmaq_laneq_f32(C52, B2, A5, 2);
                C60 = vfmaq_laneq_f32(C60, B0, A6, 2);
                C61 = vfmaq_laneq_f32(C61, B1, A6, 2);
                C62 = vfmaq_laneq_f32(C62, B2, A6, 2);

                B0 = vld1q_f32(&Bptr[baseB + 3 * ldB]);
                B1 = vld1q_f32(&Bptr[baseB + 3 * ldB + 4]);
                B2 = vld1q_f32(&Bptr[baseB + 3 * ldB + 8]);

                C00 = vfmaq_laneq_f32(C00, B0, A0, 3);
                C01 = vfmaq_laneq_f32(C01, B1, A0, 3);
                C02 = vfmaq_laneq_f32(C02, B2, A0, 3);
                C10 = vfmaq_laneq_f32(C10, B0, A1, 3);
                C11 = vfmaq_laneq_f32(C11, B1, A1, 3);
                C12 = vfmaq_laneq_f32(C12, B2, A1, 3);
                C20 = vfmaq_laneq_f32(C20, B0, A2, 3);
                C21 = vfmaq_laneq_f32(C21, B1, A2, 3);
                C22 = vfmaq_laneq_f32(C22, B2, A2, 3);
                C30 = vfmaq_laneq_f32(C30, B0, A3, 3);
                C31 = vfmaq_laneq_f32(C31, B1, A3, 3);
                C32 = vfmaq_laneq_f32(C32, B2, A3, 3);
                C40 = vfmaq_laneq_f32(C40, B0, A4, 3);
                C41 = vfmaq_laneq_f32(C41, B1, A4, 3);
                C42 = vfmaq_laneq_f32(C42, B2, A4, 3);
                C50 = vfmaq_laneq_f32(C50, B0, A5, 3);
                C51 = vfmaq_laneq_f32(C51, B1, A5, 3);
                C52 = vfmaq_laneq_f32(C52, B2, A5, 3);
                C60 = vfmaq_laneq_f32(C60, B0, A6, 3);
                C61 = vfmaq_laneq_f32(C61, B1, A6, 3);
                C62 = vfmaq_laneq_f32(C62, B2, A6, 3);
            }
        }
    }

    // STORE_TILE_C(O,W_ob, C_ob);
    vst1q_f32(&Crow(0, 0), C00);
    vst1q_f32(&Crow(0, 4), C01);
    vst1q_f32(&Crow(0, 8), C02);
    vst1q_f32(&Crow(1, 0), C10);
    vst1q_f32(&Crow(1, 4), C11);
    vst1q_f32(&Crow(1, 8), C12);
    vst1q_f32(&Crow(2, 0), C20);
    vst1q_f32(&Crow(2, 4), C21);
    vst1q_f32(&Crow(2, 8), C22);
    vst1q_f32(&Crow(3, 0), C30);
    vst1q_f32(&Crow(3, 4), C31);
    vst1q_f32(&Crow(3, 8), C32);
    vst1q_f32(&Crow(4, 0), C40);
    vst1q_f32(&Crow(4, 4), C41);
    vst1q_f32(&Crow(4, 8), C42);
    vst1q_f32(&Crow(5, 0), C50);
    vst1q_f32(&Crow(5, 4), C51);
    vst1q_f32(&Crow(5, 8), C52);
    vst1q_f32(&Crow(6, 0), C60);
    vst1q_f32(&Crow(6, 4), C61);
    vst1q_f32(&Crow(6, 8), C62);
}

// cleanup convolution kernels
template <uint32_t W_ob, uint32_t C_ob, uint32_t C_ib, uint32_t step>
inline void conv_kernel_start_end_enrique(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t W_last)
{

    ZERO_END_C(W_ob, C_ob);

    int updates = 0;
    // uint32_t step = C_ob;//stride*C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int filter_offset_h = n * W_f * C_ib * C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int filter_offset_w = m * C_ib * C_ob + filter_offset_h;
            int input_stencil_w = m * C_ib + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < C_ib; ii++)
            {

                // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                int p_cur = ii;

                FMA_END_C(step, a, b, p_cur, W_ob, C_ob, W_last);
                // printf("%d %d %d %.2f %.2f %.2f\n", n, m, ii, a[0], b[0], c_tile[0]);
            }
        }
    }

    STORE_END_C(O, W_ob, C_ob, W_last);
}

template <uint32_t W_ob, uint32_t C_ob, uint32_t C_ib, uint32_t step>
inline void conv_kernel_end_enrique(
    uint32_t H_f, uint32_t W_f,
    uint32_t input_col_stride,
    float *I,
    float *F,
    float *O,
    uint32_t W_last)
{

    LOAD_END_C(O, W_ob, C_ob)   ;
    int updates = 0;
    // uint32_t step = stride*C_ob;
    // int count = 0;
    for (uint32_t n = 0; n < H_f; n++)
    {

        int filter_offset_h = n * W_f * C_ib * C_ob;
        int input_stencil_h = /*input_col_offset +*/ n * input_col_stride /*+ input_row_offset*/;

        for (uint32_t m = 0; m < W_f; m++)
        {

            int filter_offset_w = m * C_ib * C_ob + filter_offset_h;
            int input_stencil_w = m * C_ib + input_stencil_h;

            float *b = F + filter_offset_w;
            float *a = I + input_stencil_w;
            for (uint32_t ii = 0; ii < C_ib; ii++)
            {

                // kernel_conv(W_ob,C_ob,rank_k,I + input_stencil_w, F + filter_offset_w, O);

                int p_cur = ii;
                FMA_END_C(step, a, b, p_cur, W_ob, C_ob, W_last);
            }
        }
    }

    STORE_END_C(O, W_ob, C_ob, W_last);
}
