//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2024 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126
//****************************************************************************

#include <math.h>
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>


#include <small/op_type.hpp>
// #include <small.h>
#include <small/utils/Timer.hpp>
#include <params.h>
#include <Buffer.hpp>
#include <intrinsics.h>

// kernel specific intrinsics are already in small.h

// To test different kernels make this program with different compile time arguments
// if none are provided, test the hardware-specific parameters from params.h

#ifndef KERNEL_C_ob
#define KERNEL_C_ob FLOAT_C_ob
#endif

#ifndef KERNEL_W_ob
#define KERNEL_W_ob FLOAT_W_ob
#endif

// The macro below can be used to produce all the kernel benchmarks using
// this program.

//****************************************************************************
//****************************************************************************
// Stolen from small::detail
typedef small::FloatBuffer::value_type dtype;
typedef small::FloatBuffer::value_type c_tile_t;

#define FLOAT_ABSTRACT_OP(op_type, op_class, step, a_cur, b_cur, _O_wb, _C_ob) \
    if constexpr (op_type == small::OP_CONV)                             \
    {                                                                    \
        if constexpr (op_class == 1)                                     \
        {                                                                \
            FLOAT_DW_TILE_C(step, a_cur, b_cur, _O_wb, _C_ob);           \
        }                                                                \
        else if constexpr (op_class == 2)                                \
        {                                                                \
            FLOAT_CONV_TILE_C(step, a_cur, b_cur, _O_wb, _C_ob);         \
        }                                                                \
    }                                                                    \
    else if constexpr (op_type == small::OP_RELU ||                      \
                       op_type == small::OP_MAX_POOL)                    \
    {                                                                    \
        FLOAT_MAX_TILE_C(step, a_cur, _O_wb, _C_ob);                     \
    }                                                                    \
    else if constexpr (op_type == small::OP_LEAKY_RELU)                  \
    {                                                                    \
        FLOAT_COND_SCALE_TILE_C(step, a_cur, b_cur, _O_wb, _C_ob);       \
    }                                                                    \
    else if constexpr (op_type == small::OP_ADD ||                       \
                       op_type == small::OP_AVERAGE_POOL)                \
    {                                                                    \
        FLOAT_ACCUM_TILE_C(step, a_cur, _O_wb, _C_ob);                   \
    }                                                                    \
    else if constexpr (op_type == small::OP_MUL)                         \
    {                                                                    \
        float drop_out_rate = b_cur[0];                                  \
        FLOAT_DIV_TILE_C(drop_out_rate, _O_wb, _C_ob)                    \
    }                                                                    \
    else if constexpr (op_type == small::OP_EXP)                         \
    {                                                                    \
        FLOAT_EXP_TILE_C(step, a_cur, _O_wb, _C_ob)                      \
    }

//****************************************************************************
// kernel benchmark: Work on 1 output tile, input fits in L1
// k_max for 8-way 32KB L1: 128 (if weights are fully resident)
// k_max for 8-way 32KB L1: 1024 (if weights are from L2)
// This is only correct for 1x1
//
// I: mxk, W: kxn, O: mxn, where  m=W_ob, n=C_ob, k=c*_UNROLL,
// k=C_i (1x1), G_b=1(CONV), mxk+kxn+mxn < L1 capacity, stride is stride
//
template <int W_ob, int C_ob, int G_b,
          int _UNROLL, int stride,
          small::OpType OP_TYPE, int8_t OP_CLASS>
void kernel_benchmark(
    const int64_t m, const int64_t n, const int64_t k,
    const float* I, const float* W, float* O)
{
    int32_t constexpr step = C_ob * stride;
    FLOAT_DEF_TILE_C(W_ob, C_ob);
    FLOAT_LOAD_TILE_C(O, W_ob, C_ob);
    float const *a_cur = I;
    float const *b_cur = W;
#pragma GCC unroll 16
    for(int p = 0; p < k; p+= G_b*_UNROLL)
    {
        FLOAT_ABSTRACT_OP(OP_TYPE, OP_CLASS, step, a_cur, b_cur, W_ob, C_ob);
        b_cur+=n*G_b*_UNROLL;
        a_cur += G_b*_UNROLL;
    }
    FLOAT_STORE_TILE_C(O,W_ob, C_ob);
}


// layer to profile
#define CONV 0
#define PARTIAL_CONV 1
#define DW_CONV 2
#define GROUP_CONV 3 // under development
#define FC 4
#define LEAKY_RELU 5
#define MAX_POOL 6
#define RELU 7
#define UPSAMPLE 8
#define ACCUM 9
#define BIAS 10
#define AVERAGE_POOL 11
#define DROPOUT 12
#define SOFTMAX 13
#define PARTIAL_BIAS 14

#ifndef LAYER
#define LAYER CONV
#endif

#if LAYER == CONV
#define    OP_TYPE small::OP_CONV
#define    OP_CLASS 2
#define    G_b 1
#elif LAYER == PARTIAL_CONV
#define    OP_TYPE small::OP_CONV
#define    OP_CLASS 2
#define G_b 1

#elif LAYER == DW_CONV
#define    OP_TYPE small::OP_CONV
#define    OP_CLASS 1
#define G_b KERNEL_C_ob

#elif LAYER == GROUP_CONV
#define    OP_TYPE small::OP_CONV
#define    OP_CLASS 2
#define G_b 4

#elif LAYER == FC
#define    OP_TYPE small::OP_CONV
#define    OP_CLASS 2
#define G_b 1

#elif LAYER == MAX_POOL
#define OP_TYPE small::OP_MAX_POOL
#define OP_CLASS 1
#define G_b KERNEL_C_ob
#endif


// num of implementations x number of sizes
// In gigahertz
#define FREQ 1.5

#define TRIALS 100
#define RUNS 1000
#define NUM_IMPLEMENTATIONS 1
#define NUM_SIZES 14

double min_layer_timers[NUM_IMPLEMENTATIONS][NUM_SIZES];
double avg_layer_timers[NUM_IMPLEMENTATIONS][NUM_SIZES];
double total_layer_timers[NUM_IMPLEMENTATIONS][NUM_SIZES] = {0};
double layer_flops[NUM_IMPLEMENTATIONS][NUM_SIZES];

//****************************************************************************
// todo: This should change the computation so that it is correct for
//       different layer types
void check_result(const int m, const int n, const int k, const int stride,
                  const float *I, const float *W, const float *O)
{
    float *out_check = (float *)malloc(m * n * sizeof(float));
    float * cur_ptr = out_check;
    //Init to 0
    for (int i = 0; i < m * n; i++)
    {
        *(cur_ptr++) = 0.0;
    }

    // Compute the result in scalar
    for (int32_t p = 0; p < k; p++)
    {
        float const *a_cur = I + (p)*G_b;
        float const *b_cur = W + p * n;
        int step = KERNEL_C_ob * stride;
        for (int32_t i = 0; i < m; i++)
        {
            for (int32_t j = 0; j < n; j++)
            {
                out_check[i * n + j] += a_cur[i * step] * b_cur[j];
            }
        }
    }
    //Check against benchmark
    //multiplied with TRIALS and RUNS to account for accumulation
    //error margin has to be pretty wide
    for (int32_t i = 0; i < m; i++)
    {
        for (int32_t j = 0; j < n; j++)
        {
            out_check[i * n + j] *= ( RUNS);
            assert(fabs(out_check[i * n + j] - O[i * n + j]) / (out_check[i * n + j]) < 1e-2);
        }
    }
    free(out_check);
}


//****************************************************************************
int main()
{
    const int n = KERNEL_C_ob;
    const int m = KERNEL_W_ob;
    const int stride = 1;

    printf("m: %d, n: %d\n Minimum timing over %d trials, each trial averages over %d runs\n", m, n, TRIALS, RUNS);

    printf("k, ops, time (ns), ops/cyc, total time (ms)\n");
    int k_sizes[NUM_SIZES] = {16, 32, 64, 96, 128, 256, 384, 512,
                              1024, 2048, 4096, 8192, 16384, 32768};

    //allocate aligned memory
    for (int size = 0; size < NUM_SIZES; size++)
    {
        int k = k_sizes[size];
        int64_t ops = m * n * k * ((OP_TYPE == small::OP_CONV) ? 2 : 1);
        printf("%d, %ld, ", k, ops);
        float *I, *W, *O;
        float *shared_buffer;
        int ret = posix_memalign((void **)&shared_buffer, 64,
                                 (m * k + k*n + m*n) * sizeof(float));
        if (ret != 0)
        {
            fprintf(stderr, "ERROR: posix_memalign failed, return code %d, for size=%d", ret, size);
            continue;
        }

        I = shared_buffer;
        W = I + m * k;
        O = W + k * n;

        float *cur_ptr = I;
        for (int i = 0; i < m*k; i++)
        {
            *(cur_ptr++) = 2.0 * ((float)rand() / RAND_MAX) - 1;
        }

        cur_ptr = W;
        for (int i = 0; i < k*n; i++)
        {
            *(cur_ptr++) = 2.0 * ((float)rand() / RAND_MAX) - 1;
        }

        cur_ptr = O;
        for (int i = 0; i < m * n; i++)
        {
            *(cur_ptr++) = 0.0;
        }

        // warm up run (though the init might have done that)
        // kernel_benchmark<KERNEL_W_ob, KERNEL_C_ob, G_b, 1, OP_TYPE, OP_CLASS>(m, n, k, I, W, O);

        // benchmark
        small::Timer timer;

        for (int trial = 0; trial < TRIALS; trial++)
        {
            cur_ptr = O;
            for (int i = 0; i < m * n; i++)
            {
                *(cur_ptr++) = 0.0;
            }
            timer.start();
            for (int r = 0; r < RUNS; r++)
            {
                kernel_benchmark<KERNEL_W_ob, KERNEL_C_ob, G_b, FLOAT_UNROLL, 1, OP_TYPE, OP_CLASS>(m, n, k, I, W, O);
            }
            timer.stop();
            total_layer_timers[0][size] = timer.elapsed();
            avg_layer_timers[0][size] = timer.elapsed()/RUNS;
            min_layer_timers[0][size] =
                (trial == 0) ?
                avg_layer_timers[0][size] :
                std::min(min_layer_timers[0][size], avg_layer_timers[0][size]);
        }

        printf("%f, %f , %f\n", min_layer_timers[0][size], (1.0*ops)/(min_layer_timers[0][size]*FREQ), total_layer_timers[0][size]/1e6);

        check_result(m, n, k, stride,  I, W, O);

        free(shared_buffer);
    }

    return 0;
}
