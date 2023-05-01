# SMaLL, Software for Machine Learning Libraries
# Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# For additional details (including references to third party source code and
# other files) see the LICENSE file or contact permission@sei.cmu.edu. See
# Contributors.txt for a full list of contributors. Created, in part, with
# funding and support from the U.S. Government (see Acknowledgments.txt file).
# DM23-0126

W_ob = 6
C_ob2 = 1
C_ob1 = 16
C_ob = C_ob2 * C_ob1
SIMD = 4
UNROLL = C_ob # only affects conv kernel.
# UNROLL = SIMD # only affects conv kernel.

NUM_FMA = 2
NUM_MAX = 1
NUM_LOAD = 2
NUM_STORE = 1
with open('params.h', 'w') as f:
    f.write(
        '''
#define W_ob {W_ob}
#define C_ob {C_ob}
#define SIMD {SIMD}
#define UNROLL {UNROLL}
#define C_ib C_ob

// not used for kernels, but used in throughput calculation.
#define NUM_FMA {NUM_FMA}
#define NUM_MAX {NUM_MAX}
#define NUM_LOAD {NUM_LOAD}
#define NUM_STORE {NUM_STORE}
        '''.format(**locals())
    )

def redefine(name):
    return ['#ifdef {n}\n#undef {n}\n#endif\n'.format(n=name)]

with open('intrinsics-gen.h', 'w') as f:
    s = []
    s += ['#include <arm_neon.h>']

    # define tile
    # names of variables
    s += redefine('DEF_TILE_C')
    s += ['#define DEF_TILE_C(W_ob, C_ob)\\']
    c_tile = [["c_{}_{}".format(kk, jj) for jj in range(C_ob//SIMD)] for kk in range(W_ob)]
    s += ['float c_tile[W_ob * C_ob];\\']
    for kk in range(W_ob):
        for jj in range(C_ob//SIMD):
            s += ['float32x4_t {};\\'.format(c_tile[kk][jj])]
    s += ['']


    # zero tile
    s += redefine('ZERO_TILE_C')
    s += ['#define ZERO_TILE_C(W_ob, C_ob)\\']
    for kk in range(W_ob):
        for jj in range(C_ob//SIMD):
            s += ['{} = vdupq_n_f32(0);\\'.format(c_tile[kk][jj])]
    s += ['']

    # load tile
    s += redefine('LOAD_TILE_C')
    s += ['#define LOAD_TILE_C(O, W_ob, C_ob)\\']
    for kk in range(W_ob):
        for jj in range(C_ob//SIMD):
            s += ['{c} = vld1q_f32(O + {k} * C_ob + {j} * SIMD);\\'.format(c=c_tile[kk][jj], k=kk, j=jj)]
    s += ['']

    # load tile strided
    s += redefine('LOAD_TILE_C_strided')
    s += ['#define LOAD_TILE_C_strided(O, step, W_ob, C_ob)\\']
    for kk in range(W_ob):
        for jj in range(C_ob//SIMD):
            s += ['{c} = vld1q_f32(O + {k} * step + {j} * SIMD);\\'.format(c=c_tile[kk][jj], k=kk, j=jj)]
    s += ['']

    # store tile
    s += redefine('STORE_TILE_C')
    s += ['#define STORE_TILE_C(O, W_ob, C_ob)\\']
    for kk in range(W_ob):
        for jj in range(C_ob//SIMD):
            s += ['vst1q_f32(O + {k} * C_ob + {j} * SIMD, {c});\\'.format(c=c_tile[kk][jj], k=kk, j=jj)]
    s += ['']

    # convolution
    s += redefine('CONV_TILE_C')
    s += ['#define CONV_TILE_C(step, a, b, W_ob, C_ob)\\']
    s += ['float *aa = a;\\']
    s += ['float *bb = b;\\']
    # define a
    for kk in range(W_ob):
        s += ['float32x4_t a_{kk};\\'.format(kk=kk)]
    # define b [half as many]
    for jj in range(C_ob1//SIMD):
        s += ['float32x4_t b_{jj};\\'.format(jj=jj)]

    for i in range(UNROLL//SIMD):
        # load a SIMD width of a
        # for kk in range(W_ob):
        #     s += ['a_{kk} = vld1q_f32(a + {kk} * step + {i} * SIMD);\\'.format(kk=kk, i=i)]

        for j in range(C_ob2):


            for ii in range(SIMD):
                # load B
                # for jj in range(C_ob1//SIMD):
                #     # s += ['b_{jj} = vld1q_f32(b + ({i} * SIMD + {ii})*C_ob + ({j} * {C_ob1} + {jj})*SIMD);\\'.format(i=i, ii=ii, j=j, C_ob1=C_ob1//SIMD, jj=jj)]
                #     s += ['b_{jj} = vld1q_f32(bb + {ii}*C_ob + ({j} * {C_ob1} + {jj})*SIMD);\\'.format(ii=ii, j=j, C_ob1=C_ob1//SIMD, jj=jj)]

                # compute
                for kk in range(W_ob):
                    if j == 0: # load a just before use
                        s += ['a_{kk} = vld1q_f32(a + {kk} * step + {i} * SIMD);\\'.format(kk=kk, i=i)]

                    for jj in range(C_ob1//SIMD):

                        if kk == 0: # load b just before use
                            # s += ['b_{jj} = vld1q_f32(b + ({i} * SIMD + {ii})*C_ob + ({j} * {C_ob1} + {jj})*SIMD);\\'.format(i=i, ii=ii, j=j, C_ob1=C_ob1//SIMD, jj=jj)]
                            s += ['b_{jj} = vld1q_f32(bb + {ii}*C_ob + ({j} * {C_ob1} + {jj})*SIMD);\\'.format(ii=ii, j=j, C_ob1=C_ob1//SIMD, jj=jj)]

                        # s += ['{c} = vfmaq_laneq_f32({c}, b_{jj}, a_{kk}, {ii});\\'.format(c=c_tile[kk][j * (C_ob1//SIMD) + jj], kk=kk, jj=jj, ii=ii)]
                        # s += ['__asm__ volatile("fmla %[c].4s, %[b].4s, %[a].s[{ii}]\\n\\t" : [c] "+w"({c}) : [b] "w"(b_{jj}), [a] "w"(a_{kk}));'.format(
                        s += ['__asm__ volatile ("fmla %0.4s, %1.4s, %2.s[{ii}]" : "+w"({c}) : "w"(b_{jj}), "w"(a_{kk}));\\'.format(
                        # s += ['__asm__ ("fmla %0.4s, %1.4s, %2.s[{ii}]" : "+w"({c}) : "w"(b_{jj}), "w"(a_{kk}));\\'.format(
                            c=c_tile[kk][j * (C_ob1//SIMD) + jj], kk=kk, jj=jj, ii=ii
                        )]

                        # s += ['{c} = fma_reg_broadcast({c}, b_{jj}, a_{kk}, {ii});\\'.format(c=c_tile[kk][j * (C_ob1//SIMD) + jj], kk=kk, jj=jj, ii=ii)]
        s += ['bb += {};\\'.format(SIMD * C_ob)]
        # s += ['aa += \\']

    s += ['']

    #

    # max pooling / relu
    s += redefine('MAX_TILE_C')
    s += ['#define MAX_TILE_C(step, a, W_ob, C_ob)\\']
    # compute
    s += ['float32x4_t av; \\']
    for kk in range(W_ob):
        for jj in range(C_ob//SIMD):
            s += ['av = vld1q_f32(a + {k} * step + {j} * SIMD);\\'.format(k=kk, j=jj)]
            s += ['{c} = vmaxq_f32({c}, av);\\'.format(c=c_tile[kk][jj], k=kk, j=jj)]
    s += ['']

    # depthwise
    s += redefine('DW_TILE_C')
    s += ['#define DW_TILE_C(step, a, b, W_ob, C_ob)\\']
    s += ['float32x4_t av; \\']
    # load B
    for jj in range(C_ob//SIMD):
        s += ['float32x4_t b_{j} = vld1q_f32(b + {j}*SIMD);\\'.format(j=jj)]
    # compute
    for kk in range(W_ob):
        for jj in range(C_ob//SIMD):
            s += ['av = vld1q_f32(a + {k} * step + {j} * SIMD);\\'.format(k=kk, j=jj)]
            s += ['{c} = vfmaq_f32({c}, av, b_{j});\\'.format(c=c_tile[kk][jj], j=jj)]
    s += ['']



    # to fix backslash at end of file warning
    s += ['']

    f.write('\n'.join(s))
