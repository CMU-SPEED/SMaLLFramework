# SMaLL, Software for Machine Learning Libraries
# Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# For additional details (including references to third party source code and
# other files) see the LICENSE file or contact permission@sei.cmu.edu. See
# Contributors.txt for a full list of contributors. Created, in part, with
# funding and support from the U.S. Government (see Acknowledgments.txt file).
# DM23-0126

# Platforrm specific parameters
#make this is its own file, import based on runtime platform parameters
platform_name="arm"
W_ob = 6
C_ob2 = 1
C_ob1 = 16
C_ob = C_ob2 * C_ob1
SIMD = 8
UNROLL = 1 # only affects conv kernel.
# UNROLL = SIMD # only affects conv kernel.

NUM_FMA = 2
NUM_MAX = 1
NUM_LOAD = 2
NUM_STORE = 1

#ISA setup
simd_intrin_lib_path="arm_neon.h"
simd_reg_typename="float32x4_t"
simd_load_func="vld1q_f32"
simd_store_func="vst1q_f32"

simd_fma_func="vfmaq_f32"
simd_max_func="vmaxq_f32"
simd_div_func="vdivq_f32"
simd_add_func="vaddq_f32"
#end ISA setup


#path to put the generated code
#get current directory
import os
cur_dir=os.getcwd()
path_to_gen_code=cur_dir+"/platforms/"+platform_name
print(path_to_gen_code)
#end platform specific parameters

#There should be a generalization of the kernel generation based on the op_class, whether it's binary, unary, etc.
#Ideally, given the instruction sequence for a single operation and the op_class, the kernel can be generated.
matmul_alg="in register broadcast"

with open('{:}/params_temp.h'.format(path_to_gen_code), 'w') as f:
    f.write(
        '''
//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126
//****************************************************************************

#pragma once

#define SMALL_HAS_FLOAT_SUPPORT  1

#define FLOAT_W_ob {W_ob}
#define FLOAT_C_ob {C_ob}
#define FLOAT_SIMD {SIMD}
#define FLOAT_UNROLL {UNROLL}
#define FLOAT_C_ib FLOAT_C_ob

// not used for kernels, but used in throughput calculation.
#define NUM_FMA {NUM_FMA}
#define NUM_MAX {NUM_MAX}
#define NUM_LOAD {NUM_LOAD}
#define NUM_STORE {NUM_STORE}
        '''.format(**locals())
    )

def redefine(name):
    return ['#ifdef {n}\n#undef {n}\n#endif\n'.format(n=name)]

with open('{:}/intrinsics-gen_temp.h'.format(path_to_gen_code), 'w') as f:
    s = ['''
//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126
//****************************************************************************

#pragma once

#define SMALL_HAS_FLOAT_SUPPORT  1
''']
    s += ['#include <{}>'.format(simd_intrin_lib_path)]
    
    # define tile
    # names of variables
    s += redefine('FLOAT_DEF_TILE_C')
    s += ['#define FLOAT_DEF_TILE_C(W_ob, C_ob)\\']
    c_tile = [["c_{}_{}".format(kk, jj) for jj in range(C_ob//SIMD)] for kk in range(W_ob)]
    s += ['float c_tile[W_ob * C_ob];\\']
    for kk in range(W_ob):
        for jj in range(C_ob//SIMD):
            s += ['{} {};\\'.format(simd_reg_typename, c_tile[kk][jj])]
    s += ['']


    # zero tile
    s += redefine('FLOAT_ZERO_TILE_C')
    s += ['#define FLOAT_ZERO_TILE_C(W_ob, C_ob)\\']
    for kk in range(W_ob):
        for jj in range(C_ob//SIMD):
            s += ['{} = vdupq_n_f32(0);\\'.format(c_tile[kk][jj])]
    s += ['']

    # load tile
    s += redefine('FLOAT_LOAD_TILE_C')
    s += ['#define FLOAT_LOAD_TILE_C(O, W_ob, C_ob)\\']
    for kk in range(W_ob):
        for jj in range(C_ob//SIMD):
            s += ['{c} = vld1q_f32(O + {k} * C_ob + {j} * SIMD);\\'.format(c=c_tile[kk][jj], k=kk, j=jj)]
    s += ['']

    # load tile strided
    s += redefine('FLOAT_LOAD_TILE_C_strided')
    s += ['#define FLOAT_LOAD_TILE_C_strided(O, step, W_ob, C_ob)\\']
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
        s += ['{} a_{kk};\\'.format(simd_reg_typename, kk=kk)]
    # define b [half as many]
    for jj in range(C_ob1//SIMD):
        s += ['{} b_{jj};\\'.format(simd_reg_typename, jj=jj)]

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
    s += ['{} av; \\'.format(simd_reg_typename)]
    for kk in range(W_ob):
        for jj in range(C_ob//SIMD):
            s += ['av = vld1q_f32(a + {k} * step + {j} * SIMD);\\'.format(k=kk, j=jj)]
            s += ['{c} = vmaxq_f32({c}, av);\\'.format(c=c_tile[kk][jj], k=kk, j=jj)]
    s += ['']

    # depthwise
    s += redefine('DW_TILE_C')
    s += ['#define DW_TILE_C(step, a, b, W_ob, C_ob)\\']
    s += ['{} av; \\'.format(simd_reg_typename)]
    # load B
    for jj in range(C_ob//SIMD):
        s += ['{} b_{j} = vld1q_f32(b + {j}*SIMD);\\'.format(simd_reg_typename, j=jj)]
    # compute
    for kk in range(W_ob):
        for jj in range(C_ob//SIMD):
            s += ['av = vld1q_f32(a + {k} * step + {j} * SIMD);\\'.format(k=kk, j=jj)]
            s += ['{c} = vfmaq_f32({c}, av, b_{j});\\'.format(c=c_tile[kk][jj], j=jj)]
    s += ['']



    # to fix backslash at end of file warning
    s += ['']

    f.write('\n'.join(s))
