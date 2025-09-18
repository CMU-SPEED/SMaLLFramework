import os
import sys


"""
#elementwise operations can be written as follows:
for i in range(W_ob):
    for j in range(C_ob):
    c[i][j] = (a[i][j], b[i][j]) 


depending on the input shape of a and b, there is rank promotion
Assumptions:
rank promotion for a is within the kernel
b is optional
rank promotion for b is performed by loading c[i][j] appropriately*

Assumes the variables are all defined
Assumes the appropriate LOAD and STORE phase will be called
Will generate the Compute phase with the right dataflow


"""

#platform specific setup
#@todo: This could be an input
W_ob = 6
C_ob = 16


vector_instruction_table={ 
"load":"_mm256_load_ps({ptr})",
"store": "_mm256_store_ps({ptr}, {reg})",
"broadcast": "_mm256_broadcast_ss({ptr})",
"zero": "_mm256_setzero_ps",
"add": "_mm256_add_ps",
"sub": "_mm256_sub_ps",
"mul" :"_mm256_mul_ps",
"min": "_mm256_min_ps",
"max": "_mm256_max_ps",
"and": "_mm256_and_ps",
"cmp": "_mm256_cmp_ps",
"fused-multipy-add" :"_mm256_fmadd_ps",
"fused multiply-add/sub": "_mm256_fmaddsub_ps",
"vector_type": "__m256",
"scalar_type": "float",
"REGS": 16,
"VECTOR_BITS": 256,
"ELEMENT_BITS": 32,
 "SIMD":8
}

c_regs = W_ob*C_ob//(vector_instruction_table["SIMD"])
assert((c_regs) < vector_instruction_table["REGS"])
a_regs = vector_instruction_table["REGS"] - c_regs

#operation set up
shape_a = sys.argv[1].split(":")[1].split(",") #should be of the form a:w,c
shape_c = [W_ob, C_ob]
vector_instruction = sys.argv[2]
name = sys.argv[3]
shape_b = sys.argv[4].split(":")[1].split(",") if len(sys.argv) > 4 else None


print(shape_a, shape_b)

def redefine(name):
    return ['#ifdef {n}\n#undef {n}\n#endif\n'.format(n=name)]


s = []

# define tile
# load c (various initiliazations)
# store c

a_load = " "
#if we need to rank promote the channels, load of a is a broadcast
#allocate the remaining a_regs registers to the kk and jj loops in the kernel
if int(shape_a[1]) == 1:
    a_load = vector_instruction_table["broadcast"]
    a_regs_jj = 1
    if int(shape_a[0]) == 1:
        a_regs_kk = 1
    else:
        a_regs_kk = a_regs
    
else:
    a_load = vector_instruction_table["load"]
    # 1 register per simd width in the channel dimension
    a_regs_jj = C_ob//vector_instruction_table["SIMD"]
    if int(shape_a[0]) == 1:
        a_regs_kk = 1
    else:
        a_regs_kk = a_regs//a_regs_jj
    

s += [f'#define FLOAT_EWISE_{name}_TILE_C(step, a, W_ob, C_ob)\\']
# compute
s += [vector_instruction_table["vector_type"] + ",".join([f" a_{jj}" for jj in range(a_regs_jj*a_regs_kk)]) + "; \\"]
#if a is reused over the width elements, load can be performed outside
if int(shape_a[0])==1:
    for jj in range(a_regs_jj):
        s += [f"a_{jj} = " + a_load.format(ptr=(f"a + {jj} * SIMD")) + ";\\"]
for kk in range(W_ob):
    #if a is not reused it must be reloaded
    if int(shape_a[0])>1 :
        for jj in range(a_regs_jj):
            s += [f"a_{(kk%a_regs_kk)*(a_regs_jj)+jj} = " + a_load.format(ptr=(f"a + {kk}*step + {jj} * SIMD")) + ";\\"]
    for jj in range(C_ob//vector_instruction_table["SIMD"]):
        if not (shape_b == None):
            s += ['c_{k}_{j} = {vop}(c_{k}_{j}, a_{ak});\\'.format(vop = vector_instruction, k=kk, j=jj, ak=((kk%a_regs_kk)*(a_regs_jj)+(jj%a_regs_jj)))]
        else:
            s += ['c_{k}_{j} = {vop}(a_{kj});\\'.format(vop = vector_instruction, k=kk, j=jj, kj=((kk%a_regs_kk)*(a_regs_jj)+(jj%a_regs_jj)))]
s += ['']


c_regs_jj = C_ob//vector_instruction_table["SIMD"]
s += [f'#define FLOAT_EWISE_{name}_END_C(step, a, W_ob, C_ob)\\']
# compute
s += [vector_instruction_table["vector_type"] + ",".join([f" a_{jj}" for jj in range(a_regs)]) + "; \\"]
#if a is reused over the width elements, load can be performed outside
if int(shape_a[0])==1:
    for jj in range(a_regs_jj):
        s += [f"a_{jj} = " + a_load.format(ptr=(f"a + {jj} * SIMD")) + ";\\"]
s+=["for(int kk = 0; kk<W_ob; kk++){\\"]
if int(shape_a[0])>1 :
    for jj in range(a_regs_jj):
        s += [f"a_{jj} = " + a_load.format(ptr=(f"a + kk*step + {jj} * SIMD")) + ";\\"]
for jj in range(C_ob//vector_instruction_table["SIMD"]):
    if not (shape_b == None):
        s += ['c_tile[kk * (C_ob/SIMD)+ {j}] = {vop}(c_tile[kk * (C_ob/SIMD))+ {j}], a_{kj});\\'.format(vop = vector_instruction, j=jj, kj=(jj%a_regs_jj))]
    else:
        s += ['c_tile[kk * (C_ob/SIMD)+ {j}] = {vop}(a_{kj});\\'.format(vop = vector_instruction, j=jj, kj=(jj%a_regs_jj))]
s+=["}"]
s += ['']
print("\n".join(s))




