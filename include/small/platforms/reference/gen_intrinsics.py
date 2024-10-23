FLOAT_W_ob = 4
FLOAT_C_ob = 16
FLOAT_SIMD = 4
FLOAT_UNROLL = 4
FLOAT_C_ib = FLOAT_C_ob
B_NUM_REGS = 2


DATA_TYPE = f"float32x{FLOAT_C_ob // FLOAT_SIMD}_t"


def gen_def_tile():
    # print("#define FLOAT_DEF_TILE_C(W_ob, C_ob)\\")
    print(f"if constexpr (W_ob == {FLOAT_W_ob} && C_ob == FLOAT_C_ob){{\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_C_ob // FLOAT_SIMD):
            print(f"{DATA_TYPE} c_{i}_{j};\\")

    print("}")


def gen_zero_tile():
    # print("#define FLOAT_DEF_TILE_C(W_ob, C_ob)\\")
    print(f"if constexpr (W_ob == {FLOAT_W_ob} && C_ob == FLOAT_C_ob){{\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_C_ob // FLOAT_SIMD):
            print(f"c_{i}_{j} = vdupq_n_f32(0);\\")

    print("}")


def gen_load_tile(strided=False):
    # print("#define FLOAT_LOAD_TILE_C(O, W_ob, C_ob)\\")
    print(f"if constexpr (W_ob == {FLOAT_W_ob} && C_ob == FLOAT_C_ob){{\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_C_ob // FLOAT_SIMD):
            if strided:
                print(f"c_{i}_{j} = vld1q_f32(O + {i} * step + {j} * FLOAT_SIMD);\\")
            else:
                print(f"c_{i}_{j} = vld1q_f32(O + {i} * C_ob + {j} * FLOAT_SIMD);\\")

    print("}")


def gen_store_tile():
    print("#define FLOAT_STORE_TILE_C(O, W_ob, C_ob)\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_C_ob // FLOAT_SIMD):
            print("vst1q_f32(O + {i} * C_ob + {j} * FLOAT_SIMD, c_{i}_{j});\\")


def gen_conv_tile_refresh_row_major_b_reg():
    """Uses one register for values in the B matrix.
    Refreshes thsoe values iterating through rows in B first, then cols."""
    # print("#define FLOAT_CONV_TILE_C(W_stride, a, b, W_ob, C_ob)\\")
    print(f"if constexpr (_UNROLL == {FLOAT_UNROLL} && W_ob == {FLOAT_W_ob} && C_ob == FLOAT_C_ob) {{\\")

    # declaring registers for matrix A
    for i in range(FLOAT_W_ob):
        print(f"{DATA_TYPE} a_{i};\\")

    # declaring registers for matrix B
    # (only using one register for matrix B)
    for i in range(B_NUM_REGS):
        print(f"{DATA_TYPE} b_{i};\\")

    for u in range(0, FLOAT_UNROLL, FLOAT_SIMD):
        # this loop controls which chunk of SIMD-number of columns of A get loaded into A's regs
        # this loop executes FLOAT_UNROLL / FLOAT_SIMD times

        # refresh A's registers (stepping row-wise)
        for i in range(FLOAT_W_ob):
            print(f"a_{i} = vld1q_f32(a + {i} * W_stride + {u * FLOAT_SIMD});\\")

        for uu in range(FLOAT_SIMD):
            # iterating through "cols" in a portion of matrix A's regs.  aka iterating through indices in A's registers
            # similarly, this also dictates which row of matrix B is in B's regs.

            for b_reg_col in range(0, FLOAT_C_ob, (FLOAT_SIMD * B_NUM_REGS)):
                # refresh B register to be the next set of SIMD number of cols
                for i in range(B_NUM_REGS):
                    print(
                        f"b_{i} = vld1q_f32(b + {u + uu} * C_ob + {b_reg_col} + {i} * FLOAT_SIMD);\\"
                    )

                for a_row in range(FLOAT_W_ob):
                    # iterating through one "col" in each of the "rows" in matrix A
                    # aka iterating through A's registers (sticking with the same column)
                    for b_reg in range(B_NUM_REGS):
                        print(
                            f'__asm__ volatile("fmla %0.4s, %1.4s, %2.s[{uu}]" : "+w"(c_{a_row}_{(b_reg_col // FLOAT_SIMD) + b_reg}) : "w"(b_{b_reg}), "w"(a_{a_row}));\\'
                        )
                    # TODO parameterize this assembly string for multiple architectures

    print("}")

def gen_max_tile():
    print(f"if constexpr (W_ob == {FLOAT_W_ob} && C_ob == FLOAT_C_ob){{\\")

    print(f"{DATA_TYPE} av;\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_C_ob // FLOAT_SIMD):
            print(f"av = vld1q_f32(a + {i} * step + {j} * FLOAT_SIMD);\\")
            print(f"c_{i}_{j} = vmaxq_f32(c_{i}_{j}, av);\\")

    print("}")

def gen_dw_tile():
    print(f"if constexpr (W_ob == {FLOAT_W_ob} && C_ob == FLOAT_C_ob){{\\")

    print(f"{DATA_TYPE} av;\\")

    for i in range(FLOAT_C_ob // FLOAT_SIMD):
        print(f"{DATA_TYPE} b_{i} = vld1q_f32(b + {i} * FLOAT_SIMD);\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_C_ob // FLOAT_SIMD):
            print(f"av = vld1q_f32(a + {i} * step + {j} * FLOAT_SIMD);\\")
            print(f"c_{i}_{j} = vfmaq_f32(c_{i}_{j}, av, b_{j});\\")

    print("}")


"""
def gen_conv_tile_refresh_col_major_b_reg():
def gen_conv_tile_multiple_b_reg():
"""

# gen_def_tile()
# gen_zero_tile()
# gen_load_tile()
# gen_store_tile()
# gen_conv_tile_refresh_row_major_b_reg()
# gen_dw_tile()
# gen_max_tile()
gen_load_tile(strided = True)
