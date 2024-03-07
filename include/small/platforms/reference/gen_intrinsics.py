FLOAT_W_ob = 6
FLOAT_C_ob = 16
FLOAT_SIMD = 4
FLOAT_UNROLL = 4
FLOAT_C_ib = FLOAT_C_ob


def gen_def_tile():
    print("#define FLOAT_DEF_TILE_C(W_ob, C_ob)\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_SIMD):
            # TODO should this value be FLOAT_UNROLL instead?
            print(f"float32x{FLOAT_SIMD}_t c_{i}_{j};\\")


def gen_zero_tile():
    print("#define FLOAT_DEF_TILE_C(W_ob, C_ob)\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_SIMD):
            # TODO should this value be FLOAT_UNROLL instead?
            print(f"c_{i}_{j} = vdupq_n_f32(0);\\")


def gen_load_tile(strided=False):
    print("#define FLOAT_LOAD_TILE_C(O, W_ob, C_ob)\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_SIMD):
            # TODO should this value be FLOAT_UNROLL instead?
            if strided:
                print(f"c_{i}_{j} = vld1q_f32(O + {i} * step + {j} * FLOAT_SIMD);\\")
            else:
                print(f"c_{i}_{j} = vld1q_f32(O + {i} * C_ob + {j} * FLOAT_SIMD);\\")


def gen_store_tile():
    print("#define FLOAT_STORE_TILE_C(O, W_ob, C_ob)\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_SIMD):
            # TODO should this value be FLOAT_UNROLL instead?
            print("vst1q_f32(O + {i} * C_ob + {j} * FLOAT_SIMD, c_{i}_{j});\\")


def gen_conv_tile_refresh_col_major_b_reg():
    """Uses one register for values in the B matrix.
    Refreshes thsoe values iterating through rows in B first, then cols."""
    print("#define FLOAT_CONV_TILE_C(W_stride, a, b, W_ob, C_ob)\\")

    # declaring registers for matrix A
    for i in range(FLOAT_W_ob):
        print(f"float32x4_t a_{i};\\")

    # declaring registers for matrix B
    # (only using one register for matrix B)
    for i in range(1):
        print(f"float32x4_t b_{i};\\")

    # TODO difference between vld1q_dup_f32 and vld1q_f32?

    # filling A registers
    for i in range(FLOAT_W_ob):
        print(f"a_{i} = vld1q_dup_f32(a + {i} * W_stride);\\")

    for b_col in range(FLOAT_C_ob):
        # this loop controls which col of matrix B gets put into B's register upon refresh
        for b_row in range(FLOAT_UNROLL):
            # refresh B register to be the next row at the beginning of each iteration of this loop
            print(f"b_{0} = vld1q_f32(b + {b_row} * C_ob + {b_col} * FLOAT_SIMD);\\")

            for a_col in range(FLOAT_SIMD):
                # iterating through "cols" in a portion of matrix A.  aka iterating through indices in A's registers
                # TODO what if the number of cols that A has (FLOAT_UNROLL) is > FLOAT_SIMD?
                for a_row in range(FLOAT_W_ob):
                    # iterating through "rows" in matrix A, aka iterating through A's registers (sticking with the same column)
                    print(
                        f'__asm__ volatile("fmla c_{a_col}_{a_row}, a_{a_row}[{a_col}], b_{0}");\\'
                    )
                    # TODO "fmla [output register] [input regsiter] [input broadcast with index]"
                    # this is not correct assembly syntax at all


def gen_conv_tile_refresh_row_major_b_reg():
    """Uses one register for values in the B matrix.
    Refreshes thsoe values iterating through cols in B first, then rows."""
    print("#define FLOAT_CONV_TILE_C(W_stride, a, b, W_ob, C_ob)\\")

    # declaring registers for matrix A
    for i in range(FLOAT_W_ob):
        print(f"float32x4_t a_{i};\\")

    # declaring registers for matrix B
    # (only using one register for matrix B)
    for i in range(1):
        print(f"float32x4_t b_{i};\\")

    # TODO difference between vld1q_dup_f32 and vld1q_f32?

    # filling A registers
    for i in range(FLOAT_W_ob):
        print(f"a_{i} = vld1q_dup_f32(a + {i} * W_stride);\\")

    for b_row in range(FLOAT_UNROLL):
        # this loop controls which col of matrix B gets put into B's register upon refresh
        # refresh B register to be the next row at the beginning of each iteration of this loop
        for b_col in range(FLOAT_C_ob):
            print(f"b_{0} = vld1q_f32(b + {b_row} * C_ob + {b_col} * FLOAT_SIMD);\\")

            for a_col in range(FLOAT_SIMD):
                # iterating through "cols" in a portion of matrix A.  aka iterating through indices in A's registers
                # TODO what if the number of cols that A has (FLOAT_UNROLL) is > FLOAT_SIMD?
                for a_row in range(FLOAT_W_ob):
                    # iterating through "rows" in matrix A, aka iterating through A's registers (sticking with the same column)
                    print(
                        f'__asm__ volatile("fmla c_{a_col}_{a_row}, a_{a_row}[{a_col}], b_{0}");\\'
                    )
                    # TODO "fmla [output register] [input regsiter] [input broadcast with index]"
                    # this is not correct assembly syntax at all


"""
def gen_conv_tile_multiple_b_reg():
"""
