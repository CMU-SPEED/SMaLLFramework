FLOAT_W_ob = 6
FLOAT_C_ob = 16
FLOAT_SIMD = 4
FLOAT_UNROLL = 4
FLOAT_C_ib = FLOAT_C_ob


def gen_def_tile():
    print("#define FLOAT_DEF_TILE_C(W_ob, C_ob)\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_SIMD):
            print(f"float32x{FLOAT_SIMD}_t c_{i}_{j};\\")


def gen_zero_tile():
    print("#define FLOAT_DEF_TILE_C(W_ob, C_ob)\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_SIMD):
            print(f"c_{i}_{j} = vdupq_n_f32(0);\\")


def gen_load_tile(strided=False):
    print("#define FLOAT_LOAD_TILE_C(O, W_ob, C_ob)\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_SIMD):
            if strided:
                print(f"c_{i}_{j} = vld1q_f32(O + {i} * step + {j} * FLOAT_SIMD);\\")
            else:
                print(f"c_{i}_{j} = vld1q_f32(O + {i} * C_ob + {j} * FLOAT_SIMD);\\")


def gen_store_tile():
    print("#define FLOAT_STORE_TILE_C(O, W_ob, C_ob)\\")

    for i in range(FLOAT_W_ob):
        for j in range(FLOAT_SIMD):
            print("vst1q_f32(O + {i} * C_ob + {j} * FLOAT_SIMD, c_{i}_{j});\\")


def gen_conv_tile_refresh_b_reg():
    print("#define FLOAT_CONV_TILE_C(stride, a, b, W_ob, C_ob)\\")

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
        print(f"a_{i} = vld1q_dup_f32(a + {i} * stride);\\")

    # filling B register
    for i in range(1):
        print(f"b_{i} = vld1q_f32(bb + {i} * C_ob);\\")

    # accumulate math in c_tile's registers, refresh B register


"""
def gen_conv_tile_multiple_b_reg():
"""
