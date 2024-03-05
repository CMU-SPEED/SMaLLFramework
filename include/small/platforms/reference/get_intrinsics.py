def gen_ewise_macro(
    method_signature,
    W_stride,
    C_stride,
    H_k,
    W_k,
    K_k,
    G_k,
    operator,
    input_ptr=None,
    output_ptr=None,
):
    if input_ptr == None and output_ptr == None:
        print("Error: input and output pointers can't both be None.")
        return

    W_ob = W_k
    C_ob = K_k * G_k
    step = W_stride

    print(f"#define {method_signature}\\")
    print(f"for (uint32_t kk = 0; kk < W_ob; kk++)\\")
    print("{\\")
    print("for (uint32_t jj = 0; jj < C_ob; jj++)\\")
    print("{\\")
    print(
        f"c_tile[kk * C_ob + jj] {operator} static_cast<c_tile_t> I[kk * step + jj * C_stride] + offset;\\"
    )
    print("}\\")
    print("}\\")
