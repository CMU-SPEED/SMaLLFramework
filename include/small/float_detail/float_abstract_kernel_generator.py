import argparse
import sys
from collections import OrderedDict
from typing import List, Optional

"""
#elementwise operations can be written as follows:
for i in range(W_ob):
    for j in range(C_ob):
    c[i][j] = vop(a[i][j], b[0][j], c[i][j])


depending on the input shape of a and b, there is rank promotion
Assumptions:
rank promotion for a is within the kernel
b is optional, only rank promoted along the C_ob dimension

c is assumed to be W_ob x C_ob

Assumes the variables are all defined
Assumes the appropriate LOAD and STORE phase will be called
Will generate the Compute phase with the right dataflow


"""

# platform specific setup


vector_instruction_table = {
    # memory operations
    "load": "_mm256_load_ps({ptr})",
    "store": "_mm256_store_ps({ptr}, {reg})",
    "broadcast": "_mm256_broadcast_ss({ptr})",
    # 0 input operands
    "zero": "_mm256_setzero_ps()",
    # 1 input operand
    "assign": "{reg1}",
    # binary operands
    "add": "_mm256_add_ps({reg1}, {reg2})",
    "sub": "_mm256_sub_ps({reg1}, {reg2})",
    "mul": "_mm256_mul_ps({reg1}, {reg2})",
    "min": "_mm256_min_ps({reg1}, {reg2})",
    "div" :"_mm256_div_ps({reg1}, {reg2})",
    "max": "_mm256_max_ps({reg1}, {reg2})",
    "and": "_mm256_and_ps({reg1}, {reg2})",
    "gt": "_mm256_cmp_ps({reg1}, {reg2}, _CMP_GT_OQ)",
    "lt": "_mm256_cmp_ps({reg1}, {reg2}, _CMP_LT_OQ)",
    "ge": "_mm256_cmp_ps({reg1}, {reg2}, _CMP_GE_OQ)",
    "le": "_mm256_cmp_ps({reg1}, {reg2}, _CMP_LE_OQ)",
    "eq": "_mm256_cmp_ps({reg1}, {reg2}, _CMP_EQ_OQ)",
    # ternary operands
    "fma": "_mm256_fmadd_ps({reg1}, {reg2}, {reg3})",
    "fmaddsub": "_mm256_fmaddsub_ps({reg1}, {reg2}, {reg3})",
    # architecture specific details
    "vector_type": "__m256",
    "scalar_type": "float",
    "REGS": 16,
    "VECTOR_BITS": 256,
    "ELEMENT_BITS": 32,
    "SIMD": 8,
}

# Mapping from operator to number of operands
operator_operand_count = {
    "zero": 0,
    "store": 1,
    "assign": 1,
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "min": 2,
    "max": 2,
    "and": 2,
    "gt": 2,
    "lt": 2,
    "ge": 2,
    "le": 2,
    "eq": 2,
    "fma": 3,
    "fmaddsub": 3,
}

def generate_kernel(tile_size: str, op: str, name: str, operands: Optional[List[str]] = None) -> str:
    """
    Generates the microkernel for the given operation, operands, and tile size.

    Input:
    - tile_size: str,
        The output tile size, of the form "WxC" where W (int) is the width dimension and C (int) is the channel dimension.
        C must be a positive integer multiple of the SIMD width specified in `vector_instruction_table["SIMD"]`.
    - op: str,
        The primitive operation, must be a key in `operator_operand_count`.
    - name: str,
        The name of the generated kernel, preferably meaningful and in all caps.
    - operands: List[str],
        A list specifying the operands and their shapes, each of the form "<identifier>:<shape>".
        The number of operands specified must match the number of operands required for the selected operation as specified in `operator_operand_count`.
        The operand identifiers must be unique and one of "a", "b", "c".
        The operand shapes must be of the form "W,C" where W (int) is the width dimension and C (int) is the channel dimension.
        - Identifier "a":
            - Can only be used as an input operand and can be of any of the following shapes:
                - W,C (no rank promotion)
                - 1,C (rank promotion across width dimension)
                - W,1 (rank promotion across channel dimension)
                - 1,1 (rank promotion across both dimensions)
            - For the "store" operation, "a" is the base destination pointer and must
              be of shape 1,1 (scalar). The tile data to store is always taken from the
              c registers; striding into the destination is handled via kk*step + jj*SIMD.
        - Identifier "b":
            - Can only be used as an input operand and can only be of the following shapes:
                - 1,C (rank promotion across width dimension)
                - 1,1 (rank promotion across both dimensions)
        - Identifier "c":
            - Can be used as an input operand, in which case it can be of any of the following shapes:
                - W,C (no rank promotion)
                - 1,C (rank promotion across width dimension)
                - W,1 (rank promotion across channel dimension)
                - 1,1 (rank promotion across both dimensions)
            - Is always used as the output operand with the same shape as "tile_size".
                If "c" is not specified as an input operand, it is assumed to be of the same shape as "tile_size".
                If "c" is specified as an input operand, the output operand will always shadow the input operand in terms of both name and shape.
    """

    W_ob = int(tile_size.split("x")[0])
    C_ob = int(tile_size.split("x")[1])
    c_regs = W_ob * C_ob // (vector_instruction_table["SIMD"])
    assert (c_regs) < vector_instruction_table["REGS"]

    # operation set up
    assert op in vector_instruction_table.keys()
    vector_instruction = vector_instruction_table[op]
    num_operands = operator_operand_count[op]
    is_store = (op == "store")

    # operand shapes
    # operands can appear in any order after the first 3 arguments a, b, c
    order_of_operands = OrderedDict()
    i = 0
    if operands != None:
        for arg in operands:
            if not (arg.startswith("a:") or arg.startswith("b:") or arg.startswith("c:")):
                raise ValueError(
                    f"Encountered invalid operand shape argument: {arg}. Operand shape arguments should be of the form a:w,c or b:w,c or c:w,c"
                )
            else:
                if i >= num_operands:
                    break
                if arg.startswith("a:"):
                    shape_a = arg.split(":")[1].split(",")  # should be of the form a:w,c
                elif arg.startswith("b:"):
                    shape_b = arg.split(":")[1].split(",")  # should be of the form b:w,c
                elif arg.startswith("c:"):
                    shape_c_input = arg.split(":")[1].split(",")  # should be of the form c:w,c
                order_of_operands[arg[0]] = arg.split(":")[1].split(",")
                i += 1

        if len(operands) - 4 > num_operands:
            print("Too many operand shape arguments provided for the selected operation")
            print("Ignoring arguments after {:}".format(sys.argv[4 + num_operands]))

    # insert default shapes if not provided
    print(order_of_operands.keys())
    if "a" not in order_of_operands.keys():
        shape_a = None
        order_of_operands["a"] = shape_a
    if "b" not in order_of_operands.keys():
        shape_b = None
        order_of_operands["b"] = shape_b
    if "c" not in order_of_operands.keys():
        shape_c_input = None
        print("no input c")
        order_of_operands["c"] = shape_c_input
    shape_c = [W_ob, C_ob]

    # store-specific validation: the base address operand must be scalar (1,1)
    if op == "store":
        if shape_a is None:
            raise ValueError(
                "store requires a base address operand, e.g. a:1,1"
            )
        if int(shape_a[0]) != 1 or int(shape_a[1]) != 1:
            raise ValueError(
                f"store: base address operand 'a' must be scalar (1,1), got {shape_a[0]},{shape_a[1]}. "
                "The destination pointer is a single base address; striding is handled internally."
            )

    operand_order = list(order_of_operands.keys())
    print(operand_order)
    print(shape_a, shape_b, shape_c_input)


    def redefine(name):
        return ["#ifdef {n}\n#undef {n}\n#endif\n".format(n=name)]



# if shape_c_input != None:
#     #If shape_c_input is not the same as W_ob, C_ob then we need to call the appropriate LOAD_FUNCTION_for C
#     l = []
#     if int(shape_c_input[1]) == 1:
#         c_load = vector_instruction_table["broadcast"]
#         type_c ="VECTOR"
#         if int(shape_c_input[0]) == 1:
#             type_c ="SCALAR"

#     else:
#         c_load = vector_instruction_table["load"]
#         type_c ="VECTOR_T"
#         if int(shape_c_input[0])!= 1:
#             type_c ="MATRIX"

#     l += [f'#define FLOAT_LOAD_{type_c}_TILE_C(I, step)\\']
#     for kk in range(W_ob):
#         for jj in range(C_ob//vector_instruction_table["SIMD"]):
#             l += [f'c_{kk}_{jj} = ' + c_load.format(ptr=(f"c + {kk%(int(shape_c_input[0]))}*step + {jj%(int(shape_c_input[1]))} * SIMD")) + ";\\"]

#   print("\n".join(l))


    # define tile


    a_load = " "
    # if we need to rank promote the channels, load of a is a broadcast
    # allocate the remaining a_regs registers to the kk and jj loops in the kernel

    # ----------------------------------------
    # Register Allocation
    # ----------------------------------------
    # sort out registers needed for b

    b_regs_jj_kk = 0
    type_b = "ACCUM"
    if shape_b != None:
        assert int(shape_b[0]) == 1  # rank promotion only along C_ob
        b_regs = C_ob // vector_instruction_table["SIMD"]
        if int(shape_b[1]) == 1:
            b_load = vector_instruction_table["broadcast"]
            b_regs_jj_kk = 1
            type_b = "SCALAR"
        else:
            b_load = vector_instruction_table["load"]
            b_regs_jj_kk = b_regs
            type_b = "VECTOR_T"

        assert b_regs_jj_kk + c_regs <= vector_instruction_table["REGS"]

#sort out registers needed for a
    a_regs = vector_instruction_table["REGS"] - b_regs_jj_kk - c_regs
    if num_operands > 0:
        if int(shape_a[1]) == 1:
            a_load = vector_instruction_table["broadcast"]
            a_regs_jj = 1
            type_a ="VECTOR"
            if int(shape_a[0]) == 1:
                a_regs_kk = 1
                type_a ="SCALAR"
            else:
                a_regs_kk = a_regs

        else:
            a_load = vector_instruction_table["load"]
            # 1 register per simd width in the channel dimension
            a_regs_jj = C_ob//vector_instruction_table["SIMD"]
            type_a ="VECTOR_T"
            if int(shape_a[0]) == 1:
                a_regs_kk = 1
            else:
                a_regs_kk = (a_regs//a_regs_jj)
                type_a ="MATRIX"

        a_regs_jj_kk = a_regs_jj * a_regs_kk
    else:
        a_regs_jj = 0
        a_regs_kk = 0
        a_regs_jj_kk = 0

    print()

    # ----------------------------------------
    # Compute Phase Name and Declaration
    # ----------------------------------------
    s = []
    s = []
    if num_operands == 0:
        s += [f'#define FLOAT_{name}_TILE_C\\']
    elif num_operands == 1:
        if op == "store":
            # store: c holds the tile data, a is the scalar base destination pointer
            s += [f'#define FLOAT_{name}_TILE_C(step, a)\\']
        else:
            s += [f'#define FLOAT_{name}_{type_a}_TILE_C(step, a)\\']
    elif num_operands == 2:
        s += [f'#define FLOAT_{name}_{type_a}_{type_b}_TILE_C(step, a']
        if shape_b != None:
            s[-1] += ", b"
    s[-1] += ')\\'
# compute

    # ----------------------------------------
    # Inputer Register Declarations
    # ----------------------------------------
    vector_type_name = vector_instruction_table["vector_type"]

    a_vector_registers = []
    if not is_store:
        for kk in range(a_regs_kk):
            for jj in range(a_regs_jj):
                a_vector_registers += [f"a_{kk}_{jj}"]
    if a_vector_registers:
        a_vec_declaration = ", ".join(a_vector_registers) + "; \\"
        s += ["{vector_type} ".format(vector_type=vector_type_name) + a_vec_declaration]

    if shape_b != None:
        b_vector_registers = []
        for jj in range(b_regs_jj_kk):
            b_vector_registers += [f"b_0_{jj}"]
        b_vec_declaration = ", ".join(b_vector_registers) + "; \\"
        s += ["{vector_type} ".format(vector_type=vector_type_name) + b_vec_declaration]

    order_of_operands["a"] = [a_regs_kk, a_regs_jj]
    order_of_operands["c"] = [W_ob, C_ob]
    order_of_operands["b"] = [1, b_regs_jj_kk]

    # ----------------------------------------
    # Input Loading
    # ----------------------------------------

# if a is reused over the width elements, load can be performed outside
    if not is_store and num_operands > 0 and int(shape_a[0])==1:
        for jj in range(a_regs_jj):
            s += [f"a_0_{jj} = " + a_load.format(ptr=(f"a + {jj} * SIMD")) + ";\\"]

    # if the b operand is present load it
    if shape_b != None:
        for jj in range(b_regs_jj_kk):
            s += [f"b_0_{jj} = " + b_load.format(ptr=(f"b + {jj} * SIMD")) + ";\\"]


    op_2_j_regs = int(C_ob // vector_instruction_table["SIMD"])
    op_2_k_regs = W_ob

    op_1_j_regs = int(C_ob // vector_instruction_table["SIMD"])
    op_1_k_regs = W_ob

    reg1_unlabeled = "{op}".format(op=operand_order[0]) + "_{idxs[0][0]}_{idxs[0][1]}"
    reg2_unlabeled = (
        "{op}".format(op=operand_order[1]) + "_{idxs[1][0]}_{idxs[1][1]}"
        if num_operands >= 2
        else None
    )
    reg3_unlabeled = (
        "{op}".format(op=operand_order[2]) + "_{idxs[2][0]}_{idxs[2][1]}"
        if num_operands == 3
        else None
    )

    # store uses {ptr}/{reg} placeholders rather than {reg1}/{reg2}/{reg3},
    # so it must be handled separately in both template formatting and emit.

    if not is_store:
        vector_instruction_instance = vector_instruction.format(
            reg1=reg1_unlabeled, reg2=reg2_unlabeled, reg3=reg3_unlabeled
        )

    for kk in range(W_ob):
        # if a is not reused it must be reloaded
        if not is_store and num_operands > 0 and int(shape_a[0]) > 1:
            for jj in range(a_regs_jj):
                s += [
                    f"a_{(kk%a_regs_kk)}_{jj} = "
                    + a_load.format(ptr=(f"a + {kk}*step + {jj} * SIMD"))
                    + ";\\"
                ]
        for jj in range(C_ob // vector_instruction_table["SIMD"]):
            # if not (shape_b == None):
            register_operand_idxs = []
            for operand in range(num_operands):
                register_operand_idxs.append(
                    [
                        kk % order_of_operands[operand_order[operand]][0],
                        jj % order_of_operands[operand_order[operand]][1],
                    ]
                )

            if is_store:
                # store is void: emit the instruction directly without assigning to c.
                # c_{kk}_{jj} holds the tile data; a is the scalar base destination pointer.
                vop = vector_instruction.format(
                    ptr=f"a + {kk}*step + {jj} * SIMD",
                    reg=f"c_{kk}_{jj}",
                )
                s += ["{vop};\\".format(vop=vop)]
            else:
                s += [
                    "c_{k}_{j} = {vop};\\".format(
                        vop=vector_instruction_instance.format(idxs=register_operand_idxs),
                        k=kk,
                        j=jj,
                    )
                ]
            # else:
            #     s += ['c_{k}_{j} = {vop});\\'.format(vop = vector_instruction_instance, k=kk, j=jj, kj=((kk%a_regs_kk)*(a_regs_jj)+(jj%a_regs_jj)))]
    s += [""]
    print("\n".join(s))



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("tile_size", type=str, help="Tile size in form WxC (e.g. 4x8)")
    parser.add_argument("op", type=str, help="Operation name")
    parser.add_argument("name", type=str, help="Kernel name")

    parser.add_argument(
        "operands", nargs="*", help="Operand shapes like a:4,8 b:4,8 c:4,8"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generate_kernel(args.tile_size, args.op, args.name, args.operands)