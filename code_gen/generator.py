#this file generates the header file with intrinsics and the kernels for different  operations for different architecture
import sys

header_file_name=sys.argv[1]+".h"

operations=["convolution", "pooling"]

parameters={"W":6, "C":16, "W_pool":8, "SIMD":8}

intrinsics={
            "aligned_load": "_mm256_load_ps",
            "aligned_store": "_mm256_store_ps"
            }
def gen_load(num, ptr_name, load_type):
    macro_str = "#define LOAD_{:}_tile({:})\\\n".format(load_type, ptr_name)
    for i in range(num):
        macro_str +="\t c_{:d} = {:}({:}+({:d}));\\\n".format(i,
                                                intrinsics["aligned_load"],
                                                ptr_name,
                                                i*parameters["SIMD"])
    return macro_str

def gen_store(num, ptr_name, store_type):
    macro_str = "#define STORE_{:}_tile({:})\\\n".format(store_type, ptr_name)
    for i in range(num):
        macro_str +="\t {:}({:}+({:d}), c_{:});\\\n".format(
                                                intrinsics["aligned_store"],
                                                ptr_name,
                                                i*parameters["SIMD"],
                                                i)
    return macro_str

print(gen_load(12,"O", "C"))
print(gen_store(12, "O", "C"))
