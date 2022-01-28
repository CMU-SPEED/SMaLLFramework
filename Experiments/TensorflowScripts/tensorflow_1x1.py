#tensorflow version of fusion experiment
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow.keras.layers as nn
import tensorflow as tf
import numpy as np

from hwcounter import count, count_end
# print(tf.__version__)
tf.config.threading.set_inter_op_parallelism_threads(6)
print(tf.config.threading.get_inter_op_parallelism_threads())
# tf.compat.v1.config.enable_eager_execution(False)

import sys

C = int(sys.argv[1])

K = int(sys.argv[2])
o_rows = int(sys.argv[3])
o_cols = int(sys.argv[4])

C_o_1 = int(sys.argv[5])
F_conv = 3
S_conv = 1

F_dw = 1
S_dw = 1


conv =  nn.Conv2D(
    K, F_conv, strides=(S_conv, S_conv), padding='valid',
    use_bias=False
)
pool = nn.Conv2D(
    K, F_dw, strides=(S_conv, S_conv), padding='valid',
    use_bias=False
)
@tf.function(experimental_compile=True)
def custom_conv(input_tensor):
    out = conv(input_tensor)
    return out

@tf.function(experimental_compile=True)
def custom_block(input_tensor):
    out = conv(input_tensor)
    pool_out = pool(out)
    return pool_out
ULLONG_MAX=2**63

N = (o_rows - 1) * S_conv + F_conv
M = (o_cols - 1) * S_conv + F_conv

input_tensor  = np.random.randn(N*M*C).reshape(1,N, M, C)


sum_combined = ULLONG_MAX
for i in range(1000):
    st = count()
    out_block = custom_block(input_tensor)
    et = count()
    sum_combined = min(sum_combined,(et - st))
    # print((et-st))
print(" \t {:.4f}".format(sum_combined), end = "\t")

ULLONG_MAX=2**63
sum_added = ULLONG_MAX
for i in range(1000):
    st = count()
    out_block = conv(input_tensor)
    out = pool(out_block)
    et = count()
    sum_added = min(sum_added,(et - st))
    # print((et-st))
print(" \t {:.4f}".format(sum_added), end = "\t")


sum_conv = ULLONG_MAX
for i in range(1000):
    st = count()
    out_block = conv(input_tensor)
    et = count()
    sum_conv = min(sum_conv,(et - st))
    # print((et-st))
print(" \t {:.4f}".format(sum_conv), end = "\t")

out_block = conv(input_tensor)
sum_pool = ULLONG_MAX
for i in range(1000):
    st = count()
    out = pool(out_block)
    et = count()
    sum_pool = min(sum_pool,(et - st))
    # print((et-st))
print(" \t {:.4f}".format(sum_pool))
c = input()
# f = open("unopt.txt", "w")
# f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='hlo'))

# f = open("opt.txt", "w")
# f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='optimized_hlo'))

# f = open("dotfile_relu.txt","w")
# f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='optimized_hlo_dot'))
# s = Source.from_file("dotfile_relu.txt")
# s.view()
