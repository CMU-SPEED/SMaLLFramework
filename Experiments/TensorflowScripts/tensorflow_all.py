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

C_o_1 = 16
F_conv = 3
S_conv = 2

F_dw = F_conv
S_dw = S_conv


conv =  nn.Conv2D(
    K, F_conv, strides=(S_conv, S_conv), padding='valid',
    use_bias=False
)

p_1x1 = conv =  nn.Conv2D(
    K, 1, strides=(S_conv, S_conv), padding='valid',
    use_bias=False
)



p_dw = nn.DepthwiseConv2D(
    F_dw, strides=(S_dw, S_dw), padding='valid', depth_multiplier=1,
    data_format=None, dilation_rate=(1, 1), activation=None, use_bias=False, depthwise_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None,
    bias_constraint=None
)





p_group = nn.Conv2D(
    C, F_dw, strides=(S_dw, S_dw), padding='valid',
    groups=C//C_o_1 
    )

@tf.function(experimental_compile=True)
def custom_p_group(input_tensor):
    out = p_group(input_tensor)
    return out


N = (o_rows - 1) * S_conv + F_conv
M = (o_cols - 1) * S_conv + F_conv

input_tensor  = np.random.randn(N*M*C).reshape(1,N, M, C)



ULLONG_MAX=2**63
sum_conv = ULLONG_MAX
for i in range(1000):
    st = count()
    out_block = conv(input_tensor)
    et = count()
    sum_conv = min(sum_conv,(et - st))
    # print((et-st))
print(" \t {:.4f}".format(sum_conv), end = "\t")


sum_1x1 = ULLONG_MAX
for i in range(1000):
    st = count()
    out = p_1x1(input_tensor)
    et = count()
    sum_1x1 = min(sum_1x1,(et - st))
    # print((et-st))
print(" \t {:.4f}".format(sum_1x1), end = "\t")

sum_dw = ULLONG_MAX
for i in range(1000):
    st = count()
    out = p_dw(input_tensor)
    et = count()
    sum_dw = min(sum_dw,(et - st))
    # print((et-st))
print(" \t {:.4f}".format(sum_dw), end = "\t")


sum_group = ULLONG_MAX
for i in range(1000):
    st = count()
    out = custom_p_group(input_tensor)
    et = count()
    sum_group = min(sum_group,(et - st))
    # print((et-st))
print(" \t {:.4f}".format(sum_group))


# c = input()
# f = open("unopt.txt", "w")
# f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='hlo'))

# f = open("opt.txt", "w")
# f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='optimized_hlo'))

# f = open("dotfile_relu.txt","w")
# f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='optimized_hlo_dot'))
# s = Source.from_file("dotfile_relu.txt")
# s.view()
