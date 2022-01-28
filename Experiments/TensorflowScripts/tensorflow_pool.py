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

F_conv = 3
S_conv = 1

F_dw = 3
S_dw = 2


conv =  nn.Conv2D(
    K, F_conv, strides=(S_conv, S_conv), padding='valid',
    use_bias=False
)
pool = nn.MaxPooling2D(pool_size=(F_dw,F_dw), strides=(S_dw, S_dw), padding="valid")
# pool = nn.Activation('relu')
# pool = nn.DepthwiseConv2D(
#     F_conv, strides=(S_dw, S_dw), padding='valid', depth_multiplier=1,
#     data_format=None, dilation_rate=(1, 1), activation=None, use_bias=False, depthwise_regularizer=None,
#     bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None,
#     bias_constraint=None
# )
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
print(" \t {:.4f}".format(sum_combined))
c = input()
# f = open("unopt.txt", "w")
# f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='hlo'))

# f = open("opt.txt", "w")
# f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='optimized_hlo'))

# f = open("dotfile_relu.txt","w")
# f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='optimized_hlo_dot'))
# s = Source.from_file("dotfile_relu.txt")
# s.view()
