#tensorflow version of fusion experiment

import tensorflow.nn as nn
import tensorflow as tf
import numpy as np

from hwcounter import count, count_end

tf.config.threading.set_inter_op_parallelism_threads(6)
print(tf.config.threading.get_inter_op_parallelism_threads())
# tf.compat.v1.config.enable_eager_execution(False)

print("H/W \t K \t C \t TensorFlow Convolution FLOPS\t TensorFlow Convolution Timing\t TensorFlow Combined FLOPS \t TensorFlow Combined Timing")
C = 64
K = 64

weights = np.random.randn(K*C*3*3).reshape(3, 3, C, K)

# @tf.function(jit_compile=False)
def custom_conv(input_tensor):
    out = nn.conv2d(input_tensor, weights,strides = 1, padding="VALID", data_format='NHWC', name="conv")
    return out

# @tf.function(jit_compile=False)
def custom_block(input_tensor):
    out = nn.conv2d(input_tensor, weights,strides = 1, padding="VALID", data_format='NHWC', name="conv")
    pool_out = nn.max_pool2d(out, 3, 2, "VALID")
    return pool_out

for j in [12, 36, 96, 126, 252, 504]:
    i = j+2
    input_tensor  = np.random.randn(i*i*C).reshape(1,i, i, C)
    print("{:d}\t{:d}\t{:d}".format(j, K, C), end="\t")




    sum_conv = 0
    out = custom_conv(input_tensor)

    # out_inter = conv(input_tensor)
    for r in range(100):
        st = count()
        out = custom_conv(input_tensor)
        et = count_end()
        sum_conv += (et - st)

    conv_ops = out.numpy().size*(3*3*C)*2
    # print(out.size(),conv_ops)
    # out_inter = conv(input_tensor)
    # out = pool(out_inter)

    out = nn.conv2d(input_tensor, weights,strides = 1, padding="VALID", data_format='NHWC', name="conv")
    pool_out = nn.max_pool2d(out, 3, 2, "VALID")
    sum_combined = 0
    for i in range(100):
        st = count()
        out_block = custom_block(input_tensor)
        et = count()
        sum_combined += (et - st)
    pool_ops = pool_out.numpy().size*9

    print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(((conv_ops)/(sum_conv/100)),
                                                 (sum_conv/100),
                                                 ((conv_ops + pool_ops)/(sum_combined/100)),
                                                 (sum_combined/100)
                                                 ))
