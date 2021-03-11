#tensorflow version of fusion experiment
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow.keras.layers as nn
import tensorflow as tf
import numpy as np
from graphviz import Source
from hwcounter import count, count_end
# print(tf.__version__)
tf.config.threading.set_inter_op_parallelism_threads(6)
print(tf.config.threading.get_inter_op_parallelism_threads())
# tf.compat.v1.config.enable_eager_execution(False)

print("H/W \t K \t C \t TensorFlow Convolution FLOPS\t TensorFlow Convolution Timing\t TensorFlow Combined FLOPS \t TensorFlow Combined Timing")
C = 64
K = 64

weights = np.random.randn(K*C*3*3).reshape(3, 3, C, K)

conv =  nn.Conv2D(
    K, 3, strides=(1, 1), padding='valid',
    use_bias=False
)
# pool = nn.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")
pool = nn.Activation('relu')

@tf.function(experimental_compile=True)
def custom_conv(input_tensor):
    out = conv(input_tensor)
    return out

@tf.function(experimental_compile=True)
def custom_block(input_tensor):
    out = conv(input_tensor)
    pool_out = pool(out)
    return pool_out

for j in [12, 36, 96, 126]:
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
    pool_out = custom_block(input_tensor)
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

print(tf.__version__)
f = open("unopt.txt", "w")
f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='hlo'))

f = open("opt.txt", "w")
f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='optimized_hlo'))

f = open("dotfile_relu.txt","w")
f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='optimized_hlo_dot'))
s = Source.from_file("dotfile_relu.txt")
s.view()
