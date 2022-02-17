#tensorflow version of fusion experiment
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np

from hwcounter import count, count_end
import pandas as pd
import sys
import psutil

import tensorflow as tf 

# from tf_1x1 import FusedBlock

def log_mem(annotation="start"):
  return  (annotation, psutil.virtual_memory())

print(log_mem())

C = int(sys.argv[1])

K = int(sys.argv[2])
o_rows = int(sys.argv[3])
o_cols = int(sys.argv[4])

C_o_1 = int(sys.argv[5])

F_conv = 3
S_conv = 1

F_dw = 1
S_dw = 1




ULLONG_MAX=2**63

N = (o_rows - 1) * S_conv + F_conv
M = (o_cols - 1) * S_conv + F_conv

custom_block = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(N, M, C)),
  tf.keras.layers.Dense(K,activation='relu'),
  tf.keras.layers.Dense(C_o_1, activation='softmax')
])

custom_block.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)


# print_mem()
input_tensor  = np.random.randn(N*M*C).reshape(1,N, M, C)
# print_mem()
out_block = custom_block(input_tensor)
# print_mem()

print(log_mem)

logs = "TF_logs/"

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

sum_combined = ULLONG_MAX
for i in range(1000):
    st = count()
    out_block = custom_block.run(callback = tboard_callback)
    et = count()
    sum_combined = min(sum_combined,(et - st))
    # print((et-st))
print(" \t {:.4f}".format(sum_combined), end = "\t")
