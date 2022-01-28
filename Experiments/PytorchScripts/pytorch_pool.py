#tensorflow version of fusion experiment
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch.nn as nn
import torch
import numpy as np
from hwcounter import count, count_end
import time

# print(torch.__version__)
torch.set_num_threads(6)

import sys

C = int(sys.argv[1])
K = int(sys.argv[2])
o_rows = int(sys.argv[3])
o_cols = int(sys.argv[4])

F_conv = 3 
S_conv = 1

F_dw = 3
S_dw = 2


conv = nn.Conv2d(C, K, kernel_size=F_conv, stride = S_conv, padding=0, bias=False)
conv.eval()

pool = nn.MaxPool2d(
    kernel_size=F_dw, stride = S_dw, padding=0, bias=False, groups = K
)
pool.eval()

@torch.jit.script
def custom_block(input_tensor):
    out = conv(input_tensor)
    pool_out = pool(out)
    return pool_out
ULLONG_MAX=2**63

N = (o_rows - 1) * S_conv + F_conv
M = (o_cols - 1) * S_conv + F_conv

input_tensor  = torch.randn(N*M*C).reshape(1,C, N, M)


sum_combined = ULLONG_MAX
for i in range(1000):
    st = count()
    out_block = custom_block(input_tensor)
    et = count()
    sum_combined = min(sum_combined,(et - st))
    # print((et-st))
print(" \t {:.4f}".format(sum_combined))
# c = input()
# f = open("unopt.txt", "w")
# f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='hlo'))

# f = open("opt.txt", "w")
# f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='optimized_hlo'))

# f = open("dotfile_relu.txt","w")
# f.write(custom_block.experimental_get_compiler_ir(input_tensor)(stage='optimized_hlo_dot'))
# s = Source.from_file("dotfile_relu.txt")
# s.view()
