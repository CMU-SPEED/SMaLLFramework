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
S_conv = 2

F_dw = 3
S_dw = S_conv


conv = nn.Conv2d(C, K, kernel_size=F_conv, stride = S_conv, padding=0, bias=False)
conv.eval()

p_1x1 = nn.Conv2d(
   C, K, kernel_size=1, stride = S_dw, padding=0, bias=False)
p_1x1.eval()


p_dw = nn.Conv2d(
   C, C, kernel_size=F_dw, stride = S_dw, padding=0, bias=False, groups = C
)
p_dw.eval()

p_group = nn.Conv2d(
   C, C, kernel_size=F_dw, stride = S_dw, padding=0, bias=False, groups = 16
)
p_group.eval()

N = (o_rows - 1) * S_conv + F_conv
M = (o_cols - 1) * S_conv + F_conv

input_tensor  = torch.randn(N*M*C).reshape(1,C, N, M)



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
    out = p_group(input_tensor)
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
