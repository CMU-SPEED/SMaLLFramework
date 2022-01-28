import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch.nn as nn
import torch
import numpy as np
from hwcounter import count, count_end
import time


C = int(sys.argv[1])
K = int(sys.argv[2])
o_rows = int(sys.argv[3])
o_cols = int(sys.argv[4])

G = int(sys.argv[5])
F_conv = int(sys.argv[6])
S_conv = int(sys.argv[7])

F_dw = int(sys.argv[8])
S_dw = int(sys.argv[9])
block = sys.argv[10]

conv = nn.Conv2d(C, K, kernel_size=F_conv, stride = S_conv, padding=0, bias=False)
conv.eval()

if block == "dw":
    pool = nn.Conv2d(
   K, K, kernel_size=F_dw, stride = S_dw, padding=0, bias=False, groups = G
)
elif block == "group":
    pool = nn.Conv2d(
   K, K, kernel_size=F_dw, stride = S_dw, padding=0, bias=False, groups = G
)
elif block == "1x1":
    pool = nn.Conv2d(K, G, kernel_size=F_dw, stride = S_dw, padding=0, bias=False)
elif block == "pool":
    op2 = nn.MaxPool2d(kernel_size=F_dw, stride = S_dw, padding=0)
else:
    print("Unsupported fused block")
    exit()

pool.eval()

@torch.jit.script
def custom_block(input_tensor):
    out = conv(input_tensor)
    pool_out = pool(out)
    return pool_out

N = (o_rows - 1) * S_conv + F_conv
M = (o_cols - 1) * S_conv + F_conv

input_tensor  = torch.randn(N*M*C).reshape(1,C, N, M)

out_block = custom_block(input_tensor)
ULLONG_MAX=2**63
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

