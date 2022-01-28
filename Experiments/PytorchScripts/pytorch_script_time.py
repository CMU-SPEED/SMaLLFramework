import torch.nn as nn
import torch

from hwcounter import count, count_end
import time

print(torch.__version__)
torch.set_num_threads(6)
print(torch.get_num_threads())

print("H/W \t K \t C \t Pytorch Convolution FLOPS\t Pytorch Convolution Timing\t Pytorch Combined FLOPS \t Pytorch Combined Timing")
C = 64
K = 64

for j in [12, 36, 96, 126, 252, 504]:
    i = j+2
    input_tensor  = torch.randn(i*i*C).reshape(1,C, i, i)
    print("{:d}\t{:d}\t{:d}".format(j, K, C), end="\t")
    weights = torch.randn(K*C*3*3).reshape(K, C, 3, 3)

    conv = nn.Conv2d(C, K, kernel_size=3, stride = 1, padding=0, bias=False)
    conv.eval()
    conv.weight = nn.Parameter(weights)

    pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)


    sum_conv = 0
    out_inter = conv(input_tensor)
    for r in range(100):
        st = time.time()
        out_inter = conv(input_tensor)
        et = time.time()
        sum_conv += (et - st)

    conv_ops = out_inter.numel()*(3*3*C)*2
    # print(out.size(),conv_ops)
    out_inter = conv(input_tensor)
    out = pool(out_inter)
    sum_combined = 0
    for i in range(100):
        st = time.time()
        out = pool(conv(input_tensor))
        et = time.time()
        sum_combined += (et - st)
    pool_ops = out.numel()*9

    print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(((conv_ops)/(sum_conv/100)),
                                                 (sum_conv/100),
                                                 ((conv_ops + pool_ops)/(sum_combined/100)),
                                                 (sum_combined/100)
                                                 ))


                                    
