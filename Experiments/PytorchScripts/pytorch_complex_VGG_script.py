import torch.nn as nn
import torch
#VGGnet sizes
from hwcounter import count, count_end
print(torch.__version__)
torch.set_num_threads(6)
print(torch.get_num_threads())

print("H/W \t K \t C \t Pytorch Convolution FLOPS\t Pytorch Convolution Timing\t Pytorch Combined FLOPS \t Pytorch Combined Timing")
C = 64
K = 64

for j, CK in [
        (504, 96),
#     (222, 96),
# (114, 192),
# (54, 384),
# (30, 768),
(12, 768),
(6, 768)]:
    i = j+2
    C = CK
    K = CK

    input_tensor_real  = torch.randn(i*i*C).reshape(1,C, i, i)
    input_tensor_imag  = torch.randn(i*i*C).reshape(1,C, i, i)
    print("{:d}\t{:d}\t{:d}".format(j, K, C), end="\t")
    weights_real = torch.randn(K*C*3*3).reshape(K, C, 3, 3)
    weights_imag = torch.randn(K*C*3*3).reshape(K, C, 3, 3)

    conv_real = nn.Conv2d(C, K, kernel_size=3, stride = 1, padding=0, bias=False)
    conv_real.eval()
    conv_real.weight = nn.Parameter(weights_real)

    conv_imag = nn.Conv2d(C, K, kernel_size=3, stride = 1, padding=0, bias=False)
    conv_imag.eval()
    conv_imag.weight = nn.Parameter(weights_imag)
    relu = nn.ReLU(inplace=True)
    pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)


    sum_conv = 2**32
    out_inter_real = conv_real(input_tensor_real)
    out_inter_real -= conv_imag(input_tensor_imag)

    out_inter_imag = conv_imag(input_tensor_real)
    out_inter_imag += conv_real(input_tensor_imag)

    out_inter_real = conv_real(input_tensor_real)
    out_inter_real -= conv_imag(input_tensor_imag)

    out_inter_imag = conv_imag(input_tensor_real)
    out_inter_imag += conv_real(input_tensor_imag)

    for r in range(100):
        st = count()
        out_inter_real = conv_real(input_tensor_real)
        out_inter_real -= conv_imag(input_tensor_imag)
        out_inter_imag = conv_imag(input_tensor_real)
        out_inter_imag += conv_real(input_tensor_imag)
        et = count_end()
        sum_conv = min(sum_conv, (et - st))

    conv_ops = out_inter_real.numel()*(3*3*C)*4*2


    print(sum_conv)
    # print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(((conv_ops)/(sum_conv/100)),
                                                #  ))
