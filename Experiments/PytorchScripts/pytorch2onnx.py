import io
import numpy as np


import torch.onnx

import torch.nn as nn
import torch

from hwcounter import count, count_end
print(torch.__version__)
torch.set_num_threads(6)
print(torch.get_num_threads())
torch.set_default_tensor_type(torch.FloatTensor)
print("H/W \t K \t C \t Pytorch Convolution FLOPS\t Pytorch Convolution Timing\t Pytorch Combined FLOPS \t Pytorch Combined Timing")
C = 64


class VGG_block(nn.Module):
    def __init__(self,C, K, f):
        super(VGG_block, self).__init__()
        self.conv = nn.Conv2d(C, K, kernel_size=f, stride=1, padding=0, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

for (j,K) in [(12,512), (30,512) , (54,256), (114,128), (222,64) , (504,64)]:
    i = j+2
    input_tensor  = torch.randn(i*i*K).reshape(1,K, i, i)
    print("{:d}\t{:d}\t{:d}".format(j, K, K), end="\t")
    weights = torch.randn(K*K*3*3).reshape(K, K, 3, 3)

    model = VGG_block(K,K,3)
    model.eval()

    # Export the model
    torch.onnx.export(model,               # model being run
                      input_tensor,                         # model input (or a tuple for multiple inputs)
                      "ONNX_models/VGG_C_K{:d}_j{:d}.onnx".format(K, j),   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      )
