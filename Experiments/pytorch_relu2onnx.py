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
C = 256


class VGG_block(nn.Module):
    def __init__(self,C, K, f):
        super(VGG_block, self).__init__()
        self.conv = nn.Conv2d(C, K, kernel_size=f, stride=1, padding=0, bias=False)
        self.pool = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

j = 54
i = j+2
K = C
input_tensor  = torch.randn(i*i*K).reshape(1,K, i, i)
print("{:d}\t{:d}\t{:d}".format(j, K, K), end="\t")
weights = torch.randn(K*K*3*3).reshape(K, K, 3, 3)

model = VGG_block(K,K,3)
model.eval()

# Export the model
torch.onnx.export(model,               # model being run
                  input_tensor,                         # model input (or a tuple for multiple inputs)
                  "ONNX_models/VGG_relu_fusion.onnx".format(K, j),   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
