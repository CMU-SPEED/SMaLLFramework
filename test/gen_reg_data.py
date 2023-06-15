# //****************************************************************************
# SMaLL, Software for Machine Learning Libraries
# Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
# # SPDX-License-Identifier: BSD-3-Clause
# For additional details (including references to third party source code and
# other files) see the LICENSE file or contact permission@sei.cmu.edu. See
# Contributors.txt for a full list of contributors. Created, in part, with
# funding and support from the U.S. Government (see Acknowledgments.txt file).
# DM23-0126
# //****************************************************************************

"""
This script is meant to be used for generating regression data.

Dependencies:
- torch
- numpy

Currently, this script only generates data for the conv2d layer,
and the sizes are hardcoded in. If you want a certain size, you will
need to modify the conv2d_config list. Note that this list is meant
to match the list in test_conv2d_layer_regression_data() within
test_conv2d.cpp

To use this script, run the following command:
`python gen_reg_data.py`
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# set seed for reproducibility
torch.manual_seed(0)

def save_numpy_to_file(arr, filename):
    with open(filename, 'wb') as f:
        # save total size as the header
        # use network order (i.e., big endian)
        f.write((arr.size).to_bytes(4, byteorder='big', signed=False))
        f.write(arr.tobytes())

# generate test data for conv2d layer
def generate_conv2d_test_data(Ci, H, W, k, stride, pad, Co, bias=False):
    
    if(pad != "same" and pad != "valid"):
        raise ValueError("pad must be 'same' or 'valid'")
    pad_type = 'f' if pad == "same" else 'v'

    input_tensor = torch.randn(1, Ci, H, W).float()
    input_tensor_np = input_tensor.detach().numpy()
    print("input shape:", input_tensor_np.shape)
    save_numpy_to_file(input_tensor_np, f'in__conv2d_Ci{Ci}_H{H}_W{W}_k{k}_s{stride}_{pad_type}_Co{Co}_{input_tensor_np.size}.bin')

    conv2d = nn.Conv2d(
        in_channels=Ci, 
        out_channels=Co, 
        kernel_size=k,
        stride=stride,
        padding=pad,
        bias=bias
    )
    
    weights = conv2d.weight.data.detach().numpy()
    print("weight shape:", weights.shape)
    save_numpy_to_file(weights, f'filter__conv2d_Ci{Ci}_H{H}_W{W}_k{k}_s{stride}_{pad_type}_Co{Co}_{weights.size}.bin')
    
    output_tensor = conv2d(input_tensor).detach().numpy()
    print("out shape:", output_tensor.shape)
    save_numpy_to_file(output_tensor, f'out__conv2d_Ci{Ci}_H{H}_W{W}_k{k}_s{stride}_{pad_type}_Co{Co}_{output_tensor.size}.bin')
    
conv2d_config = [
    
    # [Ci, H, W, k, stride, pad, Co]
    [3, 3, 3, 1, 1, "valid", 16],
    [3, 3, 3, 3, 1, "valid", 16],
    [3, 1, 1, 1, 1, "same", 16],
    [3, 3, 3, 1, 1, "same", 16],
    
    # [3, 1, 1, 1, 1, "valid", 16],
    # [3, 1, 6, 1, 1, "valid", 16],
    # [3, 3, 3, 3, 1, "valid", 16],
    # [3, 4, 4, 3, 1, "valid", 16],
    # [3, 3, 8, 3, 1, "valid", 16],
    # [3, 30, 30, 3, 1, "valid", 16],
    
    # [16, 1, 1, 1, 1, "valid", 16],
    # [16, 1, 6, 1, 1, "valid", 16],
    # [16, 3, 3, 3, 1, "valid", 16],
    # [16, 3, 8, 3, 1, "valid", 16],
    # [16, 30, 30, 3, 1, "valid", 16],

    # [16, 1, 6, 1, 1, "valid", 96],
    # [16, 3, 8, 3, 1, "valid", 96],
    
    # [96, 1, 6, 1, 1, "valid", 16],
    # [96, 3, 8, 3, 1, "valid", 16],
    
    # [96, 30, 30, 1, 1, "valid", 96],
    # [96, 30, 30, 3, 1, "valid", 96]
]

for config in conv2d_config:
    generate_conv2d_test_data(*config)

    
    
    