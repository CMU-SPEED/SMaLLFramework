#-------------------------------------------------------------------------------

# SMaLL, Software for Machine Learning Libraries
# Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# For additional details (including references to third party source code and
# other files) see the LICENSE file or contact permission@sei.cmu.edu. See
# Contributors.txt for a full list of contributors. Created, in part, with
# funding and support from the U.S. Government (see Acknowledgments.txt file).
# DM23-0126

#-------------------------------------------------------------------------------


"""

    Automatically run ONNX models using onnx-mlir.
    
"""

from termcolor import colored
import numpy as np
import sys
import time
import os
import json
import argparse
import torch
device = torch.device("cpu")
total_physical_cores = torch.get_num_threads()
torch.set_num_threads(total_physical_cores)
print(f"Using {total_physical_cores} threads for PyTorch")
import onnx
from onnx2pytorch import ConvertModel

def save_numpy_to_file(arr, filename):
    with open(filename, 'wb') as f:
        f.write(arr.tobytes())
        
def read_numpy_from_file(filename, shape):
    arr = np.fromfile(filename, dtype=np.float32)
    return arr.reshape(shape)

def run_onnx_model(onnx_model_path, ic, ih, iw):
    
    onnx_model_loaded = onnx.load(onnx_model_path)
    pytorch_model = ConvertModel(onnx_model_loaded)
    
    input_shape = (1, int(ic), int(ih), int(iw))
    input_ = np.random.rand(*input_shape).astype(np.float32)
    # if(input_file != ""):
    #     input_ = read_numpy_from_file(input_file, input_shape)
    input_torch = torch.from_numpy(input_)

    total_time = 0
    best_time = 1e9
    RUNS = 10000
    for _ in range(RUNS):
        s = time.time()
        outputs = pytorch_model(input_torch)
        e = time.time()
        total_time += (e-s)
        best_time = min(best_time, e-s)
        
    fps = colored(str(input_shape[0]/best_time), "red")
    print(f"{fps}")
    
    outputs = outputs.detach().numpy()
    # save_numpy_to_file(outputs, f"out_{onnx_model_path[:-5]}.bin")
    
        
    return input_, outputs

    
def get_args(): 
    
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument(
        "model", 
        type=str,
        help="Path to onnx model. TODO: Check Op set dependency."
    )
    arg_parser.add_argument(
        "ic", 
        type=str,
        help="input channels."
    )
    arg_parser.add_argument(
        "ih", 
        type=str,
        help="input height."
    )
    arg_parser.add_argument(
        "iw", 
        type=str,
        help="input width."
    )
    arg_parser.add_argument(
        "-i", "--input_file", 
        type=str,
        default="",
        help="binary input data."
    )
    arg_parser.add_argument(
        "-v", "--verbose",
        action='store_true'
    )
    
    return arg_parser.parse_args()
    

if __name__ == "__main__":
    
    args = get_args()
    
    onnx_model = args.model
    ic = args.ic
    ih = args.ih
    iw = args.iw
    
    run_onnx_model(onnx_model, ic, ih, iw)