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

import numpy as np
import time
import argparse
import torch
from torch import nn
import onnx
from onnx2pytorch import ConvertModel

#*-------------------------------------------------------------------------------
# Convert ONNX model to PyTorch model
# assumes dependency on onnx2pytorch
def convert_onnx_2_pytorch(onnx_model_path):
    onnx_model_loaded = onnx.load(onnx_model_path)
    pytorch_model = ConvertModel(onnx_model_loaded)
    print(pytorch_model)
    exit(-1)
    return pytorch_model

#*-------------------------------------------------------------------------------
# Run ONNX model for performance
# return the best fps for 10000 runs
def run_onnx_model_performance(onnx_model_path, input_file):
    
    pytorch_model = convert_onnx_2_pytorch(onnx_model_path)
    input_np = np.load(input_file)
    input_torch = torch.from_numpy(input_np)

    total_time = 0
    best_time = 1e9
    RUNS = 10000
    for _ in range(RUNS):
        s = time.time()
        outputs = pytorch_model(input_torch)
        e = time.time()
        total_time += (e-s)
        best_time = min(best_time, e-s)
        
    print(f"{best_time}")

#*-------------------------------------------------------------------------------
# Run ONNX model for correctness
# save outputs to pytorch_output.npy
def run_onnx_model_correctness(onnx_model_path, input_file):
    
    onnx_model_loaded = onnx.load(onnx_model_path)
    pytorch_model = ConvertModel(onnx_model_loaded)
    # print(pytorch_model)
    
    l = [module for module in pytorch_model.modules() if not isinstance(module, nn.Sequential)]
    # print(l)
    
    input_np = np.load(input_file)
    
    input_torch = torch.from_numpy(input_np)
    
    # warm up run
    for _ in range(50):
        outputs = pytorch_model(input_torch)
    
    s = time.time()
    outputs = pytorch_model(input_torch)
    e = time.time()
    
    # print(outputs)
    
    outputs = outputs.detach().numpy() 
    np.save("pytorch_output.npy", outputs)
    
    print(f"{e-s}")

#*-------------------------------------------------------------------------------
def get_args(): 
    
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument(
        "model", 
        type=str,
        help="Path to onnx model. TODO: Check Op set dependency."
    )
    arg_parser.add_argument(
        "input_file", 
        type=str,
        help="numpy array of the input data."
    )
    arg_parser.add_argument(
        "--correctness",
        action='store_true',
        help="Run correctness test."
    )
    
    return arg_parser.parse_args()
    
#*-------------------------------------------------------------------------------
if __name__ == "__main__":
    
    device = torch.device("cpu")
    total_physical_cores = torch.get_num_threads()
    torch.set_num_threads(total_physical_cores)
    print(f"Using {total_physical_cores} threads for PyTorch")
    
    args = get_args()
    
    onnx_model = args.model
    input_file = args.input_file
    correctness = args.correctness
    
    if(correctness):
        run_onnx_model_correctness(onnx_model, input_file)
    else:
        run_onnx_model_performance(onnx_model, input_file)