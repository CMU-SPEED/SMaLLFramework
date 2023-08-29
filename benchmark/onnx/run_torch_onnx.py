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
torch.set_num_threads(4)
import onnx
from onnx2pytorch import ConvertModel

def run_onnx_model(onnx_model_path, verbosity):
    
    onnx_model_loaded = onnx.load(onnx_model_path)
    pytorch_model = ConvertModel(onnx_model_loaded)
 
    input_dims = input(f"Enter new input dimensions as space seperated list of integers for {onnx_model_path}: ")
    input_dims = [int(x) for x in input_dims.split(" ")]
            
    input_ = torch.from_numpy(np.random.rand(*input_dims).astype(np.float32))
    
    total_time = 0
    best_time = 1e9
    RUNS = 10
    for _ in range(RUNS):
        s = time.time()
        outputs = pytorch_model(input_)
        e = time.time()
        total_time += (e-s)
        best_time = min(best_time, e-s)
        
    fps = colored(str(input_dims[0]/best_time), "red")
    print(f"FPS = {fps}")
        
    return input_, outputs

    
def get_args(): 
    
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument(
        "model", 
        type=str,
        help="Path to onnx model. TODO: Check Op set dependency."
    )
    arg_parser.add_argument(
        "-v", "--verbose",
        action='store_true'
    )
    
    return arg_parser.parse_args()
    

if __name__ == "__main__":
    
    args = get_args()
    
    onnx_model = args.model
    verbosity = args.verbose
    
    run_onnx_model(onnx_model, verbosity)