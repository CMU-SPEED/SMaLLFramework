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

    Automatically run ONNX models with SMaLL as the backend.
    
"""

import numpy as np
import sys
import time
import os
import json
import argparse

# These are imported from onnx-mlir
from PyCompile import OMCompileSession
from PyRuntime import OMExecutionSession

ONNX_MLIR_ROOT = "/workdir/onnx-mlir"

def compile_model(onnx_model_path, small_lib_path):
    
    compiler = OMCompileSession(onnx_model_path)
    print(f"Compiling {onnx_model_path}")
    
    # rc = compiler.compile(f"-O3 -L{small_lib_path} -lsmall")
    rc = compiler.compile(f"-O3 -EmitObj")
    
    if (rc):
        print(f"Failed to compile with error code {rc}")
        exit(rc)
        
    onnx_model_o = compiler.get_compiled_file_name()
    onnx_model_so = onnx_model_o[:-2] + ".so"
    onnx_lib_link = f"-L{ONNX_MLIR_ROOT}/build/Debug/lib -lcruntime"
    small_lib_link = f"-L{small_lib_path} -lsmall"

    link_cmd = f"c++ {onnx_model_o} -o {onnx_model_so} -shared -fopenmp -fPIC {onnx_lib_link} {small_lib_link}"
    os.system(link_cmd)

    print(f"{onnx_model_so} compiled!")
    
    return onnx_model_so

def run_onnx_model(onnx_model_path, small_lib_path):
    
    onnx_model_so = compile_model(onnx_model_path, small_lib_path)
    
    session = OMExecutionSession(shared_lib_path=onnx_model_so)
    
    input_sign = json.loads(session.input_signature())[0]
    input_dims = input_sign["dims"]
    
    # output_sign = json.loads(session.output_signature())[0]
    # output_dims = output_sign["dims"]
    
    input_ = np.random.rand(*input_dims).astype(np.float32)
    
    total_time = 0
    best_time = 1e9
    RUNS = 10
    for _ in range(RUNS):
        s = time.time()
        outputs = session.run(input=[input_])
        e = time.time()
        total_time += (e-s)
        best_time = min(best_time, e-s)
        
    print(f"FPS = {input_dims[0]/best_time}")
        
    return input_, outputs

    
def get_args(): 
    
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument(
        "model", 
        type=str,
        help="Path to onnx model. TODO: Check Op set dependency."
    )
    
    arg_parser.add_argument(
        "-l", "-L", "--lib", 
        type=str,
        default="../../lib",
        help="Path to SMaLL libray"
    )
    
    arg_parser.add_argument(
        "-v", "--verbose",
        action='store_true'
    )
    
    return arg_parser.parse_args()
    

if __name__ == "__main__":
    
    args = get_args()
    
    onnx_model = args.model
    small_lib_path = args.lib
    
    run_onnx_model(onnx_model, small_lib_path)