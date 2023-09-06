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
import subprocess

# These are imported from onnx-mlir
from PyRuntime import OMExecutionSession

print(os.getcwd())

def save_numpy_to_file(arr, filename):
    with open(filename, 'wb') as f:
        f.write(arr.tobytes())

def compile_model(onnx_model_path, small_lib_path, ONNX_MLIR_ROOT, verbose=False):
    
    onnx_model_o = onnx_model_path[:-5] + ".o"
    onnx_model_so = onnx_model_path[:-5] + ".so"
    
    # if(os.path.isfile(onnx_model_so)):
    #     print(f"{onnx_model_so} exists. Skipping compilation step.")
    #     return onnx_model_so
    
    if(ONNX_MLIR_ROOT != ""):
        onnx_mlir_exe = ONNX_MLIR_ROOT + "/build/Debug/bin/onnx-mlir"
    else:
        onnx_mlir_exe = "onnx-mlir"
        
    os.system(f"{onnx_mlir_exe} -O3 --EmitObj {onnx_model_path}")
    
    if(verbose):
        os.system(f"{onnx_mlir_exe} -O3 --EmitLLVMIR {onnx_model_path}")
   
    onnx_lib_link = f"-L{ONNX_MLIR_ROOT}/build/Debug/lib -lcruntime"
    small_lib_link = f"-L{small_lib_path} -lsmall"

    link_cmd = f"c++ {onnx_model_o} -o {onnx_model_so} -shared -fopenmp -fPIC {onnx_lib_link} {small_lib_link}"
    os.system(link_cmd)

    onnx_model_so_colored = colored(onnx_model_so, "light_cyan")
    print(f"{onnx_model_so_colored} compiled!")
    
    return onnx_model_so

def run_onnx_model(onnx_model_path, small_lib_path, ONNX_MLIR_ROOT="", verbose=False):
    
    onnx_model_so = compile_model(onnx_model_path, small_lib_path, ONNX_MLIR_ROOT, verbose)
    
    onnx_model_so_colored = colored(onnx_model_so, "light_cyan")
    print(f"Running model {onnx_model_so_colored}\n")
    
    session = None
    try:
        session = OMExecutionSession(shared_lib_path=onnx_model_so)
    except Exception as E:
        print(E)
        print("Did you make sure LD_LIBRARY_PATH has the path to directory with libsmall.so?")
        print("\tRun: export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:<SMALL_ROOT_DIR>/lib\"")
        exit(-1)

    input_sign = json.loads(session.input_signature())[0]
    input_dims = [1 if x==-1 else x for x in input_sign["dims"]]
    print(f"Input dims to {onnx_model_so_colored}: {input_dims}")
    
    while(True):
        c = input("Would you like to change the input dimensions? (y/n): ").lower()
        if(c != "y" and c != "n"):
            continue
        else:
            break
        
    new_input_dims_list = []
    if(c == "y"):
        while(True):
            new_input_dims = input("Enter new input dimensions as space seperated list of integers: ")
            new_input_dims_list = [int(x) for x in new_input_dims.split(" ")]
            if(len(new_input_dims_list) != len(input_dims)):
                print("[ERROR] Number of dimensions entered doesn't match expected number of dimension.")
                continue
            else:
                break
            
        input_dims = new_input_dims_list
        print(f"New Input dims to {onnx_model_so_colored}: {input_dims}")
   
    input_ = np.random.rand(*input_dims).astype(np.float32)
    input_2 = np.copy(input_)
    input_file = "input.bin"
    save_numpy_to_file(input_, input_file)
    cmd_str = f"python run_torch_onnx.py {onnx_model_path} {input_dims[1]} {input_dims[2]} {input_dims[3]}"
    print(cmd_str)
    proc = subprocess.Popen(list(cmd_str.split(" ")), stdout=subprocess.PIPE)
    print(str(proc.stdout.readline().rstrip(), encoding='utf-8'))
    pytorch_fps = float(proc.stdout.readline().rstrip().decode())
    # pytorch_fps = 1

    # input_packed = input_2.transpose(0, 2, 3, 1)
    
    total_time = 0
    best_time = 1e9
    RUNS = 1
    for _ in range(RUNS):
        s = time.time()
        outputs = session.run(input=[input_2])
        e = time.time()
        total_time += (e-s)
        best_time = min(best_time, e-s)
        
    
    small_fps = input_dims[0]/best_time
    print("small, pytorch")
    # small_fps = 1
    print(f"{small_fps}, {pytorch_fps}, {small_fps/pytorch_fps}")
        
    # return input_, outputs

    
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
    arg_parser.add_argument(
        "-o", "--ONNX_MLIR_ROOT",
        default="",
        help="Path to onnx-mlir"
    )
    
    return arg_parser.parse_args()
    

if __name__ == "__main__":
    
    args = get_args()
    
    onnx_model = args.model
    small_lib_path = args.lib
    ONNX_MLIR_ROOT = args.ONNX_MLIR_ROOT
    ONNX_MLIR_ROOT_COLORED = colored(ONNX_MLIR_ROOT, "green")
    verbosity = args.verbose
    
    print(f"\nUsing {ONNX_MLIR_ROOT_COLORED} to compile onnx models.\n")
    
    run_onnx_model(onnx_model, small_lib_path, ONNX_MLIR_ROOT, verbosity)