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
import time
import os
import json
import argparse
import subprocess

# These are imported from onnx-mlir
from PyRuntime import OMExecutionSession

import ctypes
import pathlib
from numpy.ctypeslib import ndpointer

libsmall = pathlib.Path().absolute() / "../../lib/libpack.so"
libsmall = ctypes.CDLL(libsmall)
pack = libsmall.pack
pack.restype = None
pack.argtypes = [
    ndpointer(ctypes.c_float),
    ctypes.c_int, # co
    ctypes.c_int, # ci
    ctypes.c_int, # h
    ctypes.c_int, # w
    ctypes.c_int  # type
]

INPUT = 0
OUTPUT = 1
        
#*-------------------------------------------------------------------------------
# Repack weights for a given platform
# Assumes all filter weights are in CO, CI, H, W format
# Assumes filter weights contain a string "const_fold" in their name
def repack_weights(onnx_model_path):
    os.system(f"python3 repack_weights.py {onnx_model_path}")
    
#*-------------------------------------------------------------------------------
# helper function to get input dimensions
def get_input_dims(input_sign):
    input_dims = [1 if x==-1 else x for x in input_sign["dims"]]
    print(f"Input dims: {input_dims}")
    
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
        
    return input_dims

#*-------------------------------------------------------------------------------
# use os.system to run pytorch model
# if correctness is true, run pytorch model and save output to pytorch_output.npy
# else, run pytorch model and return fps
def run_pytorch_model(onnx_model_path, input_file, correctness):
    cmd_str = f"python run_torch_onnx.py {onnx_model_path} {input_file}"
    if(correctness):
        cmd_str = cmd_str + " --correctness"

    print(f"Running {cmd_str}")
    proc = subprocess.Popen(list(cmd_str.split(" ")), stdout=subprocess.PIPE)
    print(str(proc.stdout.readline().rstrip(), encoding='utf-8'))
    return float(proc.stdout.readline().rstrip().decode())
        

#*-------------------------------------------------------------------------------
# compiles an onnx model using onnx-mlir
# returns path to shared library that contains the model
def compile_model(onnx_model_path, small_lib_path, ONNX_MLIR_ROOT):
    
    repack_weights(onnx_model_path)
    print("Repacked weights!")
    onnx_model_path = onnx_model_path[:-5] + "_repacked.onnx"
    
    onnx_model_o = onnx_model_path[:-5] + ".o"
    onnx_model_so = onnx_model_path[:-5] + ".so"
    
    if(ONNX_MLIR_ROOT != ""):
        onnx_mlir_exe = ONNX_MLIR_ROOT + "/build/Debug/bin/onnx-mlir"
    else:
        onnx_mlir_exe = "onnx-mlir"
        
    # --enable-conv-opt-pass=false must be passed to onnx-mlir to avoid a bug
    # add more passes here if needed
    print("Compiling model...\n")
    os.system(f"{onnx_mlir_exe} --enable-conv-opt-pass=false --EmitObj {onnx_model_path}")
   
    # link small to the compiled model
    onnx_lib_link = f"-L{ONNX_MLIR_ROOT}/build/Debug/lib -lcruntime"
    small_lib_link = f"-L{small_lib_path} -lsmall"
    link_cmd = f"c++ {onnx_model_o} -o {onnx_model_so} -shared -fopenmp -fPIC {onnx_lib_link} {small_lib_link}"
    os.system(link_cmd)

    onnx_model_so_colored = colored(onnx_model_so, "light_cyan")
    print(f"{onnx_model_so_colored} compiled!")
    return onnx_model_so

#*-------------------------------------------------------------------------------
# compiles and executes an onnx model using onnx-mlir
# also runs the same model using pytorch
# prints fps results for small and pytorch
def run_onnx_model(onnx_model_path, small_lib_path, ONNX_MLIR_ROOT):
    
    # compile model using onnx-mlir
    onnx_model_so = compile_model(onnx_model_path, small_lib_path, ONNX_MLIR_ROOT)
    onnx_model_so_colored = colored(onnx_model_so, "light_cyan")
    print(f"Running model {onnx_model_so_colored}\n")
    
    # use the PyRuntime module provided by onnx-mlir to execute the model
    session = None
    try:
        session = OMExecutionSession(shared_lib_path=onnx_model_so)
    except Exception as E:
        print(E)
        print("Did you make sure LD_LIBRARY_PATH has the path to directory with libsmall.so?")
        print("\tRun: export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:<SMALL_ROOT_DIR>/lib\"")
        exit(-1)

    # create input data and store it in a file for pytorch to use
    # input data is assumed to be in NCHW format
    input_sign = json.loads(session.input_signature())[0]
    input_dims = get_input_dims(input_sign)
    model_input = np.random.rand(*input_dims).astype(np.float32)
    np.save("input.npy", model_input)
    
    # run pytorch model and get fps
    pytorch_time = run_pytorch_model(onnx_model_path, "input.npy", False)    
    
    # run small model and get fps out of 10000 runs
    # it is assumed that pytorch will also get the best fps out of 10000 runs
    total_time = 0
    best_time = 1e9
    RUNS = 10000
    for _ in range(RUNS):
        s = time.time()
        outputs = session.run(input=[model_input])
        e = time.time()
        total_time += (e-s)
        best_time = min(best_time, e-s)
    
    small_fps = input_dims[0]/best_time
    print("small, pytorch")
    print(f"{best_time}, {pytorch_time}, {pytorch_time/best_time}")
        


#*-------------------------------------------------------------------------------
# compiles and executes an onnx model using onnx-mlir
# also runs the same model using pytorch
# prints correctness results by comparing outputs from small and pytorch
def run_onnx_model_correctness(onnx_model_path, small_lib_path, ONNX_MLIR_ROOT):
    
    # compile model
    onnx_model_so = compile_model(onnx_model_path, small_lib_path, ONNX_MLIR_ROOT)
    onnx_model_so_colored = colored(onnx_model_so, "light_cyan")
    print(f"Running model {onnx_model_so_colored}\n")
    
    # use the PyRuntime module provided by onnx-mlir to execute the model
    session = None
    try:
        session = OMExecutionSession(shared_lib_path=onnx_model_so)
    except Exception as E:
        print(E)
        print("Did you make sure LD_LIBRARY_PATH has the path to directory with libsmall.so?")
        print("\tRun: export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:<SMALL_ROOT_DIR>/lib\"")
        exit(-1)

    # create input data and store it in a file for pytorch to use
    # input data is assumed to be in NCHW format
    input_sign = json.loads(session.input_signature())[0]
    input_dims = get_input_dims(input_sign)
    model_input = np.random.rand(*input_dims).astype(np.float32)
    np.save("input.npy", model_input)
    
    # run pytorch model and get outputs
    # outputs are assumed to be in NCHW format and are stored in pytorch_output.npy
    # once the outputs are loaded, they need to be packed into the same format as small for comparison
    os.system("rm -rf pytorch_output.npy")
    pytorch_time = run_pytorch_model(onnx_model_path, "input.npy", True)
    outputs_torch = np.load("pytorch_output.npy")
    
    # pack output for comparision
   
    outputs_torch_packed = outputs_torch.copy()
    if(len(outputs_torch_packed.shape) == 4):
        _, oc, oh, ow = outputs_torch_packed.shape
        pack(outputs_torch_packed, 1, oc, oh, ow, OUTPUT)

    # pack input data for small
    _, ih, iw, ic = input_dims
    model_input_packed = model_input.copy()
    model_input_packed = np.ravel(model_input_packed.transpose(0, 3, 1, 2), order='C')
    pack(model_input_packed, 1, ic, ih, iw, INPUT)
    model_input_packed = model_input_packed.reshape(1, ih, iw, ic)
    # model_input_packed = np.ravel(model_input_packed.reshape(1, ih, iw, ic).transpose(0, 3, 1, 2)).reshape(1, ih, iw, ic)
    # _, ic, ih, iw = model_input_packed.shape
    # print(model_input_packed.shape)
    # model_input_packed.transpose()
     
    # for _ in range(100):
    #     outputs = session.run(input=[model_input_packed])
    
    # run small model and get outputs
    s = time.time()
    outputs = session.run(input=[model_input_packed])
    e = time.time()
    
    # compare outputs
    passed = np.allclose(outputs[0], outputs_torch_packed, atol=1e-5)
    passed_str = colored("PASSED", "green") if passed else colored("FAILED", "red")
    print(f"{onnx_model_so_colored} {passed_str}")
    if not passed:
        print("small: ")
        print(outputs[0], outputs[0].shape)
        print("pytorch: ")
        print(outputs_torch_packed, outputs_torch_packed.shape)
        max_abs_diff = np.max(np.abs(outputs[0] - outputs_torch_packed))
        print("max abs diff: ", max_abs_diff)
    else:
        print(f"{e-s}, {pytorch_time}, {pytorch_time/(e-s)}")

#*------------------------------------------------------------------------------- 
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
        "-o", "--ONNX_MLIR_ROOT",
        default="",
        help="Path to onnx-mlir"
    )
    arg_parser.add_argument(
        "--correctness",
        action='store_true',
        help="Run correctness test."
    )
    
    return arg_parser.parse_args()
    
#*-------------------------------------------------------------------------------
if __name__ == "__main__":
    
    args = get_args()
    
    np.random.seed(0)
    
    onnx_model = args.model
    small_lib_path = args.lib
    ONNX_MLIR_ROOT = args.ONNX_MLIR_ROOT
    ONNX_MLIR_ROOT_COLORED = colored(ONNX_MLIR_ROOT, "green")
    correctness = args.correctness
    
    print(f"\nUsing {ONNX_MLIR_ROOT_COLORED} to compile onnx models.\n")
    
    if(correctness):
        run_onnx_model_correctness(onnx_model, small_lib_path, ONNX_MLIR_ROOT)
    else:
        run_onnx_model(onnx_model, small_lib_path, ONNX_MLIR_ROOT)