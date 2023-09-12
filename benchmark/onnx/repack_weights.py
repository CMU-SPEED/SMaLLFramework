import numpy as np

import onnx
from onnx import numpy_helper
import sys

import ctypes
import pathlib
from numpy.ctypeslib import ndpointer

libsmall = pathlib.Path().absolute() / "libpack.so"
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

FILTER_CONV = 3

#*-------------------------------------------------------------------------------
def get_platform_params(platform):
    if(platform == "ref"):
        return 1, 1
    elif(platform == "zen2"):
        return 16, 16
    elif(platform == "arm"):
        return 16, 16
    else:
        print(f"[ERROR] Invalid platform {platform}")
        exit(-1)

#*-------------------------------------------------------------------------------
# Repack weights for a given platform
# Assumes all filter weights are in CO, CI, H, W format
# Assumes filter weights contain a string "const_fold" in their name
def repack_weights(onnx_model_path, platform):
    cob, cib = get_platform_params(platform)
    onnx_model_loaded = onnx.load(onnx_model_path)
    initializers = onnx_model_loaded.graph.initializer
    for idx, init in enumerate(initializers):
        if("const_fold" in init.name):
            W = numpy_helper.to_array(init)
            if(len(W.shape) == 4):
                print(f"repacking {init.name} that has {W.shape}")
                co, ci, kh, kw = W.shape
                # strictly speacking this assumes that co and ci are divisible by cob and cib or equal to 3
                # cob = 3 if co == 3 else cob
                # cib = 3 if ci == 3 else cib
                W_packed = W.reshape(co//cob, cob, ci//cib, cib, kh, kw).transpose(0, 2, 4, 5, 1, 3)
                W_packed = np.ravel(W_packed).reshape(co, ci, kh, kw)
                # W_packed = W.copy()
                # pack(W_packed, co, ci, kh, kw, FILTER_CONV)
                onnx_model_loaded.graph.initializer[idx].CopyFrom(numpy_helper.from_array(W_packed, init.name))
               
    print(f"Data has been repacked for {platform}!")
    onnx.save(onnx_model_loaded, onnx_model_path[:-5]+"_repacked.onnx")
    
#*-------------------------------------------------------------------------------
if __name__ == "__main__":
    repack_weights(sys.argv[1], sys.argv[2])