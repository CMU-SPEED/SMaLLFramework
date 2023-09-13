import numpy as np

import onnx
from onnx import numpy_helper
import sys

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

FILTER_DW = 2
FILTER_CONV = 3
FILTER_FC = 4

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
    init_dict = {}
    for i, w in enumerate(initializers):
        init_dict[w.name] = i
    
    for node in onnx_model_loaded.graph.node:
        if(node.op_type == "Conv"):
            # inputs should be an ordered list of [input, weight, bias]
            inputs = node.input
            weights_name = inputs[1]
            weight_idx = init_dict[weights_name]
            W = numpy_helper.to_array(initializers[weight_idx])
            assert(len(W.shape) == 4 and "weight shape is not 4D")
            co, ci, kh, kw = W.shape
            W_packed = W.copy().astype(np.float32)
            #! for the filter type, we are going to make the naive assumption that if ci > 1, then it is a conv filter
            filter_type = FILTER_CONV if ci > 1 else FILTER_DW
            filter_type_str = "conv" if ci > 1 else "conv_dw"
            pack(W_packed, co, ci, kh, kw, filter_type)
            print(f"repacking {weights_name} that has {W.shape} for {filter_type_str}")
            onnx_model_loaded.graph.initializer[weight_idx].CopyFrom(numpy_helper.from_array(W_packed, weights_name))
               
    print(f"\nData has been repacked for {platform}!")
    onnx.save(onnx_model_loaded, onnx_model_path[:-5]+"_repacked.onnx")
    print("New model has been saved")

    
#*-------------------------------------------------------------------------------
if __name__ == "__main__":
    repack_weights(sys.argv[1], sys.argv[2])