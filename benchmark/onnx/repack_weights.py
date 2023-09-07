import numpy as np
import onnx
from onnx import numpy_helper
import sys

cob = 1
cib = 1

def repack_weights(onnx_model_path):
    onnx_model_loaded = onnx.load(onnx_model_path)
    initializers = onnx_model_loaded.graph.initializer
    for idx, init in enumerate(initializers):
        if("const_fold" in init.name):
            W = numpy_helper.to_array(init)
            if(len(W.shape) == 4):
                co, ci, kh, kw = W.shape
                W_packed = W.reshape(co//cob, cob, ci//cib, cib, kh, kw).transpose(0, 2, 4, 5, 1, 3)
                W_packed = np.ravel(W_packed).reshape(co, ci, kh, kw)
                onnx_model_loaded.graph.initializer[idx].CopyFrom(numpy_helper.from_array(W_packed, init.name))
                
    onnx.save(onnx_model_loaded, onnx_model_path[:-5]+"_repacked.onnx")
    
if __name__ == "__main__":
    repack_weights(sys.argv[1])