import numpy as np

import onnx
from onnx import numpy_helper
import sys

FILTER_DW = 2
FILTER_CONV = 3
FILTER_FC = 4

def extract_weights(onnx_model_path):
    onnx_model_loaded = onnx.load(onnx_model_path)
    
    initializers = onnx_model_loaded.graph.initializer
    init_dict = {}
    for i, w in enumerate(initializers):
        init_dict[w.name] = i
    
    weight_list = []
    for node in onnx_model_loaded.graph.node:
        inputs = node.input
        weight_names = set(inputs).intersection(set(init_dict.keys()))
        if(len(weight_names) > 0):
            for weight_name in weight_names:
                weight_idx = init_dict[weight_name]
                W = numpy_helper.to_array(initializers[weight_idx])
                weight_list.append(W)
    
    zeros_pad = np.zeros(5, dtype=np.float32)
    with open(f"{onnx_model_path[:-5]}_weights.bin", "wb") as f:
        for w in weight_list:
            f.write(w.tobytes())
            f.write(zeros_pad.tobytes())     
        
onnx_model_path = sys.argv[1]
extract_weights(onnx_model_path)

weights_file = onnx_model_path[:-5] + "_weights.bin"
weights_data = np.fromfile(weights_file, dtype=np.float32)
print(weights_data, weights_data.shape)

