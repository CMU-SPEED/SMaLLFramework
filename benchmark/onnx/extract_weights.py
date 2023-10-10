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
    
    weight_names_copy = []
    weight_list = []
    weight_size_list = []
    for node in onnx_model_loaded.graph.node:
        inputs = node.input
        weight_names = set(inputs).intersection(set(init_dict.keys()))
        if(len(weight_names) > 0 and (node.op_type == "Conv" or node.op_type == "Matmul" or node.op_type == "Gemm" or node.op_type == "Add")):
            for weight_name in weight_names:
                weight_names_copy.append(weight_name)
                weight_idx = init_dict[weight_name]
                W = numpy_helper.to_array(initializers[weight_idx])
                weight_list.append(W)
                
    zeros_pad = np.zeros(5, dtype=np.float32)
    with open(f"{onnx_model_path[:-5]}_weights.bin", "wb") as f:
        for w in weight_list:
            weight_size_list.append(w.shape)
            f.write(w.tobytes())
            f.write(zeros_pad.tobytes())
            
    return weight_names_copy, weight_size_list    
        
onnx_model_path = sys.argv[1]
names, weight_size_list = extract_weights(onnx_model_path)

weights_file = onnx_model_path[:-5] + "_weights.bin"
weights_data = np.fromfile(weights_file, dtype=np.float32)

sum_total_elems = 0
for i, shape in enumerate(weight_size_list):
    total_elems = np.prod(shape)
    sum_total_elems += total_elems
    print(f"{names[i]}\n", shape, weights_data[sum_total_elems:sum_total_elems+5])
    sum_total_elems += 5

diff = weights_data.shape[0] - sum_total_elems
if(diff != 0):
    print(f"[ERROR] Mismatch number of elems. Remaining elements {diff}")
