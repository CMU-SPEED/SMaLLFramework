import onnx_graphsurgeon as gs
import numpy as np
import onnx
import sys


def create_small_code(onnx_graph):
    # checks blocks and spits out C++ model using SMaLL as a backend
    pass

def parse(onnx_graph):
    print(onnx_graph)

if __name__ == "__main__":
    onnx_model_path = sys.argv[1]
    graph = gs.import_onnx(onnx.load(onnx_model_path))
    parse(graph)