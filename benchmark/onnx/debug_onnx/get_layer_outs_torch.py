import sys
import onnx
import numpy as np
import torch
from onnx2torch import convert
            

#*-------------------------------------------------------------------------------
if __name__ == "__main__":
    
    model = sys.argv[1]
    model_name = model[:-5]
    print(model_name)
    
    onnx_model_loaded = onnx.load(model)
    pytorch_model = convert(onnx_model_loaded)
    
    # input_shape = pytorch_model.parameters().size()
    # TODO: Figure out a way to determine input shape at runtime
    input_shape = (1, 96, 96, 3)
    input_tensor = np.random.rand(*input_shape).astype(np.float32)
    outs = pytorch_model(torch.from_numpy(input_tensor))
    
    
    for i, out in enumerate(outs[1:]):
        total_elems = out.numel()
        file_name = f"{model_name}_{i}_{total_elems}.bin"
        with open(file_name, "wb") as f:
            f.write(out.detach().numpy().tobytes())
    
    total_elems = outs[0].numel()
    file_name = f"{model_name}_{len(outs)}_{total_elems}.bin"
    with open(file_name, "wb") as f:
            f.write(outs[0].detach().numpy().tobytes())