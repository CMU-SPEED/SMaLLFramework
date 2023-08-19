import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import sys
import argparse

def save_numpy_to_file(arr, filename):
    with open(filename, 'wb') as f:
        f.write(arr.tobytes())
        
def create_conv_model(ic, ih, iw, oc, oh, ow, k, p, conv_id="0"):
    
    # todo: make types more flexible and maybe add batch?
    conv_input = onnx.helper.make_tensor_value_info("I", TensorProto.FLOAT, [1, ic, ih, iw])
    
    weights_np = np.random.rand(oc, ic, k, k).astype(np.float32)
    conv_weights = onnx.helper.make_tensor(name="W", data_type=TensorProto.FLOAT, dims=weights_np.shape, vals=weights_np.flatten().tolist())
    
    conv_output = onnx.helper.make_tensor_value_info("O", TensorProto.FLOAT, [1, oc, oh, ow])
    
    conv_node = onnx.helper.make_node(
        name="conv2d",
        op_type="Conv",
        inputs=[
           "I", "W"
        ],
        outputs=[
            "O"
        ],
        kernel_shape=(k, k),
        pads=p
    )
    # print(conv_node)
    
    conv_graph = onnx.helper.make_graph(
        nodes=[conv_node],
        name="simple_conv",
        inputs=[conv_input],
        outputs=[conv_output],
        initializer=[conv_weights]
    )
    
    conv_model = onnx.helper.make_model(conv_graph, producer_name="conv_example")
    onnx.checker.check_model(conv_model)
    padding = ""
    if(ih == oh and iw == ow):
        padding = "same"
    else:
        padding = "valid"
    conv_model_name = f"conv_Ci{ic}_H{ih}_W{iw}_k{k}_Co{oc}_{padding}.onnx"
    
    onnx.save(conv_model, conv_model_name)
    print(f"{conv_model_name} is checked!")
    
def get_args():
    
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument(
        "-ic", "--input_channels",
        type=int,
        default=3
    )
    arg_parser.add_argument(
        "-ih", "--input_height",
        type=int,
        default=416
    )
    arg_parser.add_argument(
        "-iw", "--input_width",
        type=int,
        default=416
    )
    arg_parser.add_argument(
        "-k", "--kernel_size",
        type=int,
        default=3
    )
    arg_parser.add_argument(
        "-oc", "--output_channels",
        type=int,
        default=16
    )
    arg_parser.add_argument(
        "-p", "--padding",
        type=str,
        default="valid"
    )
    
    return arg_parser.parse_args()

def compute_output_dims(p, ih, iw, k):
    
    if(p == "same"):
        return ih, iw
    elif(p == "valid"):
        oh, ow = (ih-k-1), (iw-k-1)
        print(f"New output shape {oh}, {ow}")
        return oh, ow
    else:
        print(f"[ERROR] {p} padding is not supported.")
        print("Only valid and same are supported.")
        exit()
    
if __name__ == "__main__":
    
    args = get_args()
    ic, ih, iw, oc = args.input_channels, args.input_height, args.input_width, args.output_channels
    k, p = args.kernel_size, args.padding.lower()
    
    if(p != "valid" and p != "same"):
        print(f"[ERROR] {p} padding is not supported.")
        print("Only valid and same are supported.")
        exit()
    
    oh, ow = compute_output_dims(p, ih, iw, k)
    
    padding = None
    # todo: check this for stride!=1
    if(p == "same"):
        padding = (k - 2, k - 2, k - 2, k - 2)
    else:
        padding = (0,0,0,0)
    
    print("Conv parameters:")
    print(f"Input dims: {ic} x {ih} x {iw}")
    print(f"Filter dims: {oc} x {ic} x {k} x {k}")
    print(f"Output dims: {oc} x {oh} x {ow}")
    print(f"Stride = 1 | Padding = {padding}\n")
    create_conv_model(ic, ih, iw, oc, oh, ow, k, padding)
    