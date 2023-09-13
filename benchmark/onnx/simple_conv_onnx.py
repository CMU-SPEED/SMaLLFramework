import onnx
import numpy as np
from onnx import TensorProto
import argparse

#*------------------------------------------------------------------------------- 
# create an onnx model that contains a single conv node
# assumes that the input is NCHW
# assumes that the filter is OIHW
# assumes that the output is NCHW
def create_conv_model(ic, ih, iw, oc, oh, ow, k, p, s, conv_id="0"):
    
    # todo: make types more flexible and maybe add batch?
    conv_input = onnx.helper.make_tensor_value_info("I", TensorProto.FLOAT, [1, ic, ih, iw])
    
    weights_np = np.random.rand(oc, ic, k, k).astype(np.float32)
    conv_weights = onnx.helper.make_tensor(name="const_fold_conv2d_0_W", data_type=TensorProto.FLOAT, dims=weights_np.shape, vals=weights_np.flatten().tolist())
    
    bias_np = np.ones(oc).astype(np.float32)
    conv_bias = onnx.helper.make_tensor(name="const_fold_conv2d_0_b", data_type=TensorProto.FLOAT, dims=bias_np.shape, vals=bias_np.flatten().tolist())
    
    conv_output = onnx.helper.make_tensor_value_info("O", TensorProto.FLOAT, [1, oc, oh, ow])
    
    conv_node = onnx.helper.make_node(
        name="conv2d",
        op_type="Conv",
        inputs=[
           "I", "const_fold_conv2d_0_W", "const_fold_conv2d_0_b"
        ],
        outputs=[
            "O"
        ],
        kernel_shape=(k, k),
        pads=p,
        strides=s
    )
    # print(conv_node)
    
    conv_graph = onnx.helper.make_graph(
        nodes=[conv_node],
        name="simple_conv",
        inputs=[conv_input],
        outputs=[conv_output],
        initializer=[conv_weights, conv_bias]
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
    
#*-------------------------------------------------------------------------------
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
    arg_parser.add_argument(
        "-s", "--stride",
        type=str,
        default="valid"
    )
    
    return arg_parser.parse_args()

#*-------------------------------------------------------------------------------
# def compute_output_dims(p, ih, iw, k):
    
#     if(p == "same" or k == 1):
#         return ih, iw
#     elif(p == "valid"):
#         oh, ow = (ih-k-1), (iw-k-1)
#         print(f"New output shape {oh}, {ow}")
#         return oh, ow
#     else:
#         print(f"[ERROR] {p} padding is not supported.")
#         print("Only valid and same are supported.")
#         exit()
    
#*-------------------------------------------------------------------------------
if __name__ == "__main__":
    
    args = get_args()
    ic, ih, iw, oc = args.input_channels, args.input_height, args.input_width, args.output_channels
    k = args.kernel_size
    p = tuple([int(x) for x in args.padding.split(",")])
    s = tuple([int(x) for x in args.stride.split(",")])
    
    # oh, ow = compute_output_dims(p, ih, iw, k)
    
    # oh = (ih-k+2*p[0])//s[0] + 1
    # ow = (iw-k+2*p[1])//s[1] + 1
    
    oh =  (ih-k+2*(p[0]+p[2]))//s[0] + 1
    ow =  (iw-k+2*(p[1]+p[3]))//s[1] + 1
    
    print("Conv parameters:")
    print(f"Input dims: {ic} x {ih} x {iw}")
    print(f"Filter dims: {oc} x {ic} x {k} x {k}")
    print(f"Output dims: {oc} x {oh} x {ow}")
    print(f"Stride = 1 | Padding = {p}\n")
    create_conv_model(ic, ih, iw, oc, oh, ow, k, p, s)
    