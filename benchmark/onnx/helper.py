import os

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