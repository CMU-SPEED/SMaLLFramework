import os
import subprocess

def generate_and_compile(template_params):
    # Extract the parameters
    input_size = template_params["input_size"]
    kernel_size = template_params["kernel_size"]
    stride = template_params["stride"]

    # Generate the C++ function instantiation
    function_instantiation = f"""
extern "C" {{
    void avgpool2d_float(const float* input, float* output, int input_size, int kernel_size, int stride) {{
        avgpool2d<float>(input, output, input_size, kernel_size, stride);
    }}
}}
    """

    # Step 1: Append the instantiation to the existing stub
    with open("static_interface_abstract_stub.txt", "a") as f:
        f.write(function_instantiation)

    # Step 2: Copy the stub file to the C++ file that will be compiled
    os.system("cp static_interface_abstract_stub.txt static_interface_driver.cpp")

    # Step 3: Compile the C++ file into a shared library
    if not os.path.exists("build"):
        os.makedirs("build")
    compile_command = "make static_interface.o"
    subprocess.run(compile_command, shell=True, check=True)
