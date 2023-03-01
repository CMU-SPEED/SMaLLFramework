
# SMaLL Framework

A portable framework for writing high performance machine learning libraries .

## Build Instructions

### Specify µArch

The µarch is specified when building with CMake using the `CMAKE_UARCH ` parameter.

The reference build (REF) is written in purely in C and should work on any architecture with a C compiler.

### Modify Compiler Flags

Depending on the architecture and the compiler you are using, the compiler flags in `./CMakeLists.txt` might need to be modified.
For instance, if you are cross-compiling for arm you might specify the GNU C++ arm compiler 

```cmake
#If Cross Compiling for Arm
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ContextDependentFusion)
include(CTest)
set(CMAKE_VERBOSE_MAKEFILE off)
set(CMAKE_CXX_COMPILER g++)  -------> set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_C_COMPILER gcc) -------> set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc) 

```

### Build using CMake

Create a build folder from the Top Level directory and run CMake.  
Currently µarch can be 

For single precision floating point inference

- REF (the default)
- ZEN2 
- ARM
- ARM-A72 (Raspberry pi 4 b)
- ARM-A55 (Snapdragon 888 Silver)
- ARM-A78 (Snapdragon 888 Gold)
- ARM-X1 (Snapdragon 888 Prime)

For quantized uint8 inference 
- Q_REF
- Q_ARM7E. (For the Arduino Nano 33 BLE)

```bash
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_UARCH=<µarch>
$ cd demo
$ make model_*
```

This *should* generate 4 executables

- `./model_autoencoder.exe `
- `./model_dscnn.exe `
- `./model_resnet.exe `
- `./model_mobilenet.exe `

Run with 
`./model_<model_name>.exe`



<!-- ## Supported Features

3 DNN Layers

- Convolution Layer
- MaxPooling Layer
- ReLU Activation -->
