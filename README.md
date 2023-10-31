
# SMaLL: Software for rapidly instantiating Machine Learning Libraries

SMaLL is a framework for rapidly instantiating DNN based libraries for different architecture. In particular, the objective of SMaLL is to provide a quick and portable approach for developing DNN inference libraries for edge devices. At the core of SMaLL is a common set of loop nests across the different supported layers. Instead of writing every layer manually, each layer is instantiated from a small set of kernels.

A detailed discussion of the framework can be found [here](https://arxiv.org/abs/2303.04769) (preprint), and [here](https://dl.acm.org/doi/10.1145/3607870) (ACM Transaction of Embedded System article).

## Citation
Please use the following citation when referencing SMaLL
```
@article{10.1145/3607870,
author = {Sridhar, Upasana and Tukanov, Nicholai and Binder, Elliott and Low, Tze Meng and McMillan, Scott and Schatz, Martin D.},
title = {SMaLL: Software for Rapidly Instantiating Machine Learning Libraries},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1539-9087},
url = {https://doi.org/10.1145/3607870},
doi = {10.1145/3607870},
note = {Just Accepted},
journal = {ACM Trans. Embed. Comput. Syst.},
month = {jul}
}
```

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
- ARM-X1  (Snapdragon 888 Prime)

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

For quantized uint8 inference 
- Q-REF
- Q-ARM7E. (For the Arduino Nano 33 BLE)

Run the same build instructions as above
```
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_UARCH=Q-REF
```

## Build for Arduino

The Arduino platform (Q-ARM7E) has restrictions on the code layout.  To create an Arduino-friendly layout of a subset of the SMaLL library perform the following starting in the SMaLLFramework root directory.
``` bash
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_UARCH=Q-ARM7E
$ cd ../demo/small-MCU/
$ ./setup_env.sh ../../build/demo ../.. quantized_arm7E
```
Note that the command line arguments for the setup script are:
``` bash
$ ./setup_env.sh <path to build/demo> <path to SMaLLFramework> <platform dir>
```
This creates a new layout under the `build/demo/small` subdirectory.  If you want to build c++ executables for each model:
``` bash
$ cd ../../build/demo/small/
$ make
```

If running on the arduino, uncomment the appropriate model in small.ino
```c++
#include "mbed.h"
#define NANO33BLE 0

// #include "quantized/autoencoder.cpp"
// #include "quantized/resnet.cpp"
#include "quantized/dscnn.cpp"

```




## Supported Features

DNN Layers

- Convolution Layer (square and rectangular filters) 
- Depthwise Convolution Layer (only square filters)
- MaxPooling Layer  (square and rectangular filters)

With valid and same padding

- Dense Layer
- ReLU Activation

