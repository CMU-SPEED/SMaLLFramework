
# SMaLL Framework

A portable high performance machine learning library framework.

## Supported Features

3 DNN Layers

- Convolution Layer
- MaxPooling Layer
- ReLU Activation

There is currently no support for padding other than “valid” padding

## Tested Platforms and Architectures
|Name|Architecture|Created|
|---|---|---|
|[AMD Zen2](platform_notes/AMD.md)|x86-64|"March 14| 2022 11:33 AM"|
|[< New >](platform_notes/New.md)|< your arch >|"March 14| 2022 11:33 AM"|

## Upcoming

1. Tests on an ARM platform
2. Support for “full” padding
3. Depthwise Convolution Layer
4. Partial Convolution
5. Group Convolution Layer

## Build Instructions

### Specify µArch

To architecture specific features such as SIMD and NEON, modify the `uarch` parameter in `./config.h`

Please choose one of the available architectures

```c
#define REF 0
#define ZEN2 1
#define ARM 2

#ifndef uarch
#define uarch ZEN2 //<---change this
#endif
```

The reference build (REF) is written in purely in C and should work on any architecture with a C compiler.

### Modify Compiler Flags

Depending on the architecture and the compiler you are using, the compiler flags in `./CMakeLists.txt` might need to be modified

```c
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ContextDependentFusion)
include(CTest)
set(CMAKE_VERBOSE_MAKEFILE off)
set(CMAKE_CXX_COMPILER g++) #<--- Specify C++ compiler
set(CMAKE_C_COMPILER gcc) #<--- Specify C compiler

#Compiler Flags 
SET(GCC_AVX_COMPILE_FLAGS "-fopenmp -mavx2 -mfma -O3 -fpermissive -march=native")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_AVX_COMPILE_FLAGS}")
```

### Build using CMake

Create a build folder from the Top Level directory and run CMake.

```bash
$ mkdir build
$ cd build
$ cmake ../
$ make
```

This *should* generate 3 executables

- `test_interface_relu.x`
- `test_interface_pool.x`
- `test_interface_conv.x`

Each executable accepts an input size (C,H,W) and a  filter size where applicable and outputs something like the following

```bash
$ ./test_interface_conv.x 256  30 30  3 1 'v' 192
layer 0 
µArch: zen2 
W_ob =  6 
C_ob = 16 
SIMD = 8 
3463381.00      2140101.00 (time in nanoseconds)
```

### Test your build

The Experiments folder has some scripts to run a variety of sizes with the framework

```bash
$./Experiments/test_all.sh
```

Runs ./test_interface_pool.x
