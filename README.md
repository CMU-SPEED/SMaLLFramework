# SMaLL Framework
This branch is a draft of the the framework, Software for Machine Learning Libraries (SMaLL)

## Assumptions
 Dimensions of <b> Output Image </b> of any convolution:
  * \# Channels is a multiple of 16
  * \# Pixels in the Width is a multiple of 6
  * Height of the image &ge  kernel size\
  \
Dimensions of <b> Input Image</b>:
  * \# Channels is a multiple of 32
  _Note: This requirement is specifically for the the fused kernel, otherwise 16 channels minimum_
Stride for the 1x1 Convolution in the fused kernel is 1\
Dialation is always 1\
No padding

# Dependencies
C++ (14)\
gcc (9.3.0)\
GNU Make (4.2.1)\
Libtorch CPU version available here (https://pytorch.org/cppdocs/installing.html) (optional) \
CMake (3.16.3)

# Directory Structure
Libtorch Uses CMake to build its scripts. The <b>_CMakeLists.txt_</b> file provided here assumes the following directory Structure:

ContextDependentFusion/\
|____ <b>_libtorch_</b>/\
|____ CMakeLists.txt\
|____ <driver>.cpp\
