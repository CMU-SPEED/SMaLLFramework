#pragma once

// #define W_ob 6
// #define C_ob 16
// #define C_ib C_ob

// //Include params file
// #define kernel_path "kernels/"

// // #define param "params.h"
// #define intrinsics "intrinsics.h"

// #define param_path(kernel_path, uarch, param) kernel_path ## uarch ## param
// #define intrinsics_path(kernel_path, uarch, intrinsics) kernel_path##uarch##intrinsics

// // #define PPCAT_NX(A, B) A##B
// // #define PPCAT(A, B) PPCAT_NX(A, B)

// // #define param_path PPCAT(PPCAT(kernel_path, uarch), param)
// // #define intrinsics_path PPCAT(PPCAT(kernel_path, uarch), intrinsics)

// #define STRINGIZE_NX(A) #A
// #define STRINGIZE(A) STRINGIZE_NX(A)

// #include STRINGIZE(param_path(kernels/ , zen2/, param.h))
// #include intrinsics_path(kernel_path, uarch, intrinsics)

//Include the same parameters, use scalar implementations for correctness.

// #if uarch == ZEN2
// #include "../src/kernels/zen2/params.h"

// #elif uarch == REF
// #include "kernels/reference/params.h"
// #endif

void print_build_info_check();



void check_Conv2D(int layer_num, int kernel_size, int stride, char padding, int output_channels, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr);


void check_PartialConv2D( int layer_num, int kernel_size, int stride, char padding, int output_channels, int input_channels, int input_height, int input_width, float * input_ptr, float * filter_ptr, float *output_ptr);


void check_GroupConv2D( int layer_num, int kernel_size, int stride, char padding, int input_channels, int input_height, int input_width, float * input_ptr, float * filter_ptr, float *output_ptr);


void check_DepthwiseConv2D(int layer_num, int kernel_size, int stride, char padding, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr);


void check_Maxpool2D( int layer_num, int kernel_size, int stride, char padding, int input_channels, int input_height, int input_width, float * input_ptr, float * output_ptr);

void check_ReLUActivation( int layer_num, int input_channels, int  input_height, int input_width, float * input_ptr, float * output_ptr);

void check_Dense(int layer_num, int output_elements, int input_elements, float *input_ptr, float *filter_ptr, float *output_ptr);

// Normalization layers may or may not be required (constant folding)
