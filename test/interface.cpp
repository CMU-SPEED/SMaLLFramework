#include <math.h>
#include <assert.h>
// #include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../config.h"
#include "interface.h"

#if uarch == ZEN2
#include "../src/kernels/zen2/params.h"
#elif uarch == ARM
#include "../src/kernels/arm/params.h"
#elif uarch == REF
#include "../src/kernels/reference/params.h"
#endif

#include "intrinsics.h"

#ifndef op_dim
#define op_dim(IN_dim, stride, K_dim, OUT_dim)   \
    {                                            \
        OUT_dim = (IN_dim - K_dim) / stride + 1; \
    }
#endif

#include "naive_direct_convolution.h"


void print_build_info_check() {
    char const * archs[] = {"reference", "zen2", "arm"};
    printf("µArch: %s \n W_ob =  %d \n C_ob = %d \n SIMD = %d \n", archs[uarch], W_ob, C_ob, SIMD);
}


void check_Conv2D(int layer_num, int kernel_size, int stride, char padding, int output_channels, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr)
{
    // printf(" conv \n");
    if(stride == 1)
    {
    direct_convolution_naive<W_ob, C_ob, C_ib, 1, 'c'>(1,
                                                                     kernel_size, kernel_size, input_channels,
                                                                     output_channels, 1,
                                                                     input_height, input_width,
                                                                     padding,
                                                                     input_ptr, filter_ptr, output_ptr);
    }
    else if (stride == 2)
    {
        direct_convolution_naive<W_ob, C_ob, C_ib, 2, 'c'>(1,
                                                           kernel_size, kernel_size, input_channels,
                                                           output_channels, 1,
                                                           input_height, input_width,
                                                           padding,
                                                           input_ptr, filter_ptr, output_ptr);
    }
    else
    {
        printf("This stride is unsupported, please change the interface.cpp file\n");
        exit(-1);
    }
}


void check_PartialConv2D(int layer_num, int kernel_size, int stride, char padding, int output_channels, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr) {
    // printf("partial conv\n");
    direct_convolution_partial<W_ob, C_ob, C_ib,  1, 'c'>(1,
                                                                       kernel_size, kernel_size, input_channels,
                                                                       output_channels, 1,
                                                                       input_height, input_width,
                                                                       padding,
                                                                       input_ptr, filter_ptr, output_ptr);
}


void check_GroupConv2D(int layer_num, int kernel_size, int stride, char padding, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr)
{
    // printf(" group conv \n");
    direct_convolution_naive<W_ob, C_ob, C_ib, 1 , 'c'>(C_ob,
                                                                     kernel_size, kernel_size, C_ob,
                                                                     C_ob, input_channels / C_ob,
                                                                     input_height, input_width,
                                                                     padding,
                                                                     input_ptr, filter_ptr, output_ptr);
}

void check_DepthwiseConv2D(int layer_num, int kernel_size, int stride, char padding, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr)
{
    // printf(" dw conv \n");
    if(stride  == 1)
    {
    direct_convolution_naive<W_ob, C_ob, 1, 1, 'c'>(1,
                                                                  kernel_size, kernel_size, input_channels,
                                                                  1, input_channels,
                                                                  input_height, input_width,
                                                                  padding,
                                                                  input_ptr, filter_ptr, output_ptr);
    }
    else if(stride == 2)
    {
        direct_convolution_naive<W_ob, C_ob, 1, 2, 'c'>(1,
                                                             kernel_size, kernel_size, input_channels,
                                                             1, input_channels,
                                                             input_height, input_width,
                                                             padding,
                                                             input_ptr, filter_ptr, output_ptr);
    }
    else
    {
        printf("This stride is unsupported, please change the interface.cpp file\n");
        exit(-1);
    }
}


void check_Maxpool2D(int layer_num, int kernel_size, int stride, char padding, int input_channels, int input_height, int input_width, float *input_ptr, float *output_ptr)
{
    // printf(" pool \n");
    if(stride == 2)
    {
    direct_convolution_naive<W_ob, C_ob, 1, 2, 'p'>(1,
                                                                  kernel_size, kernel_size, 1,
                                                                  1, input_channels,
                                                                  input_height, input_width,
                                                                  padding,
                                                                  input_ptr, NULL, output_ptr);
    }
    else
    {
        printf("This stride is unsupported, please change the interface.cpp file\n");
        exit(-1);
    }
}

void check_ReLUActivation(int layer_num, int input_channels, int input_height, int input_width, float *input_ptr, float *output_ptr)
{
    // printf(" relu \n");
    direct_convolution_naive<W_ob, C_ob, 1, 1, 'a'>(1,
                                                         1, 1, 1,
                                                         1, input_channels,
                                                         input_height, input_width,
                                                         'v',
                                                         input_ptr, NULL, output_ptr);
}
//
