#include <math.h>
#include <assert.h>
// #include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <small/config.h>     // was ./config.h
#include <small/interface.h>  // was src/interface.h
#include <params.h>           // comes from active kernels subdirectory
#include <intrinsics.h>       // comes from active kernels subdirectory

#ifndef op_dim
#define op_dim(IN_dim, stride, K_dim, padding, OUT_dim)   \
    {                                            \
        OUT_dim = (IN_dim +  2*padding - K_dim) / stride + 1; \
    }
#endif

#ifndef op_dim_stride
#define op_dim_stride(IN_dim, stride ,OUT_dim)        \
    {                                                          \
        OUT_dim = ceil(IN_dim/stride) \
    }
#endif

#include <small/direct_convolution.h>


void Conv2D(int layer_num, int kernel_size, int stride, char padding, int output_channels, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr)
{
    if (input_channels < C_ob)
    {
        printf("small input channels");
        initial_direct_convolution<W_ob, C_ob, 1>(1,
                                                     kernel_size, kernel_size, input_channels,
                                                     output_channels, 1,
                                                     input_height, input_width,
                                                     padding,
                                                     input_ptr, filter_ptr, output_ptr);
    }
    else
    {
        if (stride == 1)
        {
            direct_convolution<W_ob, C_ob, C_ib, 1, 'c'>(1,
                                                            kernel_size, kernel_size, input_channels,
                                                            output_channels, 1,
                                                            input_height, input_width,
                                                            padding,
                                                            input_ptr, filter_ptr, output_ptr);
        }
        else if (stride == 2)
        {
            direct_convolution<W_ob, C_ob, C_ib, 2, 'c'>(1,
                                                            kernel_size, kernel_size, input_channels,
                                                            output_channels, 1,
                                                            input_height, input_width,
                                                            padding,
                                                            input_ptr, filter_ptr, output_ptr);
        }
        else
        {
            printf("This stride is unsupported, please change the interface.cpp file\n");
        }
    }
}


void PartialConv2D(int layer_num, int kernel_size, int stride, char padding, int output_channels, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr)
{
    direct_convolution_partial<W_ob, C_ob, C_ib, 1, 'c'>(1,
                                                         kernel_size, kernel_size, input_channels,
                                                         output_channels, 1,
                                                         input_height, input_width,
                                                         padding,
                                                         input_ptr, filter_ptr, output_ptr);
}


void GroupConv2D(int layer_num, int kernel_size, int stride, char padding, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr)
{
    direct_convolution<W_ob, C_ob, C_ib, 1, 'c'>(C_ob,
                                                                     kernel_size, kernel_size, C_ob,
                                                                     C_ob,
                                                                     input_channels / C_ob,
                                                                     input_height, input_width,
                                                                     padding,
                                                                     input_ptr, filter_ptr, output_ptr);
}

void DepthwiseConv2D(int layer_num, int kernel_size, int stride, char padding, int input_channels, int input_height, int input_width, float *input_ptr, float *filter_ptr, float *output_ptr)
{
    if (stride == 1)
    {
        direct_convolution<W_ob, C_ob, 1, 1, 'c'>(1,
                                                        kernel_size, kernel_size, 1,
                                                        1, input_channels,
                                                        input_height, input_width,
                                                        padding,
                                                        input_ptr, filter_ptr, output_ptr);
    }
    else if (stride == 2)
    {
        direct_convolution<W_ob, C_ob, 1, 2, 'c'>(1,
                                                        kernel_size, kernel_size, 1,
                                                        1, input_channels,
                                                        input_height, input_width,
                                                        padding,
                                                        input_ptr, filter_ptr, output_ptr);
    }
    else
    {
        printf("This stride is unsupported, please change the interface.cpp file\n");
    }
}


void Maxpool2D(int layer_num, int kernel_size, int stride, char padding, int input_channels, int input_height, int input_width, float *input_ptr, float *output_ptr)
{
    if (stride == 2)
    {
        direct_convolution<W_ob, C_ob, 1, 2, 'p'>(1,
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

void ReLUActivation(int layer_num, int input_channels, int input_height, int input_width, float *input_ptr, float *output_ptr)
{
    direct_convolution<W_ob, C_ob, 1, 1, 'a'>(1,
                                                    1, 1, 1,
                                                    1, input_channels,
                                                    input_height, input_width,
                                                    'v',
                                                    input_ptr, NULL, output_ptr);
}

void Dense(int layer_num, int output_elements, int input_elements, float *input_ptr, float *filter_ptr, float *output_ptr)
{
    direct_convolution<W_ob, C_ob, C_ib, 1, 'c'>(1,
                                                 1, 1, input_elements,
                                                 output_elements, 1,
                                                 1, 1,
                                                 'v',
                                                 input_ptr, filter_ptr, output_ptr);
}
