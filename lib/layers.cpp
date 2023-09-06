// c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// onnx data struct
// #include "OMTensor.hpp"
#include "OMTensor.inc"

// small
#include "small.h"
#include "small/Tensor.hpp"
#include "small/Conv2DLayer.hpp"

// #define WRAPPER_DEBUG 1

extern "C" {

//****************************************************************************
// print shape for debugging
void print_shape(const int64_t *shape, int64_t shape_size) {
    if(shape_size != 0) {
        printf("(%ld", shape[0]);
        for(int i=1; i<shape_size; i++) {
            printf(", %ld", shape[i]);
        }
        printf(")");
    }
    printf("\n");
}

//****************************************************************************
void print_tensor(small::Tensor<small::FloatBuffer> tensor) {
    float *data = tensor.buffer().data();
    size_t elems = tensor.capacity();

    printf(" [%f", data[0]);
    for(int i=1; i<10; i++) {
        printf(" %f", data[i]);
    }
    printf("]\n");
}

//****************************************************************************
// we want to get rid of this to avoid copying
small::Tensor<small::FloatBuffer> omtensor_to_smalltensor(
    OMTensor *tensor
){

    const int64_t *shape = omTensorGetShape(tensor);
    small::shape_type shape_small = {(size_t)shape[0],(size_t)shape[1],(size_t)shape[2],(size_t)shape[3]};

    float *onnx_data = (float*)omTensorGetDataPtr(tensor);
    small::Tensor<small::FloatBuffer> small_tensor(shape_small);

    memcpy(
        &small_tensor.buffer()[0],
        onnx_data,
        small_tensor.capacity()
    );
    
    return small_tensor;
}

//****************************************************************************
// we want to get rid of this to avoid copying
small::Tensor<small::FloatBuffer> omtensor_to_smalltensor_unpacked(
    OMTensor *tensor, small::BufferTypeEnum buf_type
){
    const int64_t *shape = omTensorGetShape(tensor);
    small::shape_type shape_small = {(size_t)shape[0],(size_t)shape[1],(size_t)shape[2],(size_t)shape[3]};
    int64_t ndims = omTensorGetRank(tensor);
    // printf("shape = ");
    // print_shape(shape, ndims);

    int64_t elems = omTensorGetNumElems(tensor);
    // printf("total_elems = %ld\n", elems);

    float *onnx_data = (float*)omTensorGetDataPtr(tensor);
    small::Tensor<small::FloatBuffer> small_tensor(shape_small);

    memcpy(
        &small_tensor.buffer()[0],
        onnx_data,
        elems
    );
    
    return small_tensor;
}

// // //****************************************************************************
// void Conv2D(
//     OMTensor *output, 
//     OMTensor *input, 
//     OMTensor *filter
// ){

//     // #ifdef WRAPPER_DEBUG
//     // printf("Entering Conv2D SMaLL wrapper\n");
//     // #endif

//     #ifdef WRAPPER_DEBUG
//     printf("Packing input\n");
//     #endif
    
//     // small::Tensor<small::FloatBuffer> input_small = omtensor_to_smalltensor(input, small::INPUT);
//     small::Tensor<small::FloatBuffer> input_small({1,3,416,416});
//     // print_tensor(input_small);
//     size_t ci = input_small.shape()[1];
//     size_t h = input_small.shape()[2];
//     size_t w = input_small.shape()[3];

//     #ifdef WRAPPER_DEBUG
//     printf("Packing filter\n");
//     #endif

//     // small::Tensor<small::FloatBuffer> filter_small = omtensor_to_smalltensor(filter, small::FILTER_CONV);
//     small::Tensor<small::FloatBuffer> filter_small({64,3,3,3});
//     size_t co = filter_small.shape()[0];
//     size_t k = filter_small.shape()[2];

//     small::Tensor<small::FloatBuffer> output_small({1U, co, h, w});

//     #ifdef WRAPPER_DEBUG
//     printf("Running Conv2D\n");
//     #endif

//     small::Conv2DLayer<small::FloatBuffer> conv2d(
//         input_small.shape(),
//         k, k, 
//         1, 
//         small::PADDING_F,
//         co,
//         filter_small.buffer()
//     );

//     conv2d.compute_output({&input_small}, {&output_small});

//     #ifdef WRAPPER_DEBUG
//     printf("Copying output\n");
//     #endif

//     small::convert_dc2tensor(
//         output_small.buffer().data(),
//         small::INPUT,
//         1, co, h, w,
//         C_ib, C_ob,
//         (float*)omTensorGetDataPtr(output)
//     );

//     // memcpy(
//     //     omTensorGetDataPtr(output),
//     //     &output_small.buffer()[0],
//     //     output_small.capacity()
//     // );

// }

//****************************************************************************
int conv2D_w_bias_cnt = 0;
void Conv2D_w_bias(
    OMTensor *output, 
    OMTensor *input,
    OMTensor *filter,
    OMTensor *bias,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    uint8_t stride_x, uint8_t stride_y
    // ,uint8_t k_x, uint8_t k_y
) {

    printf("entering Conv2D_w_bias %d\n", conv2D_w_bias_cnt++);

    int oc = omTensorGetShape(filter)[0];
    int ic = omTensorGetShape(filter)[1];
    int k = omTensorGetShape(filter)[2];

    int ih = omTensorGetShape(input)[2];
    int iw = omTensorGetShape(input)[3];

    int oh = omTensorGetShape(output)[2];
    int ow = omTensorGetShape(output)[3];

    // printf("t_pad = %d | b_pad = %d | l_pad = %d | r_pad = %d\n", t_pad, b_pad, l_pad, r_pad);
    // printf("stride_x = %d | stride_y = %d\n", stride_x, stride_y);
    // exit(-1);
    
    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer filt(omTensorGetNumElems(filter), (float*)omTensorGetDataPtr(filter));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));
    small::FloatBuffer bias_buffer(omTensorGetNumElems(bias), (float*)omTensorGetDataPtr(bias));

    // printf("oc=%d, ic=%d, k=%d, ih=%d, iw=%d, oh=%d, ow=%d\n", oc, ic, k, ih, iw, oh, ow);

    small::Bias(
        oc,
        oh,
        ow,
        bias_buffer, 
        out
    );

    small::PartialConv2D(
        k, stride_x,
        t_pad, b_pad, l_pad, r_pad,
        oc,
        ic,
        ih, iw,
        in,
        filt,
        out
    );

}



//****************************************************************************
int conv2D_wo_bias_cnt = 0;
void Conv2D_wo_bias(
    OMTensor *output, 
    OMTensor *input,
    OMTensor *filter,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    uint8_t stride_x, uint8_t stride_y
    // ,uint8_t k_x, uint8_t k_y
) {

    printf("entering Conv2D_wo_bias %d\n", conv2D_wo_bias_cnt++);

    int oc = omTensorGetShape(filter)[0];
    int ic = omTensorGetShape(filter)[1];
    int k = omTensorGetShape(filter)[2];

    int ih = omTensorGetShape(input)[2];
    int iw = omTensorGetShape(input)[3];

    int oh = omTensorGetShape(output)[2];
    int ow = omTensorGetShape(output)[3];

    // printf("t_pad = %d | b_pad = %d | l_pad = %d | r_pad = %d\n", t_pad, b_pad, l_pad, r_pad);
    // printf("stride_x = %d | stride_y = %d\n", stride_x, stride_y);
    // exit(-1);
    
    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer filt(omTensorGetNumElems(filter), (float*)omTensorGetDataPtr(filter));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    // printf("oc=%d, ic=%d, k=%d, ih=%d, iw=%d, oh=%d, ow=%d\n", oc, ic, k, ih, iw, oh, ow);

    small::Conv2D(
        k, stride_x,
        t_pad, b_pad, l_pad, r_pad,
        oc,
        ic,
        ih, iw,
        in,
        filt,
        out
    );

}

void MaxPool2D(
    OMTensor *output,
    OMTensor *input
)
{
    printf("MaxPool2D\n");

    int ic = omTensorGetShape(input)[1];
    int ih = omTensorGetShape(input)[2];
    int iw = omTensorGetShape(input)[3];

    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    int k = 2;
    uint8_t pad = 0;
    small::MaxPool2D<small::FloatBuffer>(
        k, 1,
        pad, pad, pad, pad,
        ic,
        ih, iw,
        in,
        out
    );
}

void MaxPool2D(
    OMTensor *output,
    OMTensor *input
)
{
    printf("MaxPool2D\n");

    int ic = omTensorGetShape(input)[1];
    int ih = omTensorGetShape(input)[2];
    int iw = omTensorGetShape(input)[3];

    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    int k = 2;
    uint8_t pad = 0;
    small::MaxPool2D<small::FloatBuffer>(
        k, 1,
        pad, pad, pad, pad,
        ic,
        ih, iw,
        in,
        out
    );
}

int relu_cnt = 0;

void Relu(
    OMTensor *image,
    OMTensor *output
)
{
    printf("entering Relu %d\n", relu_cnt++);

    int ic = omTensorGetShape(image)[1];
    int ih = omTensorGetShape(image)[2];
    int iw = omTensorGetShape(image)[3];

    // printf("ic=%d, ih=%d, iw=%d\n", ic, ih, iw);

    small::FloatBuffer in(omTensorGetNumElems(image), (float*)omTensorGetDataPtr(image));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    small::ReLUActivation<small::FloatBuffer>(
        ic,
        ih, iw,
        in,
        out
    );
}

//****************************************************************************
// void Conv2D(
//     OMTensor *output, 
//     OMTensor *input, 
//     OMTensor *filter, OMTensor *bias
// ){

//     const int64_t *shape;
//     shape = omTensorGetShape(input);
//     int64_t ci = shape[1];
//     int64_t h = shape[2];
//     int64_t w = shape[3];
    
//     shape = omTensorGetShape(filter);
//     int64_t co = shape[0];
//     int64_t k = shape[2];

//     size_t input_elems = omTensorGetNumElems(input);
//     size_t filter_elems = omTensorGetNumElems(filter);
//     size_t output_elems = omTensorGetNumElems(output);

//     small::FloatBuffer input_buf(input_elems);
//     small::FloatBuffer filter_buf(filter_elems);
//     small::FloatBuffer output_buf(output_elems);

//     small::Conv2D<small::FloatBuffer>(
//         3, 1,
//         1, 1, 1, 1,
//         co,
//         ci,
//         h, w,
//         input_buf,
//         filter_buf,
//         output_buf
//     );

// }

} // extern