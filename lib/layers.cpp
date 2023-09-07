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
int conv2D_bias = 0;
void Conv2D_bias(
    OMTensor *output, 
    OMTensor *input,
    OMTensor *filter,
    OMTensor *bias,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    uint8_t stride_x, uint8_t stride_y
    // ,uint8_t k_x, uint8_t k_y
) {

    printf("entering %s %d\n", __func__, conv2D_bias++);

    int oc = omTensorGetShape(filter)[0];
    int ic = omTensorGetShape(filter)[1];
    int kh = omTensorGetShape(filter)[2];
    int kw = omTensorGetShape(filter)[3];

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

    small::Bias<small::FloatBuffer>(
        oc,
        oh,
        ow,
        bias_buffer, 
        out
    );

    small::PartialConv2D<small::FloatBuffer>(
        kh, kw, stride_x,
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
int conv2D = 0;
void Conv2D(
    OMTensor *output, 
    OMTensor *input,
    OMTensor *filter,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    uint8_t stride_x, uint8_t stride_y
) {

    printf("entering %s %d\n", __func__, conv2D++);

    int oc = omTensorGetShape(filter)[0];
    int ic = omTensorGetShape(filter)[1];
    int kh = omTensorGetShape(filter)[2];
    int kw = omTensorGetShape(filter)[3];

    int ih = omTensorGetShape(input)[2];
    int iw = omTensorGetShape(input)[3];

    int oh = omTensorGetShape(output)[2];
    int ow = omTensorGetShape(output)[3];

    // printf("t_pad = %d | b_pad = %d | l_pad = %d | r_pad = %d\n", t_pad, b_pad, l_pad, r_pad);
    
    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer filt(omTensorGetNumElems(filter), (float*)omTensorGetDataPtr(filter));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    // printf("oc=%d, ic=%d, k=%d, ih=%d, iw=%d, oh=%d, ow=%d\n", oc, ic, k, ih, iw, oh, ow);

    small::Conv2D<small::FloatBuffer>(
        kh, kw, stride_x,
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
int max_pool = 0;
void MaxPoolSingleOut(
    OMTensor *output,
    OMTensor *input,
    uint8_t k_h, uint8_t k_w, uint8_t stride_h, uint8_t stride_w
    // ,uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad
)
{
    printf("entering %s %d\n", __func__, max_pool++);

    int ic = omTensorGetShape(input)[1];
    int ih = omTensorGetShape(input)[2];
    int iw = omTensorGetShape(input)[3];

    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    assert(stride_h == stride_w && "[ERROR in AveragePool2D] stride_h != stride_w\n");

    uint8_t t_pad = 0, b_pad = 0, l_pad = 0, r_pad = 0;
    small::MaxPool2D<small::FloatBuffer>(
        k_h, k_w, stride_h,
        t_pad, b_pad, l_pad, r_pad,
        ic,
        ih, iw,
        in,
        out
    );
}

//****************************************************************************
int avg_pool = 0;
void AveragePool(
    OMTensor *output,
    OMTensor *input,
    uint8_t k_h, uint8_t k_w, uint8_t stride_h, uint8_t stride_w
    // ,uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad
)
{
    printf("entering %s %d\n", __func__, avg_pool++);

    int ic = omTensorGetShape(input)[1];
    int ih = omTensorGetShape(input)[2];
    int iw = omTensorGetShape(input)[3];

    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    assert(stride_h == stride_w && "[ERROR in AveragePool2D] stride_h != stride_w\n");

    printf("k_h = %d | k_w = %d | stride_h = %d | stride_w = %d\n", k_h, k_w, stride_h, stride_w);

    uint8_t t_pad = 0, b_pad = 0, l_pad = 0, r_pad = 0;
    small::AveragePool2D<small::FloatBuffer>(
        k_h, k_w, stride_h,
        t_pad, b_pad, l_pad, r_pad,
        ic,
        ih, iw,
        in,
        out
    );
}



//****************************************************************************
int relu = 0;
void Relu(
    OMTensor *output,
    OMTensor *input
)
{
    printf("entering %s %d\n", __func__, relu++);

    int ih = omTensorGetShape(input)[2];
    int iw = omTensorGetShape(input)[3];
    int ic = omTensorGetShape(input)[1];

    // printf("ic=%d, ih=%d, iw=%d\n", ic, ih, iw);

    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    small::ReLUActivation<small::FloatBuffer>(
        ic,
        ih, iw,
        in,
        out
    );
}


//****************************************************************************
int matmul = 0;
void MatMul(
    OMTensor *C,
    OMTensor *A,
    OMTensor *B
)
{
    printf("entering %s %d\n", __func__, matmul++);

    printf("A is %ld x %ld\n", omTensorGetShape(A)[0], omTensorGetShape(A)[1]);
    printf("B is %ld x %ld\n", omTensorGetShape(B)[0], omTensorGetShape(B)[1]);
    printf("C is %ld x %ld\n", omTensorGetShape(C)[0], omTensorGetShape(C)[1]);

    // int ic = omTensorGetShape(image)[1];
    // int ih = omTensorGetShape(image)[2];
    // int iw = omTensorGetShape(image)[3];

    // // printf("ic=%d, ih=%d, iw=%d\n", ic, ih, iw);

    // small::FloatBuffer in(omTensorGetNumElems(image), (float*)omTensorGetDataPtr(image));
    // small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    // small::ReLUActivation<small::FloatBuffer>(
    //     ic,
    //     ih, iw,
    //     in,
    //     out
    // );
}

} // extern