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
void print_buffer(small::FloatBuffer buf, int n, int c, int h, int w) {
    float *data = buf.data();
    size_t elems = buf.size();

    printf(" [ %8.4le", data[0]);
    for(int i=1; i<256; i++) {
        if(i % 10 == 0)
            printf("\n");
        printf(" %8.4le", data[i]);
    }
    printf(" ]\n");
}

//****************************************************************************
int conv2D_bias = 0;
void Conv2D_bias(
    OMTensor *output, 
    OMTensor *input,
    OMTensor *filter,
    OMTensor *bias,
    uint8_t t_pad, uint8_t l_pad, uint8_t b_pad, uint8_t r_pad,
    uint8_t stride_x, uint8_t stride_y
) {

    printf("entering %s %d -- ", __func__, conv2D_bias++);

    size_t filt_oc = omTensorGetShape(filter)[0];
    size_t filt_ic = omTensorGetShape(filter)[1];
    size_t kh = omTensorGetShape(filter)[2];
    size_t kw = omTensorGetShape(filter)[3];

    size_t ic = omTensorGetShape(input)[1];
    size_t ih = omTensorGetShape(input)[2];
    size_t iw = omTensorGetShape(input)[3];

    size_t oc = omTensorGetShape(output)[1];
    size_t oh = omTensorGetShape(output)[2];
    size_t ow = omTensorGetShape(output)[3];
    
    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer filt(omTensorGetNumElems(filter), (float*)omTensorGetDataPtr(filter));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));
    small::FloatBuffer bias_buffer(omTensorGetNumElems(bias), (float*)omTensorGetDataPtr(bias));

    assert(filt_oc == oc && "[ERROR in Conv2D_bias] filt_oc != oc\n");
    
    small::Bias<small::FloatBuffer>(
        oc,
        oh,
        ow,
        bias_buffer, 
        out
    );

    if(filt_ic == 1 && ic>1) {

        printf("PartialDepthwiseConv2D\n");
        small::PartialDepthwiseConv2D<small::FloatBuffer>(
            kh, kw, stride_x,
            t_pad, b_pad, l_pad, r_pad,
            ic,
            ih, iw,
            in,
            filt,
            out
        );

    }
    else {
        
        printf("PartialConv2D\n");
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


    // small::Conv2DLayer<small::FloatBuffer> conv(
    //     {1, ic, ih, iw},
    //     kh, kw, 
    //     stride_x,
    //     small::PADDING_F, // t_pad, b_pad, l_pad, r_pad,
    //     oc, 
    //     filt,
    //     bias_buffer
    // );

    // small::Tensor<small::FloatBuffer> input_tensor({1, ic, ih, iw}, in);
    // small::Tensor<small::FloatBuffer> output_tensor({1, oc, oh, ow}, out);
    // conv.compute_output({&input_tensor}, &output_tensor);
    // memcpy(omTensorGetDataPtr(output), output_tensor.buffer().data(), omTensorGetNumElems(output) * sizeof(float));

}



//****************************************************************************
int conv2D = 0;
void Conv2D(
    OMTensor *output, 
    OMTensor *input,
    OMTensor *filter,
    uint8_t t_pad, uint8_t l_pad, uint8_t b_pad, uint8_t r_pad,
    uint8_t stride_x, uint8_t stride_y
) {

    printf("entering %s %d -- ", __func__, conv2D++);

    size_t filt_oc = omTensorGetShape(filter)[0];
    size_t filt_ic = omTensorGetShape(filter)[1];
    size_t kh = omTensorGetShape(filter)[2];
    size_t kw = omTensorGetShape(filter)[3];

    size_t ic = omTensorGetShape(input)[1];
    size_t ih = omTensorGetShape(input)[2];
    size_t iw = omTensorGetShape(input)[3];

    size_t oc = omTensorGetShape(output)[1];
    size_t oh = omTensorGetShape(output)[2];
    size_t ow = omTensorGetShape(output)[3];

    assert(filt_oc == oc && "[ERROR in Conv2D] filt_oc != oc\n");

    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer filt(omTensorGetNumElems(filter), (float*)omTensorGetDataPtr(filter));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    if(filt_ic == 1 && ic>1) {
        printf("DepthwiseConv2D\n");
        small::DepthwiseConv2D<small::FloatBuffer>(
            kh, kw, stride_x,
            t_pad, b_pad, l_pad, r_pad,
            ic,
            ih, iw,
            in,
            filt,
            out
        );
    }
    else {
        printf("Conv2D\n");
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

    //printf("Conv2D\n");
    // small::Conv2DLayer<small::FloatBuffer> conv(
    //     {1, ic, ih, iw},
    //     kh, kw, 
    //     stride_x,
    //     small::PADDING_F, // t_pad, b_pad, l_pad, r_pad,
    //     oc, 
    //     filt
    // );
    // small::Tensor<small::FloatBuffer> input_tensor({1, ic, ih, iw}, in);
    // small::Tensor<small::FloatBuffer> output_tensor({1, oc, oh, ow}, out);
    // conv.compute_output({&input_tensor}, &output_tensor);
    // memcpy(omTensorGetDataPtr(output), output_tensor.buffer().data(), omTensorGetNumElems(output) * sizeof(float));


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
    //printf("entering %s %d\n", __func__, max_pool++);

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

    int oc = omTensorGetShape(output)[1];
    int oh = omTensorGetShape(output)[2];
    int ow = omTensorGetShape(output)[3];

    //printf("ic=%d, ih=%d, iw=%d | oc=%d, oh=%d, ow=%d\n", ic, ih, iw, oc, oh, ow);

    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    // assert(stride_h == stride_w && "[ERROR in AveragePool2D] stride_h != stride_w\n");
    // //printf("k_h = %d | k_w = %d | stride_h = %d | stride_w = %d\n", k_h, k_w, stride_h, stride_w);

    // print_buffer(in, 1, ic, ih, iw);

    uint8_t t_pad = 0, b_pad = 0, l_pad = 0, r_pad = 0;
    small::AveragePool2D<small::FloatBuffer>(
        k_h, k_w, 1,
        t_pad, b_pad, l_pad, r_pad,
        ic,
        ih, iw,
        in,
        out
    );

    // print_buffer(out, 1, oc, oh, ow);
}



//****************************************************************************
int relu = 0;
void Relu(
    OMTensor *output,
    OMTensor *input
)
{
    //printf("entering %s %d\n", __func__, relu++);

    int ih = omTensorGetShape(input)[2];
    int iw = omTensorGetShape(input)[3];
    int ic = omTensorGetShape(input)[1];

    // //printf("ic=%d, ih=%d, iw=%d\n", ic, ih, iw);

    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    small::ReLUActivation<small::FloatBuffer>(
        ic,
        ih, iw,
        in,
        out
    );
}


// //****************************************************************************
// int matmul = 0;
// void MatMul(
//     OMTensor *C,
//     OMTensor *A,
//     OMTensor *B
// )
// {
//     //printf("entering %s %d\n", __func__, matmul++);

//     //printf("A is %ld x %ld\n", omTensorGetShape(A)[0], omTensorGetShape(A)[1]);
//     //printf("B is %ld x %ld\n", omTensorGetShape(B)[0], omTensorGetShape(B)[1]);
//     //printf("C is %ld x %ld\n", omTensorGetShape(C)[0], omTensorGetShape(C)[1]);


// }


//****************************************************************************
int gemm = 0;
void Gemm(
    OMTensor *C,
    OMTensor *A,
    OMTensor *B,
    OMTensor *bias,
    float alpha,
    float beta
)
{
    printf("entering %s %d\n", __func__, gemm++);

    int m = omTensorGetShape(A)[0];
    int k = omTensorGetShape(A)[1];
    int n = omTensorGetShape(B)[1];

    // printf("A is %ld x %ld\n", omTensorGetShape(A)[0], omTensorGetShape(A)[1]);
    // printf("B is %ld x %ld\n", omTensorGetShape(B)[0], omTensorGetShape(B)[1]);
    // printf("C is %ld x %ld\n", omTensorGetShape(C)[0], omTensorGetShape(C)[1]);
    // printf("bias is %ld x %ld\n", omTensorGetShape(bias)[0], omTensorGetShape(bias)[1]);
    // printf("alpha = %f | beta = %f\n", alpha, beta);

    for(int i=0; i<n; i++) {
        float sum = ((float*)omTensorGetDataPtr(bias))[i];
        // float sum = 0;
        for(int p=0; p<k; p++) {
            sum += ((float*)omTensorGetDataPtr(A))[p] * ((float*)omTensorGetDataPtr(B))[p*n + i];
        }
        ((float*)omTensorGetDataPtr(C))[i] = sum;
    }

    // small::FloatBuffer in(omTensorGetNumElems(A), (float*)omTensorGetDataPtr(A));
    // small::FloatBuffer filt(omTensorGetNumElems(B), (float*)omTensorGetDataPtr(B));
    // small::FloatBuffer out(omTensorGetNumElems(C), (float*)omTensorGetDataPtr(C));

    // small::Dense<small::FloatBuffer>(
    //     omTensorGetNumElems(C), in_elems,
    //     in,
    //     filt,
    //     out
    // );

    // int ic = omTensorGetShape(image)[1];
    // int ih = omTensorGetShape(image)[2];
    // int iw = omTensorGetShape(image)[3];

    // // //printf("ic=%d, ih=%d, iw=%d\n", ic, ih, iw);

    // small::FloatBuffer in(omTensorGetNumElems(image), (float*)omTensorGetDataPtr(image));
    // small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

}


//****************************************************************************
int softmax = 0;
void Softmax(
    OMTensor *output,
    OMTensor *input,
    int axis
)
{
    // //printf("entering %s %d\n", __func__, softmax++);
    // //printf("input is %ld x %ld\n", omTensorGetShape(input)[0], omTensorGetShape(input)[1]);
    // //printf("output is %ld x %ld\n", omTensorGetShape(output)[0], omTensorGetShape(output)[1]);
    // //printf("axis = %d\n", axis);

    // //printf("[%f, %f]\n", ((float*)omTensorGetDataPtr(input))[0], ((float*)omTensorGetDataPtr(input))[1]);

    float *in = (float*)omTensorGetDataPtr(input);
    float *out = (float*)omTensorGetDataPtr(output);

    float sum = 0;
    for(int i=0; i<omTensorGetNumElems(input); i++) {
        in[i] = exp(in[i]);
        sum += in[i];
    }

    for(int i=0; i<omTensorGetNumElems(input); i++) {
        out[i] = in[i] / sum;
    }


    // float *tmp = (float*)malloc(omTensorGetNumElems(input) * sizeof(float));


    // //printf("A is %ld x %ld\n", omTensorGetShape(A)[0], omTensorGetShape(A)[1]);
    // //printf("B is %ld x %ld\n", omTensorGetShape(B)[0], omTensorGetShape(B)[1]);
    // //printf("C is %ld x %ld\n", omTensorGetShape(C)[0], omTensorGetShape(C)[1]);
    // //printf("alpha = %f | beta = %f\n", alpha, beta);

    // int ic = omTensorGetShape(image)[1];
    // int ih = omTensorGetShape(image)[2];
    // int iw = omTensorGetShape(image)[3];

    // // //printf("ic=%d, ih=%d, iw=%d\n", ic, ih, iw);

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