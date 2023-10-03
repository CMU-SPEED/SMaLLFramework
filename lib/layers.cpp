// c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// onnx data struct
// #include "OMTensor.hpp"
#include "OMTensor.inc"

// small
#include "small.h"

// #define USE_LAYER_API

#ifdef USE_LAYER_API
#include "small/Tensor.hpp"
#include "small/Conv2DLayer.hpp"
#include "small/DepthwiseConv2DLayer.hpp"
#include "small/ReLULayer.hpp"
#endif



extern "C" {

//****************************************************************************
// print shape for debugging
// void print_shape(const int64_t *shape, int64_t shape_size) {
//     if(shape_size != 0) {
//         // printf("(%ld", shape[0]);
//         for(int i=1; i<shape_size; i++) {
//             // printf(", %ld", shape[i]);
//         }
//         // printf(")");
//     }
//     // printf("\n");
// }

// //****************************************************************************
// void print_buffer(small::FloatBuffer buf, int n, int c, int h, int w) {
//     float *data = buf.data();
//     size_t elems = buf.size();

//     // printf(" [ %8.4le", data[0]);
//     for(int i=1; i<256; i++) {
//         if(i % 10 == 0)
//             // printf("\n");
//         // printf(" %8.4le", data[i]);
//     }
//     // printf(" ]\n");
// }

//****************************************************************************
int transpose = 0;
void Transpose(
    OMTensor *output,
    OMTensor *input
){
    // printf("entering %s %d\n", __func__, transpose++);
    memcpy(omTensorGetDataPtr(output), omTensorGetDataPtr(input), omTensorGetNumElems(input) * sizeof(float));
}

//****************************************************************************
int conv2D_bias = 0;
void Conv2D_bias(
    OMTensor *output, 
    OMTensor *input,
    OMTensor *filter,
    OMTensor *bias,
    uint8_t t_pad, uint8_t l_pad, uint8_t b_pad, uint8_t r_pad,
    uint8_t stride_h, uint8_t stride_w
) {

    // printf("entering %s %d -- ", __func__, conv2D_bias++);

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
    
    #ifndef USE_LAYER_API
        small::Bias<small::FloatBuffer>(
            oc,
            oh,
            ow,
            bias_buffer, 
            out
        );
    #else
        small::PaddingEnum pad = \
            (t_pad == 0 && b_pad == 0 && l_pad == 0 && r_pad == 0) ? small::PADDING_V : small::PADDING_F;
    #endif

    if(filt_ic == 1 && ic>1) {

        // printf("PartialDepthwiseConv2D\n");

        #ifdef USE_LAYER_API
            small::DepthwiseConv2DLayer<small::FloatBuffer> conv(
                {1, ic, ih, iw},
                kh, kw,
                stride_h,
                pad,
                filt,
                bias_buffer
            );
            small::Tensor<small::FloatBuffer> input_tensor({1, ic, ih, iw}, in);
            small::Tensor<small::FloatBuffer> output_tensor({1, oc, oh, ow}, out);
            conv.compute_output({&input_tensor}, &output_tensor);
            memcpy(omTensorGetDataPtr(output), output_tensor.buffer().data(), omTensorGetNumElems(output) * sizeof(float));
        #else
            small::PartialDepthwiseConv2D<small::FloatBuffer>(
                kh, kw, stride_h,
                t_pad, b_pad, l_pad, r_pad,
                ic,
                ih, iw,
                in,
                filt,
                out
            );
        #endif
    }
    else {
        
        // printf("PartialConv2D\n");

        #ifdef USE_LAYER_API
            small::Conv2DLayer<small::FloatBuffer> conv(
                {1, ic, ih, iw},
                kh, kw,
                stride_h,
                pad,
                oc,
                filt,
                bias_buffer
            );
            small::Tensor<small::FloatBuffer> input_tensor({1, ic, ih, iw}, in);
            small::Tensor<small::FloatBuffer> output_tensor({1, oc, oh, ow}, out);
            conv.compute_output({&input_tensor}, &output_tensor);
            memcpy(omTensorGetDataPtr(output), output_tensor.buffer().data(), omTensorGetNumElems(output) * sizeof(float));
        #else
            small::PartialConv2D<small::FloatBuffer>(
                kh, kw, stride_h,
                t_pad, b_pad, l_pad, r_pad,
                oc,
                ic,
                ih, iw,
                in,
                filt,
                out
            );
        #endif
    }

}



//****************************************************************************
int conv2D = 0;
void Conv2D(
    OMTensor *output, 
    OMTensor *input,
    OMTensor *filter,
    uint8_t t_pad, uint8_t l_pad, uint8_t b_pad, uint8_t r_pad,
    uint8_t stride_h, uint8_t stride_w
) {

    // printf("entering %s %d -- ", __func__, conv2D++);

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

    // // printf("%d, %d, %d\n", omTensorGetNumElems(input), omTensorGetNumElems(filter), omTensorGetNumElems(output));

    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer filt(omTensorGetNumElems(filter), (float*)omTensorGetDataPtr(filter));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    // small::FloatBuffer in(omTensorGetNumElems(input));
    // small::FloatBuffer filt(omTensorGetNumElems(filter));
    // small::FloatBuffer out(omTensorGetNumElems(output));

    small::PaddingEnum pad = (t_pad == 0 && b_pad == 0 && l_pad == 0 && r_pad == 0) ? small::PADDING_V : small::PADDING_F;

    if(filt_ic == 1 && ic>1) {
        // printf("DepthwiseConv2D\n");
        
        #ifdef USE_LAYER_API
            small::DepthwiseConv2DLayer<small::FloatBuffer> conv(
                {1, ic, ih, iw},
                kh, kw,
                stride_h,
                pad,
                filt
            );
            small::Tensor<small::FloatBuffer> input_tensor({1, ic, ih, iw}, in);
            small::Tensor<small::FloatBuffer> output_tensor({1, oc, oh, ow}, out);
            conv.compute_output({&input_tensor}, &output_tensor);
            memcpy(omTensorGetDataPtr(output), output_tensor.buffer().data(), omTensorGetNumElems(output) * sizeof(float));
        #else
            small::DepthwiseConv2D<small::FloatBuffer>(
                kh, kw, stride_h,
                t_pad, b_pad, l_pad, r_pad,
                ic,
                ih, iw,
                in,
                filt,
                out
            );
        #endif
    }
    else {
        // printf("Conv2D\n");

        #ifdef USE_LAYER_API
            small::Conv2DLayer<small::FloatBuffer> conv(
                {1, ic, ih, iw},
                kh, kw,
                stride_h,
                pad,
                oc,
                filt
            );
            // // printf("%d, %d\n", stride_h, stride_w);
            // // printf("%d, %d, %d, %d\n", conv.output_shape()[0], conv.output_shape()[1], conv.output_shape()[2], conv.output_shape()[3]);
            // // printf("%d, %d, %d, %d\n", 1, oc, oh, ow);
            small::Tensor<small::FloatBuffer> input_tensor({1, ic, ih, iw}, in);
            small::Tensor<small::FloatBuffer> output_tensor({1, oc, oh, ow}, out);
            conv.compute_output({&input_tensor}, &output_tensor);
            memcpy(omTensorGetDataPtr(output), output_tensor.buffer().data(), omTensorGetNumElems(output) * sizeof(float));
        #else

        //     printf("(%d, %d) | %d | (%d, %d, %d, %d) | %d, %d, %d, %d\n",
        // kh, kw, stride_h, t_pad, b_pad, l_pad, r_pad, oc, ic, ih, iw);

            small::Conv2D<small::FloatBuffer>(
                kh, kw, stride_h,
                t_pad, b_pad, l_pad, r_pad,
                oc,
                ic,
                ih, iw,
                in,
                filt,
                out
            );
        #endif
    }

}

//****************************************************************************
int max_pool = 0;
void MaxPoolSingleOut(
    OMTensor *output,
    OMTensor *input,
    uint8_t k_h, uint8_t k_w, uint8_t stride_h, uint8_t stride_w
)
{
    // printf("entering %s %d\n", __func__, max_pool++);

    int ic = omTensorGetShape(input)[1];
    int ih = omTensorGetShape(input)[2];
    int iw = omTensorGetShape(input)[3];
    
    // printf("ic=%d, ih=%d, iw=%d, kh=%d, kw=%d\n", ic, ih, iw, k_h, k_w);

    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    // assert(stride_h == stride_w && "[ERROR in AveragePool2D] stride_h != stride_w\n");

    uint8_t t_pad = 0, b_pad = 0, l_pad = 0, r_pad = 0;
    small::MaxPool2D<small::FloatBuffer>(
        k_h, k_w, 1,
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
    uint8_t k_h, uint8_t k_w, 
    uint8_t stride_h, uint8_t stride_w
)
{
    // printf("entering %s %d\n", __func__, avg_pool++);

    int ic = omTensorGetShape(input)[1];
    int ih = omTensorGetShape(input)[2];
    int iw = omTensorGetShape(input)[3];

    int oc = omTensorGetShape(output)[1];
    int oh = omTensorGetShape(output)[2];
    int ow = omTensorGetShape(output)[3];

    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    uint8_t t_pad = 0, b_pad = 0, l_pad = 0, r_pad = 0;
    small::AveragePool2D<small::FloatBuffer>(
        k_h, k_w, 1,
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
    // printf("entering %s %d\n", __func__, relu++);

    // size_t ic = omTensorGetShape(input)[1];
    // size_t ih = omTensorGetShape(input)[2];
    // size_t iw = omTensorGetShape(input)[3];
    size_t ic;
    size_t ih;
    size_t iw;

    if(omTensorGetRank(input) == 2) {
        ic = omTensorGetShape(input)[1];
        ih = 1;
        iw = 1;
    }
    else if(omTensorGetRank(input) == 4) {
        ic = omTensorGetShape(input)[1];
        ih = omTensorGetShape(input)[2];
        iw = omTensorGetShape(input)[3];
    }
    else {
        printf("[ERROR in Relu] input rank != 2 or 4\n");
        exit(1);
    }

    small::FloatBuffer in(omTensorGetNumElems(input), (float*)omTensorGetDataPtr(input));
    small::FloatBuffer out(omTensorGetNumElems(output), (float*)omTensorGetDataPtr(output));

    #ifdef USE_LAYER_API
        small::ReLULayer<small::FloatBuffer> relu({1, ic, ih, iw});
        small::Tensor<small::FloatBuffer> input_tensor({1, ic, ih, iw}, in);
        small::Tensor<small::FloatBuffer> output_tensor({1, ic, ih, iw}, out);
        relu.compute_output({&input_tensor}, &output_tensor);
        memcpy(omTensorGetDataPtr(output), output_tensor.buffer().data(), omTensorGetNumElems(output) * sizeof(float));
    #else
        // printf("ic=%ld, ih=%ld, iw=%ld\n", ic, ih, iw);

        small::ReLUActivation<small::FloatBuffer>(
            ic,
            ih, iw,
            in,
            out
        );
    #endif
    
    // printf("leaving relu\n");

}





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
    // printf("entering %s %d\n", __func__, gemm++);
    int ic = omTensorGetShape(A)[1];
    int oc = omTensorGetShape(B)[1];

    // printf("A is %ld x %ld\n", omTensorGetShape(A)[0], omTensorGetShape(A)[1]);
    // printf("B is %ld x %ld\n", omTensorGetShape(B)[0], omTensorGetShape(B)[1]);
    // printf("C is %ld x %ld\n", omTensorGetShape(C)[0], omTensorGetShape(C)[1]);

    assert(omTensorGetShape(A)[0] == 1 && omTensorGetShape(C)[0] && "[ERROR in Gemm] m != 1\n");

    // for(int p_tile = 0; p_tile < k/FLOAT_C_ob; p_tile++){
    // for(int i=0; i<n; i++) {

    //     float sum = (bias != NULL) ? ((float*)omTensorGetDataPtr(bias))[i] : 0.0;
    //     // float sum =  0.0;

    //     for(int p=0; p<FLOAT_C_ob; p++) {
    //         // if the data is packed then B is oc/cob x ic/cib x 1 x 1 x cib x cob = oc/cob x ic x cob
    //         // B comes in as oc x ic = k x n
    //         // k = oc/ocob
    //         // n = ic
    //         sum += ((float*)omTensorGetDataPtr(A))[p_tile*FLOAT_C_ob + p] * ((float*)omTensorGetDataPtr(B))[p_tile*n*FLOAT_C_ob + i*FLOAT_C_ob + p];
    //     }

    //     ((float*)omTensorGetDataPtr(C))[i] += sum;
    // }
    // }

    // for(int i=0; i<n; i++) {

    //     float sum = (bias != NULL) ? ((float*)omTensorGetDataPtr(bias))[i] : 0.0;

    //     for(int p=0; p<k; p++) {
    //         sum += ((float*)omTensorGetDataPtr(A))[p] * ((float*)omTensorGetDataPtr(B))[p*n + i];
    //     }

    //     ((float*)omTensorGetDataPtr(C))[i] = sum;
    // }

    small::FloatBuffer in(omTensorGetNumElems(A), (float*)omTensorGetDataPtr(A));
    small::FloatBuffer filt(omTensorGetNumElems(B), (float*)omTensorGetDataPtr(B));
    small::FloatBuffer out(omTensorGetNumElems(C), (float*)omTensorGetDataPtr(C));

    if(bias != NULL) {
        int b_elems = omTensorGetShape(bias)[0];
        assert(b_elems == oc && "[ERROR in Gemm] b_elems != output channels\n");

        small::Bias<small::FloatBuffer>(
            oc,
            1,
            1,
            small::FloatBuffer(omTensorGetNumElems(bias), (float*)omTensorGetDataPtr(bias)), 
            out
        );

        small::PartialConv2D<small::FloatBuffer>(
            1, 1, 
            1,
            0, 0, 0, 0,
            oc, 
            ic,
            1, 1,
            in,
            filt,
            out
        );
    }
    else {
        small::Conv2D<small::FloatBuffer>(
            1, 1, 
            1,
            0, 0, 0, 0,
            oc, 
            ic,
            1, 1,
            in,
            filt,
            out
        );
    }

}

//****************************************************************************
int matmul = 0;
void MatMul(
    OMTensor *C,
    OMTensor *A,
    OMTensor *B
)
{
    // printf("entering %s %d\n", __func__, matmul++);

    Gemm(C, A, B, NULL, 1.0, 0.0);
}


//****************************************************************************
int softmax = 0;
void Softmax(
    OMTensor *output,
    OMTensor *input,
    int axis
)
{
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
}


} // extern