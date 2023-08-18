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

    // small::convert_tensor2dc<float>(
    //     onnx_data,
    //     buf_type,
    //     shape[0], shape[1], shape[2], shape[3],
    //     C_ib, C_ob,
    //     small_tensor.buffer().data()
    // );
    
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

//****************************************************************************
// void Conv2D(
//     OMTensor *output, 
//     OMTensor *input, 
//     OMTensor *filter, OMTensor *bias
// ){

//     #ifdef WRAPPER_DEBUG
//     printf("Entering Conv2D SMaLL wrapper\n");
//     #endif

//     #ifdef WRAPPER_DEBUG
//     printf("Packing input\n");
//     #endif
    
//     small::Tensor<small::FloatBuffer> input_small = omtensor_to_smalltensor(input, small::INPUT);
//     // print_tensor(input_small);
//     size_t ci = input_small.shape()[1];
//     size_t h = input_small.shape()[2];
//     size_t w = input_small.shape()[3];

//     #ifdef WRAPPER_DEBUG
//     printf("Packing filter\n");
//     #endif

//     small::Tensor<small::FloatBuffer> filter_small = omtensor_to_smalltensor(filter, small::FILTER_CONV);
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

//     // small::convert_dc2tensor(
//     //     output_small.buffer().data(),
//     //     small::INPUT,
//     //     1, co, h, w,
//     //     C_ib, C_ob,
//     //     (float*)omTensorGetDataPtr(output)
//     // );

//     memcpy(
//         omTensorGetDataPtr(output),
//         &output_small.buffer()[0],
//         output_small.capacity()
//     );

// }

//****************************************************************************
void Conv2D(
    OMTensor *output, 
    OMTensor *input, 
    OMTensor *filter, OMTensor *bias
){

    const int64_t *shape;
    shape = omTensorGetShape(input);
    int64_t ci = shape[1];
    int64_t h = shape[2];
    int64_t w = shape[3];
    
    shape = omTensorGetShape(filter);
    int64_t co = shape[0];
    int64_t k = shape[2];

    size_t input_elems = omTensorGetNumElems(input);
    size_t filter_elems = omTensorGetNumElems(filter);
    size_t output_elems = omTensorGetNumElems(output);

    small::FloatBuffer input_buf(input_elems);
    small::FloatBuffer filter_buf(filter_elems);
    small::FloatBuffer output_buf(output_elems);

    small::Conv2D<small::FloatBuffer>(
        3, 1,
        0, 1, 0, 1,
        co,
        ci,
        h, w,
        input_buf,
        filter_buf,
        output_buf
    );

}

//****************************************************************************
void Conv2D_unpacked(
    OMTensor *output, 
    OMTensor *input, 
    OMTensor *filter, 
    OMTensor *bias
){

    #ifdef WRAPPER_DEBUG
    printf("Entering Conv2D SMaLL wrapper\n");
    #endif

    #ifdef WRAPPER_DEBUG
    printf("Packing input\n");
    #endif
    
    small::Tensor<small::FloatBuffer> input_small = \
        omtensor_to_smalltensor_unpacked(input, small::INPUT);
    small::Tensor<small::FloatBuffer> filter_small = \
        omtensor_to_smalltensor_unpacked(filter, small::FILTER_CONV);

    // size_t ci = input_small.shape()[1];
    // size_t h = input_small.shape()[2];
    // size_t w = input_small.shape()[3];

    size_t co = filter_small.shape()[0];
    size_t k = filter_small.shape()[2];

    #ifdef WRAPPER_DEBUG
    printf("Running Conv2D\n");
    #endif

    small::Conv2DLayer<small::FloatBuffer> conv2d(
        input_small.shape(),
        k, k, 
        1, 
        small::PADDING_F,
        co,
        filter_small.buffer()
    );

    const int64_t *out_shape = omTensorGetShape(output);
    small::shape_type out_shape_small = {(size_t)out_shape[0],(size_t)out_shape[1],(size_t)out_shape[2],(size_t)out_shape[3]};
    small::Tensor<small::FloatBuffer> output_small(out_shape_small);

    conv2d.compute_output({&input_small}, {&output_small});

    #ifdef WRAPPER_DEBUG
    printf("Finished Conv2D\n");
    #endif
    
    // copy data out
    memcpy(
        omTensorGetDataPtr(output),
        &output_small.buffer()[0],
        output_small.capacity()
    );

}

} // extern