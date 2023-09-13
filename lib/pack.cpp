// c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// small
#include "small.h"
#include "small/Tensor.hpp"
#include "small/Conv2DLayer.hpp"

// #define WRAPPER_DEBUG 1

extern "C" {

void pack(float *input, int oc, int ic, int h, int w, int type) {

    small::BufferTypeEnum type_;
    if(type == 0) {
        type_ = small::BufferTypeEnum::INPUT;
    }
    else if (type == 1) {
        type_ = small::BufferTypeEnum::OUTPUT;
    }
    else if (type == 2) {
        type_ = small::BufferTypeEnum::FILTER_DW;
    }
    else if (type == 3) {
        type_ = small::BufferTypeEnum::FILTER_CONV;
    }
    else if (type == 4) {
        type_ = small::BufferTypeEnum::FILTER_FC;
    }
    else {
        printf("ERROR: unknown type %d\n", type);
        exit(-1);
    }

    // this changes for edge cases
    float *input_packed = (float*)malloc(oc * ic * h * w * sizeof(float));
    uint32_t ret = small::convert_tensor2dc<float>(
        input,
        type_,
        oc, ic, h, w,
        FLOAT_C_ib, FLOAT_C_ob,
        input_packed
    );
    memcpy(input, input_packed, oc * ic * h * w * sizeof(float));

    printf("pack: %d %d %d %d | %d %d\n", oc, ic, h, w, type, ret);
}

void unpack(float *input, int oc, int ic, int h, int w, int type) {

    small::BufferTypeEnum type_;
    if(type == 0) {
        type_ = small::BufferTypeEnum::INPUT;
    }
    else if (type == 1) {
        type_ = small::BufferTypeEnum::OUTPUT;
    }
    else if (type == 2) {
        type_ = small::BufferTypeEnum::FILTER_DW;
    }
    else if (type == 3) {
        type_ = small::BufferTypeEnum::FILTER_CONV;
    }
    else if (type == 4) {
        type_ = small::BufferTypeEnum::FILTER_FC;
    }
    else {
        printf("ERROR: unknown type %d\n", type);
        exit(-1);
    }

    // this changes for edge cases
    float *input_unpacked = (float*)malloc(oc * ic * h * w * sizeof(float));
    uint32_t ret = small::convert_dc2tensor<float>
    (
        input,
        type_,
        oc, ic, h, w,
        FLOAT_C_ib, FLOAT_C_ob,
        input_unpacked
    );
    memcpy(input, input_unpacked, oc * ic * h * w * sizeof(float));
    printf("unpack: %d %d %d %d | %d %d\n", oc, ic, h, w, type, ret);
}

} // extern "C"