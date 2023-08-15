//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126
//****************************************************************************

#define PARALLEL 1

#include <acutest.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>

#include <small.h>
#include <small/utils/Timer.hpp>
#include <small/UpSample2DLayer.hpp>

#include "test_utils.hpp"

//Index a packed direct_convolution input/output tensor
#define INDEX_tensor(A, num_channels, num_rows, num_cols, channel_blocking, channel_idx, height_idx, width_idx) \
A[(channel_idx/channel_blocking) * (num_rows  *  num_cols *  channel_blocking)\
+                                  height_idx * (num_cols *  channel_blocking)\
+                                               width_idx * (channel_blocking)\
+                                                           (channel_idx % channel_blocking)]

// Index a packed direct_convolution weights tensor
#define INDEX_weights(F, num_filters, num_channels, num_rows, num_cols, filter_blocking, channel_blocking, filter_idx, channel_idx, height_idx, width_idx)\
F[(filter_idx/filter_blocking) * ((num_channels/channel_blocking) *  num_rows * num_cols  *                channel_blocking *               filter_blocking)\
+                                 (channel_idx/channel_blocking)  * (num_rows * num_cols  *                channel_blocking *               filter_blocking)\
+                                                                  height_idx * (num_cols *                channel_blocking *               filter_blocking)\
+                                                                               width_idx * (              channel_blocking *               filter_blocking)\
+                                                                                           (channel_idx % channel_blocking *               filter_blocking)\
+                                                                                                                             (filter_idx % filter_blocking)]

//upsample checker
#define CHECK_UPSAMPLE(C_i, H, W, scale_factor, input_dc, output_dc, passing)\
{\
    for (size_t k = 0; k < C_i; k++)\
    {\
        for (size_t i = 0; i < H * scale_factor; i++)\
        {\
            for (size_t j = 0; j < W * scale_factor; j++)\
            {\
                /*printf("%.2f ", INDEX_tensor(output_dc, C_i, H * scale_factor, W * scale_factor, BufferT::C_ob, k, i, j));*/\
                passing &= (INDEX_tensor(output_dc, C_i, H * scale_factor, W * scale_factor, BufferT::C_ob, k, i, j) == INDEX_tensor(input_dc, C_i, H, W, BufferT::C_ob, k, i / scale_factor, j / scale_factor));\
            }\
            /*printf("\n");*/\
        }\
        /*printf("\n");*/\
    }\
    /*printf("\n");*/\
}
//****************************************************************************
template <class BufferT>
BufferT create_upsample2d_data(size_t num_elements)
{
    BufferT input_dc(num_elements);

    for (size_t ix = 0; ix < num_elements; ++ix)
    {
        input_dc[ix] = (typename BufferT::value_type)(ix);
    }

    return input_dc;
}

//****************************************************************************
void test_upsample2d_single_element(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 1;
    uint32_t const scale_factor = 2;

    size_t const num_input_elts = C_i * H * W;
    size_t const num_output_elts = C_i * H * scale_factor * W * scale_factor;

#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    BufferT input_dc =
        create_upsample2d_data<BufferT>(num_input_elts);
    BufferT output_dc(num_output_elts);

    small::UpSample2D(scale_factor, C_i, H, W, input_dc, output_dc);

    // for (size_t ix = 0; ix < num_input_elts; ++ix)
    // {
    //     // TEST_CHECK((input_dc[ix] >= 0) ?
    //     //            (output_dc[ix] == input_dc[ix]) :
    //     //            (output_dc[ix] == 0));
    //     std::cout << ix << ": Upsample in(" << (int)input_dc[ix] << ")\n";
    // }

    bool passing = true;

    CHECK_UPSAMPLE(C_i, H, W, scale_factor, input_dc, output_dc, passing)

    // for (size_t ix = 0; ix < num_output_elts; ++ix)
    // {
    //     // TEST_CHECK((input_dc[ix] >= 0) ?
    //     //            (output_dc[ix] == input_dc[ix]) :
    //     //            (output_dc[ix] == 0));
    //     std::cout << ix << ": Upsample out(" << (int)output_dc[ix] << ")\n";
    // }

    TEST_CHECK(passing);
}

//****************************************************************************
void test_upsample2d_single_tile(void)
{
    size_t const C_i = 16;
    size_t const H = 1;
    size_t const W = 3;
    uint32_t const scale_factor = 2;

    size_t const num_input_elts = C_i * H * W;
    size_t const num_output_elts = C_i * H * scale_factor * W * scale_factor;

#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    BufferT input_dc =
        create_upsample2d_data<BufferT>(num_input_elts);
    BufferT output_dc(num_output_elts);

    small::UpSample2D(scale_factor, C_i, H, W, input_dc, output_dc);

    // for (size_t ix = 0; ix < num_input_elts; ++ix)
    // {
    //     // TEST_CHECK((input_dc[ix] >= 0) ?
    //     //            (output_dc[ix] == input_dc[ix]) :
    //     //            (output_dc[ix] == 0));
    //     std::cout << ix << ": Upsample in(" << (int)input_dc[ix] << ")\n";
    // }

    bool passing = true;
    CHECK_UPSAMPLE(C_i, H, W, scale_factor, input_dc, output_dc, passing)
    TEST_CHECK(passing);
}

//****************************************************************************
void test_upsample2d_large_tile(void)
{
    size_t const C_i = 96;
    size_t const H = 13;
    size_t const W = 13;
    uint32_t const scale_factor = 2;

    size_t const num_input_elts = C_i * H * W;
    size_t const num_output_elts = C_i * H * scale_factor * W * scale_factor;

#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    BufferT input_dc =
        create_upsample2d_data<BufferT>(num_input_elts);
    BufferT output_dc(num_output_elts);

    small::UpSample2D(scale_factor, C_i, H, W, input_dc, output_dc);

    // for (size_t ix = 0; ix < num_input_elts; ++ix)
    // {
    //     // TEST_CHECK((input_dc[ix] >= 0) ?
    //     //            (output_dc[ix] == input_dc[ix]) :
    //     //            (output_dc[ix] == 0));
    //     std::cout << ix << ": Upsample in(" << (int)input_dc[ix] << ")\n";
    // }

    bool passing = true;
    CHECK_UPSAMPLE(C_i, H, W, scale_factor, input_dc, output_dc, passing)
    TEST_CHECK(passing);
}
//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"upsample2d_single_element", test_upsample2d_single_element},
    {"upsample2d_single_tile", test_upsample2d_single_tile},
    {"upsample2d_large_tile", test_upsample2d_large_tile},
    {NULL, NULL}
};
