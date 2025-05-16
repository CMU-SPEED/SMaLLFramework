//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2025 by The SMaLL Contributors, All Rights Reserved.
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
#include <small/Conv2DLayer.hpp>
#include <small/DenseLayer.hpp>
#include <small/buffers.hpp>
#include <small/Tensor.hpp>

#include "test_utils.hpp"

//****************************************************************************
void test_conv2d_dense_layer(void) {
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif
    using ScalarT = typename BufferT::value_type;

    // C_i,H,W,k,s,p,C_o
    LayerParams fc_params = {16, 1, 1, 1, 1, small::PADDING_V, 16};

    ScalarT input[16]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    ScalarT weights[256]{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ScalarT bias[16]{1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ScalarT expected_output[16]{1497, 3674, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    small::shape_type fc_input_shape{1UL, fc_params.C_i, fc_params.H, fc_params.W};

    BufferT weights_buf(fc_params.C_i*fc_params.k*fc_params.C_o);
    std::copy(weights, weights+weights_buf.size(), reinterpret_cast<ScalarT*>(weights_buf.data()));

    BufferT bias_buf(fc_params.C_o);
    std::copy(bias, bias+fc_params.C_o, reinterpret_cast<ScalarT*>(bias_buf.data()));

    small::Conv2DLayer<BufferT> fc(fc_input_shape,
                                   1U, 1U, 1U, fc_params.p, fc_params.C_o,
                                   weights_buf, bias_buf, false, small::ActivationType::NONE);

    BufferT inbuf(fc_params.C_i*fc_params.H*fc_params.W);
    small::Tensor<BufferT> packed_input(fc_input_shape);
    std::copy(input, input+inbuf.size(), reinterpret_cast<ScalarT*>(inbuf.data()));
    small::pack_buffer(inbuf, small::INPUT,
                       1, fc_params.C_i, fc_params.H, fc_params.W,
                       BufferT::C_ib, BufferT::C_ob, packed_input.buffer());

    small::Tensor<BufferT> output(fc.output_shape());
    small::init_zeros(output.buffer(), output.size());

    fc.compute_output({&packed_input}, &output);

    BufferT unpacked_output(output.size());
    small::unpack_buffer(output.buffer(), small::BufferTypeEnum::OUTPUT,
                         1UL,
                         output.shape()[small::CHANNEL],
                         output.shape()[small::HEIGHT],
                         output.shape()[small::WIDTH],
                         BufferT::C_ib, BufferT::C_ob,
                         unpacked_output);

    bool passing = true;
    for (size_t i = 0; i < fc_params.C_o*fc_params.H*fc_params.W; i++)
    {
        if (!almost_equal(expected_output[i], unpacked_output[i], 5e-5, 1e-7))
        {
            passing = false;
            std::cerr << i << ": ERROR: unequal outputs: unpacked_output("
                      << unpacked_output[i] << ") != expected_output("
                      << expected_output[i] << ")\n";
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_dense_dense_layer(void) {
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif
    using ScalarT = typename BufferT::value_type;

    // C_i,H,W,k,s,p,C_o
    LayerParams fc_params = {16, 1, 1, 1, 1, small::PADDING_V, 16};

    ScalarT input[16]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    ScalarT weights[256]{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ScalarT bias[16]{1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ScalarT expected_output[16]{1497, 3674, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    small::shape_type fc_input_shape{1UL, fc_params.C_i, fc_params.H, fc_params.W};

    BufferT weights_buf(fc_params.C_i*fc_params.C_o);
    std::copy(weights, weights+weights_buf.size(), reinterpret_cast<ScalarT*>(weights_buf.data()));

    BufferT bias_buf(fc_params.C_o);
    std::copy(bias, bias+fc_params.C_o, reinterpret_cast<ScalarT*>(bias_buf.data()));

    small::DenseLayer<BufferT> fc(fc_input_shape, fc_params.C_o,
                                  weights_buf, bias_buf, false,
                                  small::ActivationType::NONE);

    BufferT inbuf(fc_params.C_i*fc_params.H*fc_params.W);
    small::Tensor<BufferT> packed_input(fc_input_shape);
    std::copy(input, input+inbuf.size(), reinterpret_cast<ScalarT*>(inbuf.data()));
    small::pack_buffer(inbuf, small::INPUT,
                       1, fc_params.C_i, fc_params.H, fc_params.W,
                       BufferT::C_ib, BufferT::C_ob, packed_input.buffer());

    small::Tensor<BufferT> output(fc.output_shape());
    small::init_zeros(output.buffer(), output.size());

    fc.compute_output({&packed_input}, &output);

    BufferT unpacked_output(output.size());
    small::unpack_buffer(output.buffer(), small::BufferTypeEnum::OUTPUT,
                         1UL,
                         output.shape()[small::CHANNEL],
                         output.shape()[small::HEIGHT],
                         output.shape()[small::WIDTH],
                         BufferT::C_ib, BufferT::C_ob,
                         unpacked_output);

    bool passing = true;
    for (size_t i = 0; i < fc_params.C_o*fc_params.H*fc_params.W; i++)
    {
        if (!almost_equal(expected_output[i], unpacked_output[i], 5e-5, 1e-7))
        {
            passing = false;
            std::cerr << i << ": ERROR: unequal outputs: unpacked_output("
                      << unpacked_output[i] << ") != expected_output("
                      << expected_output[i] << ")\n";
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_dense_dense_layer_no_bias(void) {
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif
    using ScalarT = typename BufferT::value_type;

    // C_i,H,W,k,s,p,C_o
    LayerParams fc_params = {16, 1, 1, 1, 1, small::PADDING_V, 16};

    ScalarT input[16]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    ScalarT weights[256]{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ScalarT expected_output[16]{1496, 3672, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    small::shape_type fc_input_shape{1UL, fc_params.C_i, fc_params.H, fc_params.W};

    BufferT weights_buf(fc_params.C_i*fc_params.C_o);
    std::copy(weights, weights+weights_buf.size(), reinterpret_cast<ScalarT*>(weights_buf.data()));

    small::DenseLayer<BufferT> fc(fc_input_shape, fc_params.C_o,
                                  weights_buf, false,
                                  small::ActivationType::NONE);

    BufferT inbuf(fc_params.C_i*fc_params.H*fc_params.W);
    small::Tensor<BufferT> packed_input(fc_input_shape);
    std::copy(input, input+inbuf.size(), reinterpret_cast<ScalarT*>(inbuf.data()));
    small::pack_buffer(inbuf, small::INPUT,
                       1, fc_params.C_i, fc_params.H, fc_params.W,
                       BufferT::C_ib, BufferT::C_ob, packed_input.buffer());

    small::Tensor<BufferT> output(fc.output_shape());
    small::init_zeros(output.buffer(), output.size());

    fc.compute_output({&packed_input}, &output);

    BufferT unpacked_output(output.size());
    small::unpack_buffer(output.buffer(), small::BufferTypeEnum::OUTPUT,
                         1UL,
                         output.shape()[small::CHANNEL],
                         output.shape()[small::HEIGHT],
                         output.shape()[small::WIDTH],
                         BufferT::C_ib, BufferT::C_ob,
                         unpacked_output);

    bool passing = true;
    for (size_t i = 0; i < fc_params.C_o*fc_params.H*fc_params.W; i++)
    {
        if (!almost_equal(expected_output[i], unpacked_output[i], 5e-5, 1e-7))
        {
            passing = false;
            std::cerr << i << ": ERROR: unequal outputs: unpacked_output("
                      << unpacked_output[i] << ") != expected_output("
                      << expected_output[i] << ")\n";
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
void test_dense_no_bias(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif
    using ScalarT = typename BufferT::value_type;

    // C_i,H,W,k,s,p,C_o
    LayerParams fc_params = {16, 1, 1, 1, 1, small::PADDING_V, 16};

    ScalarT input[16]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    ScalarT weights[256]{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ScalarT expected_output[16]{1496, 3672, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    //small::shape_type fc_input_shape{1UL, fc_params.C_i, fc_params.H, fc_params.W};

    BufferT weights_buf(fc_params.C_i*fc_params.C_o);
    std::copy(weights, weights+weights_buf.size(), reinterpret_cast<ScalarT*>(weights_buf.data()));

    BufferT packed_weights(weights_buf.size());
    small::pack_buffer(weights_buf, small::FILTER_CONV,
                       fc_params.C_o, fc_params.C_i,
                       1U, 1U,
                       BufferT::C_ib, BufferT::C_ob,
                       packed_weights);

    BufferT inbuf(fc_params.C_i*fc_params.H*fc_params.W);
    BufferT packed_input(inbuf.size());
    std::copy(input, input+inbuf.size(), reinterpret_cast<ScalarT*>(inbuf.data()));
    small::pack_buffer(inbuf, small::INPUT,
                       1, fc_params.C_i, fc_params.H, fc_params.W,
                       BufferT::C_ib, BufferT::C_ob, packed_input);

    BufferT packed_output(fc_params.C_o*fc_params.H*fc_params.W);
    small::init_zeros(packed_output, packed_output.size());

    small::Dense(packed_input.size(), packed_output.size(),
                 packed_input, packed_weights, packed_output);

    BufferT unpacked_output(packed_output.size());
    small::unpack_buffer(packed_output, small::BufferTypeEnum::OUTPUT,
                         1UL,
                         fc_params.C_o,
                         fc_params.H,
                         fc_params.W,
                         BufferT::C_ib, BufferT::C_ob,
                         unpacked_output);

    bool passing = true;
    for (size_t i = 0; i < fc_params.C_o*fc_params.H*fc_params.W; i++)
    {
        if (!almost_equal(expected_output[i], unpacked_output[i], 5e-5, 1e-7))
        {
            passing = false;
            std::cerr << i << ": ERROR: unequal outputs: unpacked_output("
                      << unpacked_output[i] << ") != expected_output("
                      << expected_output[i] << ")\n";
        }
    }

    if (passing) std::cerr << "Test PASSED\n";
    TEST_ASSERT(passing);
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"dense_using_conv2d_layer",        test_conv2d_dense_layer},
    {"dense_using_dense_layer",         test_dense_dense_layer},
    {"dense_using_dense_layer_no_bias", test_dense_dense_layer_no_bias},
    {"dense_function_no_bias",          test_dense_no_bias},
    {NULL, NULL}
};
