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
#include <small/MaxPool1DLayer.hpp>

#include "test_utils.hpp"

//****************************************************************************
void test_maxpool1d_layer_error(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // C_i,H,W,k,s,p,C_o
    LayerParams params {16, 2, 16, 3, 1, small::PADDING_F, 64};

    // H != 1 should fail
    try
    {
        small::shape_type input_shape{1U, params.C_i, params.H, params.W};
        small::MaxPool1DLayer<BufferT> maxpool1d(input_shape,
                                                 params.k,
                                                 params.s, params.p);

        TEST_ASSERT(params.H == 1U);
    }
    catch (std::invalid_argument &e_obj)
    {
        TEST_CHECK(params.H != 1U);
    }
}

//****************************************************************************
void test_maxpool1d_layer(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif
    using ScalarT = typename BufferT::value_type;

    // C_i,H,W,k,s,p,C_o
    LayerParams params {16, 1, 4, 2, 2, small::PADDING_F, 16};

    auto channels = params.C_i;
    if ((channels % BufferT::C_ob) != 0)
        channels += (BufferT::C_ob - (channels % BufferT::C_ob));
    TEST_ASSERT(channels == params.C_i);

    small::shape_type input_shape{1UL, channels, 1UL, params.W, params.C_i};

    ScalarT inputs[64]
                  {1, 2, 3, 4,  1, 2, 3, 4,  1, 2, 3, 4,  1, 2, 3, 4,
                   1, 2, 3, 4,  1, 2, 3, 4,  1, 2, 3, 4,  1, 2, 3, 4,
                   1, 2, 3, 4,  1, 2, 3, 4,  1, 2, 3, 4,  1, 2, 3, 4,
                   1, 2, 3, 4,  1, 2, 3, 4,  1, 2, 3, 4,  1, 2, 3, 4 };
    ScalarT expected_output[32]
                           {2,    4,     2,   4,      2,    4,      2,    4,
                            2,    4,     2,   4,      2,    4,      2,    4,
                            2,    4,     2,   4,      2,    4,      2,    4,
                            2,    4,     2,   4,      2,    4,      2,    4};

    BufferT unpacked_inputs(channels*params.W);
    small::init_zeros(unpacked_inputs, unpacked_inputs.size());
    std::copy(&inputs[0], &inputs[0] + 64, unpacked_inputs.data());

    small::Tensor<BufferT> packed_input(input_shape);
    small::pack_buffer(unpacked_inputs, small::BufferTypeEnum::INPUT,
                       1UL, channels, params.H, params.W,
                       BufferT::C_ib, BufferT::C_ob, packed_input.buffer());

    uint8_t l_pad, r_pad;
    size_t W_o;
    small::compute_padding_output_dim(params.W, params.k, params.s, params.p, l_pad, r_pad, W_o);

    small::Tensor<BufferT> output(channels*W_o);

    // Test function call
    small::MaxPool1D(params.k, params.s, l_pad, r_pad, channels, 1, params.W,
                     packed_input.buffer(), output.buffer());

    BufferT unpacked_output(output.size());
    small::unpack_buffer(output.buffer(), small::BufferTypeEnum::OUTPUT,
                         1UL,
                         channels,
                         1UL,
                         W_o,
                         BufferT::C_ib, BufferT::C_ob,
                         unpacked_output);

    bool passing = true;
    for (size_t ix = 0; ix < params.C_o*W_o; ix++)
    {
        if (unpacked_output[ix] != expected_output[ix])
        {
            passing = false;
            std::cerr << ix << ": ERROR: unequal outputs: unpacked_output("
                      << unpacked_output[ix] << ") != expected_output("
                      << expected_output[ix] << ")\n";
        }
    }
    TEST_CHECK(passing);

    // Test extra channel values are zero
    for (size_t ix = params.C_o*W_o; ix < channels*W_o; ++ix)
    {
        if (unpacked_output[ix] != 0)
        {
            passing = false;
            std::cerr << ix << ": ERROR: extra outputs not zero: unpacked_output("
                      << unpacked_output[ix] << ")\n";
        }
    }
    TEST_CHECK(passing);

    // Test layer class
    small::MaxPool1DLayer<BufferT> maxpool1d(input_shape, params.k, params.s, params.p);
    maxpool1d.compute_output({&packed_input}, &output);

    TEST_ASSERT(maxpool1d.logical_output_channels() == params.C_o);
    TEST_ASSERT(maxpool1d.output_shape()[small::CHANNEL] == channels);

    small::unpack_buffer(output.buffer(), small::BufferTypeEnum::OUTPUT,
                         1UL,
                         output.shape()[small::CHANNEL],
                         output.shape()[small::HEIGHT],
                         output.shape()[small::WIDTH],
                         BufferT::C_ib, BufferT::C_ob,
                         unpacked_output);

    for (size_t ix = 0; ix < params.C_o*W_o; ix++)
    {
        if (unpacked_output[ix] != expected_output[ix])
        {
            passing = false;
            std::cerr << ix << ": ERROR: unequal outputs: unpacked_output("
                      << unpacked_output[ix] << ") != expected_output("
                      << expected_output[ix] << ")\n";
        }
    }
    TEST_CHECK(passing);

    // Test extra channel values are zero
    for (size_t ix = params.C_o*W_o; ix < channels*W_o; ++ix)
    {
        if (unpacked_output[ix] != 0)
        {
            passing = false;
            std::cerr << ix << ": ERROR: extra outputs not zero: unpacked_output("
                      << unpacked_output[ix] << ")\n";
        }
    }
    TEST_CHECK(passing);
}

//****************************************************************************
void test_maxpool1d_layer_odd_output_channels(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif
    using ScalarT = typename BufferT::value_type;

    // C_i,H,W,k,s,p,C_o
    LayerParams params {3, 1, 4, 2, 2, small::PADDING_F, 3};

    auto channels = params.C_i;
    if ((channels % BufferT::C_ob) != 0)
        channels += (BufferT::C_ob - (channels % BufferT::C_ob));

    small::shape_type input_shape{1UL, channels, 1UL, params.W, params.C_i};

    ScalarT inputs[12]         {1, 2, 3, 4,  1, 2, 3, 4,  1, 2, 3, 4};
    ScalarT expected_output[6] {2,    4,     2,   4,      2,    4};

    BufferT unpacked_inputs(channels*params.W);
    small::init_zeros(unpacked_inputs, unpacked_inputs.size());
    std::copy(&inputs[0], &inputs[0] + 12, unpacked_inputs.data());

    small::Tensor<BufferT> packed_input(input_shape);
    small::pack_buffer(unpacked_inputs, small::BufferTypeEnum::INPUT,
                       1UL, channels, params.H, params.W,
                       BufferT::C_ib, BufferT::C_ob, packed_input.buffer());

    uint8_t l_pad, r_pad;
    size_t W_o;
    small::compute_padding_output_dim(params.W, params.k, params.s, params.p, l_pad, r_pad, W_o);

    small::Tensor<BufferT> output(channels*W_o);

    // Test function call
    small::MaxPool1D(params.k, params.s, l_pad, r_pad, channels, 1, params.W,
                     packed_input.buffer(), output.buffer());

    BufferT unpacked_output(output.size());
    small::unpack_buffer(output.buffer(), small::BufferTypeEnum::OUTPUT,
                         1UL,
                         channels,
                         1UL,
                         W_o,
                         BufferT::C_ib, BufferT::C_ob,
                         unpacked_output);

    bool passing = true;
    for (size_t ix = 0; ix < params.C_o*W_o; ix++)
    {
        if (unpacked_output[ix] != expected_output[ix])
        {
            passing = false;
            std::cerr << ix << ": ERROR: unequal outputs: unpacked_output("
                      << unpacked_output[ix] << ") != expected_output("
                      << expected_output[ix] << ")\n";
        }
    }
    TEST_CHECK(passing);

    // Test extra channel values are zero
    for (size_t ix = params.C_o*W_o; ix < channels*W_o; ++ix)
    {
        if (unpacked_output[ix] != 0)
        {
            passing = false;
            std::cerr << ix << ": ERROR: extra outputs not zero: unpacked_output("
                      << unpacked_output[ix] << ")\n";
        }
    }
    TEST_CHECK(passing);

    // Test layer class
    small::MaxPool1DLayer<BufferT> maxpool1d(input_shape, params.k, params.s, params.p);
    maxpool1d.compute_output({&packed_input}, &output);

    TEST_ASSERT(maxpool1d.logical_output_channels() == params.C_o);
    TEST_ASSERT(maxpool1d.output_shape()[small::CHANNEL] == channels);

    small::unpack_buffer(output.buffer(), small::BufferTypeEnum::OUTPUT,
                         1UL,
                         output.shape()[small::CHANNEL],
                         output.shape()[small::HEIGHT],
                         output.shape()[small::WIDTH],
                         BufferT::C_ib, BufferT::C_ob,
                         unpacked_output);

    for (size_t ix = 0; ix < params.C_o*W_o; ix++)
    {
        if (unpacked_output[ix] != expected_output[ix])
        {
            passing = false;
            std::cerr << ix << ": ERROR: unequal outputs: unpacked_output("
                      << unpacked_output[ix] << ") != expected_output("
                      << expected_output[ix] << ")\n";
        }
    }
    TEST_CHECK(passing);

    // Test extra channel values are zero
    for (size_t ix = params.C_o*W_o; ix < channels*W_o; ++ix)
    {
        if (unpacked_output[ix] != 0)
        {
            passing = false;
            std::cerr << ix << ": ERROR: extra outputs not zero: unpacked_output("
                      << unpacked_output[ix] << ")\n";
        }
    }
    TEST_CHECK(passing);
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"maxpool1d_layer_error",   test_maxpool1d_layer_error},
    {"maxpool1d_layer",         test_maxpool1d_layer},
    {"maxpool1d_layer_odd_output_channels", test_maxpool1d_layer_odd_output_channels},
    /// @todo need more tests
    {NULL, NULL}
};
