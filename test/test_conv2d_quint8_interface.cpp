//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2026 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//****************************************************************************

#define PARALLEL 1

#include <acutest.h>

#include <cstdint>
#include <iostream>

#include <params.h>
#include <Buffer.hpp>
#include <intrinsics.h>

#include <small/buffers.hpp>
#include <small/interface_abstract.hpp>

//****************************************************************************
void test_conv2d_quint8_interface_1x1(void)
{
#if defined(SMALL_HAS_QUINT8_SUPPORT)
    using BufferT = small::QUInt8Buffer;

    constexpr uint32_t C_i = 1;
    constexpr uint32_t C_o = 2;
    constexpr uint32_t H = 2;
    constexpr uint32_t W = 3;
    constexpr uint32_t K = 1;
    constexpr uint32_t stride = 1;
    constexpr uint8_t t_pad = 0;
    constexpr uint8_t b_pad = 0;
    constexpr uint8_t l_pad = 0;
    constexpr uint8_t r_pad = 0;
    constexpr uint32_t H_o = 2;
    constexpr uint32_t W_o = 3;

    BufferT input(C_i * H * W);
    input.m_zero = -128;
    input[0] = 41;  //  -87
    input[1] = 78;  //  -50
    input[2] = 115; //  -13
    input[3] = 152; //   24
    input[4] = 189; //   61
    input[5] = 226; //   98

    BufferT filter(C_o * C_i * K * K);
    filter.m_zero = -128;
    filter[0] = 56;  //  -72
    filter[1] = 227; //   99

    BufferT packed_input(input.size());
    small::pack_buffer(input, small::INPUT,
                       1U, C_i, H, W,
                       BufferT::C_ib, BufferT::C_ob,
                       packed_input);
    packed_input.m_zero = -128;

    BufferT packed_filter(filter.size());
    small::pack_buffer(filter, small::FILTER_CONV,
                       C_o, C_i, K, K,
                       BufferT::C_ib, BufferT::C_ob,
                       packed_filter);
    packed_filter.m_zero = -128;

    constexpr uint32_t output_size = C_o * H_o * W_o;
    BufferT packed_output(output_size * sizeof(BufferT::accum_type));

    small::Conv2D(K, K, stride,
                  t_pad, b_pad, l_pad, r_pad,
                  C_o, C_i, H, W,
                  packed_input, packed_filter, packed_output);

    BufferT output(output_size);
    small::unpack_buffer(packed_output, small::OUTPUT,
                         1U, C_o, H_o, W_o,
                         BufferT::C_ib, BufferT::C_ob,
                         output);

    // Generated with test/run_conv2d_quint8_interface.py using signed-centered
    // raw uint8 values: logical_value = raw_value - 128.
    uint8_t const expected[output_size] = {
        49, 28, 7, 0, 0, 0,
        0, 0, 0, 17, 47, 76
    };

    for (uint32_t ix = 0; ix < output_size; ++ix)
    {
        if (output[ix] != expected[ix])
        {
            std::cerr << "Mismatch at " << ix
                      << ": computed=" << static_cast<int>(output[ix])
                      << ", expected=" << static_cast<int>(expected[ix])
                      << std::endl;
        }
        TEST_CHECK(output[ix] == expected[ix]);
    }
#else
    TEST_CHECK(true);
#endif
}

//****************************************************************************
TEST_LIST = {
    {"conv2d_quint8_interface_1x1", test_conv2d_quint8_interface_1x1},
    {NULL, NULL}
};
