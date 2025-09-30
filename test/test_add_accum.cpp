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

#include <acutest.h>

#include <small.h>
#include <small/buffers.hpp>
#include <small/AddLayer.hpp>

#if defined(QUANTIZED)
using Buffer = small::QUInt8Buffer;
#else
using Buffer = small::FloatBuffer;
#endif

//****************************************************************************
template <class BufferT>
BufferT create_packed_data(small::shape_type            shape,
                           typename BufferT::value_type offset = 0)
{
    BufferT buf(shape[small::CHANNEL]*shape[small::HEIGHT]*shape[small::WIDTH]);

    for (uint32_t c = 0; c < shape[small::CHANNEL]; ++c)
        for (uint32_t h = 0; h < shape[small::HEIGHT]; ++h)
            for (uint32_t w = 0; w < shape[small::WIDTH]; ++w)
            {
                auto index =
                    small::packed_buffer_index(shape[small::CHANNEL],
                                               shape[small::HEIGHT],
                                               shape[small::WIDTH],
                                               BufferT::C_ib,
                                               c, h, w);
                buf[index] = offset + (typename BufferT::value_type)c;
            }
    return buf;
}

//****************************************************************************
template <class BufferT>
bool check_buffer_contents(small::shape_type const &shape,
                           BufferT const &in_buf0,
                           BufferT const &in_buf1,
                           BufferT const &buf)
{
    bool passed = true;

    for (uint32_t c = 0; c < shape[small::CHANNEL]; ++c)
        for (uint32_t h = 0; h < shape[small::HEIGHT]; ++h)
            for (uint32_t w = 0; w < shape[small::WIDTH]; ++w)
            {
                auto index =
                    small::packed_buffer_index(shape[small::CHANNEL],
                                               shape[small::HEIGHT],
                                               shape[small::WIDTH],
                                               BufferT::C_ib,
                                               c, h, w);
                if (buf[index] != in_buf0[index] + in_buf1[index])
                {
                    std::cerr << "ERROR: wrong value: buf["
                              << index << "] == "
                              << buf[index]
                              << std::endl;
                    passed = false;
                }
            }
    return passed;
}

//****************************************************************************
void test_add_bad_shapes(void)
{
    small::shape_type input1_shape({1, 16, 52, 52});
    small::shape_type input2_shape({1, 16, 51, 52});
    small::shape_type input3_shape({1, 16, 52, 51});
    small::shape_type input4_shape({1, 16, 52, 52, 15});
    small::shape_type input5_shape({1, 32, 52, 52});

    try {
        small::AddLayer<small::FloatBuffer> add(input1_shape,
                                                input2_shape);
        TEST_CHECK(false);
    }
    catch (std::invalid_argument const &e) {
        std::cout << "\nERROR expected: " << e.what() << std::endl;
        TEST_CHECK(true);
    }

    try {
        small::AddLayer<small::FloatBuffer> add(input1_shape,
                                                input3_shape);
        TEST_CHECK(false);
    }
    catch (std::invalid_argument const &e) {
        std::cout << "ERROR expected: " << e.what() << std::endl;
        TEST_CHECK(true);
    }

    try {
        small::AddLayer<small::FloatBuffer> add(input1_shape,
                                                input4_shape);
        TEST_CHECK(false);
    }
    catch (std::invalid_argument const &e) {
        std::cout << "ERROR expected: " << e.what() << std::endl;
        TEST_CHECK(true);
    }

    try {
        small::AddLayer<small::FloatBuffer> add(input1_shape,
                                                input5_shape);
        TEST_CHECK(false);
    }
    catch (std::invalid_argument const &e) {
        std::cout << "ERROR expected: " << e.what() << std::endl;
        TEST_CHECK(true);
    }
}

//****************************************************************************
void test_add(void)
{
    small::shape_type input1_shape({1, 32, 52, 52});
    small::shape_type input2_shape({1, 32, 52, 52});

    small::AddLayer<small::FloatBuffer> add(input1_shape,
                                            input2_shape);

    auto const &output_shape(add.output_shape());

    TEST_CHECK( 1 == output_shape[small::BATCH]);
    TEST_CHECK(32 == output_shape[small::CHANNEL]);
    TEST_CHECK(52 == output_shape[small::HEIGHT]);
    TEST_CHECK(52 == output_shape[small::WIDTH]);
}

//****************************************************************************
void test_add_2_buffers(void)
{
    small::shape_type shape0({1, 16, 13, 13});
    small::Tensor<Buffer> input0(shape0,
                                 (create_packed_data<Buffer>(shape0, 0.f)));

    small::Tensor<Buffer> input1(shape0,
                                 (create_packed_data<Buffer>(shape0, 16.f)));
    small::Tensor<Buffer> output(shape0,
                                 (create_packed_data<Buffer>(shape0, 16.f)));

    small::AddLayer<Buffer> add(shape0, shape0);

    std::cerr << "FIRST TEST\n";
    add.compute_output({&input0}, &output);
    TEST_CHECK(shape0 == output.shape());
    TEST_CHECK(check_buffer_contents(shape0,
                                     input0.buffer(),
                                     input1.buffer(),
                                     output.buffer()));

}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"AddLayer bad shapes", test_add_bad_shapes},
    {"AddLayer ctor",       test_add},
    {"AddLayer two buffers",test_add_2_buffers},
    {NULL, NULL}
};
