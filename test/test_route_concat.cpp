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

#include <acutest.h>

#include <small.h>
#include <small/buffers.hpp>
#include <small/RouteLayer.hpp>

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
                if (buf[index] != (typename BufferT::value_type)c)
                {
                    std::cerr << "ERROR: bad channel number: buf["
                              << c << ":(" << h << "," << w << ")] == "
                              << buf[index]
                              << std::endl;
                    passed = false;
                }
            }
    return passed;
}

//****************************************************************************
void test_route_bad_shapes(void)
{
    small::shape_type input1_shape({1, 16, 52, 52});
    small::shape_type input2_shape({1, 16, 51, 52});
    small::shape_type input3_shape({1, 16, 52, 51});
    small::shape_type input4_shape({2, 16, 52, 52});

    try {
        small::RouteLayer<small::FloatBuffer> route(input1_shape,
                                                    input2_shape);
        TEST_CHECK(false);
    }
    catch (std::invalid_argument const &e) {
        std::cout << "\nERROR expected: " << e.what() << std::endl;
        TEST_CHECK(true);
    }

    try {
        small::RouteLayer<small::FloatBuffer> route(input1_shape,
                                                    input3_shape);
        TEST_CHECK(false);
    }
    catch (std::invalid_argument const &e) {
        std::cout << "ERROR expected: " << e.what() << std::endl;
        TEST_CHECK(true);
    }

    try {
        small::RouteLayer<small::FloatBuffer> route(input1_shape,
                                                    input4_shape);
        TEST_CHECK(false);
    }
    catch (std::invalid_argument const &e) {
        std::cout << "ERROR expected: " << e.what() << std::endl;
        TEST_CHECK(true);
    }
}

//****************************************************************************
void test_route_concat(void)
{
    small::shape_type input1_shape({1, 16, 52, 52});
    small::shape_type input2_shape({1, 32, 52, 52});

    small::RouteLayer<small::FloatBuffer> route(input1_shape,
                                                input2_shape);

    auto const &output_shape(route.output_shape());

    TEST_CHECK( 1 == output_shape[small::BATCH]);
    TEST_CHECK(48 == output_shape[small::CHANNEL]);
    TEST_CHECK(52 == output_shape[small::HEIGHT]);
    TEST_CHECK(52 == output_shape[small::WIDTH]);
}

//****************************************************************************
void test_route_1_buffer(void)
{
    small::shape_type shape({1, 16, 13, 13});
    small::Tensor<Buffer> input(shape, (create_packed_data<Buffer>(shape, 0.f)));
    small::Tensor<Buffer> output(input.size());

    small::RouteLayer<Buffer> route(shape);

    route.compute_output({&input}, &output);
    TEST_CHECK(shape == output.shape());
    TEST_CHECK(check_buffer_contents(shape, output.buffer()));

    route.compute_output({&input}, &input);
    TEST_CHECK(shape == input.shape());
    TEST_CHECK(check_buffer_contents(shape, input.buffer()));

    try
    {
        route.compute_output({&input, &input}, &output);
        TEST_CHECK(false);
    }
    catch (std::invalid_argument &e_obj)
    {
        std::cerr << "\nERROR expected: " << e_obj.what() << std::endl;
        TEST_CHECK(true);
    }
}

//****************************************************************************
void test_route_2_buffers(void)
{
    small::shape_type shape0({1, 16, 13, 13});
    small::Tensor<Buffer> input0(shape0,
                                 (create_packed_data<Buffer>(shape0, 0.f)));
    small::shape_type shape1({1, 32, 13, 13});
    small::Tensor<Buffer> input1(shape1,
                                 (create_packed_data<Buffer>(shape1, 16.f)));

    small::shape_type output_shape({1, 48, 13, 13});
    small::Tensor<Buffer> output(input0.size() + input1.size());

    small::RouteLayer<Buffer> route(shape0, shape1);

    std::cerr << "FIRST TEST\n";
    route.compute_output({&input0, &input1}, &output);
    TEST_CHECK(output_shape == output.shape());
    TEST_CHECK(check_buffer_contents(output_shape, output.buffer()));

    try
    {
        route.compute_output({&input0}, &output);
        TEST_CHECK(false);
    }
    catch (std::invalid_argument &e_obj)
    {
        std::cerr << "\nERROR expected: " << e_obj.what() << std::endl;
        TEST_CHECK(true);
    }

    try
    {
        route.compute_output({&input0, &input0}, &output);
        TEST_CHECK(false);
    }
    catch (std::invalid_argument &e_obj)
    {
        std::cerr << "\nERROR expected: " << e_obj.what() << std::endl;
        TEST_CHECK(true);
    }

    try
    {
        route.compute_output({&input1, &input0}, &output);
        TEST_CHECK(false);
    }
    catch (std::invalid_argument &e_obj)
    {
        std::cerr << "\nERROR expected: " << e_obj.what() << std::endl;
        TEST_CHECK(true);
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"RouteLayer bad shapes", test_route_bad_shapes},
    {"RouteLayer concat",     test_route_concat},
    {"RouteLayer, single buffer", test_route_1_buffer},
    {"RouteLayer, two buffers",   test_route_2_buffers},
    {NULL, NULL}
};
