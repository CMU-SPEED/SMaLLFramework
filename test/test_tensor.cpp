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
#include <small/Tensor.hpp>

/// @todo test both float and quantized integer buffers eventually
#if !defined(QUANTIZED)
using Buffer = small::FloatBuffer;

//****************************************************************************
void test_capacity_ctor(void)
{
    small::Tensor<Buffer> tensor(42);

    TEST_CHECK(tensor.capacity() == 42);
    TEST_CHECK(tensor.size() == 42);
    TEST_CHECK(tensor.buffer().data() != nullptr);
    TEST_CHECK(tensor.shape() == small::Tensor<Buffer>::shape_type({42, 1, 1}));
}

//****************************************************************************
void test_shape_ctor(void)
{
    small::Tensor<Buffer> tensor({10, 10, 10});

    TEST_CHECK(tensor.capacity() == 1000);
    TEST_CHECK(tensor.size() == 1000);
    TEST_CHECK(tensor.buffer().data() != nullptr);
    TEST_CHECK(tensor.shape() == small::Tensor<Buffer>::shape_type({10, 10, 10}));
}

//****************************************************************************
void test_buffer_ctor(void)
{
    small::Tensor<Buffer>::shape_type shp({10, 10, 10});
    Buffer small_buffer(100);
    bool test_passed = false;
    try
    {
        small::Tensor<Buffer> bad_tensor(shp, small_buffer);
    }
    catch (std::invalid_argument &e_obj)
    {
        test_passed = true;
    }
    TEST_CHECK(test_passed);

    Buffer large_buffer(10000);
    large_buffer[0] = 999;
    small::Tensor<Buffer> tensor(shp, large_buffer);

    TEST_CHECK(10000 == large_buffer.size());
    TEST_CHECK(nullptr != large_buffer.data());
    large_buffer[0] = -42;

    TEST_CHECK(tensor.capacity() == 10000);
    TEST_CHECK(tensor.size() == 1000);
    TEST_CHECK(tensor.buffer().data() != nullptr);
    TEST_CHECK(tensor.shape() == small::Tensor<Buffer>::shape_type({10, 10, 10}));
    TEST_CHECK(999 == tensor.buffer()[0]);
}

//****************************************************************************
void test_buffer_move_ctor(void)
{
    small::Tensor<Buffer>::shape_type shp({10, 10, 10});
    Buffer small_buffer(100);
    bool test_passed = false;
    try
    {
        small::Tensor<Buffer> bad_tensor(shp, small_buffer);
    }
    catch (std::invalid_argument &e_obj)
    {
        test_passed = true;
    }
    TEST_CHECK(test_passed);

    Buffer large_buffer(10000);
    auto *data_ptr = large_buffer.data();
    large_buffer[0] = 999;
    small::Tensor<Buffer> tensor(shp, std::move(large_buffer));

    TEST_CHECK(0 == large_buffer.size());
    TEST_CHECK(nullptr == large_buffer.data());

    TEST_CHECK(tensor.capacity() == 10000);
    TEST_CHECK(tensor.size() == 1000);
    TEST_CHECK(tensor.buffer().data() == data_ptr);
    TEST_CHECK(tensor.shape() == small::Tensor<Buffer>::shape_type({10, 10, 10}));
    TEST_CHECK(999 == tensor.buffer()[0]);
}

//****************************************************************************
void test_set_shape(void)
{
    small::Tensor<Buffer>::shape_type shp({10, 10, 10});
    Buffer buffer(1000);
    buffer[0] = 999;
    small::Tensor<Buffer> tensor(shp, std::move(buffer));
    TEST_CHECK(tensor.size() == 1000);
    TEST_CHECK(tensor.capacity() == 1000);
    TEST_CHECK(tensor.buffer()[0] == 999);

    auto shp1 = tensor.shape();
    TEST_CHECK(shp1[0] == 10);
    TEST_CHECK(shp1[1] == 10);
    TEST_CHECK(shp1[2] == 10);

    bool test_passed = false;
    small::Tensor<Buffer>::shape_type bad_shape({10, 10, 11});
    try
    {
        tensor.set_shape(bad_shape);
    }
    catch (std::invalid_argument &e_obj)
    {
        test_passed = true;
    }

    TEST_CHECK(test_passed);
    TEST_CHECK(tensor.size() == 1000);
    TEST_CHECK(tensor.capacity() == 1000);


    small::Tensor<Buffer>::shape_type small_shape({10, 10, 3});
    tensor.set_shape(small_shape);
    TEST_CHECK(tensor.size() == 300);
    TEST_CHECK(tensor.capacity() == 1000);
    TEST_CHECK(tensor.buffer()[0] == 999);

    shp1 = tensor.shape();
    TEST_CHECK(shp1[0] == 10);
    TEST_CHECK(shp1[1] == 10);
    TEST_CHECK(shp1[2] == 3);
}

//****************************************************************************
void test_swap(void)
{
    Buffer buffer(1000);
    buffer[0] = 999;
    small::Tensor<Buffer> tensor({10, 10, 10}, std::move(buffer));
    TEST_CHECK(tensor.size() == 1000);
    TEST_CHECK(tensor.capacity() == 1000);
    TEST_CHECK(tensor.buffer()[0] == 999);

    auto shp = tensor.shape();
    TEST_CHECK(shp[0] == 10);
    TEST_CHECK(shp[1] == 10);
    TEST_CHECK(shp[2] == 10);

    small::Tensor<Buffer> tensor2({3, 3, 3});
    tensor2.buffer()[0] = -999;
    TEST_CHECK(tensor2.size() == 27);
    TEST_CHECK(tensor2.capacity() == 27);
    TEST_CHECK(tensor2.buffer()[0] == -999);

    shp = tensor2.shape();
    TEST_CHECK(shp[0] == 3);
    TEST_CHECK(shp[1] == 3);
    TEST_CHECK(shp[2] == 3);

    tensor.swap(tensor2);

    TEST_CHECK(tensor.size() == 27);
    TEST_CHECK(tensor.capacity() == 27);
    TEST_CHECK(tensor.buffer()[0] == -999);

    shp = tensor.shape();
    TEST_CHECK(shp[0] == 3);
    TEST_CHECK(shp[1] == 3);
    TEST_CHECK(shp[2] == 3);

    TEST_CHECK(tensor2.size() == 1000);
    TEST_CHECK(tensor2.capacity() == 1000);
    TEST_CHECK(tensor2.buffer()[0] == 999);

    shp = tensor2.shape();
    TEST_CHECK(shp[0] == 10);
    TEST_CHECK(shp[1] == 10);
    TEST_CHECK(shp[2] == 10);
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"capacity ctor",     test_capacity_ctor},
    {"shape ctor",        test_shape_ctor},
    {"buffer(copy) ctor", test_buffer_ctor},
    {"buffer(move) ctor", test_buffer_move_ctor},
    {"set_shape",         test_set_shape},
    {"swap",              test_swap},
    {NULL, NULL}
};
#endif
