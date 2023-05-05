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

/// @todo test both float and quantized integer buffers eventually
#if !defined(QUANTIZED)
using Buffer = small::FloatBuffer;

//****************************************************************************
void test_ctor_default(void)
{
    Buffer buf;

    TEST_CHECK(buf.size() == 0);
    TEST_CHECK(buf.data() == nullptr);
}

//****************************************************************************
void test_ctor_size(void)
{
    Buffer buf(42);
    TEST_CHECK(42 == buf.size());
    TEST_CHECK(nullptr != buf.data());

    buf[21] = 42;
    TEST_CHECK(42 == buf[21]);
}

//****************************************************************************
void test_ctor_copy(void)
{
    size_t const SIZE{10UL};
    Buffer buf(SIZE);
    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        buf[ix] = static_cast<Buffer::value_type>(ix);
    }

    TEST_CHECK(10 == buf.size());
    TEST_CHECK(buf[5] == 5.f);

    Buffer buf2(buf);
    TEST_CHECK(10 == buf2.size());
    TEST_CHECK(10 == buf.size());

    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        TEST_CHECK(buf[ix] == static_cast<Buffer::value_type>(ix));
        buf[ix] = 0.f;
        TEST_CHECK(buf2[ix] == static_cast<Buffer::value_type>(ix));
    }
}

//****************************************************************************
void test_ctor_move(void)
{
    size_t const SIZE{10UL};
    Buffer buf(SIZE);
    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        buf[ix] = static_cast<Buffer::value_type>(ix);
    }

    TEST_CHECK(10 == buf.size());
    TEST_CHECK(buf[5] == 5.f);

    Buffer buf2(std::move(buf));

    TEST_CHECK(10 == buf2.size());
    TEST_CHECK(0 == buf.size());

    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        TEST_CHECK(buf2[ix] == static_cast<Buffer::value_type>(ix));
    }
}

//****************************************************************************
void test_copy_assignment(void)
{
    size_t const SIZE{10UL};
    Buffer buf(SIZE);
    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        buf[ix] = static_cast<Buffer::value_type>(ix);
    }

    TEST_CHECK(10 == buf.size());
    TEST_CHECK(buf[5] == 5.f);

    Buffer buf2;
    TEST_CHECK(0 == buf2.size());
    buf2 = buf;
    TEST_CHECK(10 == buf2.size());
    TEST_CHECK(10 == buf.size());

    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        TEST_CHECK(buf[ix] == static_cast<Buffer::value_type>(ix));
        buf[ix] = 0.f;
        TEST_CHECK(buf2[ix] == static_cast<Buffer::value_type>(ix));
    }
}

//****************************************************************************
void test_move_assignment(void)
{
    size_t const SIZE{10UL};
    Buffer buf(SIZE);
    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        buf[ix] = static_cast<Buffer::value_type>(ix);
    }

    TEST_CHECK(10 == buf.size());
    TEST_CHECK(buf[5] == 5.f);

    Buffer buf2;
    TEST_CHECK(0 == buf2.size());
    buf2 = std::move(buf);
    TEST_CHECK(10 == buf2.size());
    TEST_CHECK(0 == buf.size());

    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        TEST_CHECK(buf2[ix] == static_cast<Buffer::value_type>(ix));
    }
}

//****************************************************************************
void test_swap(void)
{
    size_t const SIZE{10UL};
    Buffer buf(SIZE);
    Buffer buf2(2*SIZE);
    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        buf[ix]       =  static_cast<Buffer::value_type>(ix);
        buf2[ix]      = -static_cast<Buffer::value_type>(ix);
        buf2[ix+SIZE] =  static_cast<Buffer::value_type>(ix);
    }

    TEST_CHECK(SIZE == buf.size());
    TEST_CHECK(2*SIZE == buf2.size());

    TEST_CHECK(buf[5] == 5.f);
    TEST_CHECK(buf2[5] == -5.f);
    TEST_CHECK(buf2[5+SIZE] == 5.f);

    buf.swap(buf2);

    TEST_CHECK(SIZE == buf2.size());
    TEST_CHECK(2*SIZE == buf.size());

    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        TEST_CHECK(buf2[ix] == static_cast<Buffer::value_type>(ix));
        TEST_CHECK(buf[ix] == -static_cast<Buffer::value_type>(ix));
        TEST_CHECK(buf[ix+SIZE] == static_cast<Buffer::value_type>(ix));
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"default ctor", test_ctor_default},
    {"size ctor",    test_ctor_size},
    {"copy ctor",    test_ctor_copy},
    {"move ctor",    test_ctor_move},
    {"copy assignment", test_copy_assignment},
    {"move assignment", test_move_assignment},
    {"swap", test_swap},
    {NULL, NULL}
};
#endif
