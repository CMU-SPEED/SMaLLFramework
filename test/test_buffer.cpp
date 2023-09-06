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

//****************************************************************************
template <class BufferT>
void ctor_default()
{
    BufferT buf;

    TEST_CHECK(buf.size() == 0);
    TEST_CHECK(buf.data() == nullptr);
}

void test_ctor_default(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    std::cout << "\nTesting FloatBuffer\n";
    ctor_default<small::FloatBuffer>();
#endif

#if defined(SMALL_HAS_QUINT8_SUPPORT)
    std::cout << "\nTesting QUInt8Buffer\n";
    ctor_default<small::QUInt8Buffer>();
#endif
}

//****************************************************************************
template <class BufferT>
void ctor_size()
{
    BufferT buf(42);
    TEST_CHECK(42 == buf.size());
    TEST_CHECK(nullptr != buf.data());

    buf[21] = 42;
    TEST_CHECK(42 == buf[21]);
}

void test_ctor_size(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    ctor_size<small::FloatBuffer>();
#endif

#if defined(SMALL_HAS_QUINT8_SUPPORT)
    ctor_size<small::QUInt8Buffer>();
#endif
}

//****************************************************************************
template <class BufferT>
void ctor_copy(void)
{
    using ScalarT = typename BufferT::value_type;

    size_t const SIZE{10UL};
    BufferT buf(SIZE);
    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        buf[ix] = static_cast<ScalarT>(ix);
    }

    TEST_CHECK(10 == buf.size());
    TEST_CHECK(buf[5] == static_cast<ScalarT>(5));

    BufferT buf2(buf);
    TEST_CHECK(10 == buf2.size());
    TEST_CHECK(10 == buf.size());

    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        TEST_CHECK(buf[ix] == static_cast<ScalarT>(ix));
        buf[ix] = 0;
        TEST_CHECK(buf2[ix] == static_cast<ScalarT>(ix));
    }
}

void test_ctor_copy(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    ctor_copy<small::FloatBuffer>();
#endif

#if defined(SMALL_HAS_QUINT8_SUPPORT)
    ctor_copy<small::QUInt8Buffer>();
#endif
}

//****************************************************************************
template <class BufferT>
void ctor_move(void)
{
    using ScalarT = typename BufferT::value_type;

    size_t const SIZE{10UL};
    BufferT buf(SIZE);
    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        buf[ix] = static_cast<ScalarT>(ix);
    }

    TEST_CHECK(10 == buf.size());
    TEST_CHECK(buf[5] == static_cast<ScalarT>(5));

    BufferT buf2(std::move(buf));

    TEST_CHECK(10 == buf2.size());
    TEST_CHECK(0 == buf.size());

    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        TEST_CHECK(buf2[ix] == static_cast<ScalarT>(ix));
    }
}

void test_ctor_move(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    ctor_move<small::FloatBuffer>();
#endif

#if defined(SMALL_HAS_QUINT8_SUPPORT)
    ctor_move<small::QUInt8Buffer>();
#endif
}

//****************************************************************************
template <class BufferT>
void copy_assignment(void)
{
    using ScalarT = typename BufferT::value_type;

    size_t const SIZE{10UL};
    BufferT buf(SIZE);
    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        buf[ix] = static_cast<ScalarT>(ix);
    }

    TEST_CHECK(10 == buf.size());
    TEST_CHECK(buf[5] == static_cast<ScalarT>(5));

    BufferT buf2;
    TEST_CHECK(0 == buf2.size());
    buf2 = buf;
    TEST_CHECK(10 == buf2.size());
    TEST_CHECK(10 == buf.size());

    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        TEST_CHECK(buf[ix] == static_cast<ScalarT>(ix));
        buf[ix] = 0;
        TEST_CHECK(buf2[ix] == static_cast<ScalarT>(ix));
    }
}

void test_copy_assignment(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    copy_assignment<small::FloatBuffer>();
#endif

#if defined(SMALL_HAS_QUINT8_SUPPORT)
    copy_assignment<small::QUInt8Buffer>();
#endif
}

//****************************************************************************
template <class BufferT>
void move_assignment(void)
{
    using ScalarT = typename BufferT::value_type;

    size_t const SIZE{10UL};
    BufferT buf(SIZE);
    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        buf[ix] = static_cast<ScalarT>(ix);
    }

    TEST_CHECK(10 == buf.size());
    TEST_CHECK(buf[5] == 5.f);

    BufferT buf2;
    TEST_CHECK(0 == buf2.size());
    buf2 = std::move(buf);
    TEST_CHECK(10 == buf2.size());
    TEST_CHECK(0 == buf.size());

    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        TEST_CHECK(buf2[ix] == static_cast<ScalarT>(ix));
    }
}

void test_move_assignment(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    move_assignment<small::FloatBuffer>();
#endif

#if defined(SMALL_HAS_QUINT8_SUPPORT)
    move_assignment<small::QUInt8Buffer>();
#endif
}

//****************************************************************************
template <class BufferT>
void swap_buffers(void)
{
    using ScalarT = typename BufferT::value_type;

    size_t const SIZE{10UL};
    BufferT buf(SIZE);
    BufferT buf2(2*SIZE);
    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        buf[ix]       =  static_cast<ScalarT>(ix);
        buf2[ix]      =  static_cast<ScalarT>(ix)/2;
        buf2[ix+SIZE] =  static_cast<ScalarT>(ix);
    }

    TEST_CHECK(SIZE == buf.size());
    TEST_CHECK(2*SIZE == buf2.size());

    TEST_CHECK(buf[5] == 5);
    TEST_CHECK(buf2[5] == static_cast<ScalarT>(5UL)/2);
    TEST_CHECK(buf2[5+SIZE] == 5);

    buf.swap(buf2);

    TEST_CHECK(SIZE == buf2.size());
    TEST_CHECK(2*SIZE == buf.size());

    for (size_t ix = 0; ix < SIZE; ++ix)
    {
        TEST_CHECK(buf2[ix] == static_cast<ScalarT>(ix));
        TEST_CHECK(buf[ix] == static_cast<ScalarT>(ix)/2);
        TEST_CHECK(buf[ix+SIZE] == static_cast<ScalarT>(ix));
    }
}

void test_swap(void)
{
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    swap_buffers<small::FloatBuffer>();
#endif

#if defined(SMALL_HAS_QUINT8_SUPPORT)
    swap_buffers<small::QUInt8Buffer>();
#endif
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
