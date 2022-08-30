//-----------------------------------------------------------------------------
// test/test_acutest.cpp:  test cases for xxx
//-----------------------------------------------------------------------------

// LAGraph, (c) 2022 by the Authors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// See additional acknowledgments in the LICENSE file,

//-----------------------------------------------------------------------------

//#include <LAGraph.h>

#include <acutest.h>

#include <small.h>

//****************************************************************************
void test_dummy(void)
{
    TEST_MSG("Testing equality %d", 42);
    TEST_CHECK(42 == 42);
    //BOOST_CHECK_EQUAL(42, 42);
}

//****************************************************************************
void test_dummy_fails(void)
{
    TEST_CHECK(42 == 0);
    //BOOST_CHECK_EQUAL(42, 0);
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"dummy", test_dummy},
    {"dummy_fails", test_dummy_fails},
    {NULL, NULL}
};
