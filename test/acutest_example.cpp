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
