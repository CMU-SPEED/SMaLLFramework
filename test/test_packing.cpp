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

//#include <LAGraph.h>
typedef float dtype;
#include <acutest.h>

#include <small.h>

//****************************************************************************
bool test_input_packing(uint32_t C_i, uint32_t H, uint32_t W) {
    std::cout << "Testing input packing (C_i = " << C_i << ", H = " << H << ", W = " << W << ")\n";
    uint32_t numel = C_i*H*W;
    small::FloatBuffer input(numel);
    small::init(input, numel);
    small::FloatBuffer packed_in(numel);
    uint32_t status = small::pack_buffer(input, small::INPUT, 1, C_i, H, W, C_ib, C_ob, packed_in);
    if(status == 0) {
        return false;
    }
    return true;
}

//****************************************************************************
bool test_filt_packing(uint32_t C_o, uint32_t C_i, uint32_t H, uint32_t W) {
    std::cout << "Testing filter packing (C_o = " << C_o << ", C_i = " << C_i << ", H = " << H << ", W = " << W << ")\n";
    uint32_t numel = C_o*C_i*H*W;
    small::FloatBuffer filt(numel);
    small::init(filt, numel);
    small::FloatBuffer packed_filt(numel);
    uint32_t status = small::pack_buffer(filt, small::FILTER_CONV, C_o, C_i, H, W, C_ib, C_ob, packed_filt);
    if(status == 0) {
        return false;
    }
    return true;
}

//****************************************************************************
bool test_output_packing(uint32_t C_o, uint32_t H, uint32_t W) {
    std::cout << "Testing output packing (C_o = " << C_o << ", H = " << H << ", W = " << W << ")\n";
    uint32_t numel = C_o*H*W;
    small::FloatBuffer output(numel);
    small::init(output, numel);
    small::FloatBuffer packed_out(numel);
    uint32_t status = small::pack_buffer(output, small::OUTPUT, 1, C_o, H, W, C_ib, C_ob, packed_out);
    if(status == 0) {
        return false;
    }
    return true;
}

//****************************************************************************
void test_packing(void)
{
    std::cout << "\n";
    
    TEST_CHECK(test_input_packing(3, 96, 96));
    TEST_CHECK(test_input_packing(C_ib, 416, 416));
    TEST_CHECK(test_input_packing(3*C_ib, 416, 416));

    TEST_CHECK(test_filt_packing(256, 3, 3, 3));
    TEST_CHECK(test_filt_packing(256, 3, 1, 1));

    TEST_CHECK(test_output_packing(16, 416, 416));
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"packing", test_packing},
    {NULL, NULL}
};
