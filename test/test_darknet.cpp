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
#include <small/models/Darknet.hpp>

//****************************************************************************
void test_darknet_parser(void)
{
    std::string cfg = "../test/cfg_data/no_wspace_yolov3-tiny.cfg";
    std::string weights = "../test/cfg_data/yolov3-tiny.weights";

    std::cout << "\nReading Darknet CFG from " << cfg << std::endl;
    std::cout << "Reading Darknet weights from " << weights << std::endl;

    small::Darknet<small::FloatBuffer> model(cfg, weights);

}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"darknet_parser", test_darknet_parser},
    {NULL, NULL}
};
