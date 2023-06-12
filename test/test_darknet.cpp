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

    // std::string cfg = "/home/nicholai/prog/cnn_cfgs/no_wspace_yolov3.cfg";
    // std::string weights = "/home/nicholai/prog/yolo_weights/yolov3.weights";

    std::cout << "\n\nReading Darknet CFG from " << cfg << std::endl;
    std::cout << "Reading Darknet weights from " << weights << std::endl;

    try {
        small::Darknet<small::FloatBuffer> model(cfg, weights);
        small::Tensor<small::FloatBuffer> input(model.get_input_shape());
        small::init(input.buffer(), input.capacity());
        std::cout << "\n";
        model.inference({&input});
        TEST_CHECK(true);
    }
    catch (std::exception const &e) {
        std::cout << "ERROR: " << e.what() << std::endl;
        TEST_CHECK(false);
    }

}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"darknet_parser", test_darknet_parser},
    {NULL, NULL}
};
