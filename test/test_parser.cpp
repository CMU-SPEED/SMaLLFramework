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

#include <small/DarknetParser.hpp>

#include <acutest.h>

void test_parser(void) {

    // hardcode path for now
    // assumes you are running from build
    std::string cfg_file = "../test/cfg_data/no_wspace_yolov3-tiny.cfg";
    std::string weights_file = "../test/cfg_data/yolov3-tiny.weights";

    std::cout << "Creating model from " << cfg_file << " and " << weights_file << "\n";

    std::vector<small::Layer<small::FloatBuffer>*> model = \
        parse_cfg_and_weights<small::FloatBuffer>(cfg_file, weights_file);

    std::cout << "Finished parsing\n";

    // if we reach this point then parsing should have succeeded
    // TODO: add some checks to make sure the created model is correct
    TEST_CHECK(true);
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"parser_test", test_parser},
    {NULL, NULL}
};