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
#include <small/models/Darknet.hpp>

#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");
std::string const cfg_dir("../test/cfg_data");

//****************************************************************************
void test_darknet_parser(void)
{

    using BufferT = small::FloatBuffer;

    std::string cfg = cfg_dir + "/no_wspace_yolov3-tiny.cfg";
    std::string weights = cfg_dir + "/yolov3-tiny.weights";

    std::cout << "\n\nReading Darknet CFG from " << cfg << std::endl;
    std::cout << "Reading Darknet weights from " << weights << std::endl;

    std::string in_fname = data_dir + "/in_yolov3tiny_Ci3_H416_W416_519168.bin";
    std::string out_fname = data_dir + "/out_yolov3tiny_P2535_F85_215475.bin";

    std::cout << "\nUsing Input data from " << in_fname << std::endl;
    std::cout << "Using Output data from " << out_fname << std::endl;

    // BufferT input = read_yolo_data<BufferT>(in_fname);
    BufferT input(read_inputs<BufferT>(in_fname));
    small::Tensor<BufferT> input_tensor({1, 3, 416, 416}, std::move(input));
    
    // BufferT output = read_yolo_data<BufferT>(out_fname);
    BufferT output(read_inputs<BufferT>(out_fname));
    small::Tensor<BufferT> output_tensor({1, 1, 2535, 85}, std::move(output));

    try {
        
        small::Darknet<BufferT> model(cfg, weights);
        if(model.get_input_shape() != input_tensor.shape())
        {
            std::cerr << "ERROR: input shape mismatch" << std::endl;
            TEST_CHECK(false);
        }
        std::vector<small::Tensor<BufferT>*> outs = model.inference({&input_tensor});

        // check number of yolo outputs
        if(outs.size() != 2)
        {
            std::cerr << "ERROR: not enough yolo outputs" << std::endl;
            TEST_CHECK(false);
        }

        uint64_t total_pred = 0;
        for(auto out : outs) {
            if(out->shape()[3] != 85U)
            {
                std::cerr << "ERROR: number of yolo outputs per anchor is wrong\n" << std::endl;
                std::cerr << "\tExpected 85, got " << out->shape()[3] << std::endl;
                TEST_CHECK(false);
            }
            total_pred += out->shape()[2];
        }

        if(total_pred != output_tensor.shape()[2])
        {
            std::cerr << "ERROR: number of preditictions is wrong\n" << std::endl;
            std::cerr << "\tExpected 2535, got " << total_pred << std::endl;
            TEST_CHECK(false);
        }

        std::cout << "Total yolo predictions = " << total_pred << std::endl;

        // copy all outputs into larger buffer
        BufferT out_buf(total_pred*85);
        float *out_ptr = &(out_buf[0]);
        for(auto out : outs) {
            std::copy(
                &(out->buffer()[0]), 
                &(out->buffer()[0]) + out->size(), 
                out_ptr
            );
            out_ptr += out->size();
        }

        for(size_t p = 0; p < total_pred; p++) {
            for(size_t c = 0; c < 85; c++) {
                if(!almost_equal(out_buf[p*85U + c], output_tensor.buffer()[p*85U + c]))
                {
                    std::cerr << "ERROR: output mismatch" << std::endl;
                    std::cerr << "\tExpected " << output_tensor.buffer()[p*85U + c] << ", got " << out_buf[p*85U + c] << std::endl;
                    TEST_CHECK(false);
                    return;
                }
            }
        }

        TEST_CHECK(true);
    }
    catch (std::exception const &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        TEST_CHECK(false);
    }

}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"darknet_parser", test_darknet_parser},
    {NULL, NULL}
};
