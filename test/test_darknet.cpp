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

#define PARALLEL 1

//#define PARSER_DEBUG
//#define PARSER_DEBUG_VERBOSE

//#define DAG_DEBUG
//#define BUFFER_DEBUG
//#define DEBUG_LAYERS
//#define SUMMARY 1

#include <acutest.h>

#include <small.h>
#include <small/models/Darknet.hpp>
#include <small/utils/Timer.hpp>

#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");
std::string const cfg_dir("../test/cfg_data");

//****************************************************************************
void test_yolo_parser(void)
{
    using BufferT = small::FloatBuffer;

    std::string cfg = cfg_dir + "/yolov3-tiny.cfg";
    std::string weights = cfg_dir + "/yolov3-tiny.weights";

    std::cout << "\n\nReading Darknet CFG from " << cfg << std::endl;
    std::cout << "Reading Darknet weights from " << weights << std::endl;

    std::string in_fname = data_dir + "/in_yolov3tiny_Ci3_H416_W416_519168.bin";
    std::string out_fname = data_dir + "/out_yolov3tiny_P2535_F85_215475.bin";

    std::cout << "\nUsing Input data from " << in_fname << std::endl;
    std::cout << "Using Output data from " << out_fname << std::endl;

    BufferT input(read_inputs<BufferT>(in_fname));
    small::Tensor<BufferT> input_tensor_dc({1, 3, 416, 416});
    small::pack_buffer(
        input,
        small::INPUT,
        1U, 3U, 416U, 416U,
        BufferT::C_ib, BufferT::C_ob,
        input_tensor_dc.buffer()
    );

    BufferT output(read_inputs<BufferT>(out_fname));
    small::Tensor<BufferT> output_tensor({1, 1, 2535, 85}, std::move(output));

    try
    {
        small::Darknet<BufferT> model(cfg, weights);
        if(model.get_input_shape() != input_tensor_dc.shape())
        {
            std::cerr << "ERROR: input shape mismatch" << std::endl;
            TEST_CHECK(false);
        }
        small::Timer timer;
        timer.start();
        std::vector<small::Tensor<BufferT>*> outs = model.inference({&input_tensor_dc});
        timer.stop();
        std::cout << "Elapsed time: " << timer.elapsed() << "ns.\n";

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
            std::cerr << "ERROR: number of predictions is wrong\n" << std::endl;
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

        bool passing = true;
        size_t fail_cnt = 0;
        for(size_t p = 0; p < total_pred; p++) {
            for(size_t c = 0; c < 85; c++) {
                /// @todo: revisit accuracy
                if(!almost_equal(out_buf[p*85U + c], output_tensor.buffer()[p*85U + c], 5e-3, 1e-3))
                {
                    fail_cnt++;
                    if(fail_cnt < 10) {
                        std::cerr << "FAIL: darknet(" << p << ", " << c << ") = "
                                << "(computed) " << out_buf[p*85U + c] << " != "
                                << output_tensor.buffer()[p*85U + c] << std::endl;
                    }
                    passing = false;
                }
            }
        }

        std::cout << "Total failed (diff was greater than 1e-3) = "
                  << fail_cnt << std::endl;
        TEST_CHECK(passing);

        float const conf_thresh = 0.25f;
        float const iou_thresh = 0.45f;
        auto final_predictions = model.process_outputs(outs,
                                                       conf_thresh,
                                                       iou_thresh);

        std::cerr << "FINAL LIST:\n";
        for (auto &detection : final_predictions)
        {
            std::cerr << detection << std::endl;
        }

    }
    catch (std::exception const &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        TEST_CHECK(false);
    }
}

//****************************************************************************
void test_bug_parser(void)
{

    std::string cfg = cfg_dir + "/yolov3-tiny_wbug.cfg";
    std::string weights = cfg_dir + "/yolov3-tiny.weights";

    try {
        small::Darknet<small::FloatBuffer> model(cfg, weights);
        TEST_CHECK(false);
    }

    catch (std::exception const &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        TEST_CHECK(true);
    }

}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"darknet_parser with bad cfg file", test_bug_parser},
    {"darknet_parser", test_yolo_parser},
    {NULL, NULL}
};
