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
template <typename BufferT>
bool compare_outputs(size_t layer_idx,
                     BufferT computed_out, BufferT expected_out,
                     size_t numel)
{
    bool passing = true;
    size_t fail_cnt = 0;
    for (size_t i = 0; i < numel; i++)
    {
        if (!almost_equal(computed_out[i], expected_out[i]))
        {
            fail_cnt++;
            if (fail_cnt < 10)
            {
                std::cerr << "FAIL: layer " << layer_idx
                          << " output " << i << " = "
                          << "(computed) " << computed_out[i]
                          << " != " << expected_out[i] << std::endl;
            }
            passing = false;
        }
    }
    return passing;
}


//****************************************************************************
void test_yolov3_tiny(void)
{

    using BufferT = small::FloatBuffer;

    std::string cfg = cfg_dir + "/no_wspace_yolov3-tiny.cfg";
    std::string weights = cfg_dir + "/yolov3-tiny.weights";

    std::cout << "\n\nReading Darknet CFG from " << cfg << std::endl;
    std::cout << "Reading Darknet weights from " << weights << std::endl;

    std::string in_fname = data_dir + "/in_yolov3tiny_Ci3_H416_W416_519168.bin";
    std::string out_fname = data_dir + "/out_all_layers_yolov3tiny.bin";

    std::cout << "\nUsing Input data from " << in_fname << std::endl;
    std::cout << "Using Output data from " << out_fname << std::endl;

    BufferT input(read_inputs<BufferT>(in_fname));
    small::Tensor<BufferT> input_tensor({1U, 3U, 416U, 416U});
    small::pack_buffer(
        input,
        small::INPUT,
        1U, 3U, 416U, 416U,
        C_ib, C_ob,
        input_tensor.buffer()
    );

    BufferT output(read_inputs<BufferT>(out_fname));
    small::Tensor<BufferT> output_tensor({1U, 1U, 2535U, 85U}, output);

    try {

        small::Darknet<BufferT> model(cfg, weights, true);
        if (model.get_input_shape() != input_tensor.shape())
        {
            std::cerr << "ERROR: input shape mismatch" << std::endl;
            TEST_CHECK(false);
        }
        std::vector<small::Tensor<BufferT>*> yolo_outs = model.inference({&input_tensor});
        std::vector<small::Tensor<BufferT>*> layer_outs = model.get_layer_outputs();

        // check number of yolo outputs
        if (yolo_outs.size() != 2)
        {
            std::cerr << "ERROR: not enough yolo outputs" << std::endl;
            TEST_CHECK(false);
        }

        bool passing = true;

        for (size_t layer_idx = 0; layer_idx < layer_outs.size(); layer_idx++)
        {
            passing &= compare_outputs<BufferT>(
                    layer_idx,
                    layer_outs[layer_idx]->buffer(),
                    layer_outs[layer_idx]->buffer(),
                    layer_outs[layer_idx]->capacity()
                );

            if (!passing)
            {
                std::cerr << "FAIL at layer " << layer_idx << std::endl;
                break;
            }
        }

        // size_t fail_cnt = 0;
        // for(size_t p = 0; p < total_pred; p++) {
        //     for(size_t c = 0; c < 85; c++) {
        //         if(!almost_equal(out_buf[p*85U + c], output_tensor.buffer()[p*85U + c]))
        //         {
        //             fail_cnt++;
        //             if(fail_cnt < 10) {
        //                 std::cerr << "FAIL: darknet(" << p << ", " << c << ") = "
        //                         << "(computed) " << out_buf[p*85U + c] << " != "
        //                         << output_tensor.buffer()[p*85U + c] << std::endl;
        //             }
        //             passing = false;
        //         }
        //     }
        // }

        TEST_CHECK(passing);
    }
    catch (std::exception const &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        TEST_CHECK(false);
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    // {"yolov3_tiny", test_yolov3_tiny},
    {NULL, NULL}
};
