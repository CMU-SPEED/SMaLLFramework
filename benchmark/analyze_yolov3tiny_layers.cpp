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

#include <small.h>
#include <small/models/Darknet.hpp>

#include "../test/test_utils.hpp"

std::string const data_dir("../test/regression_data");
std::string const yolo_data_dir(
    "../benchmark/regression_data/yolov3tiny_layer_outputs");
std::string const cfg_dir("../test/cfg_data");


//****************************************************************************
template <typename ScalarT>
bool compare_outputs(size_t layer_idx,
                     ScalarT *computed_out, ScalarT *expected_out,
                     size_t numel)
{

    bool passing = true;
    size_t fail_cnt = 0;
    for (size_t i = 0; i < numel; i++)
    {
        if (!almost_equal(computed_out[i], expected_out[i], 5e-3, 1e-3))
        {
            fail_cnt++;
            if (fail_cnt < 20)
            {
                std::cerr << "FAIL: layer " << layer_idx-1
                          << " output " << i << " = "
                          << "(computed) " << computed_out[i]
                          << " != " << expected_out[i] << std::endl;
            }
            passing = false;
        }
    }

    if (!passing)
    {
        std::cerr << "FAIL at layer " << layer_idx - 1 << std::endl;
        std::cerr << "Total fails = " << fail_cnt << std::endl;
    }
    else
    {
        std::cerr << "PASS at layer " << layer_idx - 1 << std::endl;
    }
    std::cerr << std::endl;

    return passing;
}


//****************************************************************************
bool test_yolov3_tiny(void)
{
    bool passing = true;

    using BufferT = small::FloatBuffer;
    using ScalarT = BufferT::value_type;

    std::string cfg = cfg_dir + "/no_wspace_yolov3-tiny.cfg";
    std::string weights = cfg_dir + "/yolov3-tiny.weights";

    std::cout << "\n\nReading Darknet CFG from " << cfg << std::endl;
    std::cout << "Reading Darknet weights from " << weights << std::endl;

    std::string in_fname = data_dir + "/in_yolov3tiny_Ci3_H416_W416_519168.bin";

    std::cout << "\nUsing Input data from " << in_fname << std::endl;

    BufferT input(read_inputs<BufferT>(in_fname));
    small::Tensor<BufferT> input_tensor_dc({1U, 3U, 416U, 416U});
    small::pack_buffer(
        input,
        small::INPUT,
        1U, 3U, 416U, 416U,
        C_ib, C_ob,
        input_tensor_dc.buffer()
    );

    try
    {
        small::Darknet<BufferT> model(cfg, weights, true);
        if (model.get_input_shape() != input_tensor_dc.shape())
        {
            std::cerr << "ERROR: input shape mismatch" << std::endl;
            return false;
        }
        std::vector<small::Tensor<BufferT>*> yolo_outs = model.inference({&input_tensor_dc});
        std::vector<small::Tensor<BufferT>*> layer_outs = model.get_layer_outputs();

        // check number of yolo outputs
        if (yolo_outs.size() != 2)
        {
            std::cerr << "ERROR: not enough yolo outputs" << std::endl;
            return false;
        }

        // layer_idx == 0 represents the input to the model
        size_t yolo_idx = 0;
        for (size_t layer_idx = 1; layer_idx < layer_outs.size(); layer_idx++)
        {
            std::cout << "Checking output of layer " << layer_idx-1
                      << std::endl;

            // read in layer output from darknet
            std::string layer_out =
                "out" + std::to_string(layer_idx-1) + "__darknet.bin";
            BufferT expected_layer_out =
                read_inputs<BufferT>(yolo_data_dir + "/" + layer_out);

            // output from darknet could be padded
            // we want to ignore that part
            size_t effective_out_size = expected_layer_out.size();
            size_t computed_out_size = layer_outs[layer_idx]->capacity();

            std::cout << "effective_out_size = " << effective_out_size
                      << std::endl;
            std::cout << "computed_out_size = " << computed_out_size
                      << std::endl;

            BufferT unpacked_computed_layer_out(computed_out_size);

            // we only unpack when it is not a yolo block
            // currently which blocks are yolo are hardcoded
            if (layer_idx-1 == 16 || layer_idx-1 == 23)
            {
                // for yolo, the number of outputs must match
                assert(effective_out_size == computed_out_size);

                passing &= compare_outputs<ScalarT>(
                    layer_idx,
                    &(yolo_outs[yolo_idx++]->buffer()[0]),
                    &(expected_layer_out[0]),
                    effective_out_size
                );
            }
            else
            {
                small::unpack_buffer(
                    layer_outs[layer_idx]->buffer(),
                    small::OUTPUT,
                    1U,
                    layer_outs[layer_idx]->shape()[1],
                    layer_outs[layer_idx]->shape()[2],
                    layer_outs[layer_idx]->shape()[3],
                    C_ib, C_ob,
                    unpacked_computed_layer_out
                );

                passing &= compare_outputs<ScalarT>(
                    layer_idx,
                    &(unpacked_computed_layer_out[0]),
                    &(expected_layer_out[0]),
                    effective_out_size
                );

            }

        }

        std::cerr << "Tests " << (passing ? "PASSED\n" : "FAILED\n");
    }
    catch (std::exception const &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        passing = false;
    }
    return passing;
}

//****************************************************************************
//****************************************************************************
int main(int, char**)
{
    bool passed = test_yolov3_tiny();
    return (passed ? 0 : 1);
};
