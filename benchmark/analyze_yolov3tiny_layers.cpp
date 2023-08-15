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

//#define DEBUG_LAYERS
//#define RECORD_CALLS

#include <small.h>
#include <small/models/Darknet.hpp>

#include "../test/test_utils.hpp"

std::string const  cfg_dir("../test/cfg_data");
std::string const data_dir("../test/regression_data");
std::string const yolo_data_dir(
    "../benchmark/regression_data/yolov3tiny_layer_outputs");

size_t const FAIL_COUNT_MAX(10);

//****************************************************************************
template <typename ScalarT>
size_t compare_outputs(size_t   layer_num,
                       ScalarT *computed_out,
                       ScalarT *expected_out,
                       size_t   numel)
{
    bool passing = true;
    size_t fail_cnt = 0;
    for (size_t i = 0; i < numel; i++)
    {
        if (!almost_equal(computed_out[i], expected_out[i], 5e-3, 1e-3))
        {
            fail_cnt++;
            if (fail_cnt < FAIL_COUNT_MAX)
            {
                std::cerr << "FAIL: layer " << layer_num
                          << " output " << i << " = "
                          << "(computed) " << computed_out[i]
                          << " != " << expected_out[i] << std::endl;
            }
            passing = false;
        }
    }

    std::cerr << "Total fails = " << fail_cnt << std::endl;
    if (!passing)
    {
        std::cerr << "FAIL at layer " << layer_num << std::endl;
    }
    else
    {
        std::cerr << "PASS at layer " << layer_num << std::endl;
    }
    std::cerr << std::endl;

    return fail_cnt;
}

//****************************************************************************
void test_yolov3_tiny(void)
{
    size_t num_fails = 0;
    size_t total_tests = 0;

    using BufferT = small::FloatBuffer;
    using ScalarT = BufferT::value_type;

    std::string cfg = cfg_dir + "/yolov3-tiny.cfg";
    std::string weights = cfg_dir + "/yolov3-tiny.weights";

    std::cout << "\n\nReading Darknet CFG from " << cfg << std::endl;
    std::cout << "Reading Darknet weights from " << weights << std::endl;

    small::Darknet<BufferT> model(cfg, weights, false);
    //std::vector<small::Tensor<BufferT>*> yolo_outs = model.inference(&input_tensor_dc);
    //std::vector<small::Tensor<BufferT>*> layer_outs = model.get_layer_outputs();

    // Recreating each individual layer computation feeding it input
    // data from a pytorch run

    std::cout << "num layers = " << model.get_num_layers() << std::endl;
    for (size_t layer_num=0; layer_num<model.get_num_layers(); ++layer_num)
    {
        std::cout << "*** Processing layer " << layer_num << std::endl;
        auto layer_ptr(model.get_layer(layer_num));

        // --------------
        // setup output tensor
        std::cout << "Reading pytorch output of layer " << layer_num << std::endl;
        std::string layer_out =
            "out" + std::to_string(layer_num) + "__darknet.bin";

        BufferT expected_layer_out;
        try
        {
            expected_layer_out =
                read_inputs<BufferT>(yolo_data_dir + "/" + layer_out);
        }
        catch (std::exception const &e)
        {
            printf("[ERROR] Missing yolo layers output files. ");
            printf("Did you untar regression_data.tar.gz?\n");
            throw;
        }

        small::Tensor<BufferT> out_tensor(layer_ptr->output_size());

        // --------------
        // setup input tensor(s)
        std::cout << "Reading pytorch input(s) of layer " << layer_num << std::endl;
        std::vector<small::Tensor<BufferT> const *> inputs;
        std::vector<size_t> const &parent_ids(model.get_parent_ids(layer_num));

        if (parent_ids.empty())
        {
            std::string in_fname = data_dir + "/in_yolov3tiny_Ci3_H416_W416_519168.bin";

            std::cout << "\nUsing Input data from " << in_fname << std::endl;

            BufferT input(read_inputs<BufferT>(in_fname));
            BufferT input_dc(input.size());
            //small::Tensor<BufferT> input_tensor_dc({1U, 3U, 416U, 416U});
            small::pack_buffer(input,
                               small::INPUT,
                               1U, 3U, 416U, 416U,
                               BufferT::C_ib, BufferT::C_ob,
                               input_dc);

            small::shape_type input_shape({1U, 3U, 416U, 416U});
            inputs.push_back(new small::Tensor<BufferT>(input_shape,
                                                        std::move(input_dc)));
            if (model.get_input_shape() != input_shape)
            {
                throw std::invalid_argument("ERROR: input shape mismatch");
            }
        }
        else
        {
            for (auto parent_id : parent_ids)
            {
                std::cout << "Reading output from predecessor = " << parent_id
                          << std::endl;

                std::string layer_input_filename =
                    "out" + std::to_string(parent_id) + "__darknet.bin";
                BufferT tmp_input =
                    read_inputs<BufferT>(yolo_data_dir + "/" + layer_input_filename);

                auto *tmp_in_tensor = new small::Tensor<BufferT>(
                    model.get_layer(parent_id)->output_shape());

                // pack new input tensor
                small::pack_buffer(tmp_input,
                                   small::INPUT,
                                   1U,
                                   tmp_in_tensor->shape()[small::CHANNEL],
                                   tmp_in_tensor->shape()[small::HEIGHT],
                                   tmp_in_tensor->shape()[small::WIDTH],
                                   BufferT::C_ib, BufferT::C_ob,
                                   tmp_in_tensor->buffer());
                inputs.push_back(tmp_in_tensor);
            }
        }

        layer_ptr->compute_output(inputs, &out_tensor);
        small::shape_type output_shape(out_tensor.shape());

        // cleanup input buffers
        for (auto buf : inputs) { delete buf; }

        // --------------
        // output from darknet could be padded
        // we want to ignore that part
        size_t effective_out_size = expected_layer_out.size();
        size_t computed_out_size  = out_tensor.size();

        std::cout << "effective_out_size = " << effective_out_size
                  << std::endl;
        std::cout << "computed_out_size = " << computed_out_size
                  << std::endl;

        // --------------
        // check result
        std::cout << "Checking output of layer " << layer_num << std::endl;

        // we only unpack when it is not a yolo block
        // currently which blocks are yolo are hardcoded
        if (typeid(*layer_ptr) == typeid(small::YOLOLayer<BufferT>))
        {
            // for yolo, the number of outputs must match
            assert(effective_out_size == computed_out_size);

            num_fails +=
                compare_outputs<ScalarT>(
                    layer_num,
                    out_tensor.buffer().data(),
                    expected_layer_out.data(),
                    effective_out_size);
            total_tests += effective_out_size;
        }
        else
        {
            BufferT unpacked_computed_layer_out(out_tensor.size());
            small::unpack_buffer(
                out_tensor.buffer(),
                small::OUTPUT,
                1U,
                output_shape[small::CHANNEL],
                output_shape[small::HEIGHT],
                output_shape[small::WIDTH],
                BufferT::C_ib, BufferT::C_ob,
                unpacked_computed_layer_out);

            num_fails +=
                compare_outputs<ScalarT>(
                    layer_num,
                    unpacked_computed_layer_out.data(),
                    expected_layer_out.data(),
                    effective_out_size);
            total_tests += effective_out_size;
        }

    }

    std::cerr << "num_fails/total_tests = " << num_fails << "/"
              << total_tests << std::endl;
}

//****************************************************************************
//****************************************************************************
int main(int, char**)
{
    bool passed = true;
    try
    {
        test_yolov3_tiny();
    }
    catch (std::exception const &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        passed = false;
    }
    return (passed ? 0 : 1);
};
