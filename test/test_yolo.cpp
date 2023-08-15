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

#include <acutest.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>

#include <small.h>
#include <small/utils/Timer.hpp>
#include <small/YOLOLayer.hpp>

#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");

//****************************************************************************
template <class BufferT>
bool run_yolo_layer_config(LayerParams const &params,
    std::vector<std::pair<uint32_t, uint32_t>> const &anchors)
{

    size_t effective_C_i = params.C_i;
    size_t packed_C_i = params.C_i;
    if(effective_C_i % BufferT::C_ib != 0) {
        packed_C_i = effective_C_i + (BufferT::C_ib - (effective_C_i % BufferT::C_ib));
    }
    // std::cout << "effective_C_i = " << effective_C_i << std::endl;
    // std::cout << "packed_C_i = " << packed_C_i << std::endl;

    // make a buffer of the packed size

    // keep around both the packed and effective shapes
    small::shape_type packed_input_shape({1UL, packed_C_i, params.H, params.W});
    small::shape_type effective_input_shape({1UL, effective_C_i, params.H, params.W});

    size_t packed_input_size = packed_C_i*params.H*params.W;
    size_t effective_input_size = effective_C_i*params.H*params.W;

    std::string in_fname =
        get_pathname(data_dir, "in", "yolo",
                     params,
                     effective_input_size);

    std::cout << "\nYOLO: input file = " << in_fname << std::endl;

    BufferT input(packed_input_size);

    small::YOLOLayer<BufferT>  yolo_layer(
        packed_input_shape,
        anchors,
        80, // hardcoded num_classes
        608 // hardcoded image size
    );

    BufferT effective_inputs = read_inputs<BufferT>(in_fname);
    BufferT oversize_input_buf(packed_input_size);
    std::copy(
        &effective_inputs[0],
        &effective_inputs[effective_input_size],
        &oversize_input_buf[0]
    );
    small::pack_buffer(
        oversize_input_buf,
        small::INPUT,
        1U, packed_C_i, params.H, params.W,
        BufferT::C_ib, BufferT::C_ob,
        input
    );
    small::Tensor<BufferT> input_tensor(packed_input_shape, input);

    TEST_ASSERT(input_tensor.capacity() == packed_input_size);

    // total predictions are H*W*num_anchors
    size_t num_pred = params.H*params.W*3U;
    size_t num_classes = 80U;

    // create output buffers

    // first 4 elements of the fastest dimension represent bounding box info
    // 5th element is the confidence score
    // remaining elements are the class confidence scores
    small::Tensor<BufferT> bb_n_conf({1U, 1U, num_pred, num_classes + 5U});
    small::Tensor<BufferT>* out = &bb_n_conf;

    // std::cout << "Computing YOLO\n";
    yolo_layer.compute_output({&input_tensor}, out);
    // std::cout << "Finished YOLO\n";
    small::Tensor<BufferT>* bb_n_conf_out = out;

    std::string out_fname =
        get_pathname(data_dir, "out", "yolo",
                     params,
                     effective_input_size);

    std::cout << "YOLO: output file = " << out_fname << std::endl;

    BufferT bb_n_conf_ref(read_inputs<BufferT>(out_fname));

    // compare against regression data
    for (size_t i = 0; i < num_pred; i++)
    {
        // make sure bounding boxes are correctly computed
        for (size_t j = 0; j < 85U; j++)
        {
            if (!almost_equal(bb_n_conf_out->buffer()[i*(num_classes+5U)+j],
                              bb_n_conf_ref[i*(num_classes+5U)+j]))
            {
                std::cerr << "bb_n_conf_out[" << i << "][" << j
                          << "] = (computed)"
                          << bb_n_conf_out->buffer()[i*(num_classes+5U)+j]
                          << " != " << bb_n_conf_ref[i*(num_classes+5U)+j]
                          << std::endl;
                return false;
            }
        }
    }

    return true;
}

//****************************************************************************
void test_yolo_layer_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {255,  19,  19, 0, 0, small::PADDING_F, 0},
        {255,  38,  38, 0, 0, small::PADDING_F, 0},
        {255,  76,  76, 0, 0, small::PADDING_F, 0} //Ci,Hi,Wi,k,s,p,Co
    };

    // first yolo block
    // [yolo]
    // mask = 6,7,8
    // anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    // classes=80
    // num=9
    // jitter=.3
    // ignore_thresh = .7
    // truth_thresh = 1
    // random=1
    TEST_CHECK(true == run_yolo_layer_config<small::FloatBuffer>(params[0], {{116,90}, {156,198}, {373,326}}));


    // second yolo block
    // [yolo]
    // mask = 3,4,5
    // anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    // classes=80
    // num=9
    // jitter=.3
    // ignore_thresh = .7
    // truth_thresh = 1
    // random=1
    TEST_CHECK(true == run_yolo_layer_config<small::FloatBuffer>(params[1], {{30,61}, {62,45}, {59,119}}));

    // third yolo block
    // [yolo]
    // mask = 0,1,2
    // anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    // classes=80
    // num=9
    // jitter=.3
    // ignore_thresh = .7
    // truth_thresh = 1
    // random=1
    TEST_CHECK(true == run_yolo_layer_config<small::FloatBuffer>(params[2], {{10,13}, {16,30}, {33,23}}));
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"yolo_layer_regression_data", test_yolo_layer_regression_data},
    {NULL, NULL}
};
