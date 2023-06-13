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
BufferT read_yolo_data(std::string const &data_file) {
    std::ifstream data_in(data_file, std::ios::binary);
    data_in.seekg(0, std::ios::end);
    size_t total_bytes = data_in.tellg(); // total size of weights in bytes
    data_in.seekg(0, std::ios::beg);

    BufferT yolo_data(total_bytes / sizeof(typename BufferT::value_type));
    data_in.read(reinterpret_cast<char*>(yolo_data.data()), total_bytes);
    data_in.close();

    return yolo_data;
}

//****************************************************************************
template <class BufferT>
bool run_relu_layer_config(LayerParams const &params)
{
    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    size_t input_size = params.C_i*params.H*params.W;
    small::YOLOLayer<BufferT>  yolo_layer(
        input_shape, {{116,90}, {156,198}, {373,326}},
        80
    );

    std::string in_fname =
        get_pathname(data_dir, "in", "yolo",
                     params,
                     input_size);
    std::cout << "\nYOLO: input file = " << in_fname << std::endl;
    BufferT input = read_yolo_data<BufferT>(in_fname);
    small::Tensor<BufferT> input_tensor(input_shape, std::move(input));

    TEST_ASSERT(input_tensor.capacity() == input_size);

    // total predictions are H*W*num_anchors
    size_t num_pred = params.H*params.W*3U;
    size_t num_classes = 80U;

    // create output buffers
    std::vector<small::Tensor<BufferT>*> outs;
    // first 4 elements of the fastest dimension represent bounding box info
    // 5th element is the confidence score
    // remaining elements are the class confidence scores
    small::Tensor<BufferT> bb_n_conf({1U, 1U, num_pred, num_classes + 5U});
    outs.push_back(&bb_n_conf);

    std::cout << "Computing YOLO\n";
    yolo_layer.compute_output({&input_tensor}, outs);
    std::cout << "Finished YOLO\n";
    small::Tensor<BufferT>* bb_n_conf_out = outs[0];

    std::string bb_fname = data_dir + "/out_bb__yolo_Ci255_H19_W19.bin";
    std::string score_fname = data_dir + "/out_score__yolo_Ci255_H19_W19.bin";

    BufferT bb = read_yolo_data<BufferT>(bb_fname);
    BufferT obj_score = read_yolo_data<BufferT>(score_fname);

    // compare against regression data
    for(size_t i = 0; i < num_pred; i++) {
        // make sure bounding boxes are correctly computed
        for(size_t j = 0; j < 4U; j++) {
            if(!almost_equal(bb_n_conf_out->buffer()[i*(num_classes+5U)+j], bb[i*4U+j])) {
                std::cerr << "bb_n_conf_out[" << i << "][" << j << "] = " << bb_n_conf_out->buffer()[i*(num_classes+5U)+j] << " != " << bb[i*4U+j] << std::endl;
                return false;
            }
        }
        // make sure obj confidences are correctly computed
        if(!almost_equal(bb_n_conf_out->buffer()[i*(num_classes+5U) + 4U], obj_score[i])) {
            std::cerr << "bb_n_conf_out[" << i << "][4] = " << bb_n_conf_out->buffer()[i*(num_classes+5U)+4U] << " != " << obj_score[i] << std::endl;
            return false;
        }
    }

    return true;
}

//****************************************************************************
void test_relu_layer_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        {255,  19,  19, 0, 0, small::PADDING_F, 0}  //Ci,Hi,Wi,k,s,p,Co
    };

    for (LayerParams const &p : params)
    {
        TEST_CHECK(true == run_relu_layer_config<small::FloatBuffer>(p));
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"yolo_layer_regression_data", test_relu_layer_regression_data},
    // {"relu_performance", measure_relu_performance},
    {NULL, NULL}
};
