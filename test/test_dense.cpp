//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2025 by The SMaLL Contributors, All Rights Reserved.
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
#include <small/Conv1DLayer.hpp>
#include <small/buffers.hpp>
#include <small/Tensor.hpp>

#include "test_utils.hpp"

void test_dense_layer(void) {
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif
    using ScalarT = typename BufferT::value_type;

    ScalarT *input = new ScalarT[16]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    ScalarT *weights = new ScalarT[256]{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ScalarT *bias = new ScalarT[16]{1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ScalarT *output = new ScalarT[16]{1497, 3674, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // C_i,H,W,k,s,p,C_o
    LayerParams fc_params = {16, 1, 1, 1, 1, small::PADDING_V, 16};
    small::shape_type fc_input_shape{1UL, fc_params.C_i, fc_params.H, fc_params.W};

    BufferT unpacked_input(16), packed_input(16);
    BufferT bias_buf(16);
    BufferT unpacked_weights(256), packed_weights(256);
    BufferT expected_output(16);

    std::move(input, input+16, reinterpret_cast<ScalarT*>(unpacked_input.data()));
    std::move(weights, weights+256, reinterpret_cast<ScalarT*>(unpacked_weights.data()));
    std::move(bias, bias+16, reinterpret_cast<ScalarT*>(bias_buf.data()));
    std::move(output, output+16, reinterpret_cast<ScalarT*>(expected_output.data()));

    small::pack_buffer(unpacked_input, small::INPUT, 1, 16, 1, 1, BufferT::C_ib, BufferT::C_ob, packed_input);
    small::pack_buffer(unpacked_weights, small::FILTER_FC, 16, 16, 1, 1, BufferT::C_ib, BufferT::C_ob, packed_weights);

    small::Conv1DLayer<BufferT> *fc = new small::Conv1DLayer<BufferT>(fc_input_shape, fc_params.k, fc_params.s, fc_params.p, fc_params.C_o,
        packed_weights, bias_buf, true, small::ActivationType::NONE);

    BufferT computed_output(16), computed_output_unpacked(16);
    small::init_zeros(computed_output, 16);

    small::Tensor<BufferT> input_tensor(fc_input_shape, packed_input);
    small::Tensor<BufferT> output_tensor(fc->output_shape(), computed_output);

    fc->compute_output({&input_tensor}, &output_tensor);

    small::unpack_buffer(output_tensor.buffer(), small::OUTPUT, 1, 16, 1, 1, BufferT::C_ib, BufferT::C_ob, computed_output_unpacked);

    for(size_t i = 0; i < expected_output.size(); i++) {
        if(!almost_equal(expected_output.data()[i], computed_output_unpacked.data()[i])) {
            printf("ERROR: on index %li %f != %f\n", i, expected_output.data()[i], computed_output_unpacked.data()[i]);
        }
        else {
            printf("Passed on index %li\n", i);
        }
    }

    delete [] input;
    delete [] weights;
    delete [] bias;
    delete [] output;
}

TEST_LIST = {
    {"conv2d_dense_layer",   test_dense_layer},
    {NULL, NULL}
};
