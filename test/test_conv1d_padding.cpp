//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
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

#include "test_utils.hpp"

//****************************************************************************
template <class BufferT>
bool run_conv1d_layer_config(LayerParams const &params)
{
    // Set up sizes
    size_t const C_i = params.C_i;
    size_t const W   = params.W;
    size_t const K   = params.k;
    size_t const S   = params.s;
    size_t const C_o = params.C_o;
    size_t const B = params.H;

    uint8_t left_pad = 0, right_pad = 0;
    if (params.p == small::PADDING_F)
    {
        small::calc_padding(W, K, S, left_pad, right_pad);
    }

    printf("padding (%d,%d) %d \n", left_pad, right_pad, FLOAT_C_ob);
    // Input, Filter, Output sizes
    size_t input_size = C_i * W * B;
    size_t filter_size = C_i * K * C_o;
    size_t Wo = small::compute_output_dim(W, K, S, params.p);
    size_t output_size = C_o * Wo * B;

    size_t padded_W = left_pad + W + right_pad;
    size_t padded_input_size = C_i * padded_W * B;

    printf(" output dims: %d %d %d %d\n", C_o, Wo, B, output_size);
    // Allocate
    BufferT input_dc(input_size);
    BufferT filter_dc(filter_size);
    BufferT output_dc(output_size);
    BufferT padded_input_dc(padded_input_size);
    BufferT output_dc_answers(output_size);

    // Initialize inputs: all ones
    // small::init_arange<BufferT, FLOAT_C_ob>(input_dc, B, W, C_i);
    small::init(input_dc, input_size);
    small::init_zeros(padded_input_dc, padded_input_size);

    // Initialize filters: all ones
    small::init(filter_dc, filter_size);

    for(size_t j = 0; j <W ; j++)
    {
        
            std::cerr << input_dc[j*FLOAT_C_ob] << " ";
    
    }
    std::cerr << std::endl;


    //@todo: put this into buffers.hpp
    //create the padded input
    small::pad_1D<BufferT, FLOAT_C_ob>(input_dc, C_i, B, W,
                left_pad, right_pad,
                padded_input_dc);


    //iterate over tiles of padded input and print the first channel in the block

    for(size_t j = 0; j < left_pad + W + right_pad; j++)
    {
        
            std::cerr << padded_input_dc[j*FLOAT_C_ob] << " ";
    
    }

    std::cerr << std::endl;
    

    // Compute
    small::Conv1D(K, S,
                  left_pad, right_pad,
                  C_o, C_i, B, W,
                  input_dc, filter_dc, output_dc);

    //check against padded input
    small::Conv1D(K, S,
                  0, 0,
                  C_o, C_i, B, padded_W,
                  padded_input_dc, filter_dc, output_dc_answers);

    // Verify
    bool passing = true;
    for(size_t i = 0; i < output_size; i++)
    {
        if (std::abs(output_dc[i] - output_dc_answers[i]) > 1e-5)
        {
            passing = false;
            std::cerr << "Mismatch at index " << i
                      << ": " << output_dc[i]
                      << " != " << output_dc_answers[i]
                      << std::endl;
        }

    }
    if (passing)
    {
        std::cerr << "Test PASSED\n";
    }
    else
    {
        std::cerr << "Test FAILED\n";
    }
    //free the small buffers


    return passing;
}

//****************************************************************************
void test_conv1d_layer_regression_data(void)
{
    std::vector<LayerParams> params =
    {
        // {C_i, H (not used for 1D), W, k, s, padding, C_o}

        {16, 1, 32, 3, 1, small::PADDING_F, 16},
        {16, 1, 31, 3, 2, small::PADDING_F, 16},
        {16, 1, 32, 3, 2, small::PADDING_F, 16},

        //kernel = 5

        {16, 1, 32, 5, 1, small::PADDING_F, 16},
        {16, 1, 31, 5, 2, small::PADDING_F, 16},
        {16, 1, 32, 5, 2, small::PADDING_F, 16},


        // kernel = 7
        {16, 1, 32, 7, 1, small::PADDING_F, 16},
        {16, 1, 31, 7, 2, small::PADDING_F, 16},
        {16, 1, 32, 7, 2, small::PADDING_F, 16},

        //kernel = 9
        {16, 1, 32, 9, 1, small::PADDING_F, 16},
        {16, 1, 31, 9, 2, small::PADDING_F, 16},
        {16, 1, 32, 9, 2, small::PADDING_F, 16},

        //kernel = 11
        {16, 1, 32, 11, 1, small::PADDING_F, 16},
        {16, 1, 31, 11, 2, small::PADDING_F, 16},
        {16, 1, 32, 11, 2, small::PADDING_F, 16},

        //kernel = 13
        {16, 1, 32, 13, 1, small::PADDING_F, 16},
        {16, 1, 31, 13, 2, small::PADDING_F, 16},
        {16, 1, 32, 13, 2, small::PADDING_F, 16},

        //kernel = 13
        {16, 2, 32, 13, 1, small::PADDING_F, 16},
        {16, 2, 31, 13, 2, small::PADDING_F, 16},
        {16, 2, 32, 13, 2, small::PADDING_F, 16},

        //kernel = 13
        {32, 2, 32, 13, 1, small::PADDING_F, 16},
        {32, 2, 31, 13, 2, small::PADDING_F, 16},
        {32, 2, 32, 13, 2, small::PADDING_F, 16},

        {32, 2, 32, 30, 1, small::PADDING_F, 16},
        {32, 2, 31, 30, 2, small::PADDING_F, 16},
        {32, 2, 32, 30, 2, small::PADDING_F, 16},


    };

    for (LayerParams const &p : params)
    {
#if defined(QUANTIZED)
        TEST_CHECK(true == run_conv1d_layer_config<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(true == run_conv1d_layer_config<small::FloatBuffer>(p));
#endif
    }
}

//****************************************************************************
TEST_LIST = {
    {"conv1d_layer_regression_data", test_conv1d_layer_regression_data},
    {NULL, NULL}
};
