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
bool run_conv2d_layer_config(LayerParams const &params)
{
    // Set up sizes
    size_t const C_i = params.C_i;
    size_t const W   = params.W;
    size_t const K   = params.k;
    size_t const S   = params.s;
    size_t const C_o = params.C_o;
    size_t const H = params.H;

    uint8_t left_pad = 0, right_pad = 0;
    uint8_t top_pad = 0, bottom_pad = 0;
    if (params.p == small::PADDING_F)
    {
        small::calc_padding(W, K, S, left_pad, right_pad);
        small::calc_padding(H, K, S, top_pad, bottom_pad);
    }

    printf("padding (l: %d,r: %d, t: %d, b: %d) %d \n", left_pad, right_pad, top_pad, bottom_pad, FLOAT_C_ob);
    // Input, Filter, Output sizes
    size_t input_size = C_i * W * H;
    size_t filter_size = C_i * K * K * C_o;
    size_t Wo = small::compute_output_dim(W, K, S, params.p);
    size_t Ho = small::compute_output_dim(H, K, S, params.p);
    size_t output_size = C_o * Wo * Ho;

    size_t padded_W = left_pad + W + right_pad;
    size_t padded_H = top_pad + H + bottom_pad;
    size_t padded_input_size = C_i * padded_W * padded_H ;

    // Allocate
    BufferT input_dc(input_size);
    BufferT filter_dc(filter_size);
    BufferT output_dc(output_size);
    BufferT padded_input_dc(padded_input_size);
    BufferT output_dc_answers(output_size);

    // Initialize inputs: all ones
    // small::init_arange<BufferT, FLOAT_C_ob>(input_dc, H, W, C_i);
    small::init(input_dc, input_size);
    small::init_zeros(padded_input_dc, padded_input_size);

    // Initialize filters: all ones
    small::init(filter_dc, filter_size);

        printf(" output dims: %d %d %d %d\n", C_o, Wo, Ho, output_size);

    //print unpadded input
    // std::cerr << "Unpadded input: " << std::endl;
    // for(size_t j = 0; j < H; j++)
    // {
    //     for(size_t i = 0; i < W; i++)
    //     {
    //         std::cerr << input_dc[j*W*FLOAT_C_ob + i*FLOAT_C_ob] << " ";
    //     }
    //     std::cerr << std::endl;
    // }
    // std::cerr << std::endl;


    //@todo: put this into buffers.hpp
    //create the padded input
    small::pad_2D<BufferT, FLOAT_C_ob>(input_dc, C_i, H, W,
                left_pad, right_pad,
                top_pad, bottom_pad,
                padded_input_dc);


    //iterate over tiles of padded input and print the first channel in the block

    // //print padded input
    // std::cerr << "Padded input: " << std::endl;
    // for(size_t j = 0; j < padded_H; j++)
    // {
    //     for(size_t i = 0; i < padded_W; i++)
    //     {
    //         std::cerr << padded_input_dc[j*padded_W*FLOAT_C_ob + i*FLOAT_C_ob] << " ";
    //     }
    //     std::cerr << std::endl;
    // }
    // std::cerr << std::endl;
    

    // Compute
    small::Conv2D(K, K, S,
                  top_pad, bottom_pad,
                  left_pad, right_pad,
                  C_o, C_i, H, W,
                  input_dc, filter_dc, output_dc);

    //check against padded input
    small::Conv2D(K, K,  S,
                  0, 0,
                  0, 0, 
                  C_o, C_i, padded_H, padded_W,
                  padded_input_dc, filter_dc, output_dc_answers);


    // Verify
    bool passing = true;
    for(size_t i = 0; i < output_size; i++)
    {
        if (std::abs((output_dc[i] - output_dc_answers[i])/(output_dc_answers[i])) > 1e-7)
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
void test_conv2d_layer_padding(void)
{
    std::vector<LayerParams> params =
    {
        // {C_i, H (not used for 1D), W, k, s, padding, C_o}

        {16, 5, 32, 3, 1, small::PADDING_F, 16},
        {16, 32, 13, 3, 2, small::PADDING_F, 16},
        {16, 32, 32, 3, 2, small::PADDING_F, 16},

        // //kernel = 5

        {16, 32, 32, 5, 1, small::PADDING_F, 16},
        {16, 32, 31, 5, 2, small::PADDING_F, 16},
        {16, 32, 32, 5, 2, small::PADDING_F, 16},
        {16, 31, 32, 5, 2, small::PADDING_F, 16},
        {16, 31, 31, 5, 2, small::PADDING_F, 16},



        // kernel = 7
        {16, 32, 32, 7, 1, small::PADDING_F, 16},
        {16, 32, 31, 7, 2, small::PADDING_F, 16},
        {16, 32, 32, 7, 2, small::PADDING_F, 16},
        {16, 31, 32, 7, 2, small::PADDING_F, 16},
        {16, 31, 31, 7, 2, small::PADDING_F, 16},


        //kernel = 9
        {16, 32, 32, 9, 1, small::PADDING_F, 16},
        {16, 32, 31, 9, 2, small::PADDING_F, 16},
        {16, 32, 32, 9, 2, small::PADDING_F, 16},
        {16, 31, 32, 9, 2, small::PADDING_F, 16},
        {16, 31, 31, 9, 2, small::PADDING_F, 16},


        //kernel = 11
        {16, 32, 32, 11, 1, small::PADDING_F, 16},
        {16, 32, 31, 11, 2, small::PADDING_F, 16},
        {16, 32, 32, 11, 2, small::PADDING_F, 16},
        {16, 31, 32, 11, 2, small::PADDING_F, 16},
        {16, 31, 31, 11, 2, small::PADDING_F, 16},


        //kernel = 13
        {16, 32, 32, 13, 1, small::PADDING_F, 16},
        {16, 32, 31, 13, 2, small::PADDING_F, 16},
        {16, 32, 32, 13, 2, small::PADDING_F, 16},
        {16, 31, 32, 13, 2, small::PADDING_F, 16},
        {16, 31, 31, 13, 2, small::PADDING_F, 16},


        //more output channels
        {16, 32, 32, 13, 1, small::PADDING_F, 256},
        {16, 32, 31, 13, 2, small::PADDING_F, 256},
        {16, 32, 32, 13, 2, small::PADDING_F, 256},
        {16, 31, 32, 13, 2, small::PADDING_F, 256},
        {16, 31, 31, 13, 2, small::PADDING_F, 256},


        // more input channels
        {32, 26, 26, 13, 1, small::PADDING_F, 32},
        {32, 26, 25, 13, 2, small::PADDING_F, 32},
        {32, 26, 26, 13, 2, small::PADDING_F, 32},
        {32, 25, 26, 13, 2, small::PADDING_F, 32},
        {32, 25, 25, 13, 2, small::PADDING_F, 32},



    };

    for (LayerParams const &p : params)
    {
#if defined(QUANTIZED)
        TEST_CHECK(true == run_conv2d_layer_config<small::QUInt8Buffer>(p));
#else
        TEST_CHECK(true == run_conv2d_layer_config<small::FloatBuffer>(p));
#endif
    }
}

//****************************************************************************
TEST_LIST = {
    {"conv1d_layer_padding", test_conv2d_layer_padding},
    {NULL, NULL}
};
