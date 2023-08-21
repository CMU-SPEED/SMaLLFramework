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

//#define DAG_DEBUG
//#define BUFFER_DEBUG
//#define DEBUG_LAYERS
//#define SUMMARY 1

#include <acutest.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>

#include <small.h>
#include <small/models/Resnet8Tiny.hpp>
#include <small/models/Resnet8TinyDAG.hpp>
#include <small/utils/Timer.hpp>

#include "test_utils.hpp"

#define RUNS 20

std::string const data_dir("../test/regression_data");

//****************************************************************************
// Function call implementation of resnet8
//****************************************************************************

#define REDUCTION_C(layer_num) layer_params[layer_num][0]
#define GROUP_C(layer_num) layer_params[layer_num][1]
#define GROUPS(layer_num) layer_params[layer_num][2]
#define REDUCTION_HW(layer_num) layer_params[layer_num][3]
#define STRIDE(layer_num) layer_params[layer_num][4]
#define TYPE(layer_num) layer_params[layer_num][9]

#define SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad) layer_params[layer_num][5] = t_pad, layer_params[layer_num][6] = b_pad, layer_params[layer_num][7] = l_pad, layer_params[layer_num][8] = r_pad;
#define PADDING(layer_num) layer_params[layer_num][5], layer_params[layer_num][6], layer_params[layer_num][7], layer_params[layer_num][8]

#define PADDING_TORCH(layer_num) layer_params[layer_num][7], layer_params[layer_num][8], layer_params[layer_num][5], layer_params[layer_num][6]

#define I_WIDTH(layer_num) intermediate_dims[layer_num][0]
#define I_HEIGHT(layer_num) intermediate_dims[layer_num][1]

#define O_HEIGHT(layer_num) (((I_HEIGHT(layer_num - 1) + layer_params[layer_num - 1][5] + layer_params[layer_num - 1][6]) - REDUCTION_HW(layer_num - 1)) / STRIDE(layer_num - 1) + 1)
#define O_WIDTH(layer_num) (((I_WIDTH(layer_num - 1) + layer_params[layer_num - 1][7] + layer_params[layer_num - 1][8]) - REDUCTION_HW(layer_num - 1)) / STRIDE(layer_num - 1) + 1)

#define OUTPUT_DIMS(layer_num)                  \
    {                                           \
        O_HEIGHT(layer_num), O_WIDTH(layer_num) \
    }

#define INPUT_NUMEL(layer_num) \
    (O_HEIGHT(layer_num) * O_WIDTH(layer_num) * GROUP_C(layer_num - 1) * GROUPS(layer_num - 1))

//****************************************************************************
template <class BufferT>
void build_baseline_resnet8(uint32_t C_i,         // 128, 1, 1, 16(8), ...
                            uint32_t N, // I_h
                            uint32_t M, // I_w
                            uint32_t num_classes,
                            size_t&  layer_num_total,
                            uint16_t layer_params[30][10],
                            uint32_t intermediate_dims[30][2],
                            std::vector<BufferT *>& filter_buf_ptrs,
                            size_t& max_numel_inter_0,
                            size_t& max_numel_inter_1,
                            size_t& num_outputs)
{
    uint8_t t_pad, b_pad, r_pad, l_pad;

    // Set up model parameters
    int layer_num = 0;
    max_numel_inter_0 = 0;
    max_numel_inter_1 = 0;

    intermediate_dims[layer_num][0] = M;  // width
    intermediate_dims[layer_num][1] = N;  // height

    // conv
    REDUCTION_C(layer_num) = C_i;
    GROUP_C(layer_num) = 16;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 3;
    STRIDE(layer_num) = 1;
    small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                        STRIDE(layer_num), t_pad, b_pad);
    small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                        STRIDE(layer_num), l_pad, r_pad);
    SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

    layer_num++; // 1
    intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
    intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);
    auto inter_dim = INPUT_NUMEL(layer_num);
    max_numel_inter_0 =
        (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

    // common set up for model architecture
    auto resnet_blocks = 3;
    int layer_strides[] = {1, 2, 2};
    // dwise 1
    for (int ds_layer = 0; ds_layer < resnet_blocks; ds_layer++)
    {
        int channel_multiplier = (ds_layer > 0) ? 2 : 1;

        uint32_t in_channels = GROUP_C(layer_num - 1); // output channels from the previous block

        REDUCTION_C(layer_num) = in_channels; // input channels
        GROUP_C(layer_num) = in_channels * channel_multiplier;
        GROUPS(layer_num) = 1;                       // output channels
        REDUCTION_HW(layer_num) = 3;                 // kernel size
        STRIDE(layer_num) = layer_strides[ds_layer]; // stride
        small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                            STRIDE(layer_num), t_pad, b_pad);
        small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                            STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

        layer_num++; // 2,4,7
        intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
        intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);

        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_1 =
            (inter_dim > max_numel_inter_1) ? inter_dim : max_numel_inter_1;

        REDUCTION_C(layer_num) = GROUP_C(layer_num - 1);
        GROUP_C(layer_num) = GROUP_C(layer_num - 1);
        GROUPS(layer_num) = 1;
        REDUCTION_HW(layer_num) = 3;
        STRIDE(layer_num) = 1;
        small::calc_padding(I_HEIGHT(layer_num), REDUCTION_HW(layer_num),
                            STRIDE(layer_num), t_pad, b_pad);
        small::calc_padding(I_WIDTH(layer_num),  REDUCTION_HW(layer_num),
                            STRIDE(layer_num), l_pad, r_pad);
        SET_PADDING(layer_num, t_pad, b_pad, l_pad, r_pad);

        layer_num++; // 3,5,8
        inter_dim = INPUT_NUMEL(layer_num);
        max_numel_inter_0 =
            (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;

        if (channel_multiplier != 1)
        {
            intermediate_dims[layer_num][0] = O_WIDTH(layer_num - 2);
            intermediate_dims[layer_num][1] = O_HEIGHT(layer_num - 2);
            REDUCTION_C(layer_num) = in_channels; // input channels
            GROUP_C(layer_num) = in_channels * channel_multiplier;
            GROUPS(layer_num) = 1;       // output channels
            REDUCTION_HW(layer_num) = 1; // kernel size
            STRIDE(layer_num) = 2;       // stride
            SET_PADDING(layer_num, 0, 0, 0, 0);
            layer_num++; // 6,9
            inter_dim = INPUT_NUMEL(layer_num);
            max_numel_inter_0 =
                (inter_dim > max_numel_inter_0) ? inter_dim : max_numel_inter_0;
        }
        intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
        intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);
    }

    // pooling dims
    REDUCTION_C(layer_num) = 1;
    GROUP_C(layer_num) = 1;
    GROUPS(layer_num) = GROUP_C(layer_num - 1);
    REDUCTION_HW(layer_num) = I_HEIGHT(layer_num);
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0);

    layer_num++;
    intermediate_dims[layer_num][0] = O_WIDTH(layer_num);
    intermediate_dims[layer_num][1] = O_HEIGHT(layer_num);
    REDUCTION_C(layer_num) = GROUPS(layer_num - 1);
    GROUP_C(layer_num) = num_classes;
    GROUPS(layer_num) = 1;
    REDUCTION_HW(layer_num) = 1;
    STRIDE(layer_num) = 1;
    SET_PADDING(layer_num, 0, 0, 0, 0);
    layer_num++;

    layer_num_total = layer_num;
    size_t num_filters = layer_num_total - 1;
    num_outputs = num_classes;

#if SUMMARY == 1
    printf("Layer num total: %ld", layer_num_total);
    for (auto i = 0UL; i < layer_num_total; i++)
    {
        printf("layer %ld: ", i);
        printf(" input_dims: %d %d ", I_HEIGHT(i), I_WIDTH(i));
        for (auto j = 0; j < 10; j++)
        {
            printf("%d, ", layer_params[i][j]);
        }
        printf(", intermediate dims: %d %d\n",
               intermediate_dims[i][0], intermediate_dims[i][1]);
    }
#endif

    for (size_t l = 0; l < num_filters - 1; l++)
    {
        uint32_t filter_dimensions =
            REDUCTION_HW(l) * REDUCTION_HW(l) * REDUCTION_C(l) *
            GROUP_C(l) * GROUPS(l);

        BufferT *filter_buf_ptr =
            small::alloc_buffer<BufferT>(filter_dimensions);
        init(*filter_buf_ptr, filter_dimensions);
        filter_buf_ptrs.push_back(filter_buf_ptr);
    }

    uint32_t filter_dimensions =
        GROUP_C(layer_num_total - 1) * REDUCTION_C(layer_num_total - 1);
    BufferT *filter_fc_dc_ptr =
        small::alloc_buffer<BufferT>(filter_dimensions);
    init(*filter_fc_dc_ptr, filter_dimensions);
    filter_buf_ptrs.push_back(filter_fc_dc_ptr);
    /// @todo assert(filter_buf_ptrs.size() == num_filters)

}

//****************************************************************************
//****************************************************************************

//****************************************************************************
// The output of the block is stored in O
//
template <class BufferT>
inline void resnet_block(
    uint32_t in_dims[2], uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t output_channels,
    uint8_t t_pad_0,
    uint8_t b_pad_0,
    uint8_t l_pad_0,
    uint8_t r_pad_0,
    uint8_t t_pad_1,
    uint8_t b_pad_1,
    uint8_t l_pad_1,
    uint8_t r_pad_1,
    BufferT const &I,
    BufferT const &F_conv0,
    BufferT const &F_conv1,
    BufferT const &F_conv_1x1,
    BufferT       &O_intermediate,
    BufferT       &O)
{
    // printf("before: %d, %.2f %.2f %.2f %.2f\n", 0, I[0], I[1], I[2], I[3]);

    small::Conv2D(kernel_size, kernel_size, stride,
                  t_pad_0, b_pad_0, l_pad_0, r_pad_0,
                  output_channels, input_channels,
                  in_dims[0], in_dims[1],
                  I, F_conv0, O_intermediate);

    uint32_t o_h = small::output_dim(in_dims[0] + t_pad_0 + b_pad_0,
                                     stride, kernel_size);
    uint32_t o_w = small::output_dim(in_dims[1] + l_pad_0 + r_pad_0,
                                     stride, kernel_size);

    small::ReLUActivation(output_channels,
                          o_h, o_w,
                          O_intermediate, O_intermediate);

    small::Conv2D(1, 1, stride,
                  0, 0, 0, 0,
                  output_channels, input_channels,
                  in_dims[0], in_dims[1],
                  I, F_conv_1x1, O);

    small::PartialConv2D(kernel_size, kernel_size, 1,
                         t_pad_1, b_pad_1, l_pad_1, r_pad_1,
                         output_channels, output_channels,
                         o_h, o_w,
                         O_intermediate, F_conv1, O);
    small::ReLUActivation(output_channels, o_h, o_w, O, O);
}

//****************************************************************************
// The output of the block is stored in O
//
template <class BufferT>
inline void resnet_block(
    uint32_t in_dims[2], uint32_t input_channels, // Input dimensions
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t output_channels,
    uint8_t t_pad_0,
    uint8_t b_pad_0,
    uint8_t l_pad_0,
    uint8_t r_pad_0,
    uint8_t t_pad_1,
    uint8_t b_pad_1,
    uint8_t l_pad_1,
    uint8_t r_pad_1,
    BufferT const &I,
    BufferT const &F_conv0,
    BufferT const &F_conv1,
    BufferT       &O_intermediate,
    BufferT       &O)
{
    small::Conv2D(kernel_size, kernel_size, stride,
                  t_pad_0, b_pad_0, l_pad_0, r_pad_0,
                  output_channels, input_channels,
                  in_dims[0], in_dims[1],
                  I, F_conv0, O_intermediate);

    uint32_t o_h = small::output_dim(in_dims[0] + t_pad_0 + b_pad_0,
                                     stride, kernel_size);
    uint32_t o_w = small::output_dim(in_dims[1] + l_pad_0 + r_pad_0,
                                     stride, kernel_size);

    small::ReLUActivation(output_channels,
                          o_h, o_w,
                          O_intermediate, O_intermediate);

    /// @todo Should this really be Partial Conv2D if no 1x1?
    small::PartialConv2D(kernel_size, kernel_size, 1,
                         t_pad_1, b_pad_1, l_pad_1, r_pad_1,
                         output_channels, output_channels,
                         o_h, o_w,
                         O_intermediate, F_conv1, O);
    small::ReLUActivation(output_channels, o_h, o_w, O, O);
}

//****************************************************************************
template <class BufferT>
BufferT &
model_inference(uint32_t layer_num_total,
                uint16_t layer_params[30][10],
                uint32_t intermediate_dims[30][2],
                std::vector<BufferT *> const &filter_buf_ptrs,
                BufferT const &input_dc,
                BufferT       &inter_0_dc,
                BufferT       &inter_1_dc,
                BufferT       &inter_2_dc)
{
    auto layer_num = 0;
    small::Conv2D(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                  STRIDE(layer_num), PADDING(layer_num),
                  GROUP_C(layer_num), REDUCTION_C(layer_num),
                  I_HEIGHT(layer_num), I_WIDTH(layer_num),
                  input_dc,
                  *filter_buf_ptrs[layer_num],
                  inter_0_dc);

    layer_num++;
    small::ReLUActivation(GROUP_C(0),
                          I_HEIGHT(layer_num), I_WIDTH(layer_num),
                          inter_0_dc,
                          inter_0_dc);

    resnet_block(intermediate_dims[layer_num], REDUCTION_C(layer_num), // Input dimensions
                 REDUCTION_HW(layer_num),
                 STRIDE(layer_num),
                 GROUP_C(layer_num),
                 PADDING(layer_num),
                 PADDING(layer_num + 1),
                 inter_0_dc,
                 *filter_buf_ptrs[layer_num],
                 *filter_buf_ptrs[layer_num + 1],
                 inter_1_dc,
                 inter_0_dc);

    layer_num += 2;
    auto resnet_blocks = 3;
    auto num_filters = layer_num_total - 1;
    for (int ds_layer = 1; ds_layer < resnet_blocks; ds_layer++)
    {
        resnet_block(intermediate_dims[layer_num], REDUCTION_C(layer_num),
                     REDUCTION_HW(layer_num),
                     STRIDE(layer_num),
                     GROUP_C(layer_num),
                     PADDING(layer_num),
                     PADDING(layer_num + 1),
                     inter_0_dc,
                     *filter_buf_ptrs[layer_num],
                     *filter_buf_ptrs[layer_num + 1],
                     *filter_buf_ptrs[layer_num + 2],
                     inter_1_dc,
                     inter_2_dc);

        layer_num += 3;
        // Since channels were scaled, switch the pointers between inter_2 and inter_0
        inter_0_dc.swap(inter_2_dc);
    }

    small::MaxPool2D(REDUCTION_HW(layer_num), REDUCTION_HW(layer_num),
                     STRIDE(layer_num),
                     PADDING(layer_num),
                     GROUPS(layer_num),
                     I_HEIGHT(layer_num), I_WIDTH(layer_num),
                     inter_0_dc,
                     inter_1_dc);
    small::Conv2D(1, 1, 1,
                  0, 0, 0, 0,
                  GROUP_C(layer_num_total - 1), REDUCTION_C(layer_num_total - 1),
                  1, 1,
                  inter_1_dc,
                  *filter_buf_ptrs[num_filters - 1],
                  inter_0_dc);

    return inter_0_dc;
}

//****************************************************************************
//****************************************************************************
void test_resnet8(void)
{
#if defined QUANTIZED
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // input parameters
    uint32_t C_i = 3;
    uint32_t N = 32;
    uint32_t M = 32;
    uint32_t num_classes = 16;

    // ================================================

    //************************************************************************
    // Baseline (function call) model
    //************************************************************************

    // "model" params
    size_t   layer_num_total = 0;
    uint16_t layer_params[30][10] = {1};
    uint32_t intermediate_dims[30][2];
    size_t   max_numel_inter_0 = 0;
    size_t   max_numel_inter_1 = 0;
    std::vector<BufferT*> filter_buf_ptrs;
    size_t   num_outputs = 0;

    build_baseline_resnet8(C_i, N, M, num_classes,
                           layer_num_total,
                           layer_params,
                           intermediate_dims,
                           filter_buf_ptrs,
                           max_numel_inter_0,
                           max_numel_inter_1,
                           num_outputs);

    // ================================================

#if defined QUANTIZED
    // std::cerr << "Intermediate buffer sizes: "
    //           << max_numel_inter_0 << ", " << max_numel_inter_1
    //           << std::endl;
    BufferT inter_0_dc(max_numel_inter_0 + C_ob*16*16*32); // HACK
    BufferT inter_1_dc(max_numel_inter_1 + C_ob*16*16*32); // HACK
    BufferT inter_2_dc((max_numel_inter_0 / 2) + C_ob*16*16*3);
#else
    BufferT inter_0_dc(max_numel_inter_0);
    BufferT inter_1_dc(max_numel_inter_1);
    BufferT inter_2_dc(max_numel_inter_0);
#endif

    // Create input tensor
    uint32_t input_dimensions = C_i * N * M;
    BufferT input_dc(input_dimensions);
    init(input_dc, input_dimensions);

    //======================================================
    small::Timer my_timer;

    std::cerr << "\nWarm up run (ORIG)" << std::endl;
    my_timer.start();
    auto &output_dc =
        model_inference(layer_num_total, layer_params,
                        intermediate_dims,
                        filter_buf_ptrs,
                        input_dc,
                        inter_0_dc,
                        inter_1_dc,
                        inter_2_dc);
    my_timer.stop();

    // copy the output for comparison in subsequent runs.
    BufferT output_answers(output_dc);
    printf("\nElapsed time: %lf ns.\n", my_timer.elapsed());

    //======================================================
    // Timing runs
    //======================================================
    std::vector<double> small_timing;

    for (int r = 0; r < RUNS; r++)
    {
        std::cout << "Baseline run: " << r;
        my_timer.start();

        auto &output_dc =
            model_inference(layer_num_total, layer_params,
                            intermediate_dims,
                            filter_buf_ptrs,
                            input_dc,
                            inter_0_dc,
                            inter_1_dc,
                            inter_2_dc);

        my_timer.stop();
        auto diff = my_timer.elapsed();
        small_timing.push_back(diff);
        std::cout << ": " << diff << " ns.\n";

        // Test that the answer stays the same through multiple invocations
        bool passed = true;
        for (size_t ix = 0; ix < num_outputs; ++ix)
        {
            bool same_value = (output_answers[ix] == output_dc[ix]);

#if SUMMARY == 1
            std::cout << (same_value ? "pass: " : "FAIL: ")
                      << "baseline (first run), baseline output " << ix
                      << ": " << (float)output_answers[ix]
                      << " ?= " << (float)output_dc[ix]
                      << std::endl;
#endif
            passed &= same_value;
        }
        TEST_CHECK(passed);
    }

    //************************************************************************
    // Model class
    //************************************************************************
    std::vector<double> layer_timing;
    {
        small::shape_type input_shape({1UL, C_i, N, M});

        small::Resnet8Tiny<BufferT> model(input_shape,
                                          filter_buf_ptrs, true);

        small::Tensor<BufferT> input_tensor(input_shape, input_dc);

        //***********
        std::cerr << "\nWarm up run (LAYERS)" << std::endl;
        auto output_tensors = model.inference(&input_tensor);
        BufferT output_buffer = output_tensors[0]->buffer();
        //***********

        TEST_CHECK(1 == output_tensors.size());
        std::cerr << "Output sizes: " << num_outputs << " ?= "
                  <<  output_tensors[0]->size() << std::endl;
        TEST_CHECK(num_outputs == output_tensors[0]->size());

        // Compare outputs
        bool passed = true;
        std::cout << "\nCHECK RESULTS: Num output elements: "
                  << num_outputs << std::endl;

        for (size_t ix = 0; ix < num_outputs; ++ix)
        {
            bool same_value = (output_answers[ix] == output_tensors[0]->buffer()[ix]);

#if SUMMARY == 1
            std::cout << (same_value ? "pass: " : "FAIL: ")
                      << "baseline, Model output " << ix
                      << ": " << (float)output_answers[ix]
                      << " ?= " << (float)output_tensors[0]->buffer()[ix]
                      << std::endl;
#endif
            passed &= same_value;
        }
        TEST_CHECK(passed);

        //======================================================
        // Timing runs
        //======================================================

        for (int r = 0; r < RUNS; r++)
        {
            std::cout << "Layers run: " << r;
            my_timer.start();

            //***********
            auto output_tensors = model.inference(&input_tensor);
            //***********

            my_timer.stop();
            auto diff = my_timer.elapsed();
            layer_timing.push_back(diff);
            std::cout << ": " << diff << " ns.\n";

            // Test that the answer stays the same through multiple invocations
            bool passed = true;
            for (size_t ix = 0; ix < num_outputs; ++ix)
            {
                bool same_value =
                    (output_answers[ix] == output_tensors[0]->buffer()[ix]);

#if SUMMARY == 1
                std::cout << (same_value ? "pass: " : "FAIL: ")
                          << "baseline (first run), Model output " << ix
                          << ": " << (float)output_answers[ix]
                          << " ?= " << (float)output_tensors[0]->buffer()[ix]
                          << std::endl;
#endif
                passed &= same_value;
            }
            TEST_CHECK(passed);
        }
    }

    //************************************************************************
    // DAGModel class
    //************************************************************************
    std::vector<double> dag_timing;
    {
        small::shape_type input_shape({1UL, C_i, N, M});

        small::Resnet8TinyDAG<BufferT> model(input_shape,
                                             filter_buf_ptrs, true);

        small::Tensor<BufferT> input_tensor(input_shape, input_dc);

        //***********
        std::cerr << "\nWarm up run (DAG)" << std::endl;
        auto output_tensors = model.inference(&input_tensor);
        BufferT output_buffer = output_tensors[0]->buffer();
        //***********

        TEST_CHECK(1 == output_tensors.size());
        std::cerr << "Output sizes: " << num_outputs << " ?= "
                  <<  output_tensors[0]->size() << std::endl;
        TEST_CHECK(num_outputs == output_tensors[0]->size());

        // Compare outputs
        bool passed = true;
        std::cout << "\nCHECK RESULTS: Num output elements: "
                  << num_outputs << std::endl;

        for (size_t ix = 0; ix < num_outputs; ++ix)
        {
            bool same_value = almost_equal(output_answers[ix],
                                           output_tensors[0]->buffer()[ix]);

#if SUMMARY == 1
            std::cout << (same_value ? "pass: " : "FAIL: ")
                      << "baseline, Model output " << ix
                      << ": " << (float)output_answers[ix]
                      << " ?= " << (float)output_tensors[0]->buffer()[ix]
                      << std::endl;
#endif
            passed &= same_value;
        }
        TEST_CHECK(passed);

        //======================================================
        // Timing runs
        //======================================================

        for (int r = 0; r < RUNS; r++)
        {
            std::cout << "DAG run: " << r;
            my_timer.start();

            //***********
            auto output_tensors = model.inference(&input_tensor);
            //***********

            my_timer.stop();
            auto diff = my_timer.elapsed();
            dag_timing.push_back(diff);
            std::cout << ": " << diff << " ns.\n";

            // Test that the answer stays the same through multiple invocations
            bool passed = true;
            for (size_t ix = 0; ix < num_outputs; ++ix)
            {
                bool same_value =
                    almost_equal(output_answers[ix],
                                 output_tensors[0]->buffer()[ix]);

#if SUMMARY == 1
                std::cout << (same_value ? "pass: " : "FAIL: ")
                          << "baseline (first run), Model output " << ix
                          << ": " << (float)output_answers[ix]
                          << " ?= " << (float)output_tensors[0]->buffer()[ix]
                          << std::endl;
#endif
                passed &= same_value;
            }
            TEST_CHECK(passed);
        }
    }

    int num_th = 1;
#if PARALLEL == 1
    char const *env_nt(std::getenv("OMP_NUM_THREADS"));
    if (nullptr != env_nt)
    {
        num_th = atoi(std::getenv("OMP_NUM_THREADS"));
    }
#endif
    std::cout << "\nSUMMARY STATS:" << std::endl;
    std::cout << "Num Threads: " << num_th << std::endl;
    print_stats(small_timing, "SMaLL:resnet8 Baseline");
    print_stats(layer_timing, "SMaLL:resnet8 Layers  ");
    print_stats(dag_timing,   "SMaLL:resnet8 DAGModel");

    // clean up
    for (auto filter : filter_buf_ptrs)
    {
        small::free_buffer(filter);
    }
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"resnet8_data_performance", test_resnet8},
    {NULL, NULL}
};
