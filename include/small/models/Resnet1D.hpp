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

#pragma once

#include <vector>
#include <small.h>
#include <small/state_dict_utils.hpp>
#include <small/Model.hpp>
#include <small/Conv1DLayer.hpp>
#include <small/AveragePool1DLayer.hpp>
#include <small/LogSoftMaxLayer.hpp>
#include <small/DenseLayer.hpp>
namespace small
{

//****************************************************************************
template <typename BufferT>
class Resnet1D : public Model<BufferT>
{
public:
    Resnet1D() = delete;

    // Assume one input layer with a single shape for now
    Resnet1D(shape_type  const &input_shape,
             uint32_t           num_stacks,  // 3
             uint32_t           num_blocks,  // 5
             uint32_t           num_classes, // 2
             std::string const &state_dict_filepath,
             bool               filters_are_packed = false)
        : Model<BufferT>(input_shape),
          m_num_stacks(num_stacks),
          m_num_blocks(num_blocks),
          m_num_classes(num_classes),
          m_buffer_0(nullptr),
          m_buffer_1(nullptr),
          m_buffer_2(nullptr)
    {
        create_model_and_buffers(input_shape,
                                 num_stacks,
                                 num_blocks,
                                 num_classes,
                                 state_dict_filepath,
                                 filters_are_packed);
    }

    virtual ~Resnet1D()
    {
        delete m_buffer_0;
        delete m_buffer_1;
        delete m_buffer_2;
    }

    size_t get_output_size() const { return m_num_classes; }

    virtual std::vector<Tensor<BufferT>*> inference(
        Tensor<BufferT> const *input);

private:
    struct LayerParams
    {
        uint32_t C_i;
        uint32_t H; // image_height;
        uint32_t W; // image_width;
        uint32_t k; // kernel_size;
        uint16_t s; // stride;
        small::PaddingEnum p; // PADDING_V or _F
        uint32_t C_o;
    };

    void construct_resnet_input_layers(
        std::string const &state_dict_content);

    void construct_resnet_stack_layers(
        const std::string &state_dict_content,
        uint32_t stack_idx,
        uint32_t num_blocks);

    void construct_resnet_output_layers(
        std::string const &state_dict_content,
        uint32_t           num_classes);

    void construct_resnet(
        std::string const &state_dict_content,
        uint32_t           num_stacks,
        uint32_t           num_blocks,
        uint32_t           num_classes);

    void create_model_and_buffers(
        shape_type  const &input_shape,
        uint32_t           num_stacks,
        uint32_t           num_blocks,
        uint32_t           num_classes,
        std::string const &state_dict_filepath,
        bool               weights_are_packed);

private:
    uint32_t const m_num_stacks;
    uint32_t const m_num_blocks;
    uint32_t const m_num_classes;

    Tensor<BufferT> *m_buffer_0;
    Tensor<BufferT> *m_buffer_1;
    Tensor<BufferT> *m_buffer_2;
};

//****************************************************************************
template<typename BufferT>
void Resnet1D<BufferT>::construct_resnet_input_layers(
    std::string const &state_dict_content)
{
    // C_i,H,W,k,s,p,C_o
    LayerParams params = {2, 1, 1024, 3, 1, small::PADDING_F, 16};
    small::shape_type input_shape{1UL, params.C_i, params.H, params.W};

    BufferT conv_weight_buf{extract_param<BufferT>(state_dict_content,
                                                   "convIn.weight")};
    BufferT bn_weight_buf{extract_param<BufferT>(state_dict_content,
                                                 "bnIn.weight")};
    BufferT bn_bias_buf{extract_param<BufferT>(state_dict_content,
                                               "bnIn.bias")};
    BufferT bn_mean_buf{extract_param<BufferT>(state_dict_content,
                                               "bnIn.running_mean")};
    BufferT bn_var_buf{extract_param<BufferT>(state_dict_content,
                                              "bnIn.running_var")};

    small::Conv1DLayer<BufferT> *conv_in =
        new small::Conv1DLayer<BufferT>(input_shape,
                                        params.k, params.s, params.p,
                                        params.C_o,
                                        conv_weight_buf,
                                        bn_weight_buf,
                                        bn_bias_buf,
                                        bn_mean_buf,
                                        bn_var_buf,
                                        /*bn_eps = */1.e-5,
                                        /*buffers_are_packed = */false,
                                        small::ActivationType::RELU);

    this->m_layers.push_back(conv_in);
}

//****************************************************************************
template<typename BufferT>
void Resnet1D<BufferT>::construct_resnet_stack_layers(
    std::string const &state_dict_content,
    uint32_t           stack_idx,
    uint32_t           num_blocks)
{
    std::cout << "Constructing stack " << stack_idx << "...\n";

    for (uint32_t block_idx = 0; block_idx < num_blocks; block_idx++)
    {
        LayerParams params1 =
            {
                (stack_idx == 1)
                ? 16
                : 16*(uint32_t)std::pow(2, stack_idx-(block_idx == 0
                                                      ? 2
                                                      : 1)),  // C_i
                1, // H
                (stack_idx == 1)
                ? 1024
                : 1024 / (uint32_t)std::pow(2, stack_idx - (block_idx == 0
                                                            ? 2
                                                            : 1)), // W
                3, // k
                stack_idx > 1 && block_idx == 0
                ? (uint16_t)2
                : (uint16_t)1, // s

                small::PADDING_F, // p
                16*(uint32_t)std::pow(2,(stack_idx-1)) // C_o
            };
        small::shape_type input_shape1{
            1UL, params1.C_i, params1.H, params1.W};

        std::string param_prefix =
            "stack" + std::to_string(stack_idx) +
            ((stack_idx > 1)
             ? ((block_idx>0) ? "b." + std::to_string(block_idx-1) : "a")
             : "." + std::to_string(block_idx));

        BufferT conv_weight_buf1{extract_param<BufferT>(
            state_dict_content,
            param_prefix + ".conv1.weight")};
        BufferT bn_weight_buf1{extract_param<BufferT>(
            state_dict_content,
            param_prefix + ".bn1.weight")};
        BufferT bn_bias_buf1{extract_param<BufferT>(
            state_dict_content,
            param_prefix + ".bn1.bias")};
        BufferT bn_mean_buf1{extract_param<BufferT>(
            state_dict_content,
            param_prefix + ".bn1.running_mean")};
        BufferT bn_var_buf1{extract_param<BufferT>(
            state_dict_content,
            param_prefix + ".bn1.running_var")};

        // #1, #
        small::Conv1DLayer<BufferT> *conv1 =
            new small::Conv1DLayer<BufferT>(
                input_shape1,
                params1.k, params1.s, params1.p, params1.C_o,
                conv_weight_buf1,
                bn_weight_buf1, bn_bias_buf1, bn_mean_buf1, bn_var_buf1,
                /*bn_eps = */1.e-5,
                /*buffers_are_packed = */false,
                small::ActivationType::RELU);
        this->m_layers.push_back(conv1);

        if (stack_idx > 1 && block_idx == 0)
        {
            LayerParams avgpool_params = {  params1.C_o,      // C_i
                                            1,                // H
                                            params1.W/2,      // W
                                            1,                // k
                                            2,                // s
                                            small::PADDING_V, // p
                                            params1.C_o       // C_o
                                        };
            small::shape_type avgpool_input_shape{
                1UL, avgpool_params.C_i, avgpool_params.H, avgpool_params.W};
            small::AveragePool1DLayer<BufferT> *avgpool =
                new small::AveragePool1DLayer<BufferT>(
                    avgpool_input_shape,
                    avgpool_params.k, avgpool_params.s, avgpool_params.p);
            this->m_layers.push_back(avgpool);
        }

        LayerParams params2 =
            {
                16*(uint32_t)std::pow(2, stack_idx-1),  // C_i
                1,                                      // H
                (stack_idx ==  1 || block_idx > 0) ? params1.W : params1.W/2, // W
                3,                                      // k
                (uint16_t)1,                            // s
                small::PADDING_F,                       // p
                16*(uint32_t)std::pow(2, stack_idx-1)   // C_o
            };
        small::shape_type input_shape2{1UL, params2.C_i, params2.H, params2.W};

        BufferT conv_weight_buf2{extract_param<BufferT>(
                state_dict_content, param_prefix+".conv2.weight")};
        BufferT bn_weight_buf2{extract_param<BufferT>(
                state_dict_content, param_prefix+".bn2.weight")};
        BufferT bn_bias_buf2{extract_param<BufferT>(
                state_dict_content, param_prefix+".bn2.bias")};
        BufferT bn_mean_buf2{extract_param<BufferT>(
                state_dict_content, param_prefix+".bn2.running_mean")};
        BufferT bn_var_buf2{extract_param<BufferT>(
                state_dict_content, param_prefix+".bn2.running_var")};

        small::Conv1DLayer<BufferT> *conv2 =
            new small::Conv1DLayer<BufferT>(
                input_shape2,
                params2.k, params2.s, params2.p, params2.C_o,
                conv_weight_buf2,
                bn_weight_buf2, bn_bias_buf2, bn_mean_buf2, bn_var_buf2,
                /*bn_eps = */1.e-5,
                /*buffers_are_packed = */false,
                small::ActivationType::RELU);
        this->m_layers.push_back(conv2);
    }
}

//****************************************************************************
template<typename BufferT>
void Resnet1D<BufferT>::construct_resnet_output_layers(
    std::string const &state_dict_content,
    uint32_t           num_classes)
{
    uint32_t num_classes_padded{num_classes};
    if (num_classes % BufferT::C_ib != 0)
    {
        num_classes_padded += (BufferT::C_ib - (num_classes % BufferT::C_ib));
    }
    std::cerr << "Constructing output layers: num_classes(" << num_classes
              << "), num_classes_padded(" << num_classes_padded << ")\n";

    BufferT fc_o_weights{
        extract_param<BufferT>(state_dict_content, "fcOut.weight")};
    BufferT fc_o_biases{
        extract_param<BufferT>(state_dict_content, "fcOut.bias")};

    // C_i, H, W, k, s, p, C_o
    LayerParams avgpool_params = {64, 1, 256, 256, 1, small::PADDING_V, 64};
    small::shape_type avgpool_input_shape{
        1UL, avgpool_params.C_i, avgpool_params.H, avgpool_params.W};
    small::AveragePool1DLayer<BufferT> *avgpool =
        new small::AveragePool1DLayer<BufferT>(
            avgpool_input_shape,
            avgpool_params.k, avgpool_params.s, avgpool_params.p);
    this->m_layers.push_back(avgpool);

    // C_i, H, W, k, s, p, C_o
    LayerParams fc_params = {64, 1, 1, 1, 1, small::PADDING_V, num_classes}; //_padded};

    small::shape_type fc_input_shape{1UL, fc_params.C_i, fc_params.H, fc_params.W};
    TEST_ASSERT(fc_input_shape == avgpool->output_shape());

    small::DenseLayer<BufferT> *fc =
        new small::DenseLayer<BufferT>(
            avgpool->output_shape(), //fc_input_shape,
            num_classes,               // fc_params.C_o,
            fc_o_weights, fc_o_biases,
            false, small::ActivationType::NONE);
    this->m_layers.push_back(fc);

    // C_i, H, W, k, s, p, C_o
    LayerParams logsoftmax_params =
        {num_classes_padded, 1, 1, 1, 1, small::PADDING_V, num_classes}; //_padded};
    small::shape_type logsoftmax_input_shape{
        1UL, logsoftmax_params.C_i, logsoftmax_params.H, logsoftmax_params.W};

    TEST_ASSERT(logsoftmax_input_shape == fc->output_shape());
    TEST_ASSERT(num_classes == fc->logical_output_channels());

    std::cerr << "LogSoftMax shape params: " << fc->output_shape() << ","
              << fc->logical_output_channels() << std::endl;
    small::LogSoftMaxLayer<BufferT> *logsoftmax =
        new small::LogSoftMaxLayer<BufferT>(fc->output_shape(),
                                            fc->logical_output_channels());
    this->m_layers.push_back(logsoftmax);
}

//****************************************************************************
template<typename BufferT>
void Resnet1D<BufferT>::construct_resnet(
    std::string const &state_dict_content,
    uint32_t           num_stacks,
    uint32_t           num_blocks,
    uint32_t           num_classes)
{
    construct_resnet_input_layers(state_dict_content);

    for (uint32_t i = 0; i < num_stacks; i++)
    {
        construct_resnet_stack_layers(state_dict_content,
                                      i+1,
                                      num_blocks);
    }

    construct_resnet_output_layers(state_dict_content,
                                   num_classes);

    std::cout << "Done constructing model.\n";
}

//****************************************************************************
template <typename BufferT>
void Resnet1D<BufferT>::create_model_and_buffers(
        shape_type  const &input_shape,
        uint32_t           num_stacks,
        uint32_t           num_blocks,
        uint32_t           num_classes,
        std::string const &state_dict_filepath,
        bool               weights_are_packed)
{
    uint32_t num_classes_padded{num_classes};
    if (num_classes % BufferT::C_ib != 0)
    {
        num_classes_padded += (BufferT::C_ib - (num_classes % BufferT::C_ib));
    }
    std::cerr << "Setting up model with: num_classes(" << num_classes
              << "), num_classes_padded(" << num_classes_padded << ")\n";

    std::string state_dict_content;
    extract_file_content(state_dict_filepath, state_dict_content);

    std::vector<small::Layer<BufferT>*> layers;
    construct_resnet(state_dict_content, num_stacks, num_blocks, num_classes);

    // HACK placeholder
    size_t max_elt = 65536;
    m_buffer_0 = new Tensor<BufferT>(max_elt);
    m_buffer_1 = new Tensor<BufferT>(max_elt);
    m_buffer_2 = new Tensor<BufferT>(max_elt);
}

//****************************************************************************
template <typename BufferT>
std::vector<Tensor<BufferT>*> Resnet1D<BufferT>::inference(
    Tensor<BufferT> const *input_tensor)
{
    // assert(input_tensor[0]->size() is correct);

    size_t layer_num = 0;
    get_layer(layer_num++)->compute_output({input_tensor},
                                           m_buffer_0);    // Conv2D+ReLU

    for (uint32_t sid = 0; sid < m_num_stacks; ++sid)
    {
        // stack_idx = sid + 1
        for (uint32_t block_idx = 0; block_idx < m_num_blocks; ++block_idx)
        {
            get_layer(layer_num++)->compute_output({m_buffer_0},
                                                   m_buffer_1);   // Conv2D+ReLU
        }
    }

    // HACK placeholder
    m_buffer_0->set_shape((this->m_layers).back()->output_shape());
    return {m_buffer_0};
}

}
