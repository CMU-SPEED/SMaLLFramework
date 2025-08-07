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

// #define DEBUG_LAYERS

#include <vector>
#include <small.h>
#include <small/state_dict_utils.hpp>
#include <small/Model.hpp>
#include <small/Conv1DLayer.hpp>
#include <small/PartialConv1DLayer.hpp>
#include <small/AveragePool1DLayer.hpp>
#include <small/LogSoftMaxLayer.hpp>
#include <small/DenseLayer.hpp>
namespace small
{

///****************************************************************************
/// The state_dict_filepath is a path to an ASCII file containing a pickled
/// pytorch stat_dict (I think).  This model requires a specific set of key
/// names in order to load properly.
///
/// @todo Is there a more generic way to load the pretrained model weights?
/// @todo Is there a way to advertise the state_dict keys (layer names) that
///       are expected?
///
/// From https://github.com/a-martyn/resnet
/// Modified to fit complex input signals (two channels) instead of images
/// Modified by A. Reddy to use 1D convolutional kernels instead of 2D
///
template <typename BufferT>
class Resnet1D : public Model<BufferT>
{
public:
    Resnet1D() = delete;

    // Assume one input layer with a single shape for now
    Resnet1D(shape_type  const &input_shape,          // NCHW = (1, 2, 1, 1024)
             uint32_t           num_stacks,           // 3
             uint32_t           num_blocks,           // 5
             uint32_t           num_classes,          // 2
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
    void construct_resnet_input_layers(
        std::string const &state_dict_content,
        size_t            &max_buffer_size);

    void construct_resnet_stack_layers(
        const std::string &state_dict_content,
        uint32_t stack_idx,
        uint32_t num_blocks,
        size_t            &max_buffer_size);

    void construct_resnet_output_layers(
        std::string const &state_dict_content,
        uint32_t           num_classes,
        size_t            &max_buffer_size);

    void construct_resnet(
        std::string const &state_dict_content,
        uint32_t           num_stacks,
        uint32_t           num_blocks,
        uint32_t           num_classes,
        size_t            &max_buffer_size);

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
    std::string const &state_dict_content,
    size_t            &max_buffer_size)
{
    max_buffer_size = (this->get_input_shape()).size();

    uint32_t C_o = 16U;

    /// @todo is there a better way to import pretrained model weights?
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
        new small::Conv1DLayer<BufferT>(this->get_input_shape(), //NCHW_i,
                                        3U,                // kernel_size
                                        1U,                // stride
                                        small::PADDING_F,  // padding
                                        C_o,               // filters
                                        conv_weight_buf,
                                        bn_weight_buf,
                                        bn_bias_buf,
                                        bn_mean_buf,
                                        bn_var_buf,
                                        1.e-5,             // bn_epsilon
                                        false,             // buffers_are_packed
                                        small::ActivationType::RELU);

    max_buffer_size = std::max(max_buffer_size, conv_in->output_size());
    this->m_layers.push_back(conv_in);
}

//****************************************************************************
template<typename BufferT>
void Resnet1D<BufferT>::construct_resnet_stack_layers(
    std::string const &state_dict_content,
    uint32_t           stack_idx,
    uint32_t           num_blocks,
    size_t            &max_buffer_size)
{
    for (uint32_t block_idx = 0; block_idx < num_blocks; block_idx++)
    {
        bool downsample{((stack_idx > 1) && (block_idx == 0))};

        //--------------------------------------------------------
        // Conv1D
        //--------------------------------------------------------
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

        // conv1
        auto& input_shape1{ (this->m_layers.back())->output_shape() };

        small::Conv1DLayer<BufferT> *conv1 =
            new small::Conv1DLayer<BufferT>(
                input_shape1,
                3U,                                // kernel_size
                (downsample ? 2U : 1U),            // stride,
                small::PADDING_F,                  // padding
                (downsample ? 2*input_shape1[CHANNEL] : input_shape1[CHANNEL]), // C_o
                conv_weight_buf1,
                bn_weight_buf1, bn_bias_buf1, bn_mean_buf1, bn_var_buf1,
                1.e-5,                             // bn_eps
                false,                             // buffers_are_packed
                small::ActivationType::RELU);

        max_buffer_size = std::max(max_buffer_size, conv1->output_size());
        this->m_layers.push_back(conv1);

        if (downsample)
        {
            small::AveragePool1DLayer<BufferT> *avgpool =
                new small::AveragePool1DLayer<BufferT>(
                    input_shape1,
                    1U,                                   // kernel size
                    2U,                                   // stride
                    small::PADDING_V);

            max_buffer_size = std::max(max_buffer_size, avgpool->output_size());
            this->m_layers.push_back(avgpool);
        }

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

        // conv2
        auto &input_shape2{conv1->output_shape()};  // pred. output

        small::PartialConv1DLayer<BufferT> *conv2 =
            new small::PartialConv1DLayer<BufferT>(
                input_shape2,
                3U,       // kernel
                1U,       // stride
                small::PADDING_F,
                input_shape2[CHANNEL], //C_o,
                conv_weight_buf2,
                bn_weight_buf2, bn_bias_buf2, bn_mean_buf2, bn_var_buf2,
                1.e-5,    // bn_eps
                false,    // buffers_are_packed
                small::ActivationType::RELU);

        max_buffer_size = std::max(max_buffer_size, conv2->output_size());
        this->m_layers.push_back(conv2);
    }
}

//****************************************************************************
template<typename BufferT>
void Resnet1D<BufferT>::construct_resnet_output_layers(
    std::string const &state_dict_content,
    uint32_t           num_classes,
    size_t            &max_buffer_size)
{
    uint32_t num_classes_padded{num_classes};
    if (num_classes % BufferT::C_ib != 0)
    {
        num_classes_padded += (BufferT::C_ib - (num_classes % BufferT::C_ib));
    }

    //-------------------------------------------------------------------------
    // AveragePool
    //-------------------------------------------------------------------------
    small::shape_type input_shape( (this->m_layers.back())->output_shape() );

    small::AveragePool1DLayer<BufferT> *avgpool =
        new small::AveragePool1DLayer<BufferT>(
            input_shape,
            input_shape[small::WIDTH],   // kernel size = input width
            1U,
            small::PADDING_V);

    max_buffer_size = std::max(max_buffer_size, avgpool->output_size());
    this->m_layers.push_back(avgpool);

    //-------------------------------------------------------------------------
    // FC/Dense (will pad odd channels in the output buffer)
    //-------------------------------------------------------------------------
    BufferT fc_o_weights{
        extract_param<BufferT>(state_dict_content, "fcOut.weight")};
    BufferT fc_o_biases{
        extract_param<BufferT>(state_dict_content, "fcOut.bias")};

    small::DenseLayer<BufferT> *fc =
        new small::DenseLayer<BufferT>(
            avgpool->output_shape(),     // fc_input_shape,
            num_classes,                 // fc_params,
            fc_o_weights, fc_o_biases,
            false, small::ActivationType::NONE);

    max_buffer_size = std::max(max_buffer_size, fc->output_size());
    this->m_layers.push_back(fc);

    //-------------------------------------------------------------------------
    // LogSoftMax
    //-------------------------------------------------------------------------
    small::LogSoftMaxLayer<BufferT> *logsoftmax =
        new small::LogSoftMaxLayer<BufferT>(fc->output_shape());

    max_buffer_size = std::max(max_buffer_size, logsoftmax->output_size());
    this->m_layers.push_back(logsoftmax);
}

//****************************************************************************
template<typename BufferT>
void Resnet1D<BufferT>::construct_resnet(
    std::string const &state_dict_content,
    uint32_t           num_stacks,
    uint32_t           num_blocks,
    uint32_t           num_classes,
    size_t            &max_buffer_size)
{
    construct_resnet_input_layers(state_dict_content,
                                  max_buffer_size);

    /// @todo weird stack id's
    for (uint32_t stack_idx = 1; stack_idx <= num_stacks; ++stack_idx)
    {
        construct_resnet_stack_layers(state_dict_content,
                                      stack_idx,
                                      num_blocks,
                                      max_buffer_size);
    }

    construct_resnet_output_layers(state_dict_content,
                                   num_classes,
                                   max_buffer_size);
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
    size_t max_buffer_size{0};

    std::string state_dict_content;
    extract_file_content(state_dict_filepath, state_dict_content);

    construct_resnet(state_dict_content,
                     num_stacks, num_blocks, num_classes,
                     max_buffer_size);

    // std::cerr << "Allocating activation buffers with size: "
    //           << max_buffer_size << std::endl;

    m_buffer_0 = new Tensor<BufferT>(max_buffer_size);
    m_buffer_1 = new Tensor<BufferT>(max_buffer_size);
    m_buffer_2 = new Tensor<BufferT>(max_buffer_size);
}

//****************************************************************************
template <typename BufferT>
std::vector<Tensor<BufferT>*> Resnet1D<BufferT>::inference(
    Tensor<BufferT> const *input_tensor)
{
    size_t layer_num = 0;

    // assert(input_tensor[0]->size() is correct);

    // Input layer
    Layer<BufferT> *curr_layer = this->get_layer(layer_num++);
    curr_layer->compute_output({input_tensor}, m_buffer_0);    // Conv2D+ReLU

    // Stacks 1, 2, 3...
    for (uint32_t stack_idx = 1; stack_idx <= m_num_stacks; ++stack_idx)
    {
        // Blocks 0, 1, 2, 3, 4...
        for (uint32_t block_idx = 0; block_idx < m_num_blocks; ++block_idx)
        {
            // for each block m_buffer_0 contains the input and stores the output

            curr_layer = this->get_layer(layer_num++);
            curr_layer->compute_output({m_buffer_0}, m_buffer_1);    // Conv2D+ReLU

            // Downsample in first block after the first stack requires a buffer merge
            if ((stack_idx > 1) && (block_idx == 0))
            {
                // Need to pad m_buffer_2 with the right size (double the channels)
                // The code that follows is equivalent to the following python:
                //   m_buffer_2  = average_pool(m_buffer_0, kernel=1, stride=2)
                //   zero_buffer = torch.mul(m_buffer_2, 0)
                //   m_buffer_0  = torch.cat((m_buffer_2, zero_buffer), dim=1)

                curr_layer = this->get_layer(layer_num++);
                curr_layer->compute_output({m_buffer_0}, m_buffer_2);    // AvgPool

                m_buffer_0->set_shape(m_buffer_1->shape());
                size_t buf2_size{m_buffer_2->size()};
                for (size_t ix = 0; ix < buf2_size; ++ix)
                {
                    m_buffer_0->buffer()[ix] = m_buffer_2->buffer()[ix];
                    m_buffer_0->buffer()[ix + buf2_size] = 0.f;
                }
            }

            curr_layer = this->get_layer(layer_num++);
            curr_layer->compute_output({m_buffer_1}, m_buffer_0);    // Conv2D+ReLU
        }
    }

    // Output layers
    curr_layer = this->get_layer(layer_num++);
    curr_layer->compute_output({m_buffer_0}, m_buffer_1);    // AvgPool

    curr_layer = this->get_layer(layer_num++);
    curr_layer->compute_output({m_buffer_1}, m_buffer_0);    // FC/Dense

    curr_layer = this->get_layer(layer_num++);
    curr_layer->compute_output({m_buffer_0}, m_buffer_1);    // LogSoftMax

    return {m_buffer_1};
}

}
