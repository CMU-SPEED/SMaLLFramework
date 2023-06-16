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

#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <iomanip>

#include <small.h>
#include <small/Model.hpp>
#include <small/Layer.hpp>
#include <small/DummyLayer.hpp>
#include <small/RouteLayer.hpp>
#include <small/Conv2DLayer.hpp>
#include <small/MaxPool2DLayer.hpp>
#include <small/ReLULayer.hpp>
#include <small/LeakyReLULayer.hpp>
#include <small/AddLayer.hpp>
#include <small/UpSample2DLayer.hpp>
#include <small/YOLOLayer.hpp>

#define PARSER_DEBUG

//****************************************************************************
// helper functions for parsing

// returns shape in a string
std::string str_shape(small::shape_type shape) {
    std::string str = "(" + std::to_string(shape[0]);
    for(uint32_t i=1; i<shape.size(); i++) {
        str += ", " + std::to_string(shape[i]);
    }
    str += ")";
    return str;
}

std::string str_shape(std::vector<int> shape) {
    std::string str = "[" + std::to_string(shape[0]);
    for(uint32_t i=1; i<shape.size(); i++) {
        str += ", " + std::to_string(shape[i]);
    }
    str += "]";
    return str;
}

// extract a list of ints from a string assuming that the string is a comma separated list of ints
template <typename T>
std::vector<T> extract_int_array(std::string s) {
    std::vector<T> list;
    std::string delimiter = ",";
    size_t last = 0, next = 0; 
    while ((next = s.find(delimiter, last)) != std::string::npos) { 
        list.push_back(stoi(s.substr(last, next-last)));  
        last = next + 1; 
    } 
    list.push_back(stoi(s.substr(last)));  
    return list;
}

// get all anchor pairs
std::vector<std::pair<uint32_t, uint32_t>> get_anchors(std::string anchors) {
    std::vector<uint32_t> anchor_list = extract_int_array<uint32_t>(anchors);
    std::vector<std::pair<uint32_t, uint32_t>> anchor_pairs;
    for(uint32_t i = 0; i < anchor_list.size(); i+=2) {
        anchor_pairs.push_back(std::make_pair(anchor_list[i], anchor_list[i+1]));
    }
    return anchor_pairs;
}

namespace small
{

//****************************************************************************
template <typename BufferT>
class Darknet : public Model<BufferT>
{
public:
    Darknet() = delete;

    // Assume one input layer with a single shape for now
    Darknet(std::string cfg, std::string weights, bool save_outputs = false) 
        : Model<BufferT>({0,0,0,0}), m_save_outputs(save_outputs)
    {
        parse_cfg_and_weights(cfg, weights);
    }

    virtual ~Darknet() {
        // delete all buffers
        for(auto out : m_outputs) {
            delete out;
        }
        for(auto out : m_cached_outputs) {
            delete out.second;
        }
        delete m_in;
        delete m_out;
    }

    // assumes all the buffers have been set up in the constructor
    size_t yolo_block_idx = 0;
    virtual std::vector<Tensor<BufferT>*>
        inference(std::vector<Tensor<BufferT> const *> inputs)
    {

        if(inputs[0]->shape() != this->m_input_shape) {
            throw std::runtime_error("Input shape does not match model input shape");
        }

        if(m_save_outputs) {
            std::copy(
                &(inputs[0]->buffer()[0]), 
                &(inputs[0]->buffer()[compute_size(inputs[0]->shape())]), 
                &(m_outputs[0]->buffer()[0])
            );
        }
        else {
            m_in->set_shape(inputs[0]->shape());
            std::copy(
                &(inputs[0]->buffer()[0]), 
                &(inputs[0]->buffer()[compute_size(inputs[0]->shape())]), 
                &(m_in->buffer()[0])
            );
        }
        
        for(size_t layer_num=0; layer_num<this->m_layers.size(); layer_num++) {

            std::cout << layer_num;
            
            // get layer info
            Layer<BufferT> *layer = this->m_layers[layer_num];
            const std::type_info& type = typeid(*layer);
            shape_type out_shape = layer->output_shape();

            // make sure the size is always set
            if(!m_save_outputs) {
                m_out->set_shape(out_shape);
            }

            // single input/output layers
            if(type == typeid(Conv2DLayer<BufferT>) || 
               type == typeid(MaxPool2DLayer<BufferT>) ||
               type == typeid(UpSample2DLayer<BufferT>))
            {
                std::cout << " Conv/MaxPool/UpSample Layer" << std::endl;
                if(m_save_outputs) {
                    layer->compute_output({m_outputs[layer_num]}, {m_outputs[layer_num+1]});
                }
                else {
                    layer->compute_output({m_in}, {m_out});
                }
            }
            else if(type == typeid(AddLayer<BufferT>)) {

                std::cout << " Add Layer" << std::endl;

                // there will always only be 2 parents
                std::vector<int> parents = dynamic_cast<AddLayer<BufferT>*>(layer)->parents();
                assert(parents.size() == 2);

                if(m_save_outputs) {
                    std::copy(
                        &m_outputs[parents[1]]->buffer()[0], 
                        &m_outputs[parents[1]]->buffer()[compute_size(out_shape)], 
                        &m_outputs[layer_num+1]->buffer()[0]
                    );
                    layer->compute_output({m_outputs[layer_num]}, {m_outputs[layer_num+1]});
                }
                else {
                    std::copy(
                        &m_cached_outputs[parents[1]]->buffer()[0], 
                        &m_cached_outputs[parents[1]]->buffer()[compute_size(out_shape)], 
                        &m_out->buffer()[0]
                    );
                    layer->compute_output({m_in}, {m_out});
                }

            }
            else if(type == typeid(RouteLayer<BufferT>)) {

                std::cout << " Route Layer" << std::endl;

                std::vector<int> parents = dynamic_cast<RouteLayer<BufferT>*>(layer)->parents();
                assert(parents.size() == 1 || parents.size() == 2);

                // for parent.size() == 1, this should just be a buffer swap/copy
                if(parents.size() == 1) {
                    if(m_save_outputs) {
                        layer->compute_output({m_outputs[parents[0]]}, {m_outputs[layer_num+1]});   
                    }
                    else {
                        layer->compute_output({m_cached_outputs[parents[0]]}, {m_out});
                    }
                }
                // here we actually need to concat
                else {
                    if(m_save_outputs) {
                        layer->compute_output({m_outputs[parents[0]], m_outputs[parents[1]]}, {m_out});
                    }
                    else {
                        layer->compute_output({m_cached_outputs[parents[0]], m_cached_outputs[parents[1]]}, {m_out});
                    }
                }

            }
            // yolo is our output block
            else if(type == typeid(YOLOLayer<BufferT>)) {
                assert(yolo_block_idx < m_yolo_outputs.size());
                std::cout << " YOLO layer\n";

                if(m_save_outputs) {
                    layer->compute_output({m_outputs[layer_num]}, {m_outputs[layer_num+1]});
                    std::copy(
                        &m_outputs[layer_num+1]->buffer()[0], 
                        &m_outputs[layer_num+1]->buffer()[compute_size(out_shape)], 
                        &m_yolo_outputs[yolo_block_idx]->buffer()[0]
                    );
                }
                else {
                    layer->compute_output({m_in}, {m_yolo_outputs[yolo_block_idx]});
                }
                yolo_block_idx++;
            }
            else {
                std::cerr << "ERROR: Layer type not supported.\n";
                throw std::exception();
            }

            // found key in m_cached_outputs
            // save output to m_cached_outputs
            if(!m_save_outputs && m_cached_outputs.count(layer_num) == 1) {
                std::cout << "Saving output to m_cached_outputs\tlayer_num = " << layer_num << "\n";
                m_cached_outputs[layer_num] = new Tensor<BufferT>(out_shape);
                std::copy(
                    &m_out->buffer()[0], 
                    &m_out->buffer()[compute_size(out_shape)], 
                    &m_cached_outputs[layer_num]->buffer()[0]
                );
            }

            // swap input and output
            m_in->swap(*m_out);
        }

        return m_yolo_outputs;
    }

    size_t total_buffer_sizes() {
        size_t total_buf_size = 0;
        if(m_save_outputs) {
            for(auto out : m_outputs) {
                total_buf_size += compute_size(out->shape());
            }
        }
        else {
            total_buf_size += 2*m_max_buffer_size;
        }
        for(auto out : m_yolo_outputs) {
            total_buf_size += compute_size(out->shape());
        }
        return total_buf_size;
    }

private:

    // map for cached outputs
    // layer_idx -> output
    std::map<size_t, Tensor<BufferT>*>  m_cached_outputs;
    
    // intermediates
    // these are init at parse time
    size_t                              m_max_buffer_size = 0;
    Tensor<BufferT>*                    m_in;
    Tensor<BufferT>*                    m_out;
    std::vector<Tensor<BufferT>*>       m_yolo_outputs;

    // explict output for layer
    // this is used for robustness testing
    std::vector<Tensor<BufferT>*>       m_outputs;
    bool                                m_save_outputs;

    // returns type of activation based on the key
    ActivationType parse_activation(std::string act_type) {
        if(act_type == "leaky") { return ActivationType::LEAKY; }
        else if(act_type == "relu" ) { return ActivationType::RELU; }
        else if(act_type == "linear") { return ActivationType::NONE; }
        else { 
            std::cerr << "WARNING: Activation type " << act_type << " not supported. Using NONE.\n";
            return ActivationType::NONE; 
        }
    }

    /// @todo: this really needs to be checked. i have no clue how this should be handled
    PaddingEnum parse_padding(std::string pad_type) {
        if(pad_type == "1") { return PaddingEnum::PADDING_F; }
        else if(pad_type == "0") { return PaddingEnum::PADDING_V; }
        else { 
            std::cerr << "WARNING: Padding type " << pad_type << " not supported. Using PADDING_F.\n";
            return PaddingEnum::PADDING_F; 
        }
    }

    template <typename ScalarT>
    Layer<BufferT>* parse_conv(std::ifstream &cfg_file, 
                                shape_type input, 
                                ScalarT *weight_data_ptr, 
                                size_t &weight_idx) 
    {
        #ifdef PARSER_DEBUG_VERBOSE
        std::cout << "Parsing Convolutional Layer\n";
        #endif

        /// @todo: double check other possible parameters such as dialation, groups, etc.
        bool bn = false;
        size_t num_filters = 0;
        size_t kernel_size = 0;
        size_t stride = 0;
        PaddingEnum pad = PaddingEnum::PADDING_F;
        ActivationType activation = ActivationType::NONE;

        int last_pos = cfg_file.tellg();
        std::string line;
        while(getline(cfg_file, line)) {

            // skip empty lines and comments
            if(line.empty() || line.at(0) == '#') { last_pos = cfg_file.tellg(); continue; }
            // stop parsing when we reach the next layer
            if(line.at(0) == '[') { cfg_file.seekg(last_pos); break; }
            last_pos = cfg_file.tellg();
            std::string key = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);

            if(key == "batch_normalize") { bn = (value == "1"); }
            else if(key == "filters") { num_filters = stoi(value); }
            else if(key == "size") { kernel_size = stoi(value); }
            else if(key == "stride") { stride = stoi(value); }
            else if(key == "pad") { pad = parse_padding(value); }
            else if(key == "activation") { activation = parse_activation(value); }
            else { std::cerr << "WARNING: unknown key in conv2d layer: " << key << std::endl;}
        }
        
        // filter size
        size_t filt_size = num_filters * kernel_size * kernel_size * input[CHANNEL];

        Conv2DLayer<BufferT> *conv;
        
        if(bn) {

            // order matters
            // data is stored in the following order:
            // bn_bias, bn_weights, bn_running_mean, bn_running_variance, conv_weights
            
            BufferT bn_bias(num_filters);
            std::copy(weight_data_ptr + weight_idx, weight_data_ptr + weight_idx + num_filters, &bn_bias[0]);
            weight_idx += num_filters;

            BufferT bn_weights(num_filters);
            std::copy(weight_data_ptr + weight_idx, weight_data_ptr + weight_idx + num_filters, &bn_weights[0]);
            weight_idx += num_filters;

            BufferT bn_running_mean(num_filters);
            std::copy(weight_data_ptr + weight_idx, weight_data_ptr + weight_idx + num_filters, &bn_running_mean[0]);
            weight_idx += num_filters;

            BufferT bn_running_variance(num_filters);
            std::copy(weight_data_ptr + weight_idx, weight_data_ptr + weight_idx + num_filters, &bn_running_variance[0]);
            weight_idx += num_filters;

            BufferT filters(filt_size);
            std::copy(weight_data_ptr + weight_idx, weight_data_ptr + weight_idx + filt_size, &filters[0]);
            weight_idx += filt_size;

            conv = new Conv2DLayer<BufferT> (
                input, 
                kernel_size, kernel_size, 
                stride,
                pad, 
                num_filters,
                std::move(filters),
                std::move(bn_weights),
                std::move(bn_bias),
                std::move(bn_running_mean),
                std::move(bn_running_variance),
                1.e-5,
                false, // WRONG, WE NEED EDGE CASES
                activation,
                0.1
            );
        }
        else {

            // order matters
            // data is stored in the following order:
            // bias, conv_weights

            BufferT bias(num_filters);
            std::copy(weight_data_ptr + weight_idx, weight_data_ptr + weight_idx + num_filters, &bias[0]);
            weight_idx += num_filters;

            BufferT filters(filt_size);
            std::copy(weight_data_ptr + weight_idx, weight_data_ptr + weight_idx + filt_size, &filters[0]);
            weight_idx += filt_size;

            conv = new Conv2DLayer<BufferT> (
                input, 
                kernel_size, kernel_size, 
                stride,
                pad, 
                num_filters,
                std::move(filters),
                std::move(bias),
                false, // WRONG, WE NEED EDGE CASES
                activation,
                0.1
            );
        }

        return conv;

    }

    Layer<BufferT>* parse_max(std::ifstream &cfg_file, shape_type input) {

        #ifdef PARSER_DEBUG_VERBOSE
        std::cout << "Parsing MaxPool Layer\n";
        #endif
        
        size_t kernel_size = 0;
        size_t stride = 0;
        /// @todo check this
        PaddingEnum pad = PaddingEnum::PADDING_F; // default

        int last_pos = cfg_file.tellg();
        std::string line;
        while(getline(cfg_file, line)) {

            // skip empty lines and comments
            if(line.empty() || line.at(0) == '#') { last_pos = cfg_file.tellg(); continue; }
            // stop parsing when we reach the next layer
            if(line.at(0) == '[') { cfg_file.seekg(last_pos); break; }
            last_pos = cfg_file.tellg();
            std::string key = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);

            if(key == "size") { kernel_size = stoi(value); }
            else if(key == "stride") { stride = stoi(value); }
            else { std::cerr << "WARNING: unknown key in maxpool layer: " << key << std::endl;}
        }

        MaxPool2DLayer<BufferT> *max = new MaxPool2DLayer<BufferT> (
            input, 
            kernel_size, kernel_size, 
            stride,
            pad
        );

        return max;
    }

    Layer<BufferT>* parse_shortcut(std::ifstream &cfg_file, shape_type input) {

        #ifdef PARSER_DEBUG_VERBOSE
        std::cout << "Parsing Shortcut Layer\n";
        #endif

        int from = 0;
        ActivationType activation = ActivationType::NONE; 

        int last_pos = cfg_file.tellg();
        std::string line;
        while(getline(cfg_file, line)) {

            // skip empty lines and comments
            if(line.empty() || line.at(0) == '#') { last_pos = cfg_file.tellg(); continue; }
            // stop parsing when we reach the next layer
            if(line.at(0) == '[') { cfg_file.seekg(last_pos); break; }
            last_pos = cfg_file.tellg();
            std::string key = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);
            

            /// @todo: shortcut also specifies an activation but its always linear
            /// ignore activation for now, but will need to support in the future
            if(key == "from") { from = stoi(value); }
            else if(key == "activation") { activation = parse_activation(value); (void)activation; }
            else { std::cerr << "WARNING: unknown key in shortcut layer: " << key << std::endl; }
        }

        // create shortcut layer
        int total_layers_so_far = this->m_layers.size();
        m_cached_outputs[total_layers_so_far+from] = nullptr;
        AddLayer<BufferT> *shortcut = new AddLayer<BufferT> \
            (input, this->m_layers[total_layers_so_far + from]->output_shape(), {total_layers_so_far-1, total_layers_so_far+from});
        return shortcut;
    }

    Layer<BufferT>* parse_route(std::ifstream &cfg_file) {

        #ifdef PARSER_DEBUG_VERBOSE
        std::cout << "Parsing Route Layer\n";
        #endif

        std::vector<int> route_layers;

        int last_pos = cfg_file.tellg(); 
        std::string line;
        while(getline(cfg_file, line)) {

            // skip empty lines and comments
            if(line.empty() || line.at(0) == '#') { last_pos = cfg_file.tellg(); continue; }
            // stop parsing when we reach the next layer
            if(line.at(0) == '[') { cfg_file.seekg(last_pos); break; }
            last_pos = cfg_file.tellg();
            std::string key = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);

            if(key == "layers") { route_layers = extract_int_array<int>(value); }
            else { std::cerr << "WARNING: unknown key in route layer: " << key << std::endl; }
        }

        RouteLayer<BufferT> *route;
        std::vector<shape_type> inputs;

        for(uint32_t i = 0; i < route_layers.size(); i++) {
            // convert negative indexing
            if(route_layers[i] < 0) { 
                route_layers[i] = this->m_layers.size() + route_layers[i];
                inputs.push_back(this->m_layers[route_layers[i]]->output_shape());
            }
            else { 
                inputs.push_back(this->m_layers[route_layers[i]]->output_shape()); 
            }
            m_cached_outputs[route_layers[i]] = nullptr;
        }

        if(inputs.size() == 1) { route = new RouteLayer<BufferT> (inputs[0], route_layers); }
        else if(inputs.size() == 2) { route = new RouteLayer<BufferT> (inputs[0], inputs[1], route_layers); }
        else {
            throw std::invalid_argument("ERROR: route layer must have only 1 or 2 layers");
        }
         
        return route;
    }

    Layer<BufferT>* parse_upsample(std::ifstream &cfg_file, shape_type input) {

        #ifdef PARSER_DEBUG_VERBOSE
        std::cout << "Parsing Upsample Layer\n";
        #endif

        uint32_t stride = 0;

        int last_pos = cfg_file.tellg(); 
        std::string line;
        while(getline(cfg_file, line)) {

            // skip empty lines and comments
            if(line.empty() || line.at(0) == '#') { last_pos = cfg_file.tellg(); continue; }
            // stop parsing when we reach the next layer
            if(line.at(0) == '[') { cfg_file.seekg(last_pos); break; }
            last_pos = cfg_file.tellg();
            std::string key = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);

            if(key == "stride") { stride = stoi(value); }
            else { std::cerr << "WARNING: unknown key in upsample layer: " << key << std::endl; }
        }

        UpSample2DLayer<BufferT> *upsample = new UpSample2DLayer<BufferT> (input, stride);

        return upsample;
    }
 
    Layer<BufferT>* parse_yolo(std::ifstream &cfg_file, shape_type input) {

        #ifdef PARSER_DEBUG_VERBOSE
        std::cout << "Parsing Yolo Layer\n";
        #endif

        std::vector<uint32_t> mask;
        std::vector<std::pair<uint32_t,uint32_t>> anchors;
        uint32_t classes = 0;
        uint32_t num = 0;
        float jitter = 0.0;
        float ignore_thresh = 0.0;
        float truth_thresh = 0.0;
        bool rand = false;

        int last_pos = cfg_file.tellg();
        std::string line;
        while(getline(cfg_file, line)) {

            // skip empty lines and comments
            if(line.empty() || line.at(0) == '#') { last_pos = cfg_file.tellg(); continue; }
            // stop parsing when we reach the next layer
            if(line.at(0) == '[') { cfg_file.seekg(last_pos); break; }
            last_pos = cfg_file.tellg();
            std::string key = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);

            if(key == "mask") { mask = extract_int_array<uint32_t>(value); (void)mask;}
            else if(key == "anchors") { anchors = get_anchors(value); (void)anchors; }
            else if(key == "classes") { classes = stoi(value); (void)classes; }
            else if(key == "num") { num = stoi(value); (void)num; }
            else if(key == "jitter") { jitter = stof(value); (void)jitter; }
            else if(key == "ignore_thresh") { ignore_thresh = stof(value); (void)ignore_thresh; }
            else if(key == "truth_thresh") { truth_thresh = stof(value); (void)truth_thresh; }
            else if(key == "random" || key == "rand") { rand = (value == "1"); (void)rand; }
            else { std::cerr << "WARNING: unknown key in yolo layer: " << key << std::endl;}
        }

        /// @todo add support for yolo layer

        // DummyLayer<BufferT> *yolo = new DummyLayer<BufferT>(input);
        std::vector<std::pair<uint32_t,uint32_t>> masked_anchors;
        for(uint32_t i = 0; i < mask.size(); i++) {
            masked_anchors.push_back(anchors[mask[i]]);
        }

        YOLOLayer<BufferT> *yolo = new YOLOLayer<BufferT>(input, masked_anchors, classes, this->m_input_shape[HEIGHT]);

        return yolo;
    }

    // parse network block and return the input shape
    shape_type parse_network(std::ifstream &cfg_file) {

        #ifdef PARSER_DEBUG
        std::cout << "Parsing Network Block\n";
        #endif

        shape_type input_shape = {0,0,0,0};
        std::string line;
        int last_pos = cfg_file.tellg(); 

        // consume network block entirely and extract input shape
        while(getline(cfg_file, line)) {
            
            // skip empty lines and comments
            if(line.empty() || line.at(0) == '#') { last_pos = cfg_file.tellg(); continue; }
            // stop parsing when we reach the next layer
            if(line.at(0) == '[') { cfg_file.seekg(last_pos); break; }
            last_pos = cfg_file.tellg();
            std::string key = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);

            if(key == "height") { input_shape[HEIGHT] = std::stoi(value); } 
            else if(key == "width") { input_shape[WIDTH] = std::stoi(value); } 
            else if(key == "channels") { input_shape[CHANNEL] = std::stoi(value); }
            #ifdef PARSER_DEBUG_VERBOSE
            else { std::cerr << "WARNING: unknown key in net layer: " << key << std::endl;}
            #endif
        }
        return input_shape;
    }

    void parse_cfg_and_weights(std::string cfg, std::string weights) {

        using ScalarT = typename BufferT::value_type;

        Layer<BufferT> *prev = nullptr;
        shape_type prev_shape = {0,0,0,0};
        std::vector<int> prev_parents = {0};     

        std::ifstream cfg_file(cfg);
        if(!cfg_file) {
            throw std::invalid_argument("Could not open cfg file: " + cfg);
        }

        std::ifstream weights_file(weights, std::ios::binary);
        if(!weights_file) {
            throw std::invalid_argument("Could not open weights file: " + weights);
        }
        
        weights_file.seekg(0, std::ios::end);
        size_t total_bytes = weights_file.tellg(); // total size of weights in bytes
        weights_file.seekg(0, std::ios::beg);
        
        // first 20 bytes are header
        // first 12 bytes are uint32_t that represent the version
        // last 8 bytes is uint64_t that represents # of images seen during training
        // more info can be found here: https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        std::vector<int> header(5);
        weights_file.read((char*)header.data(), 20);

        // total elems of weight file
        size_t total_elems = (total_bytes-20) / sizeof(ScalarT);
        BufferT weight_data(total_elems);
        weights_file.read((char*)weight_data.data(), total_bytes-20);
        ScalarT *weight_data_ptr = weight_data.data();
        size_t weight_idx = 0;

        #ifdef PARSER_DEBUG
        std::cout << "\nWEIGHT FILE METADATA:\n";
        std::cout << "Header: " << header[0] << "." << header[1] << "." << header[2] << \
                   "\tImages Seen: " << (int64_t)header[3] << "\n";
        std::cout << "Total weight elements: " << total_elems << "\n\n";
        #endif

        size_t layer_idx = 0;

        std::string line;
        while (getline(cfg_file, line)) {
            // skip empty lines and comments
            // MAKE SURE EMPTY() IS FIRST OR ELSE AT() WILL THROW
            if(line.empty() || line.at(0) == '#') { continue; }

            // all blocks are enclosed in []
            if(line.at(0) == '[') {

                // network block contains training info and input shape
                if(line == "[network]" || line == "[net]") {
                    this->m_input_shape = parse_network(cfg_file); // set model shape
                    this->m_input_shape[BATCH] = 1;
                    std::cout << "\nDarknet Input Shape: " << str_shape(this->m_input_shape) << "\n\n";
                    prev_shape = this->m_input_shape; // set prev shape to input shape
                    m_max_buffer_size = std::max(m_max_buffer_size, compute_size(prev_shape));
                    if(m_save_outputs) {
                        this->m_outputs.push_back(new Tensor<BufferT>(prev_shape));
                    }
                    continue;
                }

                if(line == "[convolutional]" || line == "[conv]") {
                    prev = parse_conv<ScalarT>(cfg_file, prev_shape, weight_data_ptr, weight_idx);
                    #ifdef PARSER_DEBUG_VERBOSE
                    std::cout << "Weights elements remaining: " << total_elems - weight_idx << "\n";
                    #endif
                }
                else if(line == "[maxpool]" || line == "[max]") {
                    prev = parse_max(cfg_file, prev_shape);
                }
                else if(line == "[shortcut]") {
                    prev = parse_shortcut(cfg_file, prev_shape);
                    prev_parents = dynamic_cast<AddLayer<BufferT>*>(prev)->parents();
                }
                else if(line == "[route]") {
                    prev = parse_route(cfg_file);
                    prev_parents = dynamic_cast<RouteLayer<BufferT>*>(prev)->parents();
                }
                else if(line == "[upsample]") {
                    prev = parse_upsample(cfg_file, prev_shape);
                }
                else if(line == "[yolo]") {
                    prev = parse_yolo(cfg_file, prev_shape);
                    m_yolo_outputs.push_back(new Tensor<BufferT>(prev->output_shape()));
                    /// @todo: allocate output buffers here
                }

                // unsupported block
                // raise warning and skip
                else {
                    std::cerr << "WARNING: Unsupported block: " << line << std::endl;
                    int last_pos = cfg_file.tellg();
                    while(getline(cfg_file, line)) {
                        if(line.empty()) { last_pos = cfg_file.tellg(); continue; }
                        if(line.at(0) == '[') { 
                            cfg_file.seekg(last_pos); // move pointer back a line
                            break; 
                        }
                        last_pos = cfg_file.tellg();
                    }
                    continue;
                }

                // print layer index and type
                std::cout << std::setw(3) << layer_idx;
                std::cout << std::setw(18) << line << "\t";

                // Print layer info
                if(line != "[route]" && line != "[shortcut]") {
                    if(line != "[yolo]") {
                        std::cout << std::setw(30) << str_shape(prev_shape) << " --> " << \
                                     str_shape(prev->output_shape()) << "\n";
                    }
                    else {
                        std::cout << std::setw(30) << str_shape(prev_shape) << "\n";
                    }
                }
                else{
                    std::cout << std::setw(30) << str_shape(prev_parents) << " --> " << \
                                 str_shape(prev->output_shape()) << "\n";
                }

                prev_shape = prev->output_shape();
                m_max_buffer_size = std::max(m_max_buffer_size, compute_size(prev_shape));

                this->m_layers.push_back(prev);
                if(m_save_outputs){
                    m_outputs.push_back(new Tensor<BufferT>(prev_shape));
                }

                layer_idx++;
                std::cout << "\n";
            }
        }

        /// @todo allocate intermediate buffers here

        #ifdef PARSER_DEBUG
        std::cout << "\nFinished parsing." << std::endl;
        std::cout << "Total layers: " << this->m_layers.size() << std::endl;
        std::cout << "Max buffer size: " << m_max_buffer_size << std::endl;
        if(m_save_outputs)
            std::cout << "Saving outputs for each layer" << std::endl;
        std::cout << "Total buffer size: " << total_buffer_sizes() << std::endl;
        std::cout << std::endl;
        #endif

        // allocate intermediate buffers
        m_in = new Tensor<BufferT>(m_max_buffer_size);
        m_out = new Tensor<BufferT>(m_max_buffer_size);

    }

};

}
