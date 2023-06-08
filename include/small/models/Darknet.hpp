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
    Darknet(std::string cfg, std::string weights) : Model<BufferT>({0,0,0,0})
    {
        parse_cfg_and_weights(cfg, weights);
    }

    virtual ~Darknet() {}

    virtual std::vector<Tensor<BufferT>*>
        inference(std::vector<Tensor<BufferT> const *> inputs)
    {
        std::cerr << "ERROR: Darknet::inference not implemented.\n";
        std::vector<Tensor<BufferT>*> outputs;
        return outputs;
    }

private:

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

        BufferT biases(num_filters);
        std::copy(&weight_data_ptr[weight_idx], &weight_data_ptr[weight_idx] + num_filters, &biases[0]);
        weight_idx += num_filters;
        /// @todo: need to find a good way to read batch norm parameters without losing scope
        // HACK: just move pointer for now
        if(bn){
            weight_idx += 3 * num_filters;
        }
        
        // Load convolution weights in
        size_t filt_size = num_filters * kernel_size * kernel_size * input[CHANNEL];
        BufferT filters(filt_size);
        std::copy(&weight_data_ptr[weight_idx], &weight_data_ptr[weight_idx] + filt_size, &filters[0]);
        weight_idx += filt_size;

        // create convolutional layer
        Conv2DLayer<BufferT> *conv = new Conv2DLayer<BufferT> (
            input, 
            kernel_size, kernel_size, 
            stride,
            pad, 
            num_filters,
            filters,
            true, /// @todo: this is not correct. however, when filter_size % C_ob != 0, it breaks;
            activation
        );

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
        size_t total_layers_so_far = this->m_layers.size();
        AddLayer<BufferT> *shortcut = new AddLayer<BufferT> (input, this->m_layers[total_layers_so_far + from]->output_shape(), {-1, from});
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

        for(auto layer : route_layers) {
            // handle negative indexing
            if(layer < 0) { inputs.push_back(this->m_layers[this->m_layers.size() + layer]->output_shape()); }
            else { inputs.push_back(this->m_layers[layer]->output_shape()); }
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

        DummyLayer<BufferT> *yolo = new DummyLayer<BufferT>(input);

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
        if(!cfg_file.is_open()) {
            std::cerr << "Could not open cfg file: " << cfg << std::endl;
            throw std::invalid_argument("parse_cfg_and_weights ERROR: failed to open file.");
        }

        std::ifstream weights_file(weights, std::ios::binary);
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
                    continue;
                }

                if(line == "[convolutional]" || line == "[conv]") {
                    prev = parse_conv<ScalarT>(cfg_file, prev_shape, weight_data_ptr, weight_idx);
                    #ifdef PARSER_DEBUG_VERBOSE
                    std::cout << "Weights elements remaining: " << total_elems - weight_idx << "\n";
                    #endif
                    prev_parents = {(int)layer_idx};
                }
                else if(line == "[maxpool]" || line == "[max]") {
                    prev = parse_max(cfg_file, prev_shape);
                    prev_parents = {(int)layer_idx};
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
                    prev_parents = {(int)layer_idx};
                }
                else if(line == "[yolo]") {
                    prev = parse_yolo(cfg_file, prev_shape);
                    prev_parents = {(int)layer_idx};
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

                std::cout << "\n";

                // add layer to model and update prev shape
                prev_shape = prev->output_shape();
                this->m_layers.push_back(prev);
                layer_idx++;
            }
        }

        #ifdef PARSER_DEBUG
        std::cout << "\nFinished parsing." << std::endl;
        std::cout << "Total layers: " << this->m_layers.size() << std::endl;
        #endif

    }
};

}
