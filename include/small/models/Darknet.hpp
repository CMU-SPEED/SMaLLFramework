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

#include <vector>

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
    Darknet(std::string cfg, std::string weights) : Model<BufferT>()
    {
        parse_cfg_and_weights(cfg, weights);
    }

    virtual ~Darknet() { }


private:

    ActivationType parse_activation(std::string act_type) {
        if(act_type == "leaky") { return ActivationType::LEAKY; }
        else if(act_type == "relu") { return ActivationType::RELU; }
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
    Layer<BufferT>* parse_conv(std::ifstream cfg_file, 
                                shape_type input, 
                                ScalarT *weight_data_ptr, 
                                size &weight_idx) 
    {
        

        /// @todo: double check other possible parameters such as dialation, groups, etc.
        bool bn = false;
        size_t num_filters = 0;
        size_t kernel_size = 0;
        size_t stride = 0;
        PaddingEnum pad = PaddingEnum::PADDING_F;
        ActivationType activation = ActivationType::NONE;

        // keep track of last position in order to move back once new block is found
        int last_pos = cfg_file->tellg();
        // consume convolutional block entirely and extract layer parameters
        std::string line;
        while(getline(cfg_file, line)) {

            // skip empty lines and comments
            if(line.empty() || line.at(0) == '#') { continue; }
            if(line.at(0) == '[') { 
                cfg_file->seekg(last_pos); // move pointer back a line
                break; 
            }
            std::string key = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);

            if(key == "batch_normalize") { bn = (value == "1"); }
            else if(key == "filters") { num_filters = stoi(value); }
            else if(key == "size") { kernel_size = stoi(value); }
            else if(key == "stride") { stride = stoi(value); }
            else if(key == "pad") { pad = parse_padding(value); }
            else if(key == "activation") { activation = parse_activation(value); }
            last_pos = cfg_file->tellg();
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
            false,
            activation
        );

        return conv;
    }

    Layer<BufferT>* parse_max(std::ifstream cfg_file, shape_type input) {
        
        size_t kernel_size = 0;
        size_t stride = 0;
        /// @todo check this
        PaddingEnum pad = PaddingEnum::PADDING_F; // default

        // keep track of last position in order to move back once new block is found
        int last_pos = cfg_file->tellg();
        // consume max block entirely and extract layer parameters
        std::string line;
        while(getline(cfg_file, line)) {

            // skip empty lines and comments
            if(line.empty() || line.at(0) == '#') { continue; }
            if(line.at(0) == '[') { 
                cfg_file->seekg(last_pos); // move pointer back a line
                break; 
            }
            std::string key = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);

            if(key == "size") { kernel_size = stoi(value); }
            else if(key == "stride") { stride = stoi(value); }
            last_pos = cfg_file->tellg();
        }

        // create max pooling layer
        MaxPool2DLayer<BufferT> *max = new MaxPool2DLayer<BufferT> (
            input, 
            kernel_size, kernel_size, 
            stride,
            pad
        );

        return max;
    }

    Layer<BufferT>* parse_shortcut(std::ifstream cfg_file, shape_type input) {

        int from;

        // keep track of last position in order to move back once new block is found
        int last_pos = cfg_file->tellg();
        // consume max block entirely and extract layer parameters
        std::string line;
        while(getline(cfg_file, line)) {

            // skip empty lines and comments
            if(line.empty() || line.at(0) == '#') { continue; }
            if(line.at(0) == '[') { 
                cfg_file->seekg(last_pos); // move pointer back a line
                break; 
            }
            std::string key = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);

            /// @todo: shortcut also specifies an activation but its always linear
            /// ignore activation for now, but will need to support in the future
            if(key == "from") { from = stoi(value); }
            last_pos = cfg_file->tellg();
        }

        // create shortcut layer
        size_t total_layers_so_far = m_layers.size();
        AddLayer<BufferT> *shortcut = new AddLayer<BufferT> (input, m_layers[total_layers_so_far + from]->get_output());
        return shortcut;
    }

    Layer<BufferT>* parse_route(ifstream cfg_file) {
        std::vector<int> route_layers;

        // keep track of last position in order to move back once new block is found
        int last_pos = cfg_file->tellg();
        // consume max block entirely and extract layer parameters
        std::string line;
        while(getline(cfg_file, line)) {

            // skip empty lines and comments
            if(line.empty() || line.at(0) == '#') { continue; }
            if(line.at(0) == '[') { 
                cfg_file->seekg(last_pos); // move pointer back a line
                break; 
            }
            std::string key = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);

            if(key == "layers") { route_layers = extract_int_array<int>(value); }
            last_pos = cfg_file->tellg();
        }

        RouteLayer<BufferT> *route;
        std::vector<shape_type> inputs;

        for(auto layer : route_layers) {
            // handle negative indexing
            if(layer < 0) {
                inputs.push_back(m_layers[m_layers.size() + layer]->get_output());
            }
            else {
                inputs.push_back(m_layers[layer]->get_output());
            }
        }

        if(inputs.size() == 1)
            route = new RouteLayer<BufferT> (inputs[0]);
        else if(inputs.size() == 2)
            route = new RouteLayer<BufferT> (inputs[0], inputs[1]);
        else {
            std::cerr << "ERROR: route layer must have 1 or 2 layers" << std::endl;
            throw std::exception(); /// @todo: throw some exception
        }
         
        return route;
    }

    Layer<BufferT>* parse_yolo(ifstream cfg_file, shape_type input) {

        std::vector<uint32_t> mask;
        std::vector<uint32_t> anchors;
        uint32_t classes = 0;
        uint32_t num = 0;
        float jitter = 0.0;
        float ignore_thresh = 0.0;
        bool rand = false;

        // keep track of last position in order to move back once new block is found
        int last_pos = cfg_file->tellg();
        // consume max block entirely and extract layer parameters
        std::string line;
        while(getline(cfg_file, line)) {

            // skip empty lines and comments
            if(line.empty() || line.at(0) == '#') { continue; }
            if(line.at(0) == '[') { 
                cfg_file->seekg(last_pos); // move pointer back a line
                break; 
            }
            std::string key = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);

            if(key == "mask") { mask = extract_int_array<uint32_t>(value); }
            else if(key == "anchors") { anchors = get_anchros(value); }
            else if(key == "classes") { classes = stoi(value); }
            else if(key == "num") { num = stoi(value); }
            else if(key == "jitter") { jitter = stof(value); }
            else if(key == "ignore_thresh") { ignore_thresh = stof(value); }
            else if(key == "rand") { rand = (value == "1"); }

            last_pos = cfg_file->tellg();
        }

        /// @todo add support for yolo layer

        DummyLayer *yolo = new DummyLayer(input);

        return yolo;
    }

    // parse network block and return the input shape
    shape_type parse_network(std::ifstream cfg_file) {
        shape_type input_shape = {0,0,0,0};
        std::string line;
        // keep track of last position in order to move back once new block is found
        int last_pos = cfg_file->tellg(); 

        // consume network block entirely and extract input shape
        while(getline(cfg_file, line)) {
            
            // skip empty lines and comments
            if(line.empty() || line.at(0) == '#') { continue; }
            if(line.at(0) == '[') { 
                cfg_file->seekg(last_pos); // move pointer back a line
                break; 
            }
            std::string key = line.substr(0, line.find("="));
            std::string value = line.substr(line.find("=") + 1);

            if(key == "height") { input_shape[HEIGHT] = std::stoi(value); } 
            else if(key == "width") { input_shape[WIDTH] = std::stoi(value); } 
            else if(key == "channels") { input_shape[CHANNEL] = std::stoi(value); }
            last_pos = cfg_file->tellg();
        }
        return input_shape;
    }

    void parse_cfg_and_weights(std::string cfg, std::weights weights) {

        using ScalarT = typename BufferT::value_type;

        Layer<BufferT> *prev = nullptr;
        shape_type prev_shape = {0,0,0,0};        

        std::ifstream cfg_file(cfg);
        if(!cfg_file.is_open()) {
            std::cerr << "Could not open cfg file: " << cfg << std::endl;
            throw std::invalid_argument("parse_cfg_and_weights ERROR: failed to open file.");
        }

        std::ifstream weights_file(weights, std::ios::binary);
        weights.seekg(0, std::ios::end);
        size_t total_bytes = weights.tellg(); // total size of weights in bytes
        weights.seekg(0, std::ios::beg);
        
        // first 20 bytes are header
        // first 12 bytes are uint32_t that represent the version
        // last 8 bytes is uint64_t that represents # of images seen during training
        // more info can be found here: https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        std::vector<int> header(5);
        weights_file.read((char*)header.data(), 20);
        std::cout << "Header: " << header[0] << "." << header[1] << "." << header[2] << \
                   "\tImages Seen:" << (int64_t)header[3] << "\n";

        // total elems of weight file
        size_t total_elems = (total_bytes-20) / sizeof(ScalarT);
        BufferT weight_data(total_elems);
        weights_file.read((char*)weight_data.data(), total_bytes-20);
        ScalarT *weight_data_ptr = weight_data.data();
        size_t weight_idx = 0;

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
                    m_input_shape = parse_network(cfg_file); // set model shape
                    std::cout << "Darknet Input Shape: " << str_shape(input_shape) << std::endl;
                    prev_shape = input_shape; // set prev shape to input shape
                    continue;
                }

                // Print layer info
                std::cout << line << " " << layer_idx << " input shape: " << str_shape(prev_shape) << std::endl;

                if(line == "[convolutional]" || line == "[conv]") {
                    prev = parse_conv<ScalarT>(cfg_file, prev_shape, weight_data_ptr, weight_idx);
                }
                else if(line == "[maxpool]" || line == "[max]") {
                    prev = parse_max(cfg_file, prev_shape);
                }
                else if(line == "[shortcut]") {
                    prev = parse_shortcut(cfg_file, input);
                }
                else if(line == "[route]") {
                    prev = parse_route(cfg_file);
                }
                else if(line == "upsample") {
                    prev = parse_upsample(cfg_file, prev_shape);
                }
                else if(line == "[yolo]") {
                    prev = parse_yolo(cfg_file, prev_shape);
                }

                // unsupported block
                // raise warning and skip
                else {
                    std::err << "WARNING: Unsupported block: " << line << std::endl;
                    while(getline(cfg_file, line)) {
                        if(line.empty()) { break; }
                    }
                    continue;
                }

                // add layer to model and update prev shape
                prev_shape = prev->output_shape();
                m_layers.push_back(prev);
                layer_idx++;
            }
        }

        #ifdef DEBUG
        std::cout << "Finished parsing." << std::endl;
        std::cout << "Total layers: " << m_layers.size() << std::endl;
        #endif

    }


};

}
