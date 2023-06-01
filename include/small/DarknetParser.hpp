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

#include <math.h>
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <climits>
#include <vector>
#include <sstream>

#include <small.h>
#include <small/utils/Timer.hpp>
#include "../demo/utils.h"

// need to include all possible darknet layers
#include <small/Layer.hpp>
#include <small/DummyLayer.hpp>
#include <small/Conv2DLayer.hpp>
#include <small/MaxPool2DLayer.hpp>
#include <small/ReLULayer.hpp>
#include <small/LeakyReLULayer.hpp>
#include <small/AddLayer.hpp>
#include <small/UpSample2DLayer.hpp>

#define PARSER_DEBUG 0

std::string str_shape(small::shape_type shape) {
    std::string str = "(" + std::to_string(shape[0]);
    for(uint32_t i=1; i<shape.size(); i++) {
        str += ", " + std::to_string(shape[i]);
    }
    str += ")";
    return str;
}

typedef int64_t layer_t;

enum LAYER_TYPES {
    // used to index into func array
    MAXPOOL=0,
    YOLO=1,
    ROUTE=2,
    SHORTCUT=3,
    UPSAMPLE=4,
    NETWORK=5,
    // not used for indexing
    CONVOLUTIONAL=-2, 
    NOT_SUPPORTED=-1
};

struct block_info_t {
    // image size
    uint32_t w=-1;
    uint32_t h=-1;
    uint32_t c=-1;
    uint32_t ic=-1;

    // kernel size
    uint32_t size=-1;

    // sliding window stride
    uint32_t stride=-1;

    small::PaddingEnum pad=small::PADDING_V;
    small::ActivationType activation=small::NONE;
    bool bn=0;

    // for route and shortcut layers
    std::vector<int> layers;

    // YOLO related stuff
    std::vector<uint32_t> mask;
    std::vector<std::pair<uint32_t,uint32_t>> anchors;
    float jitter=-1;
    float ignore_thresh=-1;
    float truth_thresh=-1;
    uint32_t random=-1;
    uint32_t classes=-1;
    uint32_t num=-1;
};


layer_t string_to_layer(std::string type)
{
    if (type == "[shortcut]") return SHORTCUT;
    if (type == "[yolo]") return YOLO;
    if (type == "[conv]" || type == "[convolutional]") return CONVOLUTIONAL;
    if (type == "[net]" || type ==  "[network]") return NETWORK;
    if (type == "[max]" || type == "[maxpool]") return MAXPOOL;
    if (type == "[route]") return ROUTE;
    if (type == "[upsample]") return UPSAMPLE;
    return NOT_SUPPORTED;
}

std::string layer_to_string(layer_t type)
{
    if (type == SHORTCUT) return "[shortcut]";
    if (type == YOLO) return "[yolo]";
    if (type == CONVOLUTIONAL) return "[conv]";
    if (type == NETWORK) return "[net]";
    if (type == MAXPOOL) return "[max]";
    if (type == ROUTE) return "[route]";
    if (type == UPSAMPLE) return "[upsample]";
    return "[not supported]";
}

small::ActivationType get_activation(std::string act_type) {
    if(act_type == "leaky")
        return small::LEAKY;
    else if(act_type == "relu")
        return small::RELU;
    else if(act_type == "linear")
        return small::NONE;
    else {
        std::cout << "Activation type " << act_type << " not supported" << std::endl;
        exit(1);
    }
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

// gets layer information
// assume that there are no white spaces between '='
block_info_t get_block_info(std::ifstream *cfg_file) {

    block_info_t block_info;

    std::string line;
    int last_pos = cfg_file->tellg();
    while(getline(*cfg_file, line)) {
        
        // skip empty lines and comments
        if(line.empty() || line.at(0) == '#' ) {
            continue;
        }
        // we hit the next block
        if(line.at(0) == '[') {
            cfg_file->seekg(last_pos); // make sure to move file ptr back
            break;
        }

        // parse the line
        std::string key = line.substr(0, line.find("="));
        std::string value = line.substr(line.find("=") + 1);
        if(key == "size") {
            block_info.size = std::stoi(value);
        } else if(key == "stride") {
            block_info.stride = std::stoi(value);
        } else if(key == "pad") {
            // todo: check this
            block_info.pad = value == "1" ?  small::PADDING_F : small::PADDING_V;
        } else if(key == "filters" || key == "channels") {
            block_info.c = std::stoi(value);
        } else if(key == "activation") {
            block_info.activation = get_activation(value);
        } else if(key == "batch_normalize") {
            block_info.bn = std::stoi(value);
        } else if(key == "layers" || key == "from") {
            block_info.layers = extract_int_array<int>(value);
        } else if(key == "mask") {
            block_info.mask = extract_int_array<uint32_t>(value);
        } else if(key == "anchors") {
            block_info.anchors = get_anchors(value);
        } else if(key == "jitter") {
            block_info.jitter = std::stof(value);
        } else if(key == "ignore_thresh") {
            block_info.ignore_thresh = std::stof(value);
        } else if(key == "truth_thresh") {
            block_info.truth_thresh = std::stof(value);
        } else if(key == "random") {
            block_info.random = std::stoi(value);
        } else if(key == "classes") {
            block_info.classes = std::stoi(value);
        } else if(key == "num") {
            block_info.num = std::stoi(value);
        } else if(key == "width") {
            block_info.w = std::stoi(value);
        } else if(key == "height") {
            block_info.h = std::stoi(value);
        } else {
            // #if PARSER_DEBUG == 1
            // std::cout << "Unknown key: " << key << std::endl;
            // #endif
            // exit(1);
        }

        last_pos = cfg_file->tellg();
    }

    return block_info;
}

template <typename BufferT>
small::Layer<BufferT>* parse_conv(small::shape_type &input, block_info_t block_info, float *weights) {
    
    size_t filt_size = input[small::CHANNEL]*block_info.size*block_info.size*block_info.c;

    #if PARSER_DEBUG == 1
    std::cout << "Parsing conv layer with filt size: " << filt_size << std::endl;
    if(block_info.bn) {
        std::cout << "Batch norm weights: " << block_info.c*4 << std::endl;
    }
    else {
        std::cout << "Conv bias: " << block_info.c << std::endl;
    }
    #endif

    BufferT biases(block_info.c);

    // has batch norm weights
    if(block_info.bn) {
        std::copy(weights, weights + block_info.c, &biases[0]);
        BufferT bn_weights(block_info.c);
        std::copy(weights+block_info.c, weights+2*block_info.c, &bn_weights[0]);
        BufferT bn_run_mean(block_info.c);
        std::copy(weights+2*block_info.c, weights+3*block_info.c, &bn_run_mean[0]);
        BufferT bn_run_var(block_info.c);
        std::copy(weights+3*block_info.c, weights+4*block_info.c, &bn_run_var[0]);
    }
    // only conv bias
    else{
        std::copy(weights, weights + block_info.c, &biases[0]);
    }   

    // copy filters in
    BufferT filters(filt_size);
    std::copy(weights+block_info.c, weights+block_info.c+filt_size, &filters[0]);

    small::Layer<BufferT> *conv = 
        new small::Conv2DLayer<BufferT>(
            input,
            block_info.size, block_info.size,
            block_info.stride, block_info.pad,
            block_info.c,
            /// @todo block_info.bn, // todo add batch norm
            /// @todo bias,
            filters, // todo replace with weights
            false,
            block_info.activation
        );

    // #if PARSER_DEBUG == 1
    // std::cout << "Finished creating conv layer with filt size: " << filt_size << std::endl;
    // #endif

    return conv;
}

template <typename BufferT>
small::Layer<BufferT>* parse_max(small::shape_type &input, block_info_t block_info) {

    #if PARSER_DEBUG == 1
    std::cout << "Parsing max layer" << std::endl;
    #endif

    small::Layer<BufferT> *maxpool = 
        new small::MaxPool2DLayer<BufferT>(
            input,
            block_info.size, block_info.size,
            block_info.stride, small::PADDING_F
        );

    return maxpool;
}

template <typename BufferT>
small::Layer<BufferT>* parse_yolo(small::shape_type &input, block_info_t block_info) {
    
    #if PARSER_DEBUG == 1
    std::cout << "Parsing yolo layer" << std::endl;
    #endif

    // TODO
    small::Layer<BufferT> *yolo = new small::DummyLayer<BufferT>(input);
    return yolo;
}

template <typename BufferT>
small::Layer<BufferT>* parse_route(small::shape_type &input, block_info_t block_info) {
    
    #if PARSER_DEBUG == 1
    std::cout << "Parsing route layer" << std::endl;
    #endif

    // TODO
    small::Layer<BufferT> *route = new small::DummyLayer<BufferT>(input);
    return route;
}

// todo: need to figure out which layers are being added
template <typename BufferT>
small::Layer<BufferT>* parse_shortcut(small::shape_type &input, block_info_t block_info) {

    #if PARSER_DEBUG == 1
    std::cout << "Parsing shortcut layer" << std::endl;
    #endif

    // TODO
    // small::Layer<BufferT> *shortcut = 
    //     new small::AddLayer<BufferT>(
    //         *prev1, *prev2
    //     );

    // return shortcut;
    small::Layer<BufferT> *shortcut = new small::DummyLayer<BufferT>(input);
    return shortcut;
}

template <typename BufferT>
small::Layer<BufferT>* parse_upsample(small::shape_type &input, block_info_t block_info) {
    
    
    #if PARSER_DEBUG == 1
    std::cout << "Parsing upsample layer" << std::endl;
    #endif

    // TODO
    small::Layer<BufferT> *upsample = // new small::DummyLayer<BufferT>(input);
        new small::UpSample2DLayer<BufferT>(
            input,
            block_info.stride
        );

    return upsample;
}

// set up input shape of network
template <typename BufferT>
small::Layer<BufferT>* parse_network(small::shape_type &input, block_info_t block_info) {
    input = {1UL, block_info.c, block_info.h, block_info.w};
    return nullptr;
}

// function pointer array for parsing each block type
// order must match enum
template <typename BufferT>
small::Layer<BufferT>*  (* parse_funcs [])(small::shape_type&, block_info_t) = {
    parse_max,
    parse_yolo,
    parse_route,
    parse_shortcut,
    parse_upsample,
    parse_network
};


// Parse a darknet config file to construct a model
template <typename BufferT>
std::vector<small::Layer<BufferT>*> parse_cfg_and_weights(std::string cfg_file, std::string weights_file) {

    // TODO: change this to create a model object
    std::vector<small::Layer<BufferT>*> model;

    small::Layer<BufferT>* prev;
    small::shape_type input = {0,0,0,0};

    std::ifstream cfg(cfg_file);
    if(!cfg.is_open()) {
        std::cerr << "Could not open cfg file: " << cfg_file << std::endl;
        throw std::invalid_argument("parse_cfg_and_weights ERROR: failed to open file.");
    }

    // harded code for floats
    std::ifstream weights(weights_file, std::ios::binary);
    weights.seekg(0, std::ios::end);
    size_t total_elems = weights.tellg() / sizeof(float);
    weights.seekg(0, std::ios::beg);
    std::vector<int> header(5);
    weights.read(reinterpret_cast<char*>(&header[0]), 5*sizeof(int));
    std::cout << "Header: " << header[0] << "." << header[1] << "." << header[2] << "\t" << (int64_t)header[3] << "\n";
    BufferT weight_data(total_elems-5);
    weights.read(reinterpret_cast<char*>(weight_data.data()), total_elems*sizeof(float));
    float *weight_data_ptr = weight_data.data();
    uint64_t weight_idx = 0;

    size_t layer_number = 0;

    std::string line;
    while (getline (cfg, line)) {

        // ignore empty lines and comments
        // make sure empty check happens first or else at() will throw an exception
        if(line.empty() || line.at(0) == '#' ) {
            continue;
        }

        // Layer/Network info starts with a '['
        if(line.at(0) == '[') {

            layer_t layer_type = string_to_layer(line); // get layer type

            if(layer_type == NOT_SUPPORTED) {
                continue; // skip unsupported layers
            }

            #if PARSER_DEBUG == 1
            std::cout << "Current Layer: " << line << std::endl;
            #endif

            block_info_t block_info = get_block_info(&cfg); // get layer info

            // this should always be the first block
            // initialize the input shape of the network
            if(layer_type == NETWORK) {
                prev = parse_funcs<BufferT>[layer_type](input, block_info);
                std::cout << "Network Input Shape: " << str_shape(input) << std::endl;
                continue; // skip adding because prev == nullptr
            }

            std::cout << "Input shape for " << layer_to_string(layer_type) << " " \
                      << layer_number << ": " << str_shape(input) << std::endl;

            // special case due to weight loading
            if(layer_type == CONVOLUTIONAL) {
                prev = parse_conv<BufferT>(input, block_info, &weight_data_ptr[weight_idx]);

                // batch norm is stored as follows: bias, weights, running_mean, running_var
                // each are the size of the number of output channels
                if(block_info.bn) {
                    weight_idx += block_info.c * 4;
                }
                // in the case that there is no batch norm, we need to increment for just the conv biases
                else{
                    weight_idx += block_info.c;
                }
                // lastly, always move pointer for filter
                weight_idx += block_info.c * block_info.size * block_info.size * input[small::CHANNEL];
                std::cout << "Weight data remaining: " << total_elems - weight_idx - 5 << std::endl;
            }
            else {
                prev = parse_funcs<BufferT>[layer_type](input, block_info);
            }

            // set next layer input shape to be prev output
            input = prev->output_shape();

            // in the case that we have route/shortcut layers
            // input shape needs to be set by the index of the layers they are referencing
            if(layer_type == ROUTE) {
                uint32_t new_c = 0;
                // we only concat on the channel dimension
                for(auto l : block_info.layers) {
                    // handle negative indexing
                    new_c += (l<0) ? \
                        model[layer_number + l]->output_shape()[small::CHANNEL] : \
                        model[l]->output_shape()[small::CHANNEL];
                }
                // get h and w
                int l = block_info.layers[0];
                uint32_t h = (l<0) ? model[layer_number + l]->output_shape()[small::HEIGHT] \
                    : model[l]->output_shape()[small::HEIGHT];
                uint32_t w = (l<0) ? model[layer_number + l]->output_shape()[small::WIDTH] \
                    : model[l]->output_shape()[small::WIDTH];
                input = {1UL, new_c, h, w}; // assign new input shape
                
            }
            else if(layer_type == SHORTCUT) {
                input = model[layer_number + block_info.layers[0]]->output_shape();
            }

            #if PARSER_DEBUG == 1
            std::cout << "Adding layer " << layer_number << " to model " << std::endl;
            #endif

            model.push_back(prev); // add layer to model
            layer_number++;
        }
    }

    #if PARSER_DEBUG == 1
    std::cout << "Finished building model" << std::endl;
    #endif

    return model;
}