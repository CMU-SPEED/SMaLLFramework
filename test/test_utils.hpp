/*
 * SMaLL framework
 *
 * Copyright 2022 Carnegie Mellon University and Authors.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DMxx-xxxx
 */

#pragma once

#include <fstream>
#include <random>
#include <exception>

#include <stdlib.h>
#include <arpa/inet.h>

struct LayerParams
{
    size_t   C_i;
    uint32_t H; // image_height;
    uint32_t W; // image_width;
    uint32_t k; // kernel_size;
    uint16_t s; // stride;
    char     p; // padding; 'v' or 'f'
    size_t   C_o;
};

//****************************************************************************
template <typename T>
T* small_alloc(size_t num_elements)
{
    T *buffer;
    int ret = posix_memalign((void**)&buffer, 64, num_elements*sizeof(T));
    if (0 == ret)
        return buffer;
    else
        return (T*)nullptr;
}

//****************************************************************************
std::string get_pathname(
    std::string const &directory,
    std::string const &buffer_name,  // in, filter, out,
    std::string const &layer_name,   // conv2d, dw_conv, relue, pool
    LayerParams const &params,
    size_t             num_elements)
{
    if ((params.p != 'v') && (params.p != 'f'))
    {
        throw std::invalid_argument("ERROR: bad padding value.");
    }

    std::string fname =
        directory + "/" + buffer_name + "__" + layer_name +
        "_Ci" + std::to_string(params.C_i) +
        "_H" + std::to_string(params.H) +
        "_W" + std::to_string(params.W) +
        "_k" + std::to_string(params.k) +
        "_s" + std::to_string(params.s) +
        ((params.p == 'v') ? "_v" : "_f") +
        ((params.C_o > 0) ? ("_Co" + std::to_string(params.C_o)) : "") +
        "_" +  std::to_string(num_elements) + ".bin";

    return fname;
}

//****************************************************************************
/// @pre in_buf points to allocation at least as big as
///             sizeof(float)*num_elts
int read_float_inputs(std::string const &input_data_fname,
                      float             *in_buf,
                      size_t             num_elts)
{
    using RealT = float;
    size_t in_n;
    std::ifstream ifs(input_data_fname, std::ios::binary);
    // TODO: check if there are endian issues for cross platform
    ifs.read(reinterpret_cast<char*>(&in_n), sizeof(size_t));

    if (num_elts != in_n)
    {
        throw std::invalid_argument("ERROR: num elements not correct: " +
                                    std::to_string(num_elts) + " != " +
                                    std::to_string(in_n));
    }

    // TODO: endian details
    ifs.read(reinterpret_cast<char*>(in_buf), num_elts*sizeof(RealT));
    return 0;
}

//****************************************************************************
/// Read a binary file in the following format:
///   - first 4 bytes is num_elements as uint32_t in network order
///   - the rest of file are the 4 byte floating point numbers
///
/// param[out] in_buf  memory will be allocated to hold the floating point
///                    numbers read from the file
///
/// @retval The number of elements stored in in_buf (0 if no allocation occured)
///
uint32_t read_float_inputs(std::string const &input_data_fname, float **in_buf)
{
    using RealT = float;
    *in_buf = nullptr;

    uint32_t in_n, num_elts;
    std::ifstream ifs(input_data_fname, std::ios::binary);
    if (!ifs)
    {
        std::cerr << "ERROR: failed to open file: " << input_data_fname
                  << std::endl;
        return 0;
    }

    // TODO: check if there are endian issues for cross platform
    ifs.read(reinterpret_cast<char*>(&in_n), sizeof(uint32_t));
    num_elts = ntohl(in_n);

    //std::cout << "Reading " << num_elts << " elements from "
    //          << input_data_fname << std::endl;

    if (num_elts < 1)
    {
        std::cerr << "ERROR: invalid number of elements in "
                  << input_data_fname << std::endl;
        *in_buf = nullptr;
        return 0;
    }

    // TODO: endian details
    *in_buf = small_alloc<float>(num_elts);
    if (*in_buf == nullptr)
    {
        std::cerr << "Error: allocation failed, num_elts = " << num_elts
                  << std::endl;
        return 0;
    }

    ifs.read(reinterpret_cast<char*>(*in_buf), num_elts*sizeof(RealT));

    return num_elts;
}

//****************************************************************************
// from https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
size_t compute_output_dim_old(size_t input_dim,
                              size_t kernel_dim,
                              size_t stride,
                              char   padding)
{
    if ((padding == 'v') && (input_dim >= kernel_dim))
    {
        return std::floor((input_dim - kernel_dim)/((float)stride)) + 1;
    }
    else if (padding == 'f')
    {
        return std::floor((input_dim - 1)/((float)stride)) + 1;
    }
    else
    {
        throw std::invalid_argument(std::string("Bad combination"));
    }

    return 0;
}

//****************************************************************************
size_t compute_output_dim(size_t input_dim,
                          size_t kernel_dim,
                          size_t stride,
                          char   padding)
{
    if ((padding == 'v') && (input_dim >= kernel_dim))
    {
        return std::floor((input_dim - kernel_dim)/((float)stride)) + 1;
    }
    else if (padding == 'f')
    {
        // int padding_elements;
        // if (input_dim % stride == 0)
        // {
        //     padding_elements = ((kernel_dim - stride > 0) ?
        //                         kernel_dim - stride :
        //                         0);
        // }
        // else
        // {
        //     padding_elements = ((kernel_dim - (input_dim % stride) > 0) ?
        //                         kernel_dim - (input_dim % stride) :
        //                         0);
        // }
        // size_t padded_input_dim = input_dim + padding_elements;
        // size_t output_dim = ((padded_input_dim - kernel_dim)/stride) - 1;

        uint8_t fpad, bpad;
        CALC_PADDING(input_dim, kernel_dim, stride, fpad, bpad);
        size_t padded_input_dim = input_dim + fpad + bpad;
        size_t output_dim = 1 + (padded_input_dim - kernel_dim)/stride;
        std::cerr << "f output dim: " << output_dim << std::endl;
        return std::max(output_dim, 0UL);
    }
    else
    {
        throw std::invalid_argument("Bad combination");
    }

    return 0;
}
