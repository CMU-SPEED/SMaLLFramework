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

//****************************************************************************
template <typename T>
T* small_alloc(size_t num_elements)
{
    T *buffer;
    int ret = posix_memalign((void**)&buffer, 4096, num_elements*sizeof(T));
    if (0 == ret)
        return buffer;
    else
        return (T*)nullptr;
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

    if (num_elts > in_n)
        return 1;

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

    uint32_t in_n, num_elts;
    std::ifstream ifs(input_data_fname, std::ios::binary);
    // TODO: check if there are endian issues for cross platform
    ifs.read(reinterpret_cast<char*>(&in_n), sizeof(uint32_t));
    num_elts = ntohl(in_n);

    std::cout << "Reading " << num_elts << " elements from "
              << input_data_fname << std::endl;

    if (num_elts < 1)
    {
        std::cerr << "Error: invalid number of elements in "
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

    for (size_t ix = 0; ix < num_elts; ++ix)
    {
        ifs.read(reinterpret_cast<char*>(&((*in_buf)[ix])), sizeof(RealT));
    }
    for (size_t ix = 0; ix < num_elts; ++ix)
    {

        //TEST_CHECK((output_dc[ix] == ));
        std::cout << ": input data(" << ix << ")-->"
                  << (*in_buf)[ix] << std::endl;
    }

    return num_elts;
}
