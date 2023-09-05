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

#include <float.h>
#include <fstream>
#include <random>
#include <exception>
#include <algorithm> // std::min_element
#include <filesystem>

#include <stdlib.h>
#include <arpa/inet.h>

#include <small/Tensor.hpp>
#include <small/buffers.hpp>

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

//****************************************************************************
template <class T>
inline bool almost_equal(T v1, T v2, float rtol = 5e-03, float atol = 1e-05)
{
    float abs_diff = fabs((float)(v1) - (float)(v2));
    float diff_tolerance = (atol + rtol * fabs(v2));
    return (abs_diff <= diff_tolerance);

    // float A = fabs((float)v1);
    // float B = fabs((float)v2);
    // float largest = (B > A) ? B : A;
    // return (abs_diff <= largest*FLT_EPSILON);

    // original checker
    // return (tol > fabs(v1 - v2));
}

//****************************************************************************
template <class BufferT>
inline void compute_mean_var(small::shape_type const &output_shape,
                             BufferT const &output_dc_answers,
                             BufferT &running_mean,
                             BufferT &running_var)
{
    auto Co(output_shape[small::CHANNEL]);
    auto Ho(output_shape[small::HEIGHT]);
    auto Wo(output_shape[small::WIDTH]);
    float num_vals = (float)(Ho * Wo);
    for (size_t co = 0; co < Co; ++co)
    {
        running_mean[co] = 0.f;
        running_var[co] = 0.f;
        for (size_t h = 0; h < Ho; ++h)
        {
            for (size_t w = 0; w < Wo; ++w)
            {
                running_mean[co] +=
                    output_dc_answers[co*Ho*Wo + h*Wo +w]/num_vals;
            }
        }

        for (size_t h = 0; h < Ho; ++h)
        {
            for (size_t w = 0; w < Wo; ++w)
            {
                running_var[co] +=
                    (output_dc_answers[co*Ho*Wo + h*Wo +w] - running_mean[co]) *
                    (output_dc_answers[co*Ho*Wo + h*Wo +w] - running_mean[co]) /
                    num_vals;
            }
        }
        //std::cerr << co << "mean, var: " << running_mean[co]
        //      << ", " << running_var[co] << std::endl;
    }
}

//****************************************************************************
/// @todo move to small::utils
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
    bool use_Co((layer_name != std::string("dw_conv")) && (params.C_o != 0));
    std::string fname =
        directory + "/" + buffer_name + "__" + layer_name +
        "_Ci" + std::to_string(params.C_i) +
        "_H" + std::to_string(params.H) +
        "_W" + std::to_string(params.W) +
        "_k" + std::to_string(params.k) +
        "_s" + std::to_string(params.s) +
        ((params.p == small::PADDING_V) ? "_v" : "_f") +
        (use_Co ? ("_Co" + std::to_string(params.C_o)) : "") +
        "_" +  std::to_string(num_elements) + ".bin";

    return fname;
}

//****************************************************************************
/// reads a binary file assuming all the data is floats
/// @todo: change float buffer when buffer types are merged
small::FloatBuffer read_binary(std::string const &data_file)
{
    std::filesystem::path data_path(data_file);

    // total size of weights in bytes
    size_t total_bytes = std::filesystem::file_size(data_path);

    std::ifstream data_in(data_file, std::ios::binary);
    small::FloatBuffer buffer(
        total_bytes / sizeof(typename small::FloatBuffer::value_type));
    data_in.read(reinterpret_cast<char*>(buffer.data()), total_bytes);
    data_in.close();

    return buffer;
}

#if 0
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
#endif

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
template <class BufferT>
BufferT read_inputs(std::string const &input_data_fname)
{
    uint32_t in_n, num_elts;
    std::ifstream ifs(input_data_fname, std::ios::binary);
    if (!ifs)
    {
        std::cerr << "ERROR: failed to open file: " << input_data_fname
                  << std::endl;
        throw std::invalid_argument("read_inputs ERROR: failed to open file.");
    }

    // TODO: check if there are endian issues for cross platform
    ifs.read(reinterpret_cast<char*>(&in_n), sizeof(uint32_t));
    num_elts = ntohl(in_n);

    if (num_elts < 1)
    {
        std::cerr << "ERROR: invalid number of elements in "
                  << input_data_fname << std::endl;
        throw std::invalid_argument("read_inputs ERROR: invalid num elements.");
    }

    BufferT in_buf(num_elts);
    ifs.read(reinterpret_cast<char*>(in_buf.data()),
             num_elts*sizeof(typename BufferT::value_type));

    return in_buf;
}

//****************************************************************************
///   - first 4 bytes is num_elements as uint32_t in network order
///   - the rest of file are the 4 byte floating point numbers
///
template <class BufferT>
void write_outputs(std::string const &output_data_fname,
                   BufferT const &out_buf,
                   uint32_t num_elts)
{
    std::ofstream ofs(output_data_fname, std::ios::binary);
    if (!ofs)
    {
        std::cerr << "ERROR: failed to open file: " << output_data_fname
                  << std::endl;
        throw std::invalid_argument("write_outputs ERROR: failed to open file.");
    }

    // TODO: check if there are endian issues for cross platform
    uint32_t out_n;
    out_n = htonl(num_elts);
    ofs.write(reinterpret_cast<char*>(&out_n), sizeof(uint32_t));

    if (num_elts < 1)
    {
        std::cerr << "ERROR: invalid number of elements in "
                  << output_data_fname << std::endl;
        throw std::invalid_argument("write_outputs ERROR: invalid num elements.");
    }

    ofs.write(reinterpret_cast<char const*>(out_buf.data()),
              num_elts*sizeof(typename BufferT::value_type));
}

#if 0
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

    //std::cerr << "Reading " << num_elts << " elements from "
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
#endif

//****************************************************************************
//logging
//****************************************************************************

template <class T>
void print_stats(std::vector<T> v, const char *benchmark)
{
    if (v.size() != 0)
    {
        T sum = std::reduce(v.begin(), v.end());
        double mean = sum / (double)v.size();
        T min_elem = *min_element(v.begin(), v.end());
        T max_elem = *max_element(v.begin(), v.end());
        std::cout << benchmark << ": #runs = " << v.size()
                  << ", Min = " << min_elem
                  << ", Max = " << max_elem
                  << ", Avg = " << mean
                  << std::endl;
    }
}
