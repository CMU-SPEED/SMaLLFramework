//-----------------------------------------------------------------------------
// test/test_utils.hpp
//-----------------------------------------------------------------------------

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
template <class T>
inline bool almost_equal(T v1, T v2, float tol = 1.0e-8)
{
    return (1e-8 > fabs(v1 - v1));
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

//****************************************************************************
//****************************************************************************

namespace small
{

enum BufferTypeEnum
{
    INPUT,       // was 'i'
    OUTPUT,      // was 'o'
    FILTER_DW,   // was 'd', for LAYER = DW_CONV
    FILTER_CONV, // was 'f', for LAYER = CONV, PARTIAL_CONV
    FILTER_FC    // was 'l', for LAYER = FC
};

//****************************************************************************
// Convert from NCHW Torch tensor formate to Direct Convolution format
// If input tensor
//      [1, C, H, W] --> [1, (C/_C_ib), H, W, _C_ib]
// If output tensor
//      [1, K, H, W] --> [1, (K/_C_ob), H, W, _C_ob]
// If filter
//     _C_ob = _K_b * _G_b
//      If layer type == "conv" OR layer type == "fully connected"
//          [K, C, F_h, F_w] --> [(K/_C_ob), (C/_C_ib), F_h, F_w, _C_ib, _C_ob]
//      If layer type == "dw"
//          [C, 1, F_h, F_w] --> [(C/_C_ob), F_h, F_w, 1, _C_ob]
//****************************************************************************
/// @todo templatize on buffer type only
//template <uint32_t _C_ob, uint32_t _C_ib>
template <class DataT>
uint32_t convert_tensor2dc(DataT                 const *flat_t,
                           BufferTypeEnum               type,
                           uint32_t                     dim0, // C_o
                           uint32_t                     dim1, // C_i
                           uint32_t                     dim2, // H
                           uint32_t                     dim3, // W
                           uint32_t                     _C_ib,
                           uint32_t                     _C_ob,
                           DataT                       *dc_array)
{
    uint32_t dim_3, dim_2, dim_1, dim_0;
    uint32_t ip_block, op_block;

    dim_3 = dim0;
    dim_2 = dim1;
    dim_1 = dim2;
    dim_0 = dim3;

    // =============  Overrides for specific filter types ==============
    if(type == FILTER_FC)
    {
        dim_1 = 1;
        dim_0 = 1;
    }

    if (type == FILTER_DW)
    {
        //override _C_ib
        _C_ib = 1;
    }

    if (type == FILTER_CONV)
    {
        if (dim1 < _C_ob)
        {
            //std::cerr << "HERE: dim1, C_ob: " << dim_1 << ", " << _C_ob << std::endl;;
            _C_ib = 3;    /// @todo why is this a 3?
        }
    }

    if (type == INPUT)
    {
        // input
        ip_block = _C_ib;
        op_block = 1;
    }
    else if (type == OUTPUT)
    {
        // output
        ip_block = _C_ob;
        op_block = 1;
    }
    else if (type == FILTER_CONV || type == FILTER_DW || type == FILTER_FC)
    {
        // filter
        ip_block = _C_ib;
        op_block = _C_ob;
    }
    else
    {
        /// @todo throw or return error code.
        printf("ERROR: unsupported tensor buffer type\n");
        return 0;
    }

    //fprintf(stderr, "copying tensor %d %d %d %d  --> %d %d %d %d %d %d\n",
    //        dim_3, dim_2, dim_1, dim_0,
    //        dim_3/op_block, dim_2/ip_block, dim_1, dim_0, ip_block, op_block);

    // copying
    uint32_t offset = 0;
    for (uint32_t g = 0; g < dim_3; g += op_block)
    {
        uint32_t g_offset = g * dim_2 * dim_1 * dim_0;
        for (uint32_t h = 0; h < dim_2; h += ip_block)
        {
            uint32_t h_offset = h * dim_1 * dim_0;
            for (uint32_t i = 0; i < dim_1; i++)
            {
                uint32_t i_offset = i * dim_0;
                for (uint32_t j = 0; j < dim_0; j++)
                {
                    uint32_t j_offset = j;
                    for (uint32_t k = 0; k < ip_block; k++)
                    {
                        uint32_t k_offset = k * dim_1 * dim_0;
                        for (uint32_t l = 0; l < op_block; l++)
                        {
                            int l_offset = l * dim_2 * dim_1 * dim_0;
                            //printf("offset: %d\n", offset);fflush(0);
                            //std::cerr << "dst index = " << offset << ", src index = "
                            //          << (g_offset + l_offset +
                            //              h_offset + k_offset +
                            //              i_offset +
                            //              j_offset)
                            //          << std::endl;
                            dc_array[offset++] =
                                flat_t[g_offset + l_offset +
                                       h_offset + k_offset +
                                       i_offset +
                                       j_offset];
                        }
                    }
                }
            }
        }
    }
    return dim_3*dim_2*dim_1*dim_0;
}

//****************************************************************************
template <class DataT>
uint32_t convert_dc2tensor(DataT                 const *dc_array,
                           BufferTypeEnum               type,
                           uint32_t                     dim0, // C_o
                           uint32_t                     dim1, // C_i
                           uint32_t                     dim2, // H
                           uint32_t                     dim3, // W
                           uint32_t                     _C_ib,
                           uint32_t                     _C_ob,
                           DataT                       *flat_t)
{
    uint32_t dim_3, dim_2, dim_1, dim_0;
    uint32_t ip_block, op_block;

    dim_3 = dim0;
    dim_2 = dim1;
    dim_1 = dim2;
    dim_0 = dim3;

    // =============  Overrides for specific filter types ==============
    if(type == FILTER_FC)
    {
        dim_1 = 1;
        dim_0 = 1;
    }

    if (type == FILTER_DW)
    {
        //override _C_ib
        _C_ib = 1;
    }

    if (type = FILTER_CONV)
    {
        if (dim_1 < _C_ob)
            _C_ib = 3;    /// @todo why is this a 3?
    }

    if (type == INPUT)
    {
        // input
        ip_block = _C_ib;
        op_block = 1;
    }
    else if (type == OUTPUT)
    {
        // output
        ip_block = _C_ob;
        op_block = 1;
    }
    else if (type == FILTER_CONV || type == FILTER_DW || type == FILTER_FC)
    {
        // filter
        ip_block = _C_ib;
        op_block = _C_ob;
    }
    else
    {
        /// @todo throw or return error code.
        printf("ERROR: unsupported tensor buffer type\n");
        return 0;
    }

    //fprintf(stderr, "copying tensor %d %d %d %d  --> %d %d %d %d %d %d\n",
    //        dim_3, dim_2, dim_1, dim_0,
    //        dim_3/op_block, dim_2/ip_block, dim_1, dim_0, ip_block, op_block);

    // copying
    uint32_t offset = 0;
    for (uint32_t g = 0; g < dim_3; g += op_block)
    {
        uint32_t g_offset = g * dim_2 * dim_1 * dim_0;
        for (uint32_t h = 0; h < dim_2; h += ip_block)
        {
            uint32_t h_offset = h * dim_1 * dim_0;
            for (uint32_t i = 0; i < dim_1; i++)
            {
                uint32_t i_offset = i * dim_0;
                for (uint32_t j = 0; j < dim_0; j++)
                {
                    uint32_t j_offset = j;
                    for (uint32_t k = 0; k < ip_block; k++)
                    {
                        uint32_t k_offset = k * dim_1 * dim_0;
                        for (uint32_t l = 0; l < op_block; l++)
                        {
                            int l_offset = l * dim_2 * dim_1 * dim_0;
                            // printf("offset: %d\n", offset);fflush(0);
                            //std::cerr << "dst index = " << offset << ", src index = "
                            //          << (g_offset + l_offset +
                            //              h_offset + k_offset +
                            //              i_offset +
                            //              j_offset)
                            //          << std::endl;
                            flat_t[g_offset + l_offset +
                                   h_offset + k_offset +
                                   i_offset +
                                   j_offset] = dc_array[offset++];
                        }
                    }
                }
            }
        }
    }
    return dim_3*dim_2*dim_1*dim_0;
}

} // namespace small
