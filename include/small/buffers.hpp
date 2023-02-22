/*
 * SMaLL Framework
 *
 * Copyright 2023 Carnegie Mellon University and Authors.
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

/// @todo OBE: Use an enum for padding  enum Padding { PAD_FULL, PAD_VALID };
/// @todo Put interface in small namespace, details in small::detail namespace
/// @todo Consider changing to unsigned integer types for dimensions
/// @todo How should errors be reported (throw exceptions, return codes?)
/// @todo add interface documentation for possible errors

#pragma once

#include <vector>
#include <iostream>  // Temporary for debugging only
#include <exception>

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

    //************************************************************************
    // Convert from NCHW Torch tensor format to Direct Convolution format
    // If input tensor
    //    [1, C, H, W] --> [1, (C/_C_ib), H, W, _C_ib]
    // If output tensor
    //    [1, K, H, W] --> [1, (K/_C_ob), H, W, _C_ob]
    // If filter
    //    _C_ob = _K_b * _G_b
    //    If layer type == "conv" OR layer type == "fully connected"
    //       [K, C, F_h, F_w] --> [(K/_C_ob), (C/_C_ib), F_h, F_w, _C_ib, _C_ob]
    //    If layer type == "dw"
    //       [C, 1, F_h, F_w] --> [(C/_C_ob), F_h, F_w, 1, _C_ob]
    //************************************************************************
    /// @todo templatize on buffer type only
    //template <uint32_t _C_ob, uint32_t _C_ib>
    template <class ScalarT>
    uint32_t convert_tensor2dc(ScalarT               const *flat_t,
                               BufferTypeEnum               type,
                               uint32_t                     dim0, // C_o
                               uint32_t                     dim1, // C_i
                               uint32_t                     dim2, // H
                               uint32_t                     dim3, // W
                               uint32_t                     _C_ib,
                               uint32_t                     _C_ob,
                               ScalarT                     *dc_array)
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

    //************************************************************************
    template <class ScalarT>
    uint32_t pack_buffer(Buffer<ScalarT>  const &flat_t,
                         BufferTypeEnum          type,
                         uint32_t                dim0, // C_o
                         uint32_t                dim1, // C_i
                         uint32_t                dim2, // H
                         uint32_t                dim3, // W
                         uint32_t                _C_ib,
                         uint32_t                _C_ob,
                         Buffer<ScalarT>        &dc_array)
    {
        return convert_tensor2dc<ScalarT>(
            flat_t.data(),
            type,
            dim0, dim1, dim2, dim3,
            _C_ib, _C_ob,
            dc_array.data());
    }

    //************************************************************************
    template <class ScalarT>
    uint32_t convert_dc2tensor(ScalarT               const *dc_array,
                               BufferTypeEnum               type,
                               uint32_t                     dim0, // C_o
                               uint32_t                     dim1, // C_i
                               uint32_t                     dim2, // H
                               uint32_t                     dim3, // W
                               uint32_t                     _C_ib,
                               uint32_t                     _C_ob,
                               ScalarT                     *flat_t)
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

    //************************************************************************
    template <class ScalarT>
    uint32_t unpack_buffer(Buffer<ScalarT>  const &dc_array,
                           BufferTypeEnum          type,
                           uint32_t                dim0, // C_o
                           uint32_t                dim1, // C_i
                           uint32_t                dim2, // H
                           uint32_t                dim3, // W
                           uint32_t                _C_ib,
                           uint32_t                _C_ob,
                           Buffer<ScalarT>        &flat_t)
    {
        return convert_dc2tensor<ScalarT>(
            dc_array.data(),
            type,
            dim0, dim1, dim2, dim3,
            _C_ib, _C_ob,
            flat_t.data());
    }

} // namespace small
