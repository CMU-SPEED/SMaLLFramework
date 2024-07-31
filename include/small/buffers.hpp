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
#include <iostream>  // Temporary for debugging only
#include <cassert>
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
// Index a packed direct_convolution weights tensor for Conv2D layers
inline size_t packed_weight_index(
    // unpacked shape
    uint32_t num_filters,      // num output channels
    uint32_t num_channels,     // num input channels
    uint32_t num_rows,         // kernel height
    uint32_t num_cols,         // kernel width

    // packing params (platform-specific)
    uint32_t filter_blocking,  // pass C_ob
    uint32_t channel_blocking, // pass C_ib

    // unpacked location
    size_t   filter_idx,       // output channel
    size_t   channel_idx,      // input channel
    size_t   height_idx,       // kernel height index
    size_t   width_idx)        // kernel width index
{

    if(num_channels < channel_blocking) {
        if(num_channels == 3) {
            channel_blocking = 3;
        }
        else {
            throw std::runtime_error("num_channels < channel_blocking && num_channels != 3");
        }
    }
    else if(num_channels % channel_blocking != 0) {
        throw std::runtime_error("num_channels %% channel_blocking != 0");
    }

    return (
        (filter_idx/filter_blocking) * (
            (num_channels/channel_blocking) *
            num_rows * num_cols * channel_blocking * filter_blocking) +
        (channel_idx/channel_blocking) * (
            num_rows * num_cols * channel_blocking * filter_blocking) +
        height_idx  * (num_cols * channel_blocking * filter_blocking) +
        width_idx *              (channel_blocking * filter_blocking) +
        (channel_idx % channel_blocking * filter_blocking) +
        (filter_idx % filter_blocking));
}

//************************************************************************
// Index a packed input or output buffer (assuming NCHW = [1, C_i/o, H, W])
inline size_t packed_buffer_index(
    // unpacked shape
    uint32_t num_channels,     // num input/output channels
    uint32_t num_rows,         // image height
    uint32_t num_cols,         // image width

    // packing params (platform-specific)
    uint32_t channel_blocking, // pass C_ib for input, C_ob for output buffers

    // unpacked location
    size_t   channel_idx,      // input/output channel
    size_t   height_idx,       // kernel height index
    size_t   width_idx)        // kernel width index
{
    if(num_channels < channel_blocking) {
        if(num_channels == 3) {
            channel_blocking = 3;
        }
        else {
            throw std::runtime_error("num_channels < channel_blocking && num_channels != 3");
        }
    }
    else if(num_channels % channel_blocking != 0) {
        throw std::runtime_error("num_channels %% channel_blocking != 0");
    }

    return (
        (channel_idx/channel_blocking) * (num_rows * num_cols * channel_blocking) +
                            height_idx * (           num_cols * channel_blocking) +
                             width_idx * (                      channel_blocking) +
        (channel_idx % channel_blocking));
}

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
                           uint32_t                     C_o, //dim0
                           uint32_t                     C_i, //dim1
                           uint32_t                     H,   //dim2
                           uint32_t                     W,   //dim3
                           uint32_t                     _C_ib,  /// @todo remove from public API
                           uint32_t                     _C_ob,  /// @todo remove from public API
                           ScalarT                     *dc_array)
{
    //uint32_t C_o, C_i, H, W;
    uint32_t ip_block, op_block;

    //C_o = dim0;
    //C_i = dim1;
    //H = dim2;
    //W = dim3;

    // =============  Overrides for specific filter types ==============
    if(type == FILTER_FC)
    {
        H = 1;
        W = 1;
    }

    if (type == FILTER_DW)
    {
        //override _C_ib
        _C_ib = 1;
    }

    if (type == FILTER_CONV || type == INPUT)
    {
        /// @todo this is fragile code.  There is one hardcoded exception
        ///       3-channel input images.  In all other cases C_i must be an
        ///       integer multiple of _C_ib.
        if (C_i < _C_ib) //(dim1 < _C_ob)
        {
            assert(C_i == 3);
            //std::cerr << "HERE: dim1, C_ob: " << H << ", " << _C_ob << std::endl;;
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
    //        C_o, C_i, H, W,
    //        C_o/op_block, C_i/ip_block, H, W, ip_block, op_block);

    // copying
    uint32_t offset = 0;
    for (uint32_t g = 0; g < C_o; g += op_block)
    {
        uint32_t g_offset = g * C_i * H * W;
        for (uint32_t h = 0; h < C_i; h += ip_block)
        {
            uint32_t h_offset = h * H * W;
            for (uint32_t i = 0; i < H; i++)
            {
                uint32_t i_offset = i * W;
                for (uint32_t j = 0; j < W; j++)
                {
                    uint32_t j_offset = j;
                    for (uint32_t k = 0; k < ip_block; k++)
                    {
                        uint32_t k_offset = k * H * W;
                        for (uint32_t l = 0; l < op_block; l++)
                        {
                            int l_offset = l * C_i * H * W;
                            //printf("offset: %d\n", offset);fflush(0);
                            //std::cerr << "dst index = " << offset << ", src index = "
                            //          << (g_offset + l_offset +
                            //              h_offset + k_offset +
                            //              i_offset +
                            //              j_offset)
                            //          << std::endl;

                            uint32_t offset_calc = 0;
                            if(type == INPUT || type == OUTPUT) {
                                offset_calc = packed_buffer_index(C_i, H, W, _C_ib, h+k, i, j);
                            }
                            else {
                                offset_calc = packed_weight_index(C_o, C_i, H, W, _C_ob, _C_ib, g+l, h+k, i, j);
                            }

                            if(offset != offset_calc) {
                                printf("ERROR: offset mismatch: %d != %d\n", offset, offset_calc);
                                return 0;
                            }

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
    return C_o*C_i*H*W;
}

//************************************************************************
template <class BufferT>
uint32_t pack_buffer(BufferT          const &flat_t,
                     BufferTypeEnum          type,
                     uint32_t                dim0, // C_o
                     uint32_t                dim1, // C_i
                     uint32_t                dim2, // H
                     uint32_t                dim3, // W
                     uint32_t                _C_ib,  /// @todo remove from public API
                     uint32_t                _C_ob,  /// @todo remove from public API
                     BufferT                &dc_array)
{
    return convert_tensor2dc<typename BufferT::value_type>(
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
                           uint32_t                     C_o, //dim0
                           uint32_t                     C_i, //dim1
                           uint32_t                     H,   //dim2
                           uint32_t                     W,   //dim3
                           uint32_t                     _C_ib,  /// @todo remove from public API
                           uint32_t                     _C_ob,  /// @todo remove from public API
                           ScalarT                     *flat_t)
{
    //uint32_t C_o, C_i, H, W;
    //uint32_t _C_ib = C_ib;
    //uint32_t _C_ob = C_ob;
    uint32_t ip_block, op_block;

    // =============  Overrides for specific filter types ==============
    if(type == FILTER_FC)
    {
        H = 1;
        W = 1;
    }

    if (type == FILTER_DW)
    {
        //override _C_ib
        _C_ib = 1;
    }

    if ((type == FILTER_CONV) || (type == INPUT))
    {
        /// @todo this is fragile code.  There is one hardcoded exception
        ///       3-channel input images.  In all other cases C_i must be an
        ///       integer multiple of _C_ib.
        if (C_i < _C_ib) //(H < _C_ob)
        {
            assert(C_i == 3);

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
    //        C_o, C_i, H, W,
    //        C_o/op_block, C_i/ip_block, H, W, ip_block, op_block);

    // copying
    uint32_t offset = 0;
    for (uint32_t g = 0; g < C_o; g += op_block)
    {
        uint32_t g_offset = g * C_i * H * W;
        for (uint32_t h = 0; h < C_i; h += ip_block)
        {
            uint32_t h_offset = h * H * W;
            for (uint32_t i = 0; i < H; i++)
            {
                uint32_t i_offset = i * W;
                for (uint32_t j = 0; j < W; j++)
                {
                    uint32_t j_offset = j;
                    for (uint32_t k = 0; k < ip_block; k++)
                    {
                        uint32_t k_offset = k * H * W;
                        for (uint32_t l = 0; l < op_block; l++)
                        {
                            int l_offset = l * C_i * H * W;
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
    return C_o*C_i*H*W;
}

//************************************************************************
template <class BufferT>
uint32_t unpack_buffer(BufferT          const &dc_array,
                       BufferTypeEnum          type,
                       uint32_t                dim0, // C_o
                       uint32_t                dim1, // C_i
                       uint32_t                dim2, // H
                       uint32_t                dim3, // W
                       uint32_t                _C_ib,  /// @todo remove from public API
                       uint32_t                _C_ob,  /// @todo remove from public API
                       BufferT                &flat_t)
{
    return convert_dc2tensor<typename BufferT::value_type>(
        dc_array.data(),
        type,
        dim0, dim1, dim2, dim3,
        _C_ib, _C_ob,
        flat_t.data());
}

//****************************************************************************
/// @todo random initializers are for testing only and should be moved to
///       demo and/or test directories
/// @todo rename to init_random(). Accept an optional seed?
template <class BufferT>
void init(BufferT &ptr, uint32_t numel)
{
    if (numel > ptr.size())
    {
        throw std::invalid_argument("init ERROR: buffer too small.");
    }

    if constexpr (std::is_same<float, typename BufferT::value_type>::value)
    {
        srand(42);
        float *cur_ptr = ptr.data();
        for (size_t i = 0 ; i < numel ; i++)
        {
            *(cur_ptr++) = 2.0*((float) rand()/ RAND_MAX) - 1;
        }
    }
    else if constexpr (std::is_same<uint8_t, typename BufferT::value_type>::value)
    {
        srand(42);
        uint8_t *cur_ptr = ptr.data();
        for (uint32_t i = 0; i < numel; i++)
        {
            *(cur_ptr++) = rand()  % 10;
        }
    }
    else
    {
        throw std::invalid_argument("init ERROR: unsupported type.");
    }
}

//****************************************************************************
template <class BufferT>
void init_ones(BufferT &ptr, uint32_t numel)
{
    using ScalarT = typename BufferT::value_type;

    if (numel > ptr.size())
    {
        throw std::invalid_argument("init_ones ERROR: buffer too small.");
    }

    ScalarT *cur_ptr = ptr.data();
    for (size_t i = 0; i < numel; i++)
    {
        *(cur_ptr++) = (ScalarT)1;
    }
}

//****************************************************************************
template <class BufferT>
void init_zeros(BufferT &ptr, uint32_t numel)
{
    using ScalarT = typename BufferT::value_type;

    if (numel > ptr.size())
    {
        throw std::invalid_argument("init_zeroes ERROR: buffer too small.");
    }

    ScalarT *cur_ptr = ptr.data();
    for (size_t i = 0; i < numel; i++)
    {
        *(cur_ptr++) = (ScalarT)0;
    }
}

//****************************************************************************
template<class BufferT, uint32_t _C_ob>
void init_arange(BufferT &ptr, uint32_t H, uint32_t W, uint32_t C)
{
    using ScalarT = typename BufferT::value_type;

    if (C * H * W > ptr.size())
    {
        throw std::invalid_argument(
            "init_arange ERROR: buffer too small.");
    }

    ScalarT *cur_ptr = ptr.data();
    for (size_t i = 0; i < C; i+=_C_ob)
    {
        for (size_t j = 0 ; j < H; j++)
        {
            for (size_t k = 0; k < W; k++)
            {
                for (size_t ii = 0; ii < _C_ob; ii++)
                {
                    /// @todo Should this be: ii*i + k*C + j*(W*C) ??
                    *(cur_ptr++) =  (ScalarT)(ii + i + k*(C) + j*(W*C));
                }
            }
        }
    }
}

//****************************************************************************
/// @todo is this valid with types other than float or double?
template <class BufferT,
          typename = std::enable_if<
              std::is_same<float, typename BufferT::value_type>::value> >
void init_norm(BufferT &ptr, uint32_t numel, uint32_t C_o)
{
    if (numel > ptr.size())
    {
        throw std::invalid_argument("init_norm ERROR: buffer too small.");
    }

    if constexpr (std::is_same<float, typename BufferT::value_type>::value)
    {
        float *cur_ptr = ptr.data();
        float norm = (1.0*C_o)/(1.0*ptr.size());
        for (size_t i = 0; i < ptr.size(); i++)
        {
            *(cur_ptr++) = norm;
        }
    }
    else if constexpr (std::is_same<uint8_t, typename BufferT::value_type>::value)
    {
        // came from quantized_reference/utils.h
        /// @todo does not look right if dtype is uint8_t
        using dtype = typename BufferT::value_type;

        dtype *cur_ptr = ptr.data();
        dtype norm = (1.0*C_o)/(1.0*numel);
        for (uint32_t i = 0; i < numel; i++)
        {
            *(cur_ptr++) = norm;
        }
    }
    else
    {
        throw std::invalid_argument("init_norm ERROR: unsupported type.");
    }

}

//****************************************************************************
// tolerance based on absolute difference
template <class BufferT>
bool equals(uint32_t numel,
            BufferT const &buf1,
            BufferT const &buf2,
            float tolerance = 1.0e-8)
{
    using ScalarT = typename BufferT::value_type;
    ScalarT const *buf1_ptr = buf1.data();
    ScalarT const *buf2_ptr = buf2.data();

    if ((buf1.size() < numel) ||
        (buf2.size() < numel))
    {
        return false;
    }

    bool check = true;
    for (size_t i = 0; i < numel; i++)
    {
        if constexpr (std::is_same<float, typename BufferT::value_type>::value)
        {
            ScalarT diff = *(buf2_ptr) - *(buf1_ptr);
            // printf("equals      : %ld, buf2/buf1 %.4f/%.4f, diff %.4f\n",
            //        i, *(buf2_ptr), *(buf1_ptr), diff);

            if (fabs(diff) > tolerance)
            {
                printf("equals ERROR: %ld, buf2/buf1 %.4f/%.4f, diff %.4f\n",
                       i, *(buf2_ptr), *(buf1_ptr), diff);
                check = false;
            }
        }
        else if constexpr (std::is_same<uint8_t, typename BufferT::value_type>::value)
        {
            if (*(buf2_ptr) != *(buf1_ptr))
            {
                std::cout << "equals ERROR: i=" << i << ", buf1[i]=" << *(buf1_ptr)
                          << ", buf2[i]=" << *(buf2_ptr) << std::endl;
                //printf("equals ERROR: %ld, buf2/buf1 %.4f/%.4f, diff %.4f\n",
                //       i, *(buf2_ptr), *(buf1_ptr), diff);
                check = false;
            }
        }
        buf1_ptr++;
        buf2_ptr++;
    }
    return check;
}


} // namespace small
