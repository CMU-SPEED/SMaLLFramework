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

#include <acutest.h>

#include <small.h>

#include "test_utils.hpp"

//****************************************************************************
template <class BufferT>
bool test_input_packing(uint32_t C_i, uint32_t H, uint32_t W)
{
    std::cout << "Testing input packing (C_i = " << C_i
              << ", H = " << H << ", W = " << W << ")\n";
    uint32_t numel = C_i*H*W;
    BufferT input(numel);
    small::init(input, numel);
    BufferT packed_in(numel);
    uint32_t status = small::pack_buffer(
        input, small::INPUT, 1, C_i, H, W,
        BufferT::C_ib, BufferT::C_ob, packed_in);
    if (status == 0)
    {
        return false;
    }
    return true;
}

//****************************************************************************
template <class BufferT>
bool test_filt_packing(uint32_t C_o, uint32_t C_i, uint32_t H, uint32_t W)
{
    std::cout << "Testing filter packing (C_o = " << C_o << ", C_i = " << C_i
              << ", H = " << H << ", W = " << W << ")\n";
    uint32_t numel = C_o*C_i*H*W;
    BufferT filt(numel);
    small::init(filt, numel);
    BufferT packed_filt(numel);
    uint32_t status = small::pack_buffer(
        filt, small::FILTER_CONV, C_o, C_i, H, W,
        BufferT::C_ib, BufferT::C_ob, packed_filt);
    if (status == 0)
    {
        return false;
    }
    return true;
}

//****************************************************************************
template <class BufferT>
bool test_output_packing(uint32_t C_o, uint32_t H, uint32_t W)
{
    std::cout << "Testing output packing (C_o = " << C_o
              << ", H = " << H << ", W = " << W << ")\n";
    uint32_t numel = C_o*H*W;
    BufferT output(numel);
    small::init(output, numel);
    BufferT packed_out(numel);
    uint32_t status = small::pack_buffer(
        output, small::OUTPUT, 1, C_o, H, W,
        BufferT::C_ib, BufferT::C_ob, packed_out);
    if (status == 0)
    {
        return false;
    }
    return true;
}

//****************************************************************************
void test_packing(void)
{
    /// @todo Test QUInt8Buffer too
    std::cout << "\n";

    TEST_CHECK(test_input_packing<small::FloatBuffer>(3, 96, 96));
    TEST_CHECK(test_input_packing<small::FloatBuffer>(
                   small::FloatBuffer::C_ib, 416, 416));
    TEST_CHECK(test_input_packing<small::FloatBuffer>(
                   3*small::FloatBuffer::C_ib, 416, 416));

    TEST_CHECK(test_filt_packing<small::FloatBuffer>(256, 3, 3, 3));
    TEST_CHECK(test_filt_packing<small::FloatBuffer>(256, 3, 1, 1));

    TEST_CHECK(test_output_packing<small::FloatBuffer>(16, 416, 416));
}

/// @todo test both float and quantized integer buffers eventually
#if !defined(QUANTIZED)
using Buffer = small::FloatBuffer;
#else
using Buffer = small::QUInt8Buffer;
#endif

//****************************************************************************
void test_filter_packing_indices(void)
{
    bool passed = true;
    using BufferT = small::FloatBuffer;  ///@todo HACK: hardcoded

    // C_i,Hi,Wi,k,s,p,C_o
    std::vector<LayerParams> layer_params =
    {
        {  16,   48,  48, 3, 1, small::PADDING_F,   16},
        {  32,   24,  24, 3, 1, small::PADDING_F,   32},

        {  32,   48,  48, 3, 1, small::PADDING_F,   32},
        {  64,   24,  24, 3, 1, small::PADDING_F,   64},
        { 128,   12,  12, 3, 1, small::PADDING_F,  128},

        {  16,   48,  48, 3, 1, small::PADDING_F,   32},
        {  32,   24,  24, 3, 1, small::PADDING_F,   64},
        {  64,   12,  12, 3, 1, small::PADDING_F,  128},
        { 128,    6,   6, 3, 1, small::PADDING_F,  256},

        { 128,   24,  24, 3, 1, small::PADDING_F,  128},
        { 256,   12,  12, 3, 1, small::PADDING_F,  256},

        { 512,   12,  12, 3, 1, small::PADDING_F,  512},
        {1024,    6,   6, 3, 1, small::PADDING_F, 1024},

        {  32,  208, 208, 3, 1, small::PADDING_F,   64},
        {  64,  104, 104, 3, 1, small::PADDING_F,  128},
        { 128,   52,  52, 3, 1, small::PADDING_F,  256},
        { 256,   26,  26, 3, 1, small::PADDING_F,  512},
        { 512,   13,  13, 3, 1, small::PADDING_F, 1024},

        /// @todo fix for non reference platforms or disable
//        { 3,   13,  13, 3, 1, small::PADDING_F, 16},
//        { 1,   52,  52, 3, 1, small::PADDING_F, 16}
    };

    std::vector<size_t> unpacked_to_packed_mapping;
    for (auto &[C_i, H, W, k, s, p, C_o] : layer_params)
    {
        std::cerr << "Testing indexing for Ci,H,W,C_o = "
                  << C_i << "," << H << "," << W << "," << C_o
                  << std::endl;
        size_t sz = C_i*H*W*C_o;

        unpacked_to_packed_mapping.clear();
        for (uint32_t co = 0; co < C_o; ++co)
            for (uint32_t ci = 0; ci < C_i; ++ci)
                for (uint32_t h = 0; h < H; ++h)
                    for (uint32_t w = 0; w < W; ++w)
                    {
                        size_t packed_index =
                            small::packed_weight_index(C_o, C_i, H, W,
                                                       BufferT::C_ob,
                                                       BufferT::C_ib,
                                                       co, ci, h, w);
                        if (packed_index > sz)
                        {
                            std::cerr << "ERROR: out of range: unpacked("
                                      << unpacked_to_packed_mapping.size()
                                      << "),  packed(macro) "
                                      << packed_index << " >= " << sz
                                      << std::endl;
                        }
                        unpacked_to_packed_mapping.push_back(packed_index);
                    }

        // *** Extracted from convert_tensor2dc ***
        uint32_t _C_ib = BufferT::C_ib;
        uint32_t _C_ob = BufferT::C_ob;

        if (C_i < _C_ib) //(dim1 < _C_ob)
        {
            //std::cerr << "HERE: dim1, C_ob: " << H << ", " << _C_ob << std::endl;;
            _C_ib = 3;    /// @todo why is this a 3?
        }

        uint32_t ip_block = _C_ib;
        uint32_t op_block = _C_ob;

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
                                auto idx = small::packed_weight_index(
                                    C_o, C_i, H, W,
                                    BufferT::C_ob, BufferT::C_ib,
                                    g+l, h+k, i, j);

                                size_t unpacked_index = g_offset + l_offset +
                                    h_offset + k_offset +
                                    i_offset + j_offset;
                                size_t packed_index = offset++;

                                if (packed_index != idx)
                                {
                                    passed = false;
                                    std::cerr << "ERROR: packed(macro) != packed(offset), "
                                              << idx << " != " << packed_index
                                              << std::endl;
                                    continue;
                                }

                                if (unpacked_index >= unpacked_to_packed_mapping.size())
                                {
                                    passed = false;
                                    std::cerr << "ERROR: Unpacked index = " << unpacked_index
                                              << " out of bounds (size = "
                                              << unpacked_to_packed_mapping.size()
                                              << ")\n";
                                    continue;
                                }

                                // std::cerr << "unpacked: " << unpacked_index
                                //           << ": packed(macro): "
                                //           << unpacked_to_packed_mapping[unpacked_index]
                                //           << " ?= packed(t2dc): "
                                //           << packed_index
                                //           << " ?= direct(macro): "
                                //           << idx << std::endl;

                                if (unpacked_to_packed_mapping[unpacked_index] !=
                                    packed_index)
                                {
                                    passed = false;
                                    std::cerr << "ERROR: unpacked: " << unpacked_index
                                              << ": packed(macro): "
                                              << unpacked_to_packed_mapping[unpacked_index]
                                              << " ?= packed(t2dc): "
                                              << packed_index << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    TEST_ASSERT(passed);
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"packing", test_packing},
    {"filter packing indices", test_filter_packing_indices},
    {NULL, NULL}
};
