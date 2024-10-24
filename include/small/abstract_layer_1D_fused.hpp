//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2024 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126
//****************************************************************************

#pragma once

#include <stdint.h>
#include <stdio.h>
#if PARALLEL == 1
#include <omp.h>
#endif

#include <small/op_type.hpp>
#include <small/utils.hpp>

#include <small/kernel_left_1D.hpp>
#include <small/kernel_1D.hpp>
#include <small/kernel_right_1D.hpp>

#define DEBUG 0

#define ELEMENTAL 1
#define BLOCK 2

#define PARALLEL_DIST BLOCK

namespace small
{

namespace detail
{

//****************************************************************************
// 1D fused abstract layers
//@todo: find a way to group all the template arguments for one set of kernels together.
// @todo Remove access to F_h, pad_top, pad_bottom
template <typename BufferT,
          dim_t _G_b,
          dim_t _K_b,
          dim_t _F_cb,
          dim_t _O_wb,
          dim_t _stride,
          dim_t _UNROLL,
          OpType op_type,
          int8_t op_class, //  2  (conv),  1  (dense,pool), or '0' (activation, upsample)
          bool rewrite_output,
          OpType op_fused_single_element_before = OP_NONE,
          OpType op_fused_single_element_after = OP_NONE,
          dim_t _stride_before = 1,
          dim_t _stride_after = 1>
void fused_abstract_layer_1D(
    Mapping<BufferT> *values_0, // Main Operation
    dim_t I_h,                  // Input Height      /// @todo This should be 1 or B (batch)
    dim_t I_w,                  // Input Width

    BufferT const * /*__restrict__*/ I, // Data
    BufferT * /*__restrict__*/ O)
{
    dim_t G = values_0->G;     // Output Channel Grouping
    dim_t K = values_0->K;     // Output Channels per group
    dim_t F_c = values_0->F_c; // Channel Reduction Dimension
    dim_t F_h = values_0->F_h; // Filter height
    dim_t F_w = values_0->F_w; // Filter width

    dim_t pad_top = values_0->pad_top; // Padding values
    dim_t pad_left = values_0->pad_left;
    dim_t pad_right = values_0->pad_right;
    dim_t pad_bottom = values_0->pad_bottom;
    BufferT const *__restrict__ F = values_0->F;
    BufferT const *__restrict__ F_before = values_0->F_before;
    BufferT const *__restrict__ F_after = values_0->F_after;

    using ScalarT = typename BufferT::value_type;
    using AccumT = typename BufferT::accum_type;

    // Pointers to buffers inside Buffer class
    ScalarT const *I_buf = I->data(); //__restrict__ ?

    ScalarT const *F_buf = nullptr;
    if constexpr (op_type == OP_CONV || op_type == OP_LEAKY_RELU || op_type == OP_MUL) // if (F != nullptr)
    {
        F_buf = F->data();
    }

    ScalarT *O_buf = O->data(); //__restrict__ ?

    ScalarT const *F_before_buf = nullptr;
    if constexpr (op_fused_single_element_before == OP_UPSAMPLE ||
                  op_fused_single_element_before == OP_MUL ||
                  op_fused_single_element_before == OP_LEAKY_RELU)
    {
        F_before_buf = F_before->data();
        // printf("bias_buf: %f %f %f %f\n", F_before_buf[0], F_before_buf[1], F_before_buf[2], F_before_buf[3]);
    }
    ScalarT const *F_after_buf = nullptr;
    if constexpr (op_fused_single_element_after == OP_UPSAMPLE ||
                  op_fused_single_element_after == OP_MUL ||
                  op_fused_single_element_after == OP_LEAKY_RELU)
    {
        F_after_buf = F_after->data();
    }

#if DEBUG == 1
    if (op_type == OP_CONV)
    {
        printf("conv class: %d \n", op_class);
    }
    else if (op_type == OP_MAX_POOL)
    {
        printf("pool class: %d \n", op_class);
    }
    else if (op_type == OP_RELU)
    {
        printf("activation class: %d \n", op_class);
    }
#endif

    // calculate output dimensions based on input params.
    constexpr dim_t _C_ib = _F_cb * _G_b;

    /*
     * Data layout (slowest to fastest changing dimensions):
     *    blocks of groups
     *       blocks of channels within groups
     *          blocks of weights in the same group
     *             spatial dimensions
     *                weights in the same group
     *                   weights across groups in a block
     *                      channels in a block
     *
     * I: [G/G_b,        F_c/F_cb, I_h, I_w, F_cb, G_b     ]  TODO: UPDATE FOR I_h = 1, B support
     * F: [G/G_b, K/K_b, F_c/F_cb, F_h, F_w, F_cb, G_b, K_b]
     * O: [G/G_b, K/K_b,           O_h, O_w,       G_b, K_b]
     */

    //************************************************************************
    // Deriving padding parameters

    //  To calculate offsets to next output row, next output block
    // @todo fix this in small::output_dim
    dim_t H_o_w_pad, W_o_w_pad;
    //@todo when fused, this computation goes into the kernel
    if constexpr (op_type == OP_UPSAMPLE)
    {
        if constexpr (_stride == std::numeric_limits<dim_t>::max())
        {
            H_o_w_pad = I_h;
            W_o_w_pad = I_w;
        }
        else
        {
            H_o_w_pad = I_h * _stride;
            W_o_w_pad = I_w * _stride;
        }
    }
    else
    {
        H_o_w_pad = small::output_dim((I_h + pad_top + pad_bottom),
                                      _stride, F_h);
        W_o_w_pad = small::output_dim((I_w + pad_left + pad_right),
                                      _stride, F_w);
    }
    const dim_t O_h_w_pad = H_o_w_pad;
    const dim_t O_w_w_pad = W_o_w_pad;

    dim_t t_pad_el = pad_top / _stride + (pad_top % _stride != 0);
    dim_t l_pad_el = pad_left / _stride + (pad_left % _stride != 0);

    dim_t H_full_index = t_pad_el * _stride - pad_top;
    dim_t W_full_index = l_pad_el * _stride - pad_left;

    // Full kernel output elements
    dim_t H_o, W_o_full;
    if constexpr (op_type == OP_UPSAMPLE)
    {
        H_o = H_o_w_pad;
        W_o_full = W_o_w_pad;
    }
    else
    {
        H_o = small::output_dim((I_h - H_full_index), _stride, F_h);
        W_o_full = small::output_dim((I_w - W_full_index), _stride, F_w);
    }

    // back padding elements
    dim_t H_back_index = H_full_index + _stride * (H_o);
    dim_t W_back_index = W_full_index + _stride * (W_o_full);
    dim_t b_pad_el, r_pad_el;
    if constexpr (op_type == OP_UPSAMPLE)
    {
        b_pad_el = 0;
        r_pad_el = 0;
    }
    else
    {
        b_pad_el = small::output_dim((I_h + pad_bottom - H_back_index),
                                     _stride, F_h);
        r_pad_el = small::output_dim((I_w + pad_right - W_back_index),
                                     _stride, F_w);
    }

    const dim_t O_h = H_o;
    const dim_t O_w = W_o_full;
    //************************************************************************

    // setting up microkernel specific parameters
    const dim_t O_w_full = (O_w / _O_wb) * _O_wb;
    const dim_t O_w_left = O_w - O_w_full;
    const dim_t O_hxO_w = O_h_w_pad * O_w_w_pad;

    // When the number of channels is not a multiple of blocking size
    // const dim_t K_full = (K / _K_b) * _K_b;
    // const dim_t K_left = K - K_full;

#if DEBUG == 1
    printf("\t\t I_h %d I_w %d F_C %d G %d \n", I_h, I_w, F_c, G);
    printf("\t\t O_h_pad: %d O_w_w_pad %d \n", O_h_w_pad, O_w_w_pad);
    printf("O_h %d O_w %d O_w_left %d \n", O_h, O_w_full, O_w_left);

    printf("bottom padding index into input: %d \t bottom padding elements: %d \n",
           H_back_index, b_pad_el);
    printf("no padding index into input: %d \t top padding elements: %d \n",
           H_full_index, t_pad_el);
    printf("right padding index into input: %d \t right padding elements: %d \n",
           W_back_index, r_pad_el);
    printf("no padding index into input: %d \t left padding elements: %d \n",
           W_full_index, l_pad_el);
    printf("O_w_full: %d O_w_left: %d \n", O_w_full, O_w_left);
    printf("params: F_Cb %d G_b %d K_b %d\n", _F_cb, _G_b, _K_b);
    printf("rewrite output?: %d, op type/class:  %d/%d\n",
           rewrite_output, op_type, op_class);
#endif

    // Set up parallelism for the channel loops

    //  Get total available threads
    int N = 1;
#if PARALLEL == 1
    char const *env_nt(std::getenv("OMP_NUM_THREADS"));
    if (nullptr != env_nt)
    {
        N = atoi(std::getenv("OMP_NUM_THREADS"));
    }
#endif

    int T_channel = N, T_group = 1, T_height = 1;

    // If dwise, parallelize on groups
    if (K == 1)
    {
        T_channel = 1;
        T_group = N;
    }

    // create parallel region with all threads
#if PARALLEL == 1
#pragma omp parallel num_threads(N)
#endif
    {
#if PARALLEL == 1
        auto t_id = omp_get_thread_num();
#else
        auto t_id = 0;
#endif
        dim_t height_tid = t_id % T_height;
        dim_t channel_tid = ((t_id) / (T_height)) % T_channel;
        dim_t group_tid = ((t_id / (T_channel * T_height))) % T_group;

        // block cyclic parallelism
        // loops over output channels
        index_t group_start, group_end;
        dim_t num_groups = G / _G_b;
        dim_t groups_p_thread = num_groups / T_group;
        dim_t groups_left = num_groups % T_group;
        group_start = groups_p_thread * group_tid +
            (group_tid <= groups_left) * (group_tid) +
            (group_tid > groups_left) * groups_left;
        group_end = group_start + groups_p_thread +
            (1) * (group_tid < groups_left);

        index_t channels_start, channels_end;
        dim_t num_channels = K / _K_b;
        dim_t channels_p_thread = num_channels / T_channel;
        dim_t channels_left = num_channels % T_channel;
        channels_start = channels_p_thread * channel_tid +
            (channel_tid <= channels_left) * (channel_tid) +
            (channel_tid > channels_left) * channels_left;
        channels_end = channels_start + channels_p_thread +
            (1) * (channel_tid < channels_left);

        // for (index_t g = group_tid; g < G / _G_b; g += T_group)
        for (index_t g = group_start; g < group_end; g++)
        {
            ScalarT const *I_group;
            if constexpr (op_type == OP_UPSAMPLE && _stride == std::numeric_limits<dim_t>::max())
            {
                I_group = I_buf + g * (F_c * 1 * 1 * _G_b);
            }
            else
            {
                I_group = I_buf + g * (F_c * I_h * I_w * _G_b);
            }
            ScalarT *O_group = O_buf + g * (K * O_hxO_w * _G_b);
            // if leaky relu, the weight pointer does not change with the group id

            ScalarT const *F_group;
            if constexpr ((op_type == OP_LEAKY_RELU) || (op_type == OP_MUL))
            {
                F_group = F_buf;
            }
            else
            {
                F_group = F_buf + g * (K * F_c * F_h * F_w * _G_b);
            }

// resuse O_group as a uint32_t array
#if PARALLEL_DIST == ELEMENTAL
            for (index_t k = channel_tid; k < K / _K_b; k += T_channel)
#else
            for (index_t k = channels_start; k < channels_end; k++)
#endif
            {
                ScalarT const *I_channel_block_output =
                    I_group + 0;
                ScalarT const *F_channel_block_output =
                    F_group + k * (F_c * F_h * F_w * _G_b * _K_b);
                ScalarT       *O_channel_block_output =
                    O_group + k * (O_hxO_w * _G_b * _K_b);

                //@todo fix the filter height and width as necessary (they should be one because it is a single element reduction)

                ScalarT const *F_before_buf_group = F_before_buf + (g * K + k) * (1 * 1 * _G_b * _K_b);
                ScalarT const *F_after_buf_group = F_after_buf + (g * K + k) * (1 * 1 * _G_b * _K_b);

                //************************************************************
                // Loop over input channel reduction
                for (index_t i = 0; i < (F_c / _F_cb) - 1; i++)
                {
                    bool first = rewrite_output && (i == 0);

                    ScalarT const *I_channel_block_input =
                        I_channel_block_output + i * (I_h * I_w * _F_cb * _G_b);
                    ScalarT const *F_channel_block_input =
                        F_channel_block_output + i * (F_h * F_w * _F_cb * _G_b * _K_b);
                    ScalarT *O_channel_block_input =
                        O_channel_block_output + 0;

                    // Loops over spatial dimensions of output

                    // Prologue with top padding
                    ScalarT const *I_row_top = I_channel_block_input;
                    //ScalarT const *F_row_top = F_channel_block_input + 0;
                    AccumT        *O_row_top = O_channel_block_input;  // ScalarT --> AccumT

                    // kernel_top<ScalarT, AccumT,
                    //            _G_b, _K_b, _F_cb, _O_wb, _stride,
                    //            _UNROLL, op_type, op_class>(
                    //     first,
                    //     F_h,
                    //     F_w,
                    //     I_w * _C_ib,
                    //     t_pad_el,
                    //     pad_top,
                    //     W_full_index,
                    //     l_pad_el,
                    //     pad_left,
                    //     O_w_w_pad,
                    //     O_w_full,
                    //     O_w_left,
                    //     r_pad_el,
                    //     pad_right,
                    //     I_row_top,
                    //     F_row_top,
                    //     O_row_top);

                    ScalarT const *I_row_full =
                        I_row_top + H_full_index * I_w * (_F_cb * _G_b);
                    AccumT        *O_row_full =
                        O_row_top + t_pad_el * O_w_w_pad * (_G_b * _K_b); // ScalarT --> AccumT

                    // Steady State over rows
                    for (index_t j = height_tid; j < O_h; j += T_height)
                    {
                        ScalarT const *I_row;
                        // @todo cast index calculation as int and make stride a float value.
                        // I_x = I_x + (int)(j * _stride) * (<remaining dimensions>)
                        if constexpr (op_type == OP_UPSAMPLE)
                        {
                            I_row = I_row_full + (j / _stride) * (I_w * _F_cb * _G_b);
                        }
                        else
                        {
                            I_row = I_row_full + (j * _stride) * (I_w * _F_cb * _G_b);
                        }
                        ScalarT const *F_row = F_channel_block_input + 0;
                        AccumT        *O_row =
                            O_row_full + j * (O_w_w_pad * _G_b * _K_b); // ScalarT --> AccumT
                        // Prologue with left padding
                        kernel_left_1D<ScalarT, AccumT,
                                       _G_b, _K_b, _F_cb, _O_wb, _stride,
                                       _UNROLL, op_type, op_class>(
                                           first,
                                           //F_h,
                                           F_w,
                                           I_w * _C_ib,
                                           l_pad_el,
                                           pad_left,
                                           I_row,
                                           F_row,
                                           O_row);

                        ScalarT const *I_col_full =
                            I_row + W_full_index * (_F_cb * _G_b);
                        AccumT        *O_col_full = O_row + l_pad_el * (_G_b * _K_b); // ScalarT --> AccumT
                        // Steady State with microkernel
                        for (index_t l = 0; l < O_w_full; l += _O_wb)
                        {
                            ScalarT const *I_col;
                            // @todo cast index calculation as int and make stride a float value.
                            // I_x = I_x + (int)(j * _stride) * (<remaining dimensions>)
                            if constexpr (op_type == OP_UPSAMPLE)
                            {
                                I_col = I_col_full + (l / _stride) * (_F_cb * _G_b);
                            }
                            else
                            {
                                I_col = I_col_full + (l * _stride) * (_F_cb * _G_b);
                            }
                            ScalarT const *F_col = F_row + 0;
                            AccumT        *O_col = O_col_full + l * (_G_b * _K_b); // ScalarT --> AccumT

                            kernel_1D<ScalarT, AccumT,
                                      _G_b, _K_b, _F_cb, _O_wb, _stride,
                                      _UNROLL, op_type, op_class>(
                                          first,
                                          //F_h,
                                          F_w,
                                          I_w * _C_ib,
                                          I_col,
                                          F_col,
                                          O_col,
                                          0,
                                          0);
                        }

#if DEBUG
                        printf(" end  kernel\n");
#endif

                        // Epilogue for microkernel + right padding elements
                        ScalarT const *I_col_left;
                        if constexpr (op_type == OP_UPSAMPLE)
                        {
                            I_col_left =
                                I_col_full + (O_w_full / _stride) * (_F_cb * _G_b);
                        }
                        else
                        {
                            I_col_left =
                                I_col_full + (O_w_full * _stride) * (_F_cb * _G_b);
                        }

                        ScalarT const *F_col_left = F_row + 0;
                        AccumT        *O_col_left = O_col_full + O_w_full * (_G_b * _K_b); // ScalarT --> AccumT

#if DEBUG
                        printf(" calling right\n");
#endif
                        kernel_right_1D<ScalarT, AccumT,
                                        _G_b, _K_b, _F_cb, _O_wb, _stride,
                                        _UNROLL, op_type, op_class>(
                                            first,
                                            //F_h,
                                            F_w,
                                            I_w * _C_ib,
                                            O_w_left,
                                            r_pad_el,
                                            pad_right,
                                            I_col_left,
                                            F_col_left,
                                            O_col_left);
                    }
                }

                for (index_t i = (F_c / _F_cb) - 1; i < (F_c / _F_cb); i++)
                {
                    bool first = rewrite_output && (i == 0);

                    ScalarT const *I_channel_block_input =
                        I_channel_block_output + i * (I_h * I_w * _F_cb * _G_b);
                    ScalarT const *F_channel_block_input =
                        F_channel_block_output + i * (F_h * F_w * _F_cb * _G_b * _K_b);
                    ScalarT *O_channel_block_input =
                        O_channel_block_output + 0;

                    // Loops over spatial dimensions of output

                    // Prologue with top padding
                    ScalarT const *I_row_top = I_channel_block_input;
                    //ScalarT const *F_row_top = F_channel_block_input + 0;
                    AccumT *O_row_top = O_channel_block_input; // ScalarT --> AccumT

                    // kernel_top<ScalarT, AccumT,
                    //            _G_b, _K_b, _F_cb, _O_wb, _stride,
                    //            _UNROLL, op_type, op_class, op_fused_single_element_before, op_fused_single_element_after, _stride_before, _stride_after>(
                    //     first,
                    //     F_h,
                    //     F_w,
                    //     I_w * _C_ib,
                    //     t_pad_el,
                    //     pad_top,
                    //     W_full_index,
                    //     l_pad_el,
                    //     pad_left,
                    //     O_w_w_pad,
                    //     O_w_full,
                    //     O_w_left,
                    //     r_pad_el,
                    //     pad_right,
                    //     I_row_top,
                    //     F_row_top,
                    //     O_row_top,
                    //     F_before_buf_group,
                    //     F_after_buf_group);

                    ScalarT const *I_row_full =
                        I_row_top + H_full_index * I_w * (_F_cb * _G_b);
                    AccumT        *O_row_full =
                        O_row_top + t_pad_el * O_w_w_pad * (_G_b * _K_b); // ScalarT --> AccumT

                    // Steady State over rows
                    for (index_t j = height_tid; j < O_h; j += T_height)
                    {
                        ScalarT const *I_row;
                        // @todo cast index calculation as int and make stride a float value.
                        // I_x = I_x + (int)(j * _stride) * (<remaining dimensions>)
                        if constexpr (op_type == OP_UPSAMPLE)
                        {
                            I_row = I_row_full + (j / _stride) * (I_w * _F_cb * _G_b);
                        }
                        else
                        {
                            I_row = I_row_full + (j * _stride) * (I_w * _F_cb * _G_b);
                        }
                        ScalarT const *F_row = F_channel_block_input + 0;
                        AccumT *O_row =
                            O_row_full + j * (O_w_w_pad * _G_b * _K_b); // ScalarT --> AccumT
                        // Prologue with left padding
                        kernel_left_1D<ScalarT, AccumT,
                                       _G_b, _K_b, _F_cb, _O_wb, _stride,
                                       _UNROLL, op_type, op_class,
                                       op_fused_single_element_before,
                                       op_fused_single_element_after,
                                       _stride_before, _stride_after>(
                                           first,
                                           //F_h,
                                           F_w,
                                           I_w * _C_ib,
                                           l_pad_el,
                                           pad_left,
                                           I_row,
                                           F_row,
                                           O_row,
                                           //0,
                                           //0,
                                           F_before_buf_group,
                                           F_after_buf_group);

                        ScalarT const *I_col_full =
                            I_row + W_full_index * (_F_cb * _G_b);
                        AccumT        *O_col_full = O_row + l_pad_el * (_G_b * _K_b); // ScalarT --> AccumT
                        // Steady State with microkernel
                        for (index_t l = 0; l < O_w_full; l += _O_wb)
                        {
                            ScalarT const *I_col;
                            // @todo cast index calculation as int and make stride a float value.
                            // I_x = I_x + (int)(j * _stride) * (<remaining dimensions>)
                            if constexpr (op_type == OP_UPSAMPLE)
                            {
                                I_col = I_col_full + (l / _stride) * (_F_cb * _G_b);
                            }
                            else
                            {
                                I_col = I_col_full + (l * _stride) * (_F_cb * _G_b);
                            }
                            ScalarT const *F_col = F_row + 0;
                            AccumT        *O_col = O_col_full + l * (_G_b * _K_b); // ScalarT --> AccumT

                            kernel_1D<ScalarT, AccumT,
                                      _G_b, _K_b, _F_cb, _O_wb, _stride,
                                      _UNROLL, op_type, op_class,
                                      op_fused_single_element_before,
                                      op_fused_single_element_after,
                                      _stride_before, _stride_after>(
                                          first,
                                          //F_h,
                                          F_w,
                                          I_w * _C_ib,
                                          I_col,
                                          F_col,
                                          O_col,
                                          //0,
                                          //0,
                                          0,
                                          0,
                                          F_before_buf_group,
                                          F_after_buf_group);
                        }

#if DEBUG
                        printf(" end  kernel\n");
#endif

                        // Epilogue for microkernel + right padding elements
                        ScalarT const *I_col_left;
                        if constexpr (op_type == OP_UPSAMPLE)
                        {
                            I_col_left =
                                I_col_full + (O_w_full / _stride) * (_F_cb * _G_b);
                        }
                        else
                        {
                            I_col_left =
                                I_col_full + (O_w_full * _stride) * (_F_cb * _G_b);
                        }

                        ScalarT const *F_col_left = F_row + 0;
                        AccumT        *O_col_left = O_col_full + O_w_full * (_G_b * _K_b); // ScalarT --> AccumT

#if DEBUG
                        printf(" calling right\n");
#endif
                        kernel_right_1D<ScalarT, AccumT,
                                        _G_b, _K_b, _F_cb, _O_wb, _stride,
                                        _UNROLL, op_type, op_class,
                                        op_fused_single_element_before,
                                        op_fused_single_element_after,
                                        _stride_before, _stride_after>(
                                            first,
                                            //F_h,
                                            F_w,
                                            I_w * _C_ib,
                                            O_w_left,
                                            r_pad_el,
                                            pad_right,
                                            I_col_left,
                                            F_col_left,
                                            O_col_left,
                                            //0,
                                            //0,
                                            F_before_buf_group,
                                            F_after_buf_group);
                    }
                }
            }
        }
    }
}


} // ns detail
} // ns small
