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
#include <stdint.h> // for uint8_t
#include <cmath>

namespace small
{
/// @ todo only need one type.
typedef uint32_t index_t;
typedef uint32_t dim_t;


enum PaddingEnum
{
    PADDING_V,   /// @todo define 'valid' padding...no padding?
    PADDING_F    /// @todo define 'full' padding
};

//****************************************************************************
// Useful utility functions
//****************************************************************************

/// @todo Combine calc_padding and compute_output_dim functions? Determine need.

//****************************************************************************
/**
 * When padding mode is 'f', compute front and back padding (for either
 * horizontal or vertical dimension) based on corresponding image dimension
 * and kernel dimension.  Call this twice to compute l,r and t,b pairs.
 *
 */
inline void calc_padding(size_t    I_dim,
                         uint32_t  K_dim,
                         uint16_t  stride,
                         uint8_t  &padding_front,
                         uint8_t  &padding_back)
{
    uint32_t padding;
    if (I_dim % stride == 0)
    {
        padding = (K_dim > stride) ?
                   K_dim - stride :
                   0;
    }
    else
    {
        padding = (K_dim > (I_dim % stride)) ?
                  (K_dim - (I_dim % stride)) :
                  0;
    }
    padding_front = padding / 2;
    padding_back  = padding - padding_front;
}

//****************************************************************************
#if 0
inline void CALC_PADDING(uint32_t I_dim,
                         uint32_t K_dim,
                         uint16_t stride,
                         uint8_t &padding_front,
                         uint8_t &padding_back)
{
    uint32_t padding;
    if (I_dim % stride == 0)
    {
        padding = (K_dim > stride) ? K_dim - stride : 0;
    }
    else
    {
        padding = (K_dim > (I_dim % stride)) ? (K_dim - (I_dim % stride)) : 0;
    }
    padding_front = padding / 2;
    padding_back = padding - padding_front;
}
#endif

//****************************************************************************
/**
 * @todo Find a better way to set const members of the Layer classes
 * @todo consider moving to Layer class.
 */
inline uint8_t calc_front_padding(PaddingEnum  padding_type,
                                  uint32_t     I_dim,
                                  uint32_t     K_dim,
                                  uint16_t     stride)
{
    if (padding_type == PADDING_V) return 0U;

    uint8_t front_padding, back_padding;
    calc_padding(I_dim, K_dim, stride, front_padding, back_padding);
    return front_padding;
}

//****************************************************************************
/**
 * @todo Find a better way to set const members of the Layer classes
 * @todo consider moving to Layer class.
 * @deprecated
 */
inline uint8_t calc_front_padding(char      padding_type,
                                  uint32_t  I_dim,
                                  uint32_t  K_dim,
                                  uint16_t  stride)
{
    return calc_front_padding(((padding_type == 'f') ? PADDING_F : PADDING_V),
                              I_dim, K_dim, stride);
}

//****************************************************************************
/**
 * @todo Find a better way to set const members of the Layer classes
 * @todo consider moving to Layer class.
 */
inline uint8_t calc_back_padding(PaddingEnum padding_type,
                                 uint32_t    I_dim,
                                 uint32_t    K_dim,
                                 uint16_t    stride)
{
    if (padding_type == PADDING_V) return 0U;

    uint8_t front_padding, back_padding;
    calc_padding(I_dim, K_dim, stride, front_padding, back_padding);
    return back_padding;
}

//****************************************************************************
/**
 * @todo Find a better way to set const members of the Layer classes
 * @todo consider moving to Layer class.
 * @deprecated
 */
inline uint8_t calc_back_padding(char      padding_type,
                                 uint32_t  I_dim,
                                 uint32_t  K_dim,
                                 uint16_t  stride)
{
    return calc_back_padding(((padding_type == 'f') ? PADDING_F : PADDING_V),
                              I_dim, K_dim, stride);
}

//****************************************************************************
//****************************************************************************

//****************************************************************************
/// @todo make constexpr(?) function
/// @todo PICK ONE output_dim function
// assumes 'v' padding
inline dim_t output_dim(dim_t input_dim, dim_t stride, dim_t kernel_dim)
{
    int out_elems = (int(input_dim) - int(kernel_dim)) /(stride) + 1;
    // int out_elems = (int(input_dim) - int(kernel_dim))/ int(stride) + 1;

    return ((out_elems > 0) ? dim_t(out_elems) : 0U);
}

//****************************************************************************
// assumes 'v' padding
inline dim_t output_dim_new(dim_t input_dim, dim_t stride, dim_t kernel_dim)
{
    return ((kernel_dim > input_dim)
            ? 0U
            : ((input_dim - kernel_dim)/stride + 1));
}

//****************************************************************************
inline size_t compute_output_dim(size_t input_dim,
                                 size_t kernel_dim,
                                 size_t stride,
                                 small::PaddingEnum padding) //char   padding)
{
    if ((padding == small::PADDING_V) && (input_dim >= kernel_dim))
    {
        return std::floor((input_dim - kernel_dim)/((float)stride)) + 1;
    }
    else if (padding == small::PADDING_F)
    {
        uint8_t fpad, bpad;
        small::calc_padding(input_dim, kernel_dim, stride, fpad, bpad);
        size_t padded_input_dim = input_dim + fpad + bpad;
        size_t output_dim = 1 + (padded_input_dim - kernel_dim)/stride;
        return std::max(output_dim, 0UL);
    }
    else
    {
        throw std::invalid_argument(
            "compute_output_dim: Bad kernel and image size combination.");
    }

    return 0;
}

//****************************************************************************
inline void compute_padding_output_dim(size_t    unpadded_input_dim,
                                       uint32_t  kernel_dim,
                                       uint32_t  stride,
                                       PaddingEnum padding_type,
                                       uint8_t  &front_pad,
                                       uint8_t  &back_pad,
                                       size_t   &output_dim)
{
    front_pad = 0;
    back_pad = 0;
    output_dim = 0;

    if (padding_type == PADDING_F)
    {
        small::calc_padding(unpadded_input_dim, kernel_dim, stride,
                            front_pad, back_pad);
    }

    size_t padded_input_dim = unpadded_input_dim + front_pad + back_pad;

    if (padded_input_dim < kernel_dim)
    {
        throw std::invalid_argument("compute_padding_output_dim() ERROR: "
                                    "Bad combination, kernel too large.");
    }

    output_dim = 1 + (padded_input_dim - kernel_dim)/stride;
}

//****************************************************************************
// from https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
// size_t compute_output_dim_old(size_t input_dim,
//                               size_t kernel_dim,
//                               size_t stride,
//                               char   padding)
// {
//     if ((padding == 'v') && (input_dim >= kernel_dim))
//     {
//         return std::floor((input_dim - kernel_dim)/((float)stride)) + 1;
//     }
//     else if (padding == 'f')
//     {
//         return std::floor((input_dim - 1)/((float)stride)) + 1;
//     }
//     else
//     {
//         throw std::invalid_argument(std::string("Bad combination"));
//     }

//     return 0;
// }

//****************************************************************************
//****************************************************************************
inline int clip(int n, int upper, int lower = 0)
{
    n = (n > lower) * n + !(n > lower) * lower;
    return (n < upper) * n + !(n < upper) * upper;
}

//****************************************************************************
// Quantization Functions
template <typename Q_T, typename T>
void Quantize(int num_elements, T const *tensor_ptr, Q_T *quant_tensor_ptr)
{
    float scale_inv = (1.0 / quant_tensor_ptr->scale);
    uint64_t max_val = (1 << quant_tensor_ptr->b) - 1;
    int quant_val = rint(quant_tensor_ptr->zero + (0.0 * scale_inv));
    for (int i = 0; i < num_elements; i++)
    {
        int quant_val = rint(quant_tensor_ptr->zero + (tensor_ptr[i] * scale_inv));
        quant_tensor_ptr->tensor[i] =
            (quant_val < max_val) ? quant_val : max_val;
    }
}

//****************************************************************************
template <typename Q_T, typename T>
void DeQuantize(int num_elements, T *tensor_ptr, Q_T const *quant_tensor_ptr)
{
    for (int i = 0; i < num_elements; i++)
    {
        tensor_ptr[i] =
            static_cast<T>(
                quant_tensor_ptr->scale *
                (quant_tensor_ptr->tensor[i] - quant_tensor_ptr->zero));
    }
}

//****************************************************************************
template <typename Q_T, typename T>
void DebugDeQuantize(int num_elements, T *tensor_ptr, Q_T const *quant_tensor_ptr)
{
    for (int i = 0; i < num_elements; i++)
    {
        tensor_ptr[i] =
            static_cast<T>(
                quant_tensor_ptr->scale *
                ((T)(quant_tensor_ptr->tensor[i] - quant_tensor_ptr->zero)));
    }
}

} // small
