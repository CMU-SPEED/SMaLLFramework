//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2026 by The SMaLL Contributors, All Rights Reserved.
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
#include <small/buffers.hpp>

#define DEBUG_LAYERS 1
#include <small/SequentialLayer.hpp>
#include <small/Conv2DLayer.hpp>
#include <small/ReLULayer.hpp>

#include "test_utils.hpp"

std::string const data_dir("../test/regression_data");

#if defined(QUANTIZED)
using Buffer = small::QUInt8Buffer;
#else
using Buffer = small::FloatBuffer;
#endif

//****************************************************************************
void test_sequential_ctor(void)
{
#if defined(QUANTIZED)
    using BufferT = small::QUInt8Buffer;
#else
    using BufferT = small::FloatBuffer;
#endif

    // Build conv2d
    // C_i,Hi,Wi,k,s,p,C_o
    LayerParams params {96, 30, 30, 3, 2, small::PADDING_F, 96};

    // Read filter data
    std::string filter_fname =
        get_pathname(data_dir, "filter", "conv2d",
                     params,
                     params.C_i*params.k*params.k*params.C_o);
    std::cout << "\nConv2D: filter file= " << filter_fname << std::endl;

    BufferT filter_dc = read_inputs<BufferT>(filter_fname);
    TEST_ASSERT(filter_dc.size() == params.C_i*params.k*params.k*params.C_o);

    //=========================================================================
    BufferT bias(params.C_o);
    float bias_const = 1.0f;

    for (size_t ix = 0; ix < params.C_o; ++ix)
    {
        bias[ix] = bias_const;
    }

    small::shape_type input_shape({1UL, params.C_i, params.H, params.W});
    //size_t input_size = params.C_i*params.H*params.W;

    std::vector<small::Layer<BufferT> *> layers;
    layers.push_back(new small::Conv2DLayer<BufferT>(input_shape,
                                                     params.k, params.k,
                                                     params.s, params.p,
                                                     params.C_o,
                                                     filter_dc, bias,
                                                     false));

    small::shape_type output_shape(
        {1UL, params.C_o,
         small::compute_output_dim(params.H, params.k, params.s, params.p),
         small::compute_output_dim(params.W, params.k, params.s, params.p)});

    layers.push_back(new small::ReLULayer<BufferT>(output_shape));

    small::SequentialLayer seq(layers);
}

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"SequentialLayer constructor", test_sequential_ctor},
    {NULL, NULL}
};
