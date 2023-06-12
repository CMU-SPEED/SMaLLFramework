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

#include <small.h>
#include <small/buffers.hpp>
#include <small/Layer.hpp>

namespace small
{

//****************************************************************************
template <typename BufferT>
class YOLOLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    YOLOLayer(shape_type const &shape,
            // std::vector<uint32_t> const mask,
            std::vector<std::pair<uint32_t, uint32_t>> anchors,
            size_t num_classes) // {B, C, H, W}
        : Layer<BufferT>(shape), m_anchors(anchors), m_num_classes(num_classes)
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "Yolo(batches:" << shape[BATCH]
                  << ",chans:" << shape[CHANNEL]
                  << ",img:" << shape[HEIGHT] << "x" << shape[WIDTH]
                  << ")" << std::endl;
#endif

        m_num_outputs = m_num_classes + 5; // # of outputs per anchor
        m_num_anchors = m_anchors.size();
    }

    virtual ~YOLOLayer() {}

    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        std::vector<Tensor<BufferT>*>        output) const
    {
        // Tensor<BufferT> *in = input[0];

        // do i really need to reshape?
        
        

    }

private:

    std::vector<std::pair<uint32_t,uint32_t>> const m_anchors;
    size_t       m_num_anchors;
    size_t const m_num_classes;
    size_t       m_num_outputs;

};

}
