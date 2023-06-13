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

#include <assert.h>

#include <small.h>
#include <small/buffers.hpp>
#include <small/Layer.hpp>

template <typename ScalarT>
ScalarT sigmoid(ScalarT v) {
    return 1.0 / (1.0 + std::exp(-v));
}

// sigmoid function on flat buffer
template <typename ScalarT>
void sigmoid_buf(ScalarT *buf, uint32_t numel)
{
    for (size_t i = 0; i < numel; ++i)
        buf[i] = sigmoid(buf[i]);
}

// computes the offset into the reshaped buffer
inline size_t compute_offset(size_t i0, size_t i1, size_t i2, size_t i3,
                             size_t num_classes, size_t H, size_t W)
{
    return (i0 * (num_classes + 5) * H * W) + (i1 * H * W) + (i2 * W) + (i3);
}


// i1 = (5 + conf_idx) / H * W

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

        assert(shape[CHANNEL] % (m_num_classes + 5) == 0);

        m_num_outputs = m_num_classes + 5; // # of outputs per anchor
        m_num_anchors = m_anchors.size();
    }

    virtual ~YOLOLayer() {}

    // note: fastest dimension is rightmost
    // input is in packed format
    // input is [C/Cb, H, W, Cb]
    // unpacked it is [C, H, W]
    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        std::vector<Tensor<BufferT>*>        output) const
    {

        using ScalarT = typename BufferT::value_type;

        // For now, assume input is unpacked (i.e. [C, H, W])
        // We need to reshape input into [num_anchors, num_classes + 5, H, W]
        // for a given index [i0, i1, i2, i3]
        // offset = i0 * (num_classes + 5) * H * W + i1 * H * W + i2 * W + i3
        // Note: num_classes = (C / num_anchors) - 5
        // The 5 comes from the fact that the first 5 of represent
        // the dimension correspond to x, y, w, h, and confidence
        // the rest of data corresponds to the class probabilities

        // num_pred = num_anchors * H * W
        
        // bbox_list is  [num_pred, 4]
        // conf_list is  [num_pred, 1]
        // class_list is [num_pred, 1]
        size_t h = input[0]->shape()[HEIGHT];
        size_t w = input[0]->shape()[WIDTH];

        // alias outputs
        Tensor <BufferT> *bbox_n_conf = output[0];
        // Tensor <BufferT> *bbox = output[0];
        // Tensor <BufferT> *conf = output[1];
        // Tensor <BufferT> *class_conf = output[2];

        // size_t num_pred = m_num_anchors * h * w;

        // image is [C, H, W]
        // reshaped image is [num_anchors, num_classes + 5, H, W]
        size_t pred_idx = 0;
        for(size_t i0 = 0; i0 < m_num_anchors; i0++) {
            size_t anchor_offset = i0 * (m_num_classes + 5) * h * w;

            for(size_t i1 = 0; i1 < h; i1++) {
                size_t h_offset = i1 * w;

                for(size_t i2 = 0; i2 < w; i2++) {
                    size_t w_offset = i2;

                    // fastest dimension for output
                    for(size_t i3 = 0; i3 < (m_num_classes + 5); i3++) {
                        size_t offset = anchor_offset + i3 * h * w + h_offset + w_offset;

                        // compute x
                        if(i3 == 0) {
                            bbox_n_conf->buffer()[pred_idx*85U + i3] = (sigmoid<ScalarT>(input[0]->buffer()[offset]) + i2) / w;
                        }
                        // compute y
                        else if(i3 == 1) {
                            bbox_n_conf->buffer()[pred_idx*85U + i3] = (sigmoid<ScalarT>(input[0]->buffer()[offset]) + i1) / h;
                        }
                        // compute w
                        else if(i3 == 2) {
                            bbox_n_conf->buffer()[pred_idx*85U + i3] = std::exp(input[0]->buffer()[offset]) * m_anchors[i0].first;
                        }
                        // compute h
                        else if(i3 == 3) {
                            bbox_n_conf->buffer()[pred_idx*85U + i3] = std::exp(input[0]->buffer()[offset]) * m_anchors[i0].second;
                        }
                        // compute obj and class conf
                        else {
                            bbox_n_conf->buffer()[pred_idx*85U + i3] = sigmoid<ScalarT>(input[0]->buffer()[offset]);
                        }

                    }
                    pred_idx++;
                }
            }
        }
        
    }

private:

    std::vector<std::pair<uint32_t,uint32_t>> const m_anchors;
    size_t       m_num_anchors;
    size_t const m_num_classes;
    size_t       m_num_outputs;

};

}
