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
// template <typename ScalarT>
// void sigmoid_buf(ScalarT *buf, uint32_t numel)
// {
//     for (size_t i = 0; i < numel; ++i)
//         buf[i] = sigmoid(buf[i]);
// }

// computes the offset into the reshaped buffer
inline size_t compute_offset(size_t i0, size_t i1, size_t i2, size_t i3,
                             size_t num_classes, size_t H, size_t W)
{
    return (i0 * (num_classes + 5) * H * W) + (i1 * H * W) + (i2 * W) + (i3);
}

namespace small
{

//****************************************************************************
template <typename BufferT>
class YOLOLayer : public Layer<BufferT>
{
public:
    typedef typename BufferT::value_type value_type;

    YOLOLayer(shape_type const &input_shape,
              std::vector<std::pair<uint32_t, uint32_t>> masked_anchors,
              size_t num_classes,
              size_t input_img_size) /// @todo assumes square image
        : Layer<BufferT>(),
          m_input_shape(input_shape),
          m_unpacked_input(
              input_shape[CHANNEL]*input_shape[HEIGHT]*input_shape[WIDTH]),
          m_stride(input_img_size / input_shape[HEIGHT]),
          m_anchors(masked_anchors),
          m_num_anchors(masked_anchors.size()),
          m_num_classes(num_classes),
          m_num_outputs(num_classes + 5), // # of outputs per anchor
          m_num_pred(m_num_anchors * input_shape[HEIGHT] * input_shape[WIDTH]),
          m_effective_channels(m_num_anchors * m_num_outputs),
          m_padded_channels(input_shape[CHANNEL])
    {
#if defined(DEBUG_LAYERS)
        std::cerr << "Yolo(batches:" << input_shape[BATCH]
                  << ",chans:" << input_shape[CHANNEL]
                  << ",img:" << input_shape[HEIGHT] << "x" << input_shape[WIDTH]
                  << ")" << std::endl;
#endif

        // HACK
        // Since Conv2D can't support channel dimensions that are not a multiple
        // of the blocking size (i.e., Cb), we need to keep around 2 sizes:
        //
        // - m_effective_channels is equal to the number of channels that are
        //                        actually valid (i.e., not padded)
        // - m_padded_channels    is the number of channels that are actually
        //                        allocated in the buffer (i.e., padded)

        this->set_output_shape(
            {input_shape[BATCH], 1U, m_num_pred, m_num_outputs});
    }

    virtual ~YOLOLayer() {}

    size_t get_num_pred() const { return m_num_pred; }

    size_t get_num_outputs() const { return m_num_outputs; }

    // note: fastest dimension is rightmost
    // input is in packed format
    // input is [C/Cb, H, W, Cb]
    // unpacked it is [1, C, H, W]
    virtual void compute_output(
        std::vector<Tensor<BufferT> const *> input,
        Tensor<BufferT>*                     output) const
    {
        if ((input.size() != 1) || (input[0]->shape() != m_input_shape))
        {
            throw std::invalid_argument(
                "YOLOLayer::compute_output() ERROR: "
                "incorrect input buffer shape.");
        }

        if (output->capacity() < this->output_size())
        {
            throw std::invalid_argument(
                "YOLOLayer::compute_output() ERROR: "
                "insufficient output buffer space.");
        }

        using ScalarT = typename BufferT::value_type;

        size_t h = input[0]->shape()[HEIGHT];
        size_t w = input[0]->shape()[WIDTH];

        /// @todo profile pulling directly from the packed input buffer
        ///       to avoid construction of this unpacked buffer.
        small::unpack_buffer(
            input[0]->buffer(), small::INPUT,
            1U, m_padded_channels, h, w,
            BufferT::C_ib, BufferT::C_ob,
            m_unpacked_input
        );

        // For now, assume input is unpacked (i.e. [1, C, H, W])
        // We need to reshape input into [num_anchors, num_classes + 5, H, W]
        // for a given index [i0, i1, i2, i3]
        //
        // offset = i0 * (num_classes + 5) * H * W + i1 * H * W + i2 * W + i3
        //
        // Note: num_classes = (C / num_anchors) - 5
        //
        // The 5 comes from the fact that the first 5 elements
        // correspond to x, y, w, h, and objectness_confidence
        // the rest of data corresponds to the class probabilities

        // num_pred = num_anchors * H * W

        // alias outputs
        Tensor<BufferT> *bbox_n_conf = output;
        bbox_n_conf->set_shape(this->output_shape());

        // check to make sure the output buffer is the right size
        // if(m_num_pred != bbox_n_conf->shape()[2]) {
        //     std::cerr << "ERROR: num_pred != bbox_n_conf->shape()[2]" << std::endl;
        //     std::cerr << "       num_pred = " << m_num_pred << std::endl;
        //     std::cerr << "       bbox_n_conf->shape()[2] = " << bbox_n_conf->shape()[2] << std::endl;
        //     exit(1);
        // }

        if (bbox_n_conf->shape() != this->output_shape())
        {
            throw std::runtime_error(
                "YOLOLayer::compute_output ERROR: "
                "bb_n_conf->shape() != this->output_shape()");
        }

        // image is [C, H, W]
        // reshaped image is [num_anchors, num_classes + 5, H, W]
        size_t pred_idx = 0;
        for (size_t i0 = 0; i0 < m_num_anchors; i0++)
        {
            size_t anchor_offset = i0 * (m_num_classes + 5) * h * w;

            for (size_t i1 = 0; i1 < h; i1++)
            {
                size_t h_offset = i1 * w;

                for (size_t i2 = 0; i2 < w; i2++)
                {
                    size_t w_offset = i2;

                    size_t out_off = pred_idx*(m_num_classes + 5);
                    // fastest dimension for output
                    for (size_t i3 = 0; i3 < (m_num_classes + 5); i3++)
                    {
                        size_t offset =
                            anchor_offset + i3 * h * w + h_offset + w_offset;

                        // compute x
                        if (i3 == 0)
                        {
                            bbox_n_conf->buffer()[out_off + i3] = m_stride *
                                (sigmoid<ScalarT>(m_unpacked_input[offset])+i2);
                        }
                        // compute y
                        else if (i3 == 1)
                        {
                            bbox_n_conf->buffer()[out_off + i3] = m_stride *
                                (sigmoid<ScalarT>(m_unpacked_input[offset]) + i1);
                        }
                        // compute w
                        else if (i3 == 2)
                        {
                            bbox_n_conf->buffer()[out_off + i3] =
                                std::exp(m_unpacked_input[offset]) *
                                m_anchors[i0].first;
                        }
                        // compute h
                        else if (i3 == 3)
                        {
                            bbox_n_conf->buffer()[out_off + i3] =
                                std::exp(m_unpacked_input[offset]) *
                                m_anchors[i0].second;
                        }
                        // compute obj and class conf
                        else
                        {
                            bbox_n_conf->buffer()[out_off + i3] =
                                sigmoid<ScalarT>(m_unpacked_input[offset]);
                        }
                    }
                    pred_idx++;
                }
            }
        }
    }

private:
    shape_type const m_input_shape;
    mutable BufferT  m_unpacked_input;  /// @todo remove; use packed buffer directly

    size_t       m_stride;
    std::vector<std::pair<uint32_t,uint32_t>> const m_anchors;
    size_t       m_num_anchors;
    size_t const m_num_classes;
    size_t       m_num_outputs;
    size_t       m_num_pred;
    size_t       m_effective_channels;
    size_t       m_padded_channels;

};

}
