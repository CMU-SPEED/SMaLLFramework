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
#include <small.h>
#include <small/DAGModel.hpp>
#include <small/Conv2DLayer.hpp>
#include <small/ReLULayer.hpp>

//****************************************************************************

/* From https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/keras_model.py#L26

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
def get_model(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder
    (128*128*128*128*8*128*128*128*128)
    """
    inputLayer = Input(shape=(inputDim,)) // batches (unspecified) of 'inputDim' sized vectors

    h = Dense(128)(inputLayer)   // input=inDim, output=128, weights=128xinDim
    h = BatchNormalization()(h)  // pointwise norm. using set mean / std values
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(8)(h)               <---------- 8
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(inputDim)(h)

    return Model(inputs=inputLayer, outputs=h)
 */

//****************************************************************************
/* RECORD_CALLS:

  Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O) <-- ichans:inDim
  ReLUActivation(chans:128,img:1x1,I,O)
  Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
  ReLUActivation(chans:128,img:1x1,I,O)
  Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
  ReLUActivation(chans:128,img:1x1,I,O)
  Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
  ReLUActivation(chans:128,img:1x1,I,O)

  Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:16,ichans:128,img:1x1,I,F,O)  <-- ochans:8
  ReLUActivation(chans:16,img:1x1,I,O)

  Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:16,img:1x1,I,F,O)
  ReLUActivation(chans:128,img:1x1,I,O)
  Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
  ReLUActivation(chans:128,img:1x1,I,O)
  Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
  ReLUActivation(chans:128,img:1x1,I,O)
  Conv2D(k:1,s:1,pad:[0,0,0,0],ochans:128,ichans:128,img:1x1,I,F,O)
  ReLUActivation(chans:128,img:1x1,I,O)

  Missing:
  Conv2D(k:1,s:1,pad:[?,?,?,?],ochans:inDim,ichans:128,img:1x1,I,F,O)
*/

namespace small
{

//****************************************************************************
template <typename BufferT>
class AutoencoderTinyDAG : public DAGModel<BufferT>
{
public:
    AutoencoderTinyDAG() = delete;

    // Assume one input layer with a single shape for now
    AutoencoderTinyDAG(shape_type            const &input_shape,
                       size_t                       dimension_reduction,
                       std::vector<BufferT*> const &filters,
                       bool                         filters_are_packed = false)
        : DAGModel<BufferT>(input_shape)
    {
        size_t max_buffer_size =
            create_model_and_buffers(dimension_reduction,
                                     filters, filters_are_packed);
        this->initializeDAG(max_buffer_size);
    }

    virtual ~AutoencoderTinyDAG()
    {
    }

private:
    size_t create_model_and_buffers(
        size_t                       dimension_reduction,
        std::vector<BufferT*> const &filters,
        bool                         filters_are_packed)
    {
        uint32_t kernel_size = 1;
        uint32_t stride = 1;
        uint32_t output_channels = 128;

        /// @todo assert dimension_reduction is a multiple of "16"

        size_t layer_idx = 0UL;
        size_t max_buffer_size = 0UL;

        Layer<BufferT> *prev = nullptr;
        shape_type prev_shape(this->m_input_shape);

        for (auto ix = 0U; ix < filters.size(); ++ix)
        {
            /// @todo Support "dimension_reduction == 8;"
            if (ix == 4)
                output_channels = dimension_reduction;
            else
                output_channels = 128;

            prev = new Conv2DLayer<BufferT>(prev_shape,
                                            kernel_size, kernel_size,
                                            stride, PADDING_V,
                                            output_channels,
                                            *filters[ix], filters_are_packed,
                                            RELU);
            this->m_layers.push_back(prev);
            prev_shape = prev->output_shape();

            max_buffer_size =
                std::max<size_t>(max_buffer_size, prev->output_size());

            // =================================================
            this->m_graph.add_vertex(layer_idx);
            if (layer_idx > 0) this->m_graph.add_edge(layer_idx-1, layer_idx);
            ++layer_idx;
            // =================================================
        }

        return max_buffer_size;
    }
};

}
