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
#include <small/Model.hpp>
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
class AutoencoderTiny : public Model<BufferT>
{
public:
    AutoencoderTiny() = delete;

    // Assume one input layer with a single shape for now
    AutoencoderTiny(shape_type            const &input_shape,
                    size_t                       dimension_reduction,
                    std::vector<BufferT*> const &filters,
                    bool                         filters_are_packed = false)
        : Model<BufferT>(input_shape),
          m_buffer_0(nullptr),
          m_buffer_1(nullptr)
    {
        create_model_and_buffers(dimension_reduction, filters, filters_are_packed);
    }

    virtual ~AutoencoderTiny()
    {
        delete m_buffer_0;
        delete m_buffer_1;
    }

    /// @todo Consider returning a vector of smart pointers (weak_ptr?) to the
    ///       output buffers.
    virtual std::vector<Tensor<BufferT>*>
        inference(Tensor<BufferT> const *input_tensor)
    {
        // assert(input_tensor->size() is correct);

        size_t layer_num = 0;
        // Conv2D + ReLU
        this->m_layers[layer_num++]->compute_output({input_tensor},
                                                    m_buffer_0);

        while (layer_num < this->m_layers.size())
        {
            // Conv2D + ReLU
            this->m_layers[layer_num++]->compute_output({m_buffer_0},
                                                        m_buffer_1);

            m_buffer_0->swap(*m_buffer_1);
        }

        return {m_buffer_0};
    }

private:
    Tensor<BufferT> *m_buffer_0;
    Tensor<BufferT> *m_buffer_1;

    void create_model_and_buffers(
        size_t                       dimension_reduction,
        std::vector<BufferT*> const &filters,
        bool                         filters_are_packed)
    {
        uint32_t kernel_size = 1;
        uint32_t stride = 1;
        uint32_t output_channels = 128;

        /// @todo assert dimension_reduction is a multiple of "16"

        size_t max_elt_0 = 0UL;
        size_t max_elt_1 = 0UL;

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

            if (ix == 0)
            {
                max_elt_0 = std::max<size_t>(max_elt_0, prev->output_size());
            }
            else
            {
                max_elt_1 = std::max<size_t>(max_elt_1, prev->output_size());
                max_elt_0 = std::max<size_t>(max_elt_0, max_elt_1);  // for the swap
            }
        }

        m_buffer_0 = new Tensor<BufferT>(max_elt_0);
        m_buffer_1 = new Tensor<BufferT>(max_elt_1);
    }
};

}
