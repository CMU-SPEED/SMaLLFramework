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

#include<vector>
#include<small.h>

namespace small
{
//****************************************************************************
template <typename BufferT>
class Layer
{
public:

    Layer() {};

    virtual ~Layer() {}

    virtual size_t  input_buffer_size() const = 0;
    virtual size_t output_buffer_size() const = 0;

    virtual void compute_output(BufferT const &input,
                                BufferT       &output) const = 0;

#if 0
    // CONSIDER THE FOLLOWING INTERFACE INSTEAD
    virtual size_t  input_buffer_size(uint32_t       input_height,
                                      uint32_t       input_width) = 0;
    virtual size_t output_buffer_size(uint32_t       input_height,
                                      uint32_t       input_width) = 0;

    virtual void compute_output(uint32_t       input_height,
                                uint32_t       input_width,
                                BufferT const &input_dc,
                                uint32_t      &output_height,
                                uint32_t      &output_width
                                BufferT       &output_dc) const = 0;
    {
        /// If I do this then padding needs to be recomputed everytime
    }

    // IS THIS NECESSARY?
    static BufferT allocate_buffer(size_t num_elements)
    {
        return BufferT(num_elements);
    }
#endif

protected:
};

}
