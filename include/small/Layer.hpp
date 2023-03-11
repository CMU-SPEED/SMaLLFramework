//------------------------------------------------------------------------------
// Layer.hpp - Scotts attempt at OO design for SMaLL
//------------------------------------------------------------------------------

// SMaLL Framework, (c) 2023
// by The SMaLL Framework Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DMxx-xxxx

//------------------------------------------------------------------------------

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

    virtual size_t  input_buffer_size() const = 0;
    virtual size_t output_buffer_size() const = 0;

    virtual void compute_output(BufferT const &input,
                                BufferT       &output) = 0;

    static BufferT allocate_buffer(size_t num_elements)
    {
        return BufferT(num_elements);
    }

protected:
};

}
