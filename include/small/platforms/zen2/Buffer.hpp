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

#include <stdexcept>
#include <stdlib.h> // for posix_memalign
#include <vector>

namespace small
{

//****************************************************************************
// Buffer class templated on size_t must be defined by a specific
// platform by defining in params.h of the platform-specific headers.
// Must have:
// - value_type typedef for the type of scalars stored in the buffer
// - accum_type typedef for the type of scalars used to accumulate values
// - data() method that returns raw pointer to data buffer
// - size() method that returns number of elements of sizeof(ScalarT) in
//          the data buffer.
// - swap() method that swaps the contents of two Buffer instances of the
//          same scalar type (shallow pointer swaps where possible)
// - operator[size_t] - element-wise access.
// - zero() the additive identity for the accum_type

//****************************************************************************
class FloatBuffer
{
public:
    typedef float value_type;
    typedef float accum_type;

    FloatBuffer(size_t num_elts) : m_num_elts(num_elts)
    {
        if (0 != posix_memalign((void**)&m_buffer,
                                64,
                                num_elts*sizeof(value_type)))
        {
            throw std::bad_alloc();
        }
    }

    ~FloatBuffer()
    {
        free(m_buffer);
    }

    inline size_t size() const { return m_num_elts; }

    inline value_type       *data()       { return m_buffer; }
    inline value_type const *data() const { return m_buffer; }

    inline value_type       &operator[](size_t index)       { return m_buffer[index]; }
    inline value_type const &operator[](size_t index) const { return m_buffer[index]; }

    inline void swap(FloatBuffer &other)
    {
        if (this != &other)
        {
            std::swap(m_buffer, other.m_buffer);
            std::swap(m_num_elts, other.m_num_elts);
        }
    }

    // type traits?
    inline accum_type zero() const { return (accum_type)0; }

private:
    size_t      m_num_elts;
    value_type *m_buffer;
};

//**********************************************************************
inline FloatBuffer *alloc_buffer(size_t num_elts)
{
    return new FloatBuffer(num_elts);
}

inline void free_buffer(FloatBuffer *buffer)
{
    delete buffer;
}

} // small

typedef small::FloatBuffer::value_type dtype;
