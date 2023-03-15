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
namespace detail
{
//****************************************************************************
// Allocation
//****************************************************************************
template <typename T, size_t alignment=64UL>
struct buffer_allocator : std::allocator<T>
{
    typedef typename std::allocator<T>::pointer pointer;
    typedef typename std::allocator<T>::size_type size_type;

    template<typename U>
    struct rebind {
        typedef buffer_allocator<U> other;
    };

    buffer_allocator() {}

    template<typename U>
    buffer_allocator(buffer_allocator<U> const& u)
        :std::allocator<T>(u) {}

    pointer allocate(size_type num_elements,
                     std::allocator<void>::const_pointer = 0) {
        pointer buffer;
        if (0 != posix_memalign((void**)&buffer,
                                alignment,
                                num_elements*sizeof(T)))
        {
            throw std::bad_alloc();
        }
        return buffer;
    }

    void deallocate(pointer p, size_type) {
        std::free(p);
    }

};
} // detail

//****************************************************************************
// Buffer class templated on size_t must be defined by a specific
// platform by defining in params.h of the platform-specific headers.
// Must have:
// - value_type typedef for the type of scalars stored in the buffer
// - data() method that returns raw pointer to data buffer
// - size() method that returns number of elements of sizeof(ScalarT) in
//          the data buffer.
// - swap() method that swaps the contents of two Buffer instances of the
//          same scalar type (shallow pointer swaps where possible)
// - operator[size_t] - element-wise access.
//

//****************************************************************************
class FloatBuffer
{
public:
    typedef float value_type;
    typedef float accum_type;

    FloatBuffer(size_t num_elts) :  m_buffer(num_elts) {}

    size_t size() const { return m_buffer.size(); }

    value_type       *data()       { return m_buffer.data(); }
    value_type const *data() const { return m_buffer.data(); }

    value_type       &operator[](size_t index)       { return m_buffer[index]; }
    value_type const &operator[](size_t index) const { return m_buffer[index]; }

    void swap(FloatBuffer &other)
    {
        if (this != &other)
        {
            std::swap(m_buffer, other.m_buffer);
        }
    }

    // type traits?
    inline accum_type zero() const { return (accum_type)0; }

private:
    // consider raw buffer instead, std::array does not support allocator
    std::vector<value_type, small::detail::buffer_allocator<value_type>> m_buffer;
};

//**********************************************************************
inline FloatBuffer *alloc_buffer(size_t num_elts)
{
    return new FloatBuffer(num_elts);
}

} // small

typedef small::FloatBuffer::value_type dtype;
