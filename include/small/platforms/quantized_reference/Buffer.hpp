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
#include <stdint.h>
#include <stdlib.h> // for posix_memalign
#include <vector>
#include <iostream>

namespace small
{
namespace detail
{
// Adapted from reference

//****************************************************************************
//Allocation
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

        std::cerr << "buffer_allocator::allocate: " << (void*)buffer << std::endl;
        return buffer;
    }

    void deallocate(pointer p, size_type) {
        std::cerr << "buffer_allocator::deallocator: " << (void*)p << std::endl;
        std::free(p);
    }

};

} // detail


//****************************************************************************
// define the Buffer type in the small namespace (qdtype becomes QUInt8Buffer)
/// @todo if Arduino compiler supports concepts using the integer_type concept
///       for ScalarT
class QUInt8Buffer
{
public:
    typedef uint8_t value_type;
    typedef int32_t accum_type;

    QUInt8Buffer(size_t num_elts) :
        scale(0.752941),
        offset(0),
        multiplier(1616928864),
        lshift(0),
        rshift(3),
        m_zero(0),
        min_val(0),   // std::numeric_limits<value_type>::lowest()
        max_val(255),     // std::numeric_limits<value_type>::max()
        b(8),
        m_buffer(num_elts)
    {
        /// @todo Revisit for other platforms
        quantized_init();

        // todo should the buffer be cleared?
        std::cerr << "QUInt8Buffer::ctor " << (void*)this
                  << ", data_ptr = " << (void*)m_buffer.data()
                  << ", size = " << m_buffer.size() << std::endl;
    }

    ~QUInt8Buffer()
    {
        std::cerr << "QUInt8Buffer::dtor " << (void*)this
                  << ", data_ptr = " << (void*)m_buffer.data()
                  << ", size = " << m_buffer.size() << std::endl;
    }

    size_t size() const { return m_buffer.size(); }

    value_type       *data()       { return m_buffer.data(); }
    value_type const *data() const { return m_buffer.data(); }

    value_type       &operator[](size_t index)       { return m_buffer[index]; }
    value_type const &operator[](size_t index) const { return m_buffer[index]; }

    void swap(QUInt8Buffer &other)
    {
        if (this != &other)
        {
            std::swap(scale,      other.scale);
            std::swap(offset,     other.offset);
            std::swap(multiplier, other.multiplier);
            std::swap(lshift,     other.lshift);
            std::swap(rshift,     other.rshift);
            std::swap(m_zero,     other.m_zero);
            std::swap(min_val,    other.min_val);
            std::swap(max_val,    other.max_val);
            std::swap(b,          other.b);
            std::swap(m_buffer,   other.m_buffer);
        }
    }

    /// @todo Should this be part of construction?
    /// @todo Note this function did not depend on numel.  REMOVED
    void quantized_init()
    {
        float max = 1.0;
        float min = -1.0;
        b = (sizeof(value_type) * 8);
        uint64_t max_q = (1 << b) - 1;
        int min_q = 0;
        double dscale = (max - min) / ((max_q - min_q) * 1.0) + 1e-17;
        int shift;
        const double q = frexp(dscale, &shift);
        auto q_fixed = static_cast<int64_t>(std::round(q * (1LL << 31)));
        if (q_fixed == (1LL << 31))
        {
            q_fixed /= 2;
            ++shift;
        }
        if (shift < -31)
        {
            shift = 0;
            q_fixed = 0;
        }
        if (shift > 30)
        {
            shift = 30;
            q_fixed = (1LL << 31) - 1;
        }
        multiplier = static_cast<int32_t>(q_fixed);

        m_zero = rint((double)(max * min_q - min * max_q) /
                      ((double)(max - min)));
        scale = dscale;
        lshift = shift > 0 ? shift : 0;
        rshift = shift > 0 ? 0 : -shift;
        min_val = 0
        max_val = 255;

        /// @todo offset not set
    }

    // type traits?
    inline accum_type zero() const { return m_zero; }

public:
    float    scale;
    int32_t  offset;     // AccumT?
    int32_t  multiplier; // AccumT?
    int      lshift;     // AccumT?
    int      rshift;     // AccumT?
    accum_type m_zero;       // AccumT?
    int      min_val;    // AccumT?
    int      max_val;    // AccumT?
    uint8_t  b;

private:
    // consider raw buffer instead, std::array does not support allocator
    std::vector<value_type, small::detail::buffer_allocator<value_type>> m_buffer;
};

//**********************************************************************
/// @todo return smart pointer?
inline QUInt8Buffer *alloc_buffer(size_t num_elts)
{
    return new QUInt8Buffer(num_elts);
}

} // small
