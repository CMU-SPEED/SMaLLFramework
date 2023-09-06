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

#include <type_traits>
#include <exception>

#if defined(SMALL_HAS_FLOAT_SUPPORT)
#include <FloatBuffer.hpp>
#endif

#if defined(SMALL_HAS_QUINT8_SUPPORT)
#include <QUInt8Buffer.hpp>
#endif

namespace small
{

//****************************************************************************
// Buffer class templated on size_t must be defined by a specific
// platform by defining in params.h of the platform-specific headers.
// Must have:
// - value_type typedef for the type of scalars stored in the buffer
// - accum_type typedef for the type of scalars used to accumulate values
// - move constructor and move assignment operator
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

    FloatBuffer() :
        m_num_elts(0),
        m_buffer(nullptr)
    {
    }

    FloatBuffer(size_t num_elts, float *buffer)
        : m_num_elts(num_elts),
          m_buffer(buffer),
          m_buffer_created(false)
    {
    }

    FloatBuffer(size_t num_elts) : m_num_elts(num_elts)
    {
        if (0 != posix_memalign((void**)&m_buffer,
                                64,
                                num_elts*sizeof(value_type)))
        {
            throw std::bad_alloc();
        }
        // std::cerr << "FloatBuffer::ctor " << (void*)this
        //           << ", data_ptr = " << (void*)m_buffer.data()
        //           << ", size = " << m_buffer.size() << std::endl;
    }

    FloatBuffer(FloatBuffer const &other)
        : m_num_elts(other.m_num_elts)
    {
        //std::cerr << "FloatBuffer copy ctor\n";
        if (0 != posix_memalign((void**)&m_buffer,
                                64,
                                m_num_elts*sizeof(value_type)))
        {
            throw std::bad_alloc();
        }

        std::copy(other.m_buffer, other.m_buffer + m_num_elts,
                  m_buffer);
    }

    FloatBuffer(FloatBuffer&& other) noexcept
        : m_num_elts(0),
          m_buffer(nullptr)
    {
        //std::cerr << "FloatBuffer move ctor\n";
        std::swap(m_num_elts, other.m_num_elts);
        std::swap(m_buffer,   other.m_buffer);
        
    }

    FloatBuffer &operator=(FloatBuffer const &other)
    {
        //std::cerr << "FloatBuffer copy assignment\n";
        if (this != &other)
        {
            // expensive, but with exception guarantees.
            FloatBuffer tmp(other);

            std::swap(m_num_elts, tmp.m_num_elts);
            std::swap(m_buffer,   tmp.m_buffer);
        }

        return *this;
    }

    FloatBuffer &operator=(FloatBuffer&& other) noexcept
    {
        //std::cerr << "FloatBuffer move assignment\n";
        if (this != &other)
        {
            m_num_elts = 0;
            free(m_buffer);
            m_buffer = nullptr;
            std::swap(m_num_elts, other.m_num_elts);
            std::swap(m_buffer,   other.m_buffer);
        }

        return *this;
    }

    ~FloatBuffer()
    {
        // std::cerr << "FloatBuffer::dtor " << (void*)this
        //           << ", data_ptr = " << (void*)m_buffer.data()
        //           << ", size = " << m_buffer.size() << std::endl;
        if (m_buffer_created && m_buffer != nullptr)
        {
            free(m_buffer);
        }
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
    bool        m_buffer_created = true;
};

//**********************************************************************
/// @todo return smart pointer?
/// @todo Consider using static member factory method and hide ctor's
/// @todo Can this be done better with a CPO or explicit specialization?
///       I.e., define unimplemented alloc_buffer here and specialize in
///       the various Buffer header files.
///
template <class BufferT>
inline BufferT *alloc_buffer(size_t num_elts)
{
#if defined(SMALL_HAS_QUINT8_SUPPORT)
    if constexpr (std::is_same_v<BufferT, QUInt8Buffer>)
    {
        return new QUInt8Buffer(num_elts);
    }
#endif
#if defined(SMALL_HAS_FLOAT_SUPPORT)
    if constexpr (std::is_same_v<BufferT, FloatBuffer>)
    {
        return new FloatBuffer(num_elts);
    }
#endif

    throw std::invalid_argument("small::alloc_buffer ERROR: "
                                "unsupported template type.");
}

//**********************************************************************
template <class BufferT>
inline void free_buffer(BufferT *buffer)
{
    delete buffer;
}

} // small

