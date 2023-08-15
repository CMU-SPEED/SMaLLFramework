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
//#include <iostream>
#include <cmath>

namespace small
{
namespace detail
{
// Adapted from small_MCU/small/quantized/include/utils.h

//****************************************************************************
//Allocation
//****************************************************************************

//  #define MAX_BUFF_SIZE 50000000
//  #define MAX_BUFF_SIZE 10000000
//  #define MAX_BUFF_SIZE 200000
    #define MAX_BUFF_SIZE 208200

    uint8_t  memory_buffer[MAX_BUFF_SIZE];
    uint8_t *current_free_ptr = memory_buffer;
    size_t   buf_offset = 0;

    //************************************************************************
    /// @todo This is allocation-only memory management.  It cannot manage
    ///       out of order calls to a free method
    /// @todo return void instead?
    void *alloc(size_t num_bytes, size_t alignment = 1)
    {
        /// deal with alignment issues
        if (alignment == 0) alignment = 1;
        size_t alignment_error =
            reinterpret_cast<size_t>(current_free_ptr) % alignment;

        if (alignment_error > 0)
        {
            size_t waste = alignment - alignment_error;
            buf_offset += waste;
            current_free_ptr += waste;
        }

        size_t used_bytes = buf_offset + num_bytes;

        if (MAX_BUFF_SIZE < used_bytes)
        {
            // Serial.println("out of space\n");
            // Serial.println(used_bytes);
            // Serial.println();

            throw std::bad_alloc();
        }

        void *ret_ptr =
            reinterpret_cast<void*>(memory_buffer + buf_offset);
        buf_offset = used_bytes;
        current_free_ptr = memory_buffer + used_bytes;
        return ret_ptr;
    }

    //**********************************************************************
    /// @need better buffer management?
    uint32_t free_all()
    {
        auto freed_space = buf_offset;
        current_free_ptr = memory_buffer;
        buf_offset = 0;

        return freed_space;
    }

}  // detail


//****************************************************************************
// define the Buffer type in the small namespace (qdtype becomes QUInt8Buffer)
/// @todo if Arduino compiler supports concepts using the integer_type concept
///       for ScalarT
class QUInt8Buffer
{
public:
    static uint32_t const   W_ob{QUINT8_W_ob};
    static uint32_t const   C_ob{QUINT8_C_ob};
    static uint32_t const   SIMD{QUINT8_SIMD};
    static uint32_t const UNROLL{QUINT8_UNROLL};
    static uint32_t const   C_ib{QUINT8_C_ib};

    static uint32_t const   NUM_FMA{QUINT8_NUM_FMA};
    static uint32_t const   NUM_MAX{QUINT8_NUM_MAX};
    static uint32_t const  NUM_LOAD{QUINT8_NUM_LOAD};
    static uint32_t const NUM_STORE{QUINT8_NUM_STORE};

public:
    typedef uint8_t value_type;
    typedef int32_t accum_type;

    QUInt8Buffer() :
        m_num_elts(0),
        m_buffer(nullptr)
    {
        quantized_init();
    }

    /// @todo add min_val and max_val parameters defaulted to the limits below
    QUInt8Buffer(size_t num_elts) :
        //scale(0.752941),
        //offset(0),
        //multiplier(1616928864),
        //lshift(0),
        //rshift(3),
        //m_zero(0),
        //min_val(0),       // std::numeric_limits<value_type>::lowest()
        //max_val(255),     // std::numeric_limits<value_type>::max()
        //b(8),
        m_num_elts(num_elts),
        m_buffer(reinterpret_cast<value_type*>(
                     detail::alloc(num_elts*sizeof(value_type), 4)))
    {
        /// @todo Merge with member initialization
        quantized_init();
    }

    QUInt8Buffer(QUInt8Buffer const &other)
        : scale(other.scale),
          offset(other.offset),  // not set by quantized_init
          multiplier(other.multiplier),
          lshift(other.lshift),
          rshift(other.rshift),
          m_zero(other.m_zero),
          min_val(other.min_val),
          max_val(other.max_val),
          b(other.b),
          m_num_elts(other.m_num_elts),
          m_buffer(reinterpret_cast<value_type*>(
                       detail::alloc(other.m_num_elts*sizeof(value_type), 4)))
    {
        std::copy(other.m_buffer, other.m_buffer + m_num_elts,
                  m_buffer);
    }

    QUInt8Buffer(QUInt8Buffer&& other) noexcept
        : m_num_elts(0),
          m_buffer(nullptr)
    {
        //std::cerr << "QUInt8Buffer move ctor\n";
        std::swap(scale,      other.scale);
        std::swap(offset,     other.offset);  // not set by quantized_init
        std::swap(multiplier, other.multiplier);
        std::swap(lshift,     other.lshift);
        std::swap(rshift,     other.rshift);
        std::swap(m_zero,     other.m_zero);
        std::swap(min_val,    other.min_val);
        std::swap(max_val,    other.max_val);
        std::swap(b,          other.b);

        std::swap(m_num_elts, other.m_num_elts);
        std::swap(m_buffer,   other.m_buffer);
    }

    QUInt8Buffer &operator=(QUInt8Buffer const &other)
    {
        //std::cerr << "QUInt8Buffer copy assignment\n";
        if (this != &other)
        {
            // expensive, but with exception guarantees.
            QUInt8Buffer tmp(other);

            std::swap(scale,      tmp.scale);
            std::swap(offset,     tmp.offset);  // not set by quantized_init
            std::swap(multiplier, tmp.multiplier);
            std::swap(lshift,     tmp.lshift);
            std::swap(rshift,     tmp.rshift);
            std::swap(m_zero,     tmp.m_zero);
            std::swap(min_val,    tmp.min_val);
            std::swap(max_val,    tmp.max_val);
            std::swap(b,          tmp.b);

            std::swap(m_num_elts, tmp.m_num_elts);
            std::swap(m_buffer,   tmp.m_buffer);
        }

        return *this;
    }

    QUInt8Buffer &operator=(QUInt8Buffer&& other) noexcept
    {
        //std::cerr << "QUInt8Buffer move assignment\n";
        if (this != &other)
        {
            m_num_elts = 0;
            free(m_buffer);
            m_buffer = nullptr;

            std::swap(scale,      other.scale);
            std::swap(offset,     other.offset);  // not set by quantized_init
            std::swap(multiplier, other.multiplier);
            std::swap(lshift,     other.lshift);
            std::swap(rshift,     other.rshift);
            std::swap(m_zero,     other.m_zero);
            std::swap(min_val,    other.min_val);
            std::swap(max_val,    other.max_val);
            std::swap(b,          other.b);

            std::swap(m_num_elts, other.m_num_elts);
            std::swap(m_buffer,   other.m_buffer);
        }

        return *this;
    }

    ~QUInt8Buffer()
    {
        // std::cerr << "QUInt8Buffer::dtor " << (void*)this
        //           << ", data_ptr = " << (void*)m_buffer.data()
        //           << ", size = " << m_buffer.size() << std::endl;
        if (m_buffer != nullptr)
        {
            /// @todo Need to implement memory pool to reclaim blocks.
            // free(m_buffer);
        }
    }

    size_t size() const { return m_num_elts; }

    value_type       *data()       { return m_buffer; }
    value_type const *data() const { return m_buffer; }

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

            std::swap(m_num_elts, other.m_num_elts);
            std::swap(m_buffer,   other.m_buffer);
        }
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
    /// @todo Note this function did not depend on numel.  REMOVED
    void quantized_init()
    {
        float max = 1.0;
        float min = -1.0;
        b = (sizeof(value_type) * 8);
        uint64_t max_q = (1 << b) - 1;
        int min_q = 0;
        double dscale = (max - min) / ((double)(max_q - min_q)) + 1e-17;
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

        multiplier = static_cast<int32_t>(q_fixed);  // quantized_multiplier

        m_zero = rint((double)(max * min_q - min * max_q) /
                      ((double)(max - min)));

        scale = dscale;  /// @todo Was narrowing conversion intended?
        lshift = shift > 0 ? shift : 0;
        rshift = shift > 0 ? 0 : -shift;
        min_val = 0;
        max_val = 255;
        offset = 0;
    }

    size_t      m_num_elts;
    value_type *m_buffer;
};

//**********************************************************************
// "dynamic" allocation of Buffer from static buffer (placement new)
/// @todo return smart pointer?
/// @todo Consider using static member factory method and hide ctor's
/// @todo Can this be done better with a CPO or explicit specialization?
///       I.e., define unimplemented alloc_buffer here and specialize in
///       the various Buffer header files.
///
template <class BufferT>
inline BufferT *alloc_buffer(size_t num_elts)
{
    BufferT::unimplemented_function();
}

template<>
inline QUInt8Buffer *alloc_buffer<QUInt8Buffer>(size_t num_elts)
{
    void *location = detail::alloc(sizeof(QUInt8Buffer), 8);
    QUInt8Buffer *buffer = new (location) QUInt8Buffer(num_elts);
    fprintf(stderr, "Allocated QUInt8Buffer(%ld) at address %p\n",
            num_elts, buffer);
    return buffer;
}

inline void free_buffer(QUInt8Buffer *)
{
    // memory mgmt does not exist on this platform
}

}  // small
