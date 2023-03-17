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
    typedef uint8_t value_type;
    typedef int32_t accum_type;

    QUInt8Buffer(size_t num_elts) :
        scale(0.752941),
        offset(0),
        multiplier(1616928864),
        lshift(0),
        rshift(3),
        m_zero(0),
        min_val(255),   // std::numeric_limits<value_type>::max()
        max_val(0),     // std::numeric_limits<value_type>::lowest()
        b(8),
        m_num_elts(num_elts),
        m_buffer(reinterpret_cast<value_type*>(
                     detail::alloc(num_elts*sizeof(value_type), 1)))
    {
        /// @todo should the buffer be cleared?
        /// @todo call quantized_init() here and make private
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
        min_val = 255;
        max_val = 0;

        /// @todo offset not set
    }

    ~QUInt8Buffer() {  /** @todo need to free buffer */ }

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
    }

    // type traits?
    inline accum_type zero() const { return m_zero; }

private:
    size_t      m_num_elts;
    value_type *m_buffer;

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
};

//**********************************************************************
// "dynamic" allocation of Buffer from static buffer (placement new)
inline QUInt8Buffer *alloc_buffer(size_t num_elts)
{
    void *location = detail::alloc(sizeof(QUInt8Buffer), 8);
    QUInt8Buffer *buffer = new (location) QUInt8Buffer(num_elts);
    fprintf(stderr, "Allocated QUInt8Buffer(%ld) at address %p\n",
            num_elts, buffer);
    return buffer;
}

}  // small

/// @todo Remove this
typedef small::QUInt8Buffer::value_type dtype;
