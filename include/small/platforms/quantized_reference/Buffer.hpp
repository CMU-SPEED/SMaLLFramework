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
    template <class ScalarT>
    ScalarT *alloc(size_t numel)
    {
        /// @todo deal with alignment issues
        size_t bytes_to_alloc = numel * sizeof(ScalarT);
        size_t used_bytes = buf_offset + bytes_to_alloc;

        if (MAX_BUFF_SIZE < used_bytes)
        {
            // Serial.println("out of space\n");
            // Serial.println(used_bytes);
            // Serial.println();

            throw std::bad_alloc();
        }
        else
        {
            ScalarT *ret_ptr =
                reinterpret_cast<ScalarT*>(memory_buffer + buf_offset);
            buf_offset = used_bytes;
            current_free_ptr = memory_buffer + used_bytes;
            return ret_ptr;
        }
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
        zero(0),
        min_val(255),   // std::numeric_limits<value_type>::max()
        max_val(0),     // std::numeric_limits<value_type>::lowest()
        b(8),
        m_num_elts(num_elts),
        m_buffer(detail::alloc<value_type>(num_elts))
    {
        // todo should the buffer be cleared?
    }

    ~QUInt8Buffer() {  /** @todo need to free buffer */ }

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
            std::swap(zero,       other.zero);
            std::swap(min_val,    other.min_val);
            std::swap(max_val,    other.max_val);
            std::swap(b,          other.b);
            std::swap(m_num_elts, other.m_num_elts);
            std::swap(m_buffer,   other.m_buffer);
        }
    }

    /// @todo Should this be part of construction?
    void quantized_init()
    {
        float max = 1.0;
        float min = -1.0;
        b = (sizeof(value_type) * 8);
        uint64_t max_q = (1 << b) - 1;
        int min_q = 0;
        double scale = (max - min) / ((max_q - min_q) * 1.0) + 1e-17;
        int shift;
        const double q = frexp(scale, &shift);
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
        int32_t quantized_multiplier = static_cast<int32_t>(q_fixed);

        int zero = rint((double)(max * min_q - min * max_q) / ((double)(max - min)));
        scale = scale;
        zero = zero;
        lshift = shift > 0 ? shift : 0;
        rshift = shift > 0 ? 0 : -shift;
        multiplier = quantized_multiplier;
        min_val = 255;
        max_val = 0;
    }

    float    scale;
    int32_t  offset;
    int32_t  multiplier;
    int      lshift;
    int      rshift;
    int      zero;
    int      min_val;
    int      max_val;
    uint8_t  b;

private:
    size_t      m_num_elts;
    value_type *m_buffer;
};

} // small

/// @todo Remove this
typedef small::QUInt8Buffer::value_type dtype;
