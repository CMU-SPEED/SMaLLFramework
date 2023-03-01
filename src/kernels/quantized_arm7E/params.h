#pragma once

#include <stdexcept>
#include <stdint.h>

#define W_ob 2
#define C_ob 1
#define SIMD 1
#define UNROLL 1
#define C_ib C_ob

//Potential blocking parameters for packing
#define NUM_FMA 1
#define NUM_MAX 1
#define NUM_LOAD 1
#define NUM_STORE 1


namespace small
{
namespace detail
{
// Adapted from small_MCU/small/quantized/include/utils.h

//****************************************************************************
//Allocation
//****************************************************************************

// #define MAX_BUFF_SIZE 50000000
// #define MAX_BUFF_SIZE 10000000
// #define MAX_BUFF_SIZE 200000
#define MAX_BUFF_SIZE 208200

    uint8_t  memory_buffer[MAX_BUFF_SIZE];
    uint8_t *current_free_ptr = memory_buffer;
    size_t   buf_offset = 0;

    /// @todo This is allocation only memory management.  It cannot manage
    ///       out of order calls to a free method
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

} // detail

// define the Buffer type in the small namespace (qdtype becomes Buffer<dtype>)
/// @todo if Arduino compiler supports concepts using the integer_type concept
///       for ScalarT
template <class ScalarT>
class Buffer
{
public:
    typedef ScalarT value_type;

    Buffer(size_t num_elts) :
        tensor(detail::alloc<ScalarT>(num_elts)),
        m_num_elts(num_elts),
        scale(0.752941),
        offset(0),
        multiplier(1616928864),
        lshift(0),
        rshift(3),
        zero(0),
        min_val(255),
        max_val(0),
        b(8)
    {
        // todo should the buffer be cleared?
    }

    ~Buffer()
    {
        /// @todo free the buffer somehow;
    }

    size_t size() const { return m_num_elts; }

    ScalarT       *data()       { return tensor; }
    ScalarT const *data() const { return tensor; }

    ScalarT       &operator[](size_t index)       { return tensor[index]; }
    ScalarT const &operator[](size_t index) const { return tensor[index]; }

    void swap(Buffer<ScalarT> &other)
    {
        if (this != &other)
        {
            std::swap(tensor,     other.tensor);
            std::swap(m_num_elts, other.m_num_elts);
            std::swap(scale,      other.scale);
            std::swap(offset,     other.offset);
            std::swap(multiplier, other.multiplier);
            std::swap(lshift,     other.lshift);
            std::swap(rshift,     other.rshift);
            std::swap(zero,       other.zero);
            std::swap(min_val,    other.min_val);
            std::swap(max_val,    other.max_val);
            std::swap(b,          other.b);
        }
    }

    ScalarT *tensor;
    size_t   m_num_elts;
    float    scale;
    int32_t  offset;
    int32_t  multiplier;
    int      lshift;
    int      rshift;
    int      zero;
    int      min_val;
    int      max_val;
    uint8_t  b;
};


} // small
