#include <iostream>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <stdio.h>

#include "Buffer.hpp"

struct qdtype
{
    typedef uint8_t value_type;

    size_t      m_num_elts;
    value_type *m_buffer;
    float    scale;
    int32_t  offset;     // AccumT?
    int32_t  multiplier; // AccumT?
    int      lshift;     // AccumT?
    int      rshift;     // AccumT?
    int      zero;       // AccumT?
    int      min_val;    // AccumT?
    int      max_val;    // AccumT?
    uint8_t  b;
};

//****************************************************************************
int main(int, char**)
{
    using ScalarT = uint8_t;

    std::cout << "sizeof(qdtype) = " << sizeof(qdtype) << std::endl;
    std::cout << "sizeof(QUInt8Buffer) = " << sizeof(small::QUInt8Buffer) << std::endl;

    std::cerr << "Dynamic buffer allocation with 100 bytes.\n";
    small::QUInt8Buffer* b = small::alloc_buffer(100);

    std::cerr << "Performing basic API tests.\n";
    if (b->size() != 100)  std::cerr << "FAILED size() test.\n";
    ScalarT *ptr = b->data();
    (*b)[20] = 99;
    if (*(ptr + 20) != 99)  std::cerr << "FAILED data() test.\n";
    if ((*b)[20] != 99)      std::cerr << "FAILED operator[] test.\n";

    std::vector<small::QUInt8Buffer *> buffer_ptrs;
    try
    {
        for (size_t ix = 0; ix < 5; ++ix)
        {
            std::cerr << "\nTEST: Allocating a 50001 element buffer:\n";
            small::QUInt8Buffer *buf = small::alloc_buffer(50001);
            std::cerr << "small::detail::buf_offset = "
                      << small::detail::buf_offset << std::endl;

            buffer_ptrs.push_back(buf);
        }
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Caught std::bad_alloc." << std::endl;
    }

    small::detail::free_all();
    return 0;
}
