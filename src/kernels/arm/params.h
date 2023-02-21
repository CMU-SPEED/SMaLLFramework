#pragma once

#include <stdlib.h> // for posix_memalign
#include <vector>

#include <arm_neon.h>

#define W_ob 6
#define C_ob 16
#define SIMD 4

#define UNROLL 16
#define C_ib C_ob

// not used for kernels, but used in throughput calculation.
#define NUM_FMA 2
#define NUM_MAX 1
#define NUM_LOAD 2
#define NUM_STORE 1


namespace small
{
    //************************************************************************
    // template <typename T>
    //     T* alloc_buffer(size_t num_elements)
    // {
    //     T *buffer;
    //     if (0 != posix_memalign((void**)&buffer, 64, num_elements*sizeof(T)))
    //     {
    //         throw(std::bad_alloc());
    //     }
    //     return buffer;
    // }

    //************************************************************************
    // Override the standard STL allocator to use posix_memalign
    template <typename T, size_t alignment=64UL>
    struct small_alloc : std::allocator<T>
    {
        typedef typename std::allocator<T>::pointer pointer;
        typedef typename std::allocator<T>::size_type size_type;

        template<typename U>
        struct rebind {
            typedef small_alloc<U> other;
        };

        small_alloc() {}

        template<typename U>
        small_alloc(small_alloc<U> const& u) : std::allocator<T>(u) {}

        pointer allocate(size_type num_elements,
                         std::allocator<void>::const_pointer = 0)
        {
            pointer buffer;
            if (0 != posix_memalign((void**)&buffer,
                                    alignment,
                                    num_elements*sizeof(T)))
            {
                throw std::bad_alloc();
            }
            return buffer;
        }

        void deallocate(pointer p, size_type) { std::free(p); }

    };

    //************************************************************************
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

    //************************************************************************
    template <class ScalarT>
    using Buffer = std::vector<ScalarT, small::small_alloc<ScalarT>>;
}
