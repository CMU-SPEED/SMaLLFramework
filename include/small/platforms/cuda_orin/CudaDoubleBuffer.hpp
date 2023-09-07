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
#include <iostream>
#include <cuda_runtime.h>

#include <params.h>

namespace small
{

//****************************************************************************
// Buffer class must be defined by a specific
// platform by defining in params.h of the platform-specific headers.
// Must have:
// - all of the platform specific parameters for this data type
//
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
class CudaDoubleBuffer
{
public:
    // static uint32_t const   W_ob{DOUBLE_W_ob};
    // static uint32_t const   C_ob{DOUBLE_C_ob};
    // static uint32_t const   SIMD{DOUBLE_SIMD};
    // static uint32_t const UNROLL{DOUBLE_UNROLL};
    // static uint32_t const   C_ib{DOUBLE_C_ib};

    // static uint32_t const   NUM_FMA{DOUBLE_NUM_FMA};
    // static uint32_t const   NUM_MAX{DOUBLE_NUM_MAX};
    // static uint32_t const  NUM_LOAD{DOUBLE_NUM_LOAD};
    // static uint32_t const NUM_STORE{DOUBLE_NUM_STORE};

public:
    typedef double value_type;
    typedef double accum_type;

    CudaDoubleBuffer() :
        m_num_elts(0),
        m_buffer(nullptr),
        m_dev_buffer(nullptr)
    {
        //std::cerr << "CudaDoubleBuffer::default_ctor " << (void*)this << std::endl;
    }

    CudaDoubleBuffer(size_t num_elts) : m_num_elts(num_elts)
    {
        if (0 != posix_memalign((void**)&m_buffer,
                                64,
                                num_elts*sizeof(value_type)))
        {
            throw std::bad_alloc();
        }

        if (cudaSuccess != cudaMalloc((void **)&m_dev_buffer,
                                      m_num_elts*sizeof(value_type)))
        {
            throw std::bad_alloc();
        }

        //std::cerr << "CudaDoubleBuffer::ctor " << (void*)this
        //          << ", host data_ptr = " << (void*)m_buffer
        //          << ", device data_ptr = " << (void*)m_dev_buffer
        //          << ", size = " << num_elts << std::endl;
    }

    CudaDoubleBuffer(CudaDoubleBuffer const &other)
        : m_num_elts(other.m_num_elts)
    {
        //std::cerr << "CudaDoubleBuffer copy ctor\n";
        if (0 != posix_memalign((void**)&m_buffer,
                                64,
                                m_num_elts*sizeof(value_type)))
        {
            throw std::bad_alloc();
        }

        if (cudaSuccess != cudaMalloc((void **)&m_dev_buffer,
                                      m_num_elts*sizeof(value_type)))
        {
            throw std::bad_alloc();
        }
        std::copy(other.m_buffer, other.m_buffer + m_num_elts,
                  m_buffer);

        /// @todo defer copy to device until kernel is called.???
    }

    CudaDoubleBuffer(CudaDoubleBuffer&& other) noexcept
        : m_num_elts(0),
          m_buffer(nullptr),
          m_dev_buffer(nullptr)
    {
        //std::cerr << "CudaDoubleBuffer move ctor\n";
        std::swap(m_num_elts,   other.m_num_elts);
        std::swap(m_buffer,     other.m_buffer);
        std::swap(m_dev_buffer, other.m_dev_buffer);
    }

    CudaDoubleBuffer &operator=(CudaDoubleBuffer const &other)
    {
        //std::cerr << "CudaDoubleBuffer copy assignment\n";
        if (this != &other)
        {
            // expensive, but with exception guarantees.
            CudaDoubleBuffer tmp(other);

            std::swap(m_num_elts, tmp.m_num_elts);
            std::swap(m_buffer,   tmp.m_buffer);
        }

        return *this;
    }

    CudaDoubleBuffer &operator=(CudaDoubleBuffer&& other) noexcept
    {
        //std::cerr << "CudaDoubleBuffer move assignment\n";
        if (this != &other)
        {
            m_num_elts = 0;
            free(m_buffer);
            m_buffer = nullptr;
            cudaFree(m_dev_buffer);
            m_dev_buffer = nullptr;
            std::swap(m_num_elts,   other.m_num_elts);
            std::swap(m_buffer,     other.m_buffer);
            std::swap(m_dev_buffer, other.m_dev_buffer);
        }

        return *this;
    }

    ~CudaDoubleBuffer()
    {
        // std::cerr << "CudaDoubleBuffer::dtor " << (void*)this
        //           << ", data_ptr = " << (void*)m_buffer.data()
        //           << ", size = " << m_buffer.size() << std::endl;
        if (m_buffer != nullptr)
        {
            free(m_buffer);
        }
        if (m_dev_buffer != nullptr)
        {
            cudaFree(m_dev_buffer);
        }
    }

    inline size_t size() const { return m_num_elts; }

    inline value_type       *data()       { return m_buffer; }
    inline value_type const *data() const { return m_buffer; }

    inline value_type       &operator[](size_t index)       { return m_buffer[index]; }
    inline value_type const &operator[](size_t index) const { return m_buffer[index]; }

    inline void swap(CudaDoubleBuffer &other)
    {
        if (this != &other)
        {
            std::swap(m_buffer,     other.m_buffer);
            std::swap(m_dev_buffer, other.m_dev_buffer);
            std::swap(m_num_elts,   other.m_num_elts);
        }
    }

    inline value_type       *device_data()       { return m_dev_buffer; }
    inline value_type const *device_data() const { return m_dev_buffer; }
    void copyToDevice() const
    {
        cudaMemcpy(m_dev_buffer, m_buffer, m_num_elts*sizeof(value_type),
                   cudaMemcpyHostToDevice);
    }

    void copyFromDevice()
    {
        cudaMemcpy(m_buffer, m_dev_buffer, m_num_elts*sizeof(value_type),
                   cudaMemcpyDeviceToHost);
    }

    // type traits?
    inline accum_type zero() const { return (accum_type)0; }

private:
    size_t      m_num_elts;
    value_type *m_buffer;
    mutable value_type *m_dev_buffer;
};

} // small
