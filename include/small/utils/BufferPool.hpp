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

#include <vector>
#include <small/Tensor.hpp>

namespace small
{

//****************************************************************************
template <class BufferT>
class BufferPool
{
public:
    BufferPool()
        : m_buffer_size(0)
    {
    }

    BufferPool(size_t buffer_size, size_t num_buffers)
        : m_buffer_size(buffer_size)
    {
        for (size_t ix = 0; ix < num_buffers; ++ix)
        {
            m_pool.push_back(new Tensor<BufferT>(m_buffer_size));
#ifdef BUFFER_DEBUG
            std::cerr << "BufferPool: allocating buffer: size = "
                      << buffer_size << ", ptr = "
                      << (void*) m_pool.back() << std::endl;
#endif
        }
    }

    ~BufferPool()
    {
        for (auto buf : m_pool)
        {
#ifdef BUFFER_DEBUG
            std::cerr << "BufferPool deleteing buffer: ptr = "
                     << (void*)buf << std::endl;
#endif
            delete buf;
        }
    }

    BufferPool<BufferT> &operator=(BufferPool<BufferT>&& other)
    {
        if (this != &other)
        {
            std::swap(m_buffer_size, other.m_buffer_size);
            m_pool.swap(other.m_pool);
        }
        return *this;
    }

    // Passing ownership out
    Tensor<BufferT>* pop_buffer()
    {
        if (!m_pool.empty())
        {
            auto buf = m_pool.back();
            m_pool.pop_back();
#ifdef BUFFER_DEBUG
            std::cerr << "BufferPool: pop from pool, new size = "
                      << m_pool.size()
                      << ", ptr = " << (void*)buf << std::endl;
#endif
            return buf;
        }

        auto *buf = new Tensor<BufferT>(m_buffer_size);
#ifdef BUFFER_DEBUG
        std::cerr << "BufferPool WARNING: pop from empty pool, ptr = "
                  << (void*)buf << std::endl;
#endif
        return buf;
    }

    // Passing ownership in
    void push_buffer(Tensor<BufferT> *buf)
    {
        assert(buf->capacity() == m_buffer_size);

        m_pool.push_back(buf);
#ifdef BUFFER_DEBUG
        std::cerr << "BufferPool: push to pool,  new size = "
                  << m_pool.size()
                  << ", ptr = " << (void*)buf << std::endl;
#endif
    }

private:
    size_t m_buffer_size;
    std::vector<Tensor<BufferT>*> m_pool;
};

}
