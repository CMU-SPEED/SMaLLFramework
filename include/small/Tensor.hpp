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

#include <array>
#include <ostream>
#include <initializer_list>

#include <small.h>
#include <small/buffers.hpp>

namespace small
{

//typedef std::array<size_t, 5UL> shape_type; //{N, C, H, W, logical_channels}

enum ShapeFields {
    BATCH   = 0,
    CHANNEL = 1,
    HEIGHT  = 2,
    WIDTH   = 3,
    L_CHAN  = 4
};

// Augmented logical shape: {N, C, H, W, logical_C}
class shape_type
{
public:
    shape_type()
        : m_dims({0, 0, 0, 0, 0}) {}

    shape_type(std::initializer_list<size_t> &&dims)
    {
        if ((dims.size() < 4) || (dims.size() > 5))
        {
            throw std::invalid_argument(
                "shape_type ERROR: wrong size for ctor.");
        }
        if ((dims.size() == 5) &&
            (dims.begin()[4] > dims.begin()[1]))
        {
            throw std::invalid_argument(
                "shape_type ERROR: bad logical channels.");
        }

        std::copy(dims.begin(), dims.end(), m_dims.begin());
        if (dims.size() == 4)
        {
            m_dims[4] = dims.begin()[1];
        }
    }

    // Access elements (example)
    size_t& operator[](size_t index) {
        return m_dims[index];
    }

    const size_t& operator[](size_t index) const {
        return m_dims[index];
    }

    bool operator==(shape_type const &other) const {
        return m_dims == other.m_dims;
    }

    bool operator!=(shape_type const &other) const {
        return !(m_dims == other.m_dims);
    }

    void swap(shape_type &other)
    {
        m_dims.swap(other.m_dims);
    }

    size_t size() const
    {
        return m_dims[0]*m_dims[CHANNEL]*m_dims[2]*m_dims[3];
    }

    size_t logical_size() const
    {
        return m_dims[0]*m_dims[L_CHAN]*m_dims[2]*m_dims[3];
    }
private:
    std::array<size_t, 5UL> m_dims;
};

inline size_t compute_size(shape_type const &shape)
{
    return shape[0]*shape[CHANNEL]*shape[2]*shape[3];
}

inline size_t compute_logical_size(shape_type const &shape)
{
    return shape[0]*shape[L_CHAN]*shape[2]*shape[3];
}
}



std::ostream &operator<<(std::ostream &ostr, small::shape_type const &shape)
{
    ostr << "(" << shape[small::BATCH] << ", "
         << shape[small::CHANNEL] << "/" << shape[small::L_CHAN] << ", "
         << shape[small::HEIGHT] << ", " << shape[small::WIDTH] << ")";
    return ostr;
}

namespace small
{

//****************************************************************************
// Wraps a buffer and defines a set of dimensions of dense data that can be
// stored in the buffer.
//
// The buffer can contain more storage than the dimensions require (not less)
//
// The dimension describe in order:
//   0: the number of batches
//   1: the number of channels
//   2: image/input height
//   3: image/input width
//
template <typename BufferT>
class Tensor
{
public:
    typedef typename BufferT::value_type value_type;

    Tensor() = delete;

    Tensor(size_t capacity)
        : m_shape({capacity, 1, 1, 1, 1}),  /// @todo revisit
          m_buffer(capacity)
    {
    }

    Tensor(shape_type const &shape)
        : m_shape(shape),
          m_buffer(compute_size(m_shape))
    {
        if (compute_size(shape) == 0)
        {
            throw std::invalid_argument("Tensor ctor ERROR: "
                                        "invalid shape.");
        }
    }

    Tensor(shape_type const &shape,
           BufferT const &buffer)
        : m_shape(shape),
          m_buffer()
    {
        if (compute_size(m_shape) > buffer.size())
        {
            throw std::invalid_argument(
                (std::string)"Tensor ctor ERROR: " +
                (std::string)"insufficient buffer size.\n" +
                (std::string) "Expected shape: {" + std::to_string(shape[small::BATCH]) +
                (std::string)", " + std::to_string(shape[small::CHANNEL]) +
                (std::string)", " + std::to_string(shape[small::HEIGHT]) +
                (std::string)", " + std::to_string(shape[small::WIDTH]) +
                (std::string)"}, with size (at least) " + std::to_string(compute_size(m_shape)) +
                (std::string)"\nBut received buffer with size " + std::to_string(buffer.size()));
        }
        m_buffer = buffer;
    }

    Tensor(shape_type const &shape,
           BufferT  &&buffer)
        : m_shape(shape)
    {
        if (compute_size(m_shape) > buffer.size())
        {
            throw std::invalid_argument(
                (std::string)"Tensor ctor ERROR: " +
                (std::string)"insufficient buffer size.\n" +
                (std::string) "Expected shape: {" + std::to_string(shape[small::BATCH]) +
                (std::string)", " + std::to_string(shape[small::CHANNEL]) +
                (std::string)", " + std::to_string(shape[small::HEIGHT]) +
                (std::string)", " + std::to_string(shape[small::WIDTH]) +
                (std::string)"}, with size (at least) " + std::to_string(compute_size(m_shape)) +
                (std::string)"\nBut received buffer with size " + std::to_string(buffer.size()));
        }
        m_buffer = std::move(buffer);
    }

    ~Tensor() {}

    void set_shape(shape_type const &new_shape)
    {
        if (compute_size(new_shape) > m_buffer.size())
        {
            throw std::invalid_argument("Tensor::set_shape() ERROR: "
                                        "insufficient buffer size.");
        }
        m_shape = new_shape;
    }

    shape_type const &shape() const { return m_shape; }
    size_t size() const { return compute_size(m_shape); }
    size_t capacity() const { return m_buffer.size(); }

    BufferT &buffer() { return m_buffer; }
    BufferT const &buffer() const { return m_buffer; }

    inline void swap(Tensor<BufferT> &other) noexcept
    {
        if (this != &other)
        {
            m_shape.swap(other.m_shape);
            m_buffer.swap(other.m_buffer);
        }
    }

private:
    shape_type m_shape;
    BufferT    m_buffer;
};

}
