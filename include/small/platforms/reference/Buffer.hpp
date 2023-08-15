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

