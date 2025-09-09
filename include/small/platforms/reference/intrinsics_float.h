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

#include <FloatBuffer.hpp>
#include <cmath>

// scalar versions of all the float microkernels for platform portability
// Use the FLOAT_ prefix for all macros in this file.

#define FLOAT_SIMD_EPILOGUE 1

namespace small
{
    namespace float_detail
    {

        /// @todo both pairs of typedefs should not be needed.
        typedef small::FloatBuffer::value_type dtype;

        typedef small::FloatBuffer::value_type c_tile_t;

    }
}

// The macros in this file are intended to be used in a shared scope.
//****************************************************************************
//****************************************************************************
// Kernel Structure
//****************************************************************************
//****************************************************************************
// Steady State Kernel
//****************************************************************************
// Each kernel will
// 1) define a tile (size known at run-time) of C,
// 2) initialize it,
// 3) perform some computation in a loop,
// 4) and (optionally) store the result.

// The macros used in this kernel have the _TILE_C suffix.
// Pseduocode below
/*
    template <W_ob, C_ob, step>
    kernel(a, b, c)
    {
        DEF_TILE_C(W_ob, C_ob)

        if(ZERO)
        {
            ZERO_TILE_C(W_ob, C_ob)
        } else {
            LOAD_TILE_C(a, step, W_ob, C_ob)
        }

        computation loop
        {
            <COMPUTE>_TILE_C(step, a, b, W_ob, C_ob)
        }

        STORE_TILE_C(c, W_ob, C_ob)
    }

*/
//****************************************************************************
// Edge-case kernel
//****************************************************************************
// Each kernel will
// 1) define a tile (size known at run-time) of C,
// 2) initialize it,
// 3) perform some computation in a loop,
// 4) and (optionally) store the result.


// Macros have the _END_C suffix.
// Pseduocode below
/*
    template <step>
    kernel_end(a, b, c, W_ob, C_ob)
    {
        DEF_END_C(W_ob, C_ob)

        if(ZERO)
        {
            ZERO_END_C(W_ob, C_ob)
        }
        else
        {
            LOAD_END_C(a, step, W_ob, C_ob)
        }

        computation loop
        {
            <COMPUTE>_END_C(step, a, b, W_ob, C_ob)
        }

        STORE_END_C(c, W_ob, C_ob)
    }

*/

//****************************************************************************
// Initializations
//****************************************************************************

/**
 * @brief Macro to define a tile of C matrix with given dimensions.
 *
 * @param W_ob The width of the tile.              // Constant at compile time
 * @param C_ob The number of channels in the tile. // Constant at compile time
 */
#define FLOAT_DEF_TILE_C(W_ob, C_ob) \
    c_tile_t c_tile[W_ob * C_ob];

/**
 * @brief Macro to define a tile of C matrix with given dimensions.
 *
 * @param W_ob The width of the tile.             // Variable, determined at runtime
 * @param C_ob The number of channels in the tile.// Variable, determined at runtime
 */
#define FLOAT_DEF_END_C(W_ob, C_ob) \
    c_tile_t c_tile[W_ob * C_ob];

/**
 * @brief Macro to zero-initialize a (previously defined) tile of C matrix with given dimensions.
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * 
 * form : O (output) -> W_ob x C_ob matrix 
 *  O[i][j] = 0 for i in [0, W_b), for j in [0,C_ob)
 */
#define FLOAT_ZERO_TILE_C(W_ob, C_ob)          \
    for (uint32_t kk = 0; kk < W_ob; kk++)     \
    {                                          \
        for (uint32_t jj = 0; jj < C_ob; jj++) \
        {                                      \
            c_tile[kk * C_ob + jj] = 0.f;      \
        }                                      \
    }

/**x
 * @brief Macro to zero-initialize the end of a tile of C matrix with given dimensions.
 *
 * @param _W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 */
#define FLOAT_ZERO_END_C(W_ob, C_ob)           \
    for (uint32_t kk = 0; kk < W_ob; kk++)     \
    {                                          \
        for (uint32_t jj = 0; jj < C_ob; jj++) \
        {                                      \
            c_tile[kk * C_ob + jj] = 0.f;      \
        }                                      \
    }

//****************************************************************************
// Loads
//****************************************************************************
/**
 * @brief Macro to load a tile of values into a (previously defined) tile of C matrix with given dimensions.
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * 
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x C_ob matrix
 * layout: Elements in I are contiguous, O is in registers
 *  O[i][j] = I[i][j] for i in [0, W_b), for j in [0,C_ob)
 */

#define FLOAT_LOAD_TILE_C(O, W_ob, C_ob)                \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
        }                                               \
    }

//  c_tile_t c_tile[W_ob * C_ob];
#define FLOAT_LOAD_END_C(O, W_ob, C_ob)                 \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = O[kk * C_ob + jj]; \
        }                                               \
    }


//****************************************************************************
// Pooling Loads
//****************************************************************************

// strided loads
/** 
 * @brief Macro to load a tile of values into a (previously defined) tile of C matrix with given dimensions.
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param step The stride in between rows of I, positive integer
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x C_ob matrix
 * layout: rows in I are not contiguous, O is in registers
 * O[i][j] = I[i*step][j] for i in [0, W_b), for j in [0,C_ob)
*/
/** @todo step should be stride? We have an implicit assumption that C_ob is the fastest dimension
The code below expects step = stride*C_ib */
#define FLOAT_LOAD_TILE_C_strided(O, step, W_ob, C_ob)         \
    for (uint32_t kk = 0; kk < W_ob; kk++)                     \
    {                                                          \
        for (uint32_t jj = 0; jj < C_ob; jj++)                 \
        {                                                      \
            c_tile[kk * C_ob + jj] = O[kk * step + jj];        \
        }                                                      \
    }

//  c_tile_t c_tile[W_ob * C_ob];
#define FLOAT_LOAD_END_C_strided(O, step, W_ob, C_ob)   \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = O[kk * step + jj]; \
        }                                               \
    }


//****************************************************************************
// Upsampling loads (stride < 1)
//****************************************************************************
/** 
 * @brief Macro to load a tile of values into a (previously defined) tile of C matrix with given dimensions.
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param stride The stride in between rows of I, positive rational between 0 and 1, passed as an integer
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x C_ob matrix
 * layout: rows in I are not contiguous, O is in registers
 * O[i][j] = I[i*stride][j] for i in [0, W_b), for j in [0,C_ob)
 * implemented as
 * O[i][j] = I[i/(stride)][j] for i in [0, W_b), for j in [0,C_ob)
*/
#define FLOAT_LOAD_TILE_C_upsample(I, stride, _C_ib, W_ob, C_ob)        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            c_tile[kk * C_ob + jj] = I[(kk / stride) * (_C_ib) + jj];   \
        }                                                               \
    }

#define FLOAT_LOAD_END_C_upsample(I, stride, _C_ib, W_ob, C_ob)         \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            c_tile[kk * C_ob + jj] = I[(kk / stride) * (_C_ib) + jj];   \
        }                                                               \
    }


//****************************************************************************
// Missing Loads within the notation
//****************************************************************************
/** 
 * @brief Macro to broadcast a vector of values into a (previously defined) tile of C matrix with given dimensions.
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * layout: Elements in I are contiguous, O is in registers 
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x 1 matrix
 * O[i][j] = I[i] for i in [0, W_b), for j in [0,C_ob)

 OR

  * form : O (output) -> W_ob x C_ob matrix, I (input) -> 1 x C_ob matrix
 * O[i][j] = I[j] for i in [0, W_b), for j in [0,C_ob)
*/

/** 
 * @brief Macro to broadcast a scalar into a (previously defined) tile of C matrix with given dimensions.
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * layout: O is in registers 
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> 1 x 1 matrix
 * O[i][j] = I[0][0] for i in [0, W_b), for j in [0,C_ob)
*/


//****************************************************************************
// Stores
//****************************************************************************


/** 
 * @brief Macro to store a tile of local values to a pointer
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x C_ob matrix
 * layout: I is in registers, O is contiguous
 * O[i][j] = I[i[j] for i in [0, W_b), for j in [0,C_ob)
*/
#define FLOAT_STORE_TILE_C(O, W_ob, C_ob)               \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
        }                                               \
    }

#define FLOAT_STORE_END_C(O, W_ob, C_ob)                \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            O[kk * C_ob + jj] = c_tile[kk * C_ob + jj]; \
        }                                               \
    }
//****************************************************************************
// Missing Stores within notation
//****************************************************************************
/** 
 * @brief Macro to store a tile of local values to a pointer
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param stride The stride in between rows of I, positive integer
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x C_ob matrix
 * layout: I is in registers, O is non contiguous
 * O[i*stride][j] = I[i][j] for i in [0, W_b), for j in [0,C_ob)
*/

//****************************************************************************
// Convolution
//****************************************************************************

/** 
 * @brief Macro to compute a Rank-UNROLL outer product of I and W, accumulate 
          with a local output tile, O
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param stride The stride in between rows of I, positive integer
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x UNROLL matrix, B(weight) -> UNROLL x C_ob matrix
 * layout: O is in registers, I and W are row-major, Rows of I need not be contiguous
 * O[i][j] = I[i*stride][k]* W[k][k] + O[i][j] for i in [0, W_b), for j in [0,C_ob), for k in [0, UNROLL)
*/

/** @todo: UNROLL is 1 for this platform, so it was removed. Add back? */
#define FLOAT_CONV_TILE_C(step, a, b, W_ob, C_ob)       \
    c_tile_t *c_pixel = c_tile;                         \
    c_tile_t const *a_channel = a;                      \
    size_t a_offset = 0;                                \
    /*size_t c_offset = 0;*/                            \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        c_tile_t a_val = *(a_channel);                  \
        c_tile_t * c_channel = c_pixel;                 \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile_t b_val = *(b + jj);                 \
            *(c_channel) += a_val * b_val;              \
            c_channel++;                                \
            /*std::cout << "I_idx=" << a_offset << ", F_idx=" << jj << "--> C_idx=" << c_offset++ << std::endl;*/ \
        }                                               \
        a_channel += step;                              \
        a_offset += step;                               \
        c_pixel += C_ob;                                \
    }

#define FLOAT_CONV_END_C(step, a, b, c_cur, W_ob, C_ob)         \
    c_tile_t *c_pixel = c_cur;                                  \
    c_tile_t const *a_channel = a;                              \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        c_tile_t a_val = *(a_channel);                          \
        c_tile_t * c_channel = c_pixel;                         \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            c_tile_t b_val = *(b + jj);                         \
            *(c_channel) += a_val * b_val;                      \
            c_channel++;                                        \
        }                                                       \
        a_channel += step;                                      \
        c_pixel += C_ob;                                        \
    }


//****************************************************************************
// Missing Convolution
//****************************************************************************


/** 
 * @brief Macro to compute a Rank-UNROLL outer product of I and W, accumulate 
          with a local output tile, O
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param stride The stride in between rows of I, positive rational between 0 < 1, passed as integer
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x UNROLL matrix, B(weight) -> UNROLL x C_ob matrix
 * layout: O is in registers, I and W are row-major, Rows of I need not be contiguous
 * O[i][j] += I[i/stride][k]* W[k][k] for i in [0, W_b), for j in [0,C_ob), for k in [0, UNROLL)
*/


//****************************************************************************
// Pooling
//   Max pooling
//****************************************************************************

/** 
 * @brief Macro to compute a tile of the maximum of the local tile O, and an input I
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param stride The stride in between rows of I, positive integer
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x C_ob matrix
 * layout: O is in registers, I and W are row-major, Rows of I need not be contiguous
 * O[i][j] = max(I[i*stride][j],O[i][j])  for i in [0, W_b), for j in [0,C_ob)
*/

#define FLOAT_MAX_TILE_C(step, a, W_ob, C_ob)                           \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (*(a_channel) > *(c_channel)) ? *(a_channel) : *(c_channel); \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define FLOAT_MAX_END_C(step, a, c_cur, W_last, C_ob)                   \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_last; kk++)                            \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (*(a_channel) > *(c_channel)) ? *(a_channel) : *(c_channel); \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }


/** 
 * @brief Macro to compute a tile of the maximum of the local tile O, and an input I
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param stride The stride in between rows of I, positive integer
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> 1 x 1 matrix
 * layout: O is in registers, I and W are row-major, Rows of I need not be contiguous
 * O[i][j] = max(I[0][0],O[i][j])  for i in [0, W_b), for j in [0,C_ob)
//FUSED MAX

AND

 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x 1 matrix
 * layout: O is in registers, I and W are row-major, Rows of I need not be contiguous
 * O[i][j] = max(I[i*stride][0],O[i][j])  for i in [0, W_b), for j in [0,C_ob)

AND

 * form : O (output) -> W_ob x C_ob matrix, I (input) -> 1 x C_ob matrix
 * layout: O is in registers, I and W are row-major, Rows of I need not be contiguous
 * O[i][j] = max(I[0][C_ob],O[i][j])  for i in [0, W_b), for j in [0,C_ob)

*/



//****************************************************************************
// DW Convolution
//****************************************************************************
/** 
 * @brief Macro to compute the element-wise product of two matrices I and 
          W (rank promoted),
          accumulate with local output tile.
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param stride The stride in between rows of I, positive integer
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x C_ob matrix, W -> 1 x C_ob 
 * layout: O is in registers, I and W are row-major, Rows of I need not be contiguous
 * O[i][j] = (I[i*stride][j] * W[0][j]) + O[i][j]  for i in [0, W_b), for j in [0,C_ob)
*/

#define FLOAT_DW_TILE_C(step, a, b, W_ob, C_ob)                 \
    {                                                           \
        c_tile_t *c_pixel = c_tile;                             \
        c_tile_t const *a_pixel = a;                            \
        for (uint32_t kk = 0; kk < W_ob; kk++)                  \
        {                                                       \
            c_tile_t *c_channel = c_pixel;                      \
            c_tile_t const *a_channel = a_pixel;                \
            c_tile_t const *b_channel = b;                      \
            for (uint32_t jj = 0; jj < C_ob; jj++)              \
            {                                                   \
                *(c_channel) += (*(a_channel) * *(b_channel));  \
                c_channel++;                                    \
                b_channel++;                                    \
                a_channel++;                                    \
            }                                                   \
            a_pixel += step;                                    \
            c_pixel += C_ob;                                    \
        }                                                       \
    }

#define FLOAT_DW_END_C(step, a, b, c_cur, W_ob, C_ob)           \
    {                                                           \
        c_tile_t *c_pixel = c_cur;                              \
        c_tile_t const *a_pixel = a;                            \
        for (uint32_t kk = 0; kk < W_ob; kk++)                  \
        {                                                       \
            c_tile_t *c_channel = c_pixel;                      \
            c_tile_t const *a_channel = a_pixel;                \
            c_tile_t const *b_channel = b;                      \
            for (uint32_t jj = 0; jj < C_ob; jj++)              \
            {                                                   \
                *(c_channel) += (*(a_channel) * *(b_channel));  \
                c_channel++;                                    \
                b_channel++;                                    \
                a_channel++;                                    \
            }                                                   \
            a_pixel += step;                                    \
            c_pixel += C_ob;                                    \
        }                                                       \
    }


/** 
 * @brief Macro to compute the element-wise product of two matrices I and W,
          accumulate with local output tile.
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param stride The stride in between rows of I, positive integer
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x C_ob matrix, W -> W_ob x C_ob 
 * layout: O is in registers, I and W are row-major, Rows of I need not be contiguous
 * O[i][j] = (I[i*stride][j] * W[i][j]) + O[i][j] for i in [0, W_b), for j in [0,C_ob)
*/

//****************************************************************************
// ReLU Activation
//****************************************************************************

// Same kernel as Pooling, set to zero to start.

// When Fused, compare with a register of zeros
/** 
 * @brief Macro to compute a tile of the maximum of the local tile O, and an input I
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param stride The stride in between rows of I, positive integer
 * form : O (output/input) -> W_ob x C_ob matrix, I (input) -> 1 x 1 matrix
 * layout: O is in registers, I and W are row-major, Rows of I need not be contiguous
 * O[i][j] = max(I[0][0],O[i][j])  for i in [0, W_b), for j in [0,C_ob)
 */
#define FLOAT_FUSED_RELU_TILE_C(W_ob, C_ob)                           \
    float *c_pixel = c_tile;                                          \
    for (uint32_t kk = 0; kk < W_ob; kk++)                            \
    {                                                                 \
        float *c_channel = c_pixel;                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                        \
        {                                                             \
            *(c_channel) = (0.0 > *(c_channel)) ? 0.0 : *(c_channel); \
            c_channel++;                                              \
        }                                                             \
        c_pixel += C_ob;                                              \
    }

#define FLOAT_FUSED_RELU_END_C(c_cur, W_last, C_ob)                   \
    float *c_pixel = c_cur;                                           \
    for (uint32_t kk = 0; kk < W_last; kk++)                          \
    {                                                                 \
        float *c_channel = c_pixel;                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                        \
        {                                                             \
            *(c_channel) = (0.0 > *(c_channel)) ? 0.0 : *(c_channel); \
            c_channel++;                                              \
        }                                                             \
        c_pixel += C_ob;                                              \
    }
//****************************************************************************
// Leaky ReLU activation
//****************************************************************************

/** 
 * @brief Macro to compute a tile of a piecewise linear function

    O[i][j] = I[i][j] if I[i][j] > O[i][j]
    O[i][j] = I[i][j]*W[0][0] if I[i][j] <= O[i][j]       
    
    for all i in [0, W_ob), for j in [0,C_ob)

 */

#define FLOAT_COND_SCALE_TILE_C(step, a, b, W_ob, C_ob)                 \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    c_tile_t scale = b[0];                                              \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (*(a_channel) > *(c_channel)) ? *(a_channel) : (*(a_channel) * (scale)); \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define FLOAT_COND_SCALE_END_C(step, a, b, c_cur, W_last, C_ob)         \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t const *a_pixel = a;                                        \
    c_tile_t scale = b[0];                                              \
    for (uint32_t kk = 0; kk < W_last; kk++)                            \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (*(a_channel) > *(c_channel)) ? *(a_channel) : (*(a_channel) * (scale)); \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }


/** 
 * @brief Macro to compute a tile of a piecewise linear function

    O[i][j] = O[i][j] if I[0][0] < O[i][j]
    O[i][j] = O[i][j]*W[0][0] if I[0][0] >= O[i][j]       
    
    for all i in [0, W_ob), for j in [0,C_ob)

 */



#define FLOAT_FUSED_COND_SCALE_TILE_C(b, W_ob, C_ob)                                       \
    float *c_pixel = c_tile;                                                               \
    float scale = b[0];                                                                    \
    for (uint32_t kk = 0; kk < W_ob; kk++)                                                 \
    {                                                                                      \
        float *c_channel = c_pixel;                                                        \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                             \
        {                                                                                  \
            *(c_channel) = (0.0 > *(c_channel)) ? (*(c_channel) * (scale)) : *(c_channel); \
            c_channel++;                                                                   \
        }                                                                                  \
        c_pixel += C_ob;                                                                   \
    }

#define FLOAT_FUSED_COND_SCALE_END_C(b, c_cur, W_last, C_ob)                               \
    float *c_pixel = c_cur;                                                                \
    float scale = b[0];                                                                    \
    for (uint32_t kk = 0; kk < W_last; kk++)                                               \
    {                                                                                      \
        float *c_channel = c_pixel;                                                        \
        for (uint32_t jj = 0; jj < C_ob; jj++)                                             \
        {                                                                                  \
            *(c_channel) = (0.0 > *(c_channel)) ? (*(c_channel) * (scale)) : *(c_channel); \
            c_channel++;                                                                   \
        }                                                                                  \
        c_pixel += C_ob;                                                                   \
    }

//****************************************************************************
// Accumulation kernels
//****************************************************************************
/** 
 * @brief Macro to accumulate a tile of an input I, onto the local tile O
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param stride The stride in between rows of I, positive integer
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x C_ob matrix
 * layout: O is in registers, I and W are row-major, Rows of I need not be contiguous
 * O[i][j] = I[i*stride][j] + O[i][j] for i in [0, W_b), for j in [0,C_ob)
 */
#define FLOAT_ACCUM_TILE_C(step, a, W_ob, C_ob) \
    float *c_pixel = c_tile;                    \
    float const *a_pixel = a;                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)      \
    {                                           \
        float *c_channel = c_pixel;             \
        float const *a_channel = a_pixel;       \
        for (uint32_t jj = 0; jj < C_ob; jj++)  \
        {                                       \
            *(c_channel) += *(a_channel);       \
            c_channel++;                        \
            a_channel++;                        \
        }                                       \
        a_pixel += step;                        \
        c_pixel += C_ob;                        \
    }

#define FLOAT_ACCUM_END_C(step, a, c_cur, W_last, C_ob) \
    float const * a_in_channel = a;                     \
    for(uint32_t u =0 ; u < _UNROLL; u++)               \
    {                                                   \
        float *c_pixel = c_cur;                         \
        float const *a_pixel = a_in_channel;            \
        for (uint32_t kk = 0; kk < W_last; kk++)        \
        {                                               \
            float *c_channel = c_pixel;                 \
            float const *a_channel = a_pixel;           \
            for (uint32_t jj = 0; jj < C_ob; jj++)      \
            {                                           \
                *(c_channel) += *(a_channel);           \
                c_channel++;                            \
                a_channel++;                            \
            }                                           \
            a_pixel += step;                            \
            c_pixel += C_ob;                            \
        }                                               \
        a_in_channel++;                                 \
    }

//****************************************************************************
// Broadcast multiplication kernels
//****************************************************************************
/** 
 * @brief Macro to scale every element of a tile by a constant value
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> 0 x 0 matrix
 * layout: O is in registers
 * O[i][j] = O[i][j] * I[0][0]  for i in [0, W_b), for j in [0,C_ob)
 */
#define FLOAT_DIV_TILE_C(norm, W_ob, C_ob)     \
    float *c_pixel = c_tile;                   \
    for (uint32_t kk = 0; kk < W_ob; kk++)     \
    {                                          \
        float *c_channel = c_pixel;            \
        for (uint32_t jj = 0; jj < C_ob; jj++) \
        {                                      \
            *(c_channel) *= norm;              \
            c_channel++;                       \
        }                                      \
        c_pixel += C_ob;                       \
    }

#define FLOAT_DIV_END_C(c_cur, norm, W_last, C_ob)    \
    float *c_pixel = c_cur;                           \
    for (uint32_t kk = 0; kk < W_last; kk++)          \
    {                                                 \
        float *c_channel = c_pixel;                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)        \
        {                                             \
            *(c_channel) *= norm;                     \
            c_channel++;                              \
        }                                             \
        c_pixel += C_ob;                              \
    }

//****************************************************************************
// Accumulate upsampling
//****************************************************************************

/** 
 * @brief Macro to accumulate a tile of an input I, onto the local tile O
 *
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param stride The stride in between rows of I, positive rational between 0 and 1
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x C_ob matrix
 * layout: O is in registers, I and W are row-major, Rows of I need not be contiguous
 * O[i][j] = I[i*stride][j] + O[i][j] for i in [0, W_b), for j in [0,C_ob)
 * implemented as
  * O[i][j] = I[i/stride][j] + O[i][j] for i in [0, W_b), for j in [0,C_ob)
 */

#define FLOAT_ACCUM_TILE_C_upsample(I, stride, _C_ib, _W_ob, C_ob)     \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                            \
    {                                                                  \
        for (uint32_t jj = 0; jj < C_ob; jj++)                         \
        {                                                              \
            c_tile[kk * C_ob + jj] += I[(kk / stride) * (_C_ib) + jj]; \
        }                                                              \
    }

#define FLOAT_ACCUM_END_C_upsample(I, stride, _C_ib, _W_ob, C_ob)      \
    for (uint32_t kk = 0; kk < _W_ob; kk++)                            \
    {                                                                  \
        for (uint32_t jj = 0; jj < C_ob; jj++)                         \
        {                                                              \
            c_tile[kk * C_ob + jj] += I[(kk / stride) * (_C_ib) + jj]; \
        }                                                              \
    }

//****************************************************************************
// Accumulate channel dimension
//****************************************************************************


/** 
 * @brief Macro to accumulate all elements of the local tile O
 *
 * @param O_w_left The width of the tile.
 * @param C_ob The number of channels in the tile.
 * form : O (output) -> 1 x 1 matrix, O(input) -> 1 x C_ob matrix
 * layout: O is in registers
 * O[0] = O[0] + O[j] for j in [1,C_ob)
 * and
 * O[j] = 0 for j in [1,C_ob)
 */

#define FLOAT_REDUCE_CHANNEL_END_C(O_w_left, C_ob)                      \
    if constexpr (C_ob == 1 && C_ob != FLOAT_SIMD_EPILOGUE)             \
    {                                                                   \
        float c_tile_array[FLOAT_C_ob];                                 \
        for (uint32_t kk = 0; kk < O_w_left; kk++)                      \
        {                                                               \
            float *c_channel_v = c_tile + kk * (FLOAT_C_ob);            \
            for (uint32_t jj = 1; jj < FLOAT_C_ob; jj++)                \
            {                                                           \
                c_channel_v[0] += c_channel_v[jj];                      \
                c_channel_v[jj] = 0;                                    \
            }                                                           \
        }                                                               \
    }

//****************************************************************************
// Reduce kernels??
//****************************************************************************

#define FLOAT_REDUCE_div_C(O, d, W_ob_g, C_ob)          \
    {                                                   \
        c_tile_t *c_pixel = c_tile;                     \
        c_tile_t *O_channel = O;                        \
        c_tile_t *c_channel = c_pixel;                  \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)        \
        {                                               \
            c_tile_t *O_channel = O;                    \
            c_tile_t *c_channel = c_pixel;              \
            for (uint32_t kk = 0; kk < C_ob; kk++)      \
            {                                           \
                *O_channel += *c_channel;               \
                O_channel++;                            \
                c_channel++;                            \
            }                                           \
            c_pixel += C_ob;                            \
        }                                               \
        O_channel = O;                                  \
        for (uint32_t kk = 0; kk < C_ob; kk++)          \
        {                                               \
            *O_channel *= d;                            \
            O_channel++;                                \
        }                                               \
    }

#define FLOAT_REDUCE_C(O, W_ob_g, C_ob)                 \
    {                                                   \
        c_tile_t *c_pixel = c_tile;                     \
        c_tile_t *O_channel = O;                        \
        c_tile_t *c_channel = c_pixel;                  \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)        \
        {                                               \
            c_tile_t *O_channel = O;                    \
            c_tile_t *c_channel = c_pixel;              \
            for (uint32_t kk = 0; kk < C_ob; kk++)      \
            {                                           \
                *O_channel += *c_channel;               \
                O_channel++;                            \
                c_channel++;                            \
            }                                           \
            c_pixel += C_ob;                            \
        }                                               \
    }

#define FLOAT_REDUCE_C_last(O, W_last, C_ob)            \
    {                                                   \
        c_tile_t *c_pixel = c_tile;                     \
        c_tile_t *O_channel = O;                        \
        c_tile_t *c_channel = c_pixel;                  \
        for (uint32_t mm = 0; mm < W_ob_g; mm++)        \
        {                                               \
            c_tile_t *O_channel = O;                    \
            c_tile_t *c_channel = c_pixel;              \
            for (uint32_t kk = 0; kk < C_ob; kk++)      \
            {                                           \
                *O_channel += *c_channel;               \
                O_channel++;                            \
                c_channel++;                            \
            }                                           \
            c_pixel += C_ob;                            \
        }                                               \
    }

//****************************************************************************
// Softmax  (Ewise exponentiation)
//****************************************************************************

/** 
 * @brief Macro to compute a tile of exponentiated values in I
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param stride The stride in between rows of I, positive integer
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x C_ob matrix
 * layout: O is in registers, I and W are row-major, Rows of I need not be contiguous
 * O[i][j] = e ^(I[i*stride][j]) for i in [0, W_b), for j in [0,C_ob)
*/

#define FLOAT_EXP_TILE_C(step, a, W_ob, C_ob)                           \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *a_pixel = a;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *a_channel = a_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = std::exp(*a_channel);                        \
            c_channel++;                                                \
            a_channel++;                                                \
        }                                                               \
        a_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

#define FLOAT_EXP_END_C(step, a, c_cur, W_last, C_ob) \
    c_tile_t *c_pixel = c_cur;                        \
    c_tile_t const *a_pixel = a;                      \
    for (uint32_t kk = 0; kk < W_last; kk++)          \
    {                                                 \
        c_tile_t *c_channel = c_pixel;                \
        c_tile_t const *a_channel = a_pixel;          \
        for (uint32_t jj = 0; jj < C_ob; jj++)        \
        {                                             \
            *(c_channel) = std::exp(*a_channel);      \
            c_channel++;                              \
            a_channel++;                              \
        }                                             \
        a_pixel += step;                              \
        c_pixel += C_ob;                              \
    }


/** 
 * @brief Macro to compute a tile of exponentiated values in the local tile O
 * @param W_ob The width of the tile.
 * @param C_ob The number of channels in the tile.
 * @param stride The stride in between rows of I, positive integer
 * form : O (output, input) -> W_ob x C_ob matrix
 * layout: O is in registers
 * O[i][j] = e ^(O[i][j]) for i in [0, W_b), for j in [0,C_ob)
*/    
#define FLOAT_FUSED_EXP_TILE_C(W_ob, C_ob)       \
    c_tile_t *c_pixel = c_tile;                  \
    for (uint32_t kk = 0; kk < W_ob; kk++)       \
    {                                            \
        c_tile_t *c_channel = c_pixel;           \
        for (uint32_t jj = 0; jj < C_ob; jj++)   \
        {                                        \
            *(c_channel) = std::exp(*c_channel); \
            c_channel++;                         \
        }                                        \
        c_pixel += C_ob;                         \
    }

#define FLOAT_FUSED_EXP_END_C(c_cur, W_last, C_ob) \
    c_tile_t *c_pixel = c_cur;                     \
    for (uint32_t kk = 0; kk < W_last; kk++)       \
    {                                              \
        c_tile_t *c_channel = c_pixel;             \
        for (uint32_t jj = 0; jj < C_ob; jj++)     \
        {                                          \
            *(c_channel) = std::exp(*c_channel);   \
            c_channel++;                           \
        }                                          \
        c_pixel += C_ob;                           \
    }
