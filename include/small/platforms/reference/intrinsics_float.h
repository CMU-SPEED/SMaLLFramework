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
        typedef small::FloatBuffer::value_type c_tile_t;
    }
}

/// @todo Rename all "TILE" macros to "FULL" macros
/// @todo Rename c_tile to f_tile in FULL macros
/// @todo Rename c_tile to e_tile in END macros
/// @todo Replace the 'i' index in pseudocode in docs to 'k' to match kk in code
/// @todo revisit 'FUSED' names: replace FUSED with INPLACE?

// The macros in this file are intended to be used in a shared scope.
//****************************************************************************
//****************************************************************************
// Kernel Structure
//****************************************************************************
//****************************************************************************
// Steady State Kernel
//****************************************************************************
// Each kernel will
// 1) define a full tile (size known at compile-time) of C,
// 2) initialize it,
// 3) perform some computation in a loop,
// 4) and (optionally) store the result.

// The macros used in this kernel have the _TILE_C suffix.
// Pseduocode below
/*
    template <FLOAT_W_ob, FLOAT_C_ob, step>
    kernel(a, b, c)
    {
        DEF_TILE_C;

        if(ZERO)
        {
            ZERO_TILE_C;
        } else {
            LOAD_TILE_C(a, step);
        }

        computation loop
        {
            <COMPUTE>_TILE_C(step, a, b);
        }

        STORE_TILE_C(c);
    }

*/
//****************************************************************************
// Edge-case kernel
//****************************************************************************
// Each kernel will
// 1) define an end tile (size known at run-time) of C,
// 2) initialize it,
// 3) perform some computation in a loop,
// 4) and (optionally) store the result.


// Macros have the _END_C suffix.
// Pseduocode below
/*
    template <step>
    kernel_end(a, b, c, W_ob, C_ob);
    {
        DEF_END_C(W_ob, C_ob);

        if(ZERO)
        {
            ZERO_END_C(W_ob, C_ob);
        }
        else
        {
            LOAD_END_C(a, step, W_ob, C_ob);
        }

        computation loop
        {
            <COMPUTE>_END_C(step, a, b, W_ob, C_ob);
        }

        STORE_END_C(c, W_ob, C_ob);
    }

*/


//****************************************************************************
// Definitions
//****************************************************************************

/**
 * @brief Define the "full tile" of C matrix with dimensions determined by
 *        the platform parameters: FLOAT_W_ob x FLOAT_C_ob
 *
 * @note FLOAT_W_ob is the number of rows in the full tile.
 * @note FLOAT_C_ob is the number of channels in each row of the full tile.
 * @note c_tile is in registers
 */
#define FLOAT_DEF_TILE_C                        \
    c_tile_t c_tile[FLOAT_W_ob * FLOAT_C_ob];


/**
 * @brief Define the "end tile" of C matrix with dimensions: W_ob x C_ob
 *
 * @param[in] W_ob  The number of rows in the end tile.
 * @param[in] C_ob  The number of channels in each row of the end tile.
 *
 * @note C_ob is currently only FLOAT_C_ob but left in for future flexibility
 */
#define FLOAT_DEF_END_C(W_ob, C_ob)             \
    c_tile_t c_tile[W_ob * C_ob];


//****************************************************************************
// Initializations
//****************************************************************************

/**
 * @brief Zero-initialize the full tile
 *
 * foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *      c_tile[i][j] = 0
 */
#define FLOAT_ZERO_TILE_C                            \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)     \
    {                                                \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++) \
        {                                            \
            c_tile[kk * FLOAT_C_ob + jj] = 0.f;      \
        }                                            \
    }

/**
 * @brief Zero-initialize the end tile
 *        with given dimensions: W_ob x C_ob
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *      c_tile[i][j] = 0
 *
 * @param[in] W_ob  The number of rows in the end tile.
 * @param[in] C_ob  The number of channels in each row of the end tile.
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
 * @brief Load a FLOAT_W_ob x FLOAT_C_ob tile of input values into the full tile
 *
 * @param[in] I  Pointer to input buffer containing the contiguous set of
 *               FLOAT_W_ob x FLOAT_C_ob values
 *
 * foreach i in [0, FLOAT_W_ob), j in [0,FLOAT_C_ob)
 *    c_tile[i][j] = I[i][j]
 *
 */
#define FLOAT_LOAD_TILE_C(I)                                        \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                    \
    {                                                               \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                \
        {                                                           \
            c_tile[kk * FLOAT_C_ob + jj] = I[kk * FLOAT_C_ob + jj]; \
        }                                                           \
    }


/**
 * @brief Load a W_ob x C_ob tile of input values into the end tile.
 *
 * @param[in] I     Pointer to input buffer containing the contiguous set of
 *                  W_ob x C_ob values
 * @param[in] W_ob  The number of rows in the end tile.
 * @param[in] C_ob  The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_tile[i][j] = I[i][j]
 */
#define FLOAT_LOAD_END_C(I, W_ob, C_ob)                 \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = I[kk * C_ob + jj]; \
        }                                               \
    }


//****************************************************************************
// Pooling Loads
//****************************************************************************

//****************************************************************************
// strided loads (stride >= 1)
//****************************************************************************

/**
 * @brief Load a strided tile of values into the full tile.
 *
 * @param[in] I     The input buffer containing FLOAT_W_ob*stride contiguous
 *                  rows of FLOAT_C_ob elements each
 * @param[in] step  The offset corresponding to number of elements between
 *                  the beginnings successive strided (non-contiguous)
 *                  rows in I to load. It is a positive integer, where
 *                  step = stride*_C_ib
 *
 * foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] = I[i*stride][j]
 */
#define FLOAT_LOAD_TILE_C_strided(I, step)                           \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                     \
    {                                                                \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                 \
        {                                                            \
            c_tile[kk * FLOAT_C_ob + jj] = I[kk * step + jj];        \
        }                                                            \
    }

/**
 * @brief Load a strided tile of values into the end tile.
 *
 * @param[in] I     The input containining W_ob*stride contiguous rows of
 *                  C_ob elements each
 * @param[in] step  The offset corresponding to the stride between non-
 *                  contiguous rows in I to load, a positive integer, where
 *                  step = stride*_C_ib
 * @param[in] W_ob  The number of rows in the end tile.
 * @param[in] C_ob  The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_tile[i][j] = I[i*stride][j]
 */
#define FLOAT_LOAD_END_C_strided(I, step, W_ob, C_ob)   \
    for (uint32_t kk = 0; kk < W_ob; kk++)              \
    {                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)          \
        {                                               \
            c_tile[kk * C_ob + jj] = I[kk * step + jj]; \
        }                                               \
    }


//****************************************************************************
// Upsampling loads (stride < 1, factor = 1/stride)
//****************************************************************************

/**
 * @brief Load a tile of values into the full tile repeating (upsampling) rows
 *        of the input tile.
 *
 * @param[in] I      The input buffer containing FLOAT_W_ob//factor contiguous
 *                   rows of FLOAT_C_ob values each.
 * @param[in] factor The repeat factor for rows of I, a positive integer
 *
 * foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] = I[i//factor][j]
 */
#define FLOAT_LOAD_TILE_C_upsample(I, factor)                           \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                    \
        {                                                               \
            c_tile[kk * FLOAT_C_ob + jj] = I[(kk / factor) * (FLOAT_C_ob) + jj]; \
        }                                                               \
    }

/**
 * @brief Load a tile of values into the end tile by repeating (upsampling)
 *        rows of the input tile
 *
 * @param[in] I      The input buffer containing W_ob//factor contiguous rows
 *                   of C_ob values each.
 * @param[in] factor The repeat factor for rows of I, positive integer
 * @param[in] W_ob   The number of rows in the end tile.
 * @param[in] C_ob   The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_tile[i][j] = I[i//factor][j]
 */
#define FLOAT_LOAD_END_C_upsample(I, factor, W_ob, C_ob)                \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            c_tile[kk * C_ob + jj] = I[(kk / factor) * (C_ob) + jj];    \
        }                                                               \
    }


//****************************************************************************
// Missing Loads within the notation
//****************************************************************************
/**
 * @todo Broadcast load of a column vector into matrix (rank promotion)
 * @brief Broadcast a column vector of values into the full/end tile of
 *        C matrix with W_ob x C_ob dimensions. (rank promotion)
 *
 * @param[in] I      W_ob x 1 matrix (vector) of contiguous values to load
 * @param[in] W_ob   The number of rows in the end tile.
 * @param[in] C_ob   The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_tile[i][j] = I[i]
 */

/**
 * @todo Broadcast load of a row vector into matrix (rank promotion)
 * @brief Broadcast a row vector of values into the full/end tile of
 *        C matrix with W_ob x C_ob dimensions. (rank promotion)
 *
 * @param[in] I      1 x C_ob matrix (vector) of contiguous values to load
 * @param[in] W_ob   The number of rows in the end tile.
 * @param[in] C_ob   The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_tile[i][j] = I[j]
 */

/**
 * @todo Broadcast load of a scalar into matrix (rank promotion)
 * @brief Broadcast a scalar into the full/end tile of
 *        C matrix with W_ob x C_ob dimensions. (rank promotion)
 *
 * @param[in] I      1 x 1 matrix (scalar) to load
 * @param[in] W_ob   The number of rows in the end tile.
 * @param[in] C_ob   The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_tile[i][j] = I[0][0]
 */


//****************************************************************************
// Stores
//****************************************************************************

/**
 * @brief Store the full tile of values into a FLOAT_W_ob x FLOAT_C_ob output
 *        buffer
 *
 * @param[out] O  Pointer to an output buffer containing FLOAT_W_ob x FLOAT_C_ob
 *                contiguous values.
 *
 * foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    O[i][j] = c_tile[i][j]
 */
#define FLOAT_STORE_TILE_C(O)                                       \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                    \
    {                                                               \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                \
        {                                                           \
            O[kk * FLOAT_C_ob + jj] = c_tile[kk * FLOAT_C_ob + jj]; \
        }                                                           \
    }

/**
 * @brief Store an end tile of values into a W_ob x C_ob output buffer
 *
 * @param[out] O     Pointer to an output buffer containing W_ob x C_ob
 *                   contiguous values.
 * @param[in]  W_ob  The number of rows in the end tile.
 * @param[in]  C_ob  The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    O[i][j] = c_tile[i][j]
 */
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
 * @todo  Store a tile into a strided output buffer
 * @brief Store a full/end tile of full values to strided rows of an
 *        output buffer.
 *
 * @param[out] O      The output buffer containing W_ob*stride contiguous rows
 *                    of C_ob elements each.
 * @param[in]  W_ob   The number of rows in the end tile.
 * @param[in]  C_ob   The number of channels in each row of the end tile.
 * @param[in]  stride The stride between rows to store in O, positive integer
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    O[i*stride][j] = c_tile[i][j]
 */

//****************************************************************************
// Convolution Computation (Strided GEMM)
//****************************************************************************

/**
 * @brief Compute a Rank-UNROLL outer product of an input tensor (I) and
 *        a weight buffer (W), and accumulate into the full tile with
 *        dimensions FLOAT_W_ob x FLOAT_C_ob
 *
 * @param[in]  step  The offset corresponding to the stride between
 *                   non-contiguous rows of I, a positive integer
 *                   where step = stride*_C_ib
 * @param[in]  I     FLOAT_W_ob x UNROLL non-contiguous, row-major,
 *                   tile of inputs
 * @param[in]  W     UNROLL x FLOAT_C_ob row-major tile of weights
 *
 * for k in [0, UNROLL)
 *    foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *       c_tile[i][j] += I[i*stride][k] * W[k][j]
 *
 * @todo: UNROLL is 1 for this platform, so it was removed. Add back?
 */
#define FLOAT_CONV_TILE_C(step, I, W)                         \
    c_tile_t *c_pixel = c_tile;                               \
    c_tile_t const *I_channel = I;                            \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)              \
    {                                                         \
        c_tile_t I_val = *(I_channel);                        \
        c_tile_t *c_channel = c_pixel;                        \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)          \
        {                                                     \
            c_tile_t W_val = *(W + jj);                       \
            *(c_channel) += I_val * W_val;                    \
            c_channel++;                                      \
        }                                                     \
        I_channel += step;                                    \
        c_pixel += FLOAT_C_ob;                                \
    }

/**
 * @brief Compute a Rank-UNROLL outer product of an input tensor (I) and
 *        a weight buffer (W), and accumulate into an the end tile with
 *        dimensions W_ob x C_ob
 *
 * @param[in]  step  The offset corresponding to the stride in between non-
 *                   contiguous rows of I, a positive integer, where
 *                   step = stride*C_ib
 * @param[in]  I     W_ob x UNROLL non-contiguous, row-major, tile of inputs
 * @param[in]  W     UNROLL x C_ob, row-major tile of weights
 * @param[out] c_cur An offset pointer into the end tile (where to start)
 * @param[in]  W_ob  The number of rows in the end tile.
 * @param[in]  C_ob  The number of channels in each row of the end tile.
 *
 * for k in [0, UNROLL)
 *    foreach i in [0, W_ob), j in [0, C_ob)
 *       c_cur[i][j] += I[i*stride][k] * W[k][j]
 *
 * @todo: UNROLL is 1 for this platform, so it was removed. Add back?
 */
#define FLOAT_CONV_END_C(step, I, W, c_cur, W_ob, C_ob)         \
    c_tile_t *c_pixel = c_cur;                                  \
    c_tile_t const *I_channel = I;                              \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        c_tile_t I_val = *(I_channel);                          \
        c_tile_t * c_channel = c_pixel;                         \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            c_tile_t W_val = *(W + jj);                         \
            *(c_channel) += I_val * W_val;                      \
            c_channel++;                                        \
        }                                                       \
        I_channel += step;                                      \
        c_pixel += C_ob;                                        \
    }


//****************************************************************************
// Missing Convolution
//****************************************************************************

/**
 * @todo Missing convolution kernel on strided input
 * @brief Compute a Rank-UNROLL outer product of a strided input tensor (I) and
 *        a weight buffer (W), and accumulate into a full/end tile with
 *        dimensions W_ob x C_ob
 *
 * @param[in]  stride The stride in between rows of I, positive integer
 * @param[in]  step   The offset corresponding to the stride in between non-
 *                    contiguous rows of I, a positive integer, where
 *                    step = stride*C_ib
 * @param[in]  I      W_ob*stride x UNROLL contiguous, row-major, tile of inputs
 * @param[in]  W      UNROLL x C_ob row-major tile of weights
 * @param[out] c_cur  An offset pointer into the end tile (where to start)
 * @param[in]  W_ob   The number of rows in the end tile.
 * @param[in]  C_ob   The number of channels in each row of the end tile.
 *
 * for k in [0, UNROLL)
 *    foreach i in [0, W_ob), j in [0, C_ob),
 *       c_cur[i][j] += I[i*stride][k] * W[k][j]
 *
 * or
 * for k in [0, UNROLL)
 *    foreach i in [0, W_ob), j in [0, C_ob),
 *       c_cur[i][j] += I[i/stride][k] * W[k][j]
 *
 */


//****************************************************************************
// Pooling
//   Max pooling
//****************************************************************************

/**
 * @brief Compute the maximum between the full tile and an input I
 *        whose dimensions are FLOAT_W_ob x FLOAT_C_ob
 *
 * @param[in]  step  The offset corresponding to the stride between
 *                   non-contiguous rows of I; a positive integer where
 *                   step = stride*_C_ib
 * @param[in]  I     stride*FLOAT_W_ob x FLOAT_C_ob, contiguous row-major tile
 *                   of inputs
 *
 * foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] = max(I[i*stride][j], c_tile[i][j])
 */
#define FLOAT_MAX_TILE_C(step, I)                                       \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *I_pixel = I;                                        \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *I_channel = I_pixel;                            \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                    \
        {                                                               \
            *(c_channel) = (*(I_channel) > *(c_channel)) ? *(I_channel) : *(c_channel); \
            c_channel++;                                                \
            I_channel++;                                                \
        }                                                               \
        I_pixel += step;                                                \
        c_pixel += FLOAT_C_ob;                                          \
    }

/**
 * @brief Compute the maximum between an end tile and an input I
 *        whose dimensions are W_ob x C_ob
 *
 * @param[in]  step   The offset corresponding to the stride between
 *                    non-contiguous rows of I; a positive integer where
 *                    step = stride*_C_ib
 * @param[in]  I      stride*W_ob x C_ob, contiguous row-major tile of inputs
 * @param[out] c_cur  An offset pointer into end tile where computation begins
 * @param[in]  W_ob   The number of rows in the end tile.
 * @param[in]  C_ob   The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, FLOAT_C_ob)
 *    c_cur[i][j] = max(I[i*stride][j], c_cur[i][j])
 */
#define FLOAT_MAX_END_C(step, I, c_cur, W_ob, C_ob)                     \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t const *I_pixel = I;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *I_channel = I_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (*(I_channel) > *(c_channel)) ? *(I_channel) : *(c_channel); \
            c_channel++;                                                \
            I_channel++;                                                \
        }                                                               \
        I_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

/**
 * @brief Compute the maximum of the full/end tile and an input scalar, I
 *        (rank promoted).
 *
 * @param[in] I      1x1 input tile (scalar)
 * @param[in] W_ob   The number of rows in the end tile.
 * @param[in] C_ob   The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_tile[i][j] = max(I[0][0], c_tile[i][j])

AND

 * @brief Compute the maximum of the full/end tile and an input col vector, I
 *        (rank promoted).
 *
 * @param[in] I      Input W_ob x 1 tile (vector)
 * @param[in] W_ob   The number of rows in the end tile.
 * @param[in] C_ob   The number of channels in each row of the end tile.
 * @param[in] stride The stride in between rows (elements) of I, positive integer
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_tile[i][j] = max(I[i*stride][0], c_tile[i][j])

AND (TODO)

 * @brief Compute the maximum of the full/end tile and an input row vector, I
 *        (rank promoted).
 *
 * @param[in] I      Input 1 x C_ob matrix (vector)
 * @param[in] W_ob   The number of rows in the end tile.
 * @param[in] C_ob   The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_tile[i][j] = max(I[0][C_ob], c_tile[i][j])
 *
 */

//****************************************************************************
// DW Convolution
//****************************************************************************

/**
 * @brief Compute the element-wise product of an input tensor (I) and
 *        a rank-promoted weight tensor (W), accumulate into the full tile
 *        with dimensions FLOAT_W_ob x FLOAT_C_ob.
 *
 * @param[in] step The offset corresponding to the stride between
 *                 non-contiguous rows of I; a positive integer where
 *                 step = stride*_C_ib
 * @param[in] I    stride*FLOAT_W_ob x FLOAT_C_ob contiguous row-major tile
 *                 of inputs.
 * @param[in] W    1 x FLOAT_C_ob row-major tile of weights
 *
 * foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] += (I[i*stride][j] * W[0][j])
 */
#define FLOAT_DW_TILE_C(step, I, W)                             \
    {                                                           \
        c_tile_t *c_pixel = c_tile;                             \
        c_tile_t const *I_pixel = I;                            \
        for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)            \
        {                                                       \
            c_tile_t *c_channel = c_pixel;                      \
            c_tile_t const *I_channel = I_pixel;                \
            c_tile_t const *W_channel = W;                      \
            for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)        \
            {                                                   \
                *(c_channel) += (*(I_channel) * *(W_channel));  \
                c_channel++;                                    \
                W_channel++;                                    \
                I_channel++;                                    \
            }                                                   \
            I_pixel += step;                                    \
            c_pixel += FLOAT_C_ob;                              \
        }                                                       \
    }

/**
 * @brief Compute the element-wise product of an input tensor (I) and
 *        a rank-promoted weight tensor (W), accumulate into the end tile
 *        with dimensions W_ob x C_ob.
 *
 * @param[in]  step  The offset corresponding to the stride between
 *                   non-contiguous rows of I, a positive integer where
 *                   step = stride*_C_ib
 * @param[in]  I     stride*W_ob x C_ob contiguous, row-major tile of inputs
 * @param[in]  W     1 x C_ob row-major tile of weights
 * @param[out] c_cur Offset pointer into end tile where computation begins
 * @param[in]  W_ob  The number of rows in the end tile.
 * @param[in]  C_ob  The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_cur[i][j] += (I[i*stride][j] * W[0][j])
 */
#define FLOAT_DW_END_C(step, I, W, c_cur, W_ob, C_ob)           \
    {                                                           \
        c_tile_t *c_pixel = c_cur;                              \
        c_tile_t const *I_pixel = I;                            \
        for (uint32_t kk = 0; kk < W_ob; kk++)                  \
        {                                                       \
            c_tile_t *c_channel = c_pixel;                      \
            c_tile_t const *I_channel = I_pixel;                \
            c_tile_t const *W_channel = W;                      \
            for (uint32_t jj = 0; jj < C_ob; jj++)              \
            {                                                   \
                *(c_channel) += (*(I_channel) * *(W_channel));  \
                c_channel++;                                    \
                W_channel++;                                    \
                I_channel++;                                    \
            }                                                   \
            I_pixel += step;                                    \
            c_pixel += C_ob;                                    \
        }                                                       \
    }

/** TODO: MISSING
 * @brief Compute the strided element-wise product of an input tensor (I) and
 *        a rank-promoted weight tensor (W), accumulate into the full/end tile
 *        with dimensions FLOAT_W_ob x FLOAT_C_ob.
 *
 * @param[in] stride  The stride in between rows of I, positive integer
 * @param[in] step    The offset corresponding to the stride between
 *                    non-contiguous rows of I, a positive integer where
 *                    step = stride*_C_ib
 * @param[in] I       stride*W_ob x C_ob contiguous, row-major tile of inputs
 * @param[in] W_ob    The number of rows in the end tile.
 * @param[in] C_ob    The number of channels in each row the end tile.
 * form : O (output) -> W_ob x C_ob matrix, I (input) -> W_ob x C_ob matrix, W -> W_ob x C_ob
 * layout: O is in registers, I and W are row-major, Rows of I need not be contiguous
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    O[i][j] += (I[i*stride][j] * W[i][j])
 */

//****************************************************************************
// ReLU Activation
//****************************************************************************

// Same kernel as Pooling, set to zero to start.
// When Fused (in-place), compare with a register of zeros

/**
 * @brief Compute a ReLU by performing the maximum of the full tile,
 *        whose dimensions are FLOAT_W_ob x FLOAT_C_ob, with a constant zero.
 *
 * foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] = max(0.0, c_tile[i][j])
 *
 */
#define FLOAT_FUSED_RELU_TILE_C                                       \
    float *c_pixel = c_tile;                                          \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                      \
    {                                                                 \
        float *c_channel = c_pixel;                                   \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                  \
        {                                                             \
            *(c_channel) = (0.0 > *(c_channel)) ? 0.0 : *(c_channel); \
            c_channel++;                                              \
        }                                                             \
        c_pixel += FLOAT_C_ob;                                        \
    }

/**
 * @brief Compute a ReLU by performing the maximum of the end tile,
 *        whose dimensions are W_ob x C_ob, with a constant zero.
 *
 * @param[out] c_cur   An offset into the end tile where to begin (necessary?)
 * @param[in]  W_ob    The number of rows in the end tile.
 * @param[in]  C_ob    The number of channels in each row of the tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_tile[i][j] = max(0.0, c_tile[i][j])
 */
#define FLOAT_FUSED_RELU_END_C(c_cur, W_ob, C_ob)                     \
    float *c_pixel = c_cur;                                           \
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

//****************************************************************************
// Leaky ReLU activation
//****************************************************************************

/**
 * @brief Compute a piecewise linear function, by comparing elements of
 *        an input tile (I) with elements of the full tile, both
 *        FLOAT_W_ob x FLOAT_C_ob
 *
 * @param[in]  step  The offset corresponding to the stride between
 *                   non-contiguous rows of I, a positive integer where
 *                   step = stride*_C_ib
 * @param[in]  I     stride*FLOAT_W_ob x UNROLL non-contiguous,
 *                   row-major, tile of inputs
 * @param[in]  W     1x1 array (scalar) holding the scale factor for
 *                   elements in the input that are less than the
 *                   corresponding full tile element.
 *
 * @note full tile holds zeros to compute leaky ReLU.
 *
 * foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob):
 *    c_tile[i][j] = I[i][j]           if I[i][j] >  c_tile[i][j]
 *    c_tile[i][j] = I[i][j]*W[0][0]   if I[i][j] <= c_tile[i][j]
 */
#define FLOAT_COND_SCALE_TILE_C(step, I, W)                             \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *I_pixel = I;                                        \
    c_tile_t scale = W[0];                                              \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *I_channel = I_pixel;                            \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                    \
        {                                                               \
            *(c_channel) = (*(I_channel) > *(c_channel)) ? *(I_channel) : (*(I_channel) * (scale)); \
            c_channel++;                                                \
            I_channel++;                                                \
        }                                                               \
        I_pixel += step;                                                \
        c_pixel += FLOAT_C_ob;                                          \
    }

/**
 * @brief Compute a piecewise linear function, by comparing elements of
 *        an input tile (I) with elements of end tile, both W_ob x C_ob
 *
 * @param[in]  step   The offset corresponding to the stride between
 *                    non-contiguous rows of I, a positive integer where
 *                    step = stride*_C_ib
 * @param[in]  I      stride*W_ob x UNROLL contiguous, row-major,
 *                    tile of inputs
 * @param[in]  W      1x1 array (scalar) holding the scale factor for
 *                    elements in the input that are less than the
 *                    corresponding end tile element.
 * @param[out] c_cur  Offset into end tile where computation begins
 * @param[in]  W_ob   The number of rows in the end tile.
 * @param[in]  C_ob   The number of channels in each row of the end tile.
 *
 * @note The end tile holds zeros to compute leaky ReLU.
 *
 * foreach all i in [0, W_ob), for j in [0, C_ob):
 *    c_tile[i][j] = I[i][j]           if I[i][j] >  c_tile[i][j]
 *    c_tile[i][j] = I[i][j]*W[0][0]   if I[i][j] <= c_tile[i][j]
 *
 */
#define FLOAT_COND_SCALE_END_C(step, I, W, c_cur, W_ob, C_ob)           \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t const *I_pixel = I;                                        \
    c_tile_t scale = W[0];                                              \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *I_channel = I_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (*(I_channel) > *(c_channel)) ? *(I_channel) : (*(I_channel) * (scale)); \
            c_channel++;                                                \
            I_channel++;                                                \
        }                                                               \
        I_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }


/**
 * @brief Compute a piecewise linear function, by comparing elements of
 *        the full tile to zero; full tile is FLOAT_W_ob x FLOAT_C_ob.
 *        Specifically computes a leaky ReLU.
 *
 * @param[in]  W    1x1 array (scalar) holding the scale factor for
 *                  elements less than zero.
 *
 * foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob):
 *    c_tile[i][j] = c_tile[i][j]           if c_tile[i][j] >  0.0
 *    c_tile[i][j] = c_tile[i][j]*W[0][0]   if c_tile[i][j] <= 0.0
 *
 * @note "FUSED" means in-place in c_tile
 */
#define FLOAT_FUSED_COND_SCALE_TILE_C(W)                                \
    float *c_pixel = c_tile;                                            \
    float scale = W[0];                                                 \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        float *c_channel = c_pixel;                                     \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                    \
        {                                                               \
            *(c_channel) = (0.0 > *(c_channel)) ? (*(c_channel) * (scale)) : *(c_channel); \
            c_channel++;                                                \
        }                                                               \
        c_pixel += FLOAT_C_ob;                                          \
    }

/**
 * @brief Compute a piecewise linear function, by comparing elements of
 *        the end tile to zero; end tile is W_ob x C_ob.
 *        Specifically computes a leaky ReLU.
 *
 * @param[in]  W      1x1 array holding the scale
 *                    factor for elements less than zero.
 * @param[out] c_cur  Offset into end tile where computation begins
 * @param[in]  W_ob   The number of rows in the end tile.
 * @param[in]  C_ob   The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob):
 *    c_tile[i][j] = O[i][j]           if c_tile[i][j] >  0.0
 *    c_tile[i][j] = O[i][j]*W[0][0]   if c_tile[i][j] <= 0.0
 */
#define FLOAT_FUSED_COND_SCALE_END_C(W, c_cur, W_ob, C_ob)              \
    float *c_pixel = c_cur;                                             \
    float scale = W[0];                                                 \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        float *c_channel = c_pixel;                                     \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = (0.0 > *(c_channel)) ? (*(c_channel) * (scale)) : *(c_channel); \
            c_channel++;                                                \
        }                                                               \
        c_pixel += C_ob;                                                \
    }

//****************************************************************************
// Accumulation kernels
//****************************************************************************

/**
 * @brief Accumulate a tile of an input I, into the full tile.
 *        Tiles are FLOAT_W_ob x FLOAT_C_ob. Access of rows of I are strided.
 *
 * @param[in]  step  The offset corresponding to the stride between
 *                   non-contiguous rows of I, a positive integer where
 *                   step = stride*_C_ib
 * @param[in]  I     stride*FLOAT_W_ob x FLOAT_C_ob contiguous, row major
 *                   tile of inputs.
 *
 * foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] += I[i*stride][j]
 *
 * @todo Add _UNROLL loop?
 */
#define FLOAT_ACCUM_TILE_C(step, I)                     \
    float *c_pixel = c_tile;                            \
    float const *I_pixel = I;                           \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)        \
    {                                                   \
        float *c_channel = c_pixel;                     \
        float const *I_channel = I_pixel;               \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)    \
        {                                               \
            *(c_channel) += *(I_channel);               \
            c_channel++;                                \
            I_channel++;                                \
        }                                               \
        I_pixel += step;                                \
        c_pixel += FLOAT_C_ob;                          \
    }

/**
 * @brief Macro to accumulate a tile of an input I, onto the full tile O.
 *        Tiles are W_ob x C_ob. Access of rows of I are strided.
 *
 * @param[in]  step   The offset corresponding to the stride between
 *                    non-contiguous rows of I, a positive integer where
 *                    step = stride*_C_ib
 * @param[in]  I      stride*FLOAT_W_ob x FLOAT_C_ob contiguous, row major
 *                    tile of inputs
 * @param[out] c_cur  Offset into end tile where computation begins
 * @param[in]  W_ob   The number of rows in the end tile.
 * @param[in]  C_ob   The number of channels in each row of the end tile.
 *
 * @todo _UNROLL not passed in.
 *
 * foreach i in [0, W_ob), for j in [0, C_ob)
 *    c_tile[i][j] += I[i*stride][j]
 */
#define FLOAT_ACCUM_END_C(step, I, c_cur, W_ob, C_ob)   \
    float const *I_in_channel = I;                      \
    for (uint32_t u = 0 ; u < _UNROLL; u++)             \
    {                                                   \
        float *c_pixel = c_cur;                         \
        float const *I_pixel = I_in_channel;            \
        for (uint32_t kk = 0; kk < W_ob; kk++)          \
        {                                               \
            float *c_channel = c_pixel;                 \
            float const *I_channel = I_pixel;           \
            for (uint32_t jj = 0; jj < C_ob; jj++)      \
            {                                           \
                *(c_channel) += *(I_channel);           \
                c_channel++;                            \
                I_channel++;                            \
            }                                           \
            I_pixel += step;                            \
            c_pixel += C_ob;                            \
        }                                               \
        I_in_channel++;                                 \
    }

//****************************************************************************
// Broadcast multiplication kernels
//****************************************************************************

/**
 * @brief Multiply every element of the full tile by a constant value.
 *        The full tile is FLOAT_W_ob x FLOAT_C_ob.
 *
 * @param[in] scale   scale factor;
 *
 * foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] *= scale
 *
 * @todo Rename FLOAT_INPLACE_MUL_SCALAR_TILE_C
 */
#define FLOAT_DIV_TILE_C(scale)                         \
    float *c_pixel = c_tile;                            \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)        \
    {                                                   \
        float *c_channel = c_pixel;                     \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)    \
        {                                               \
            *(c_channel) *= scale;                      \
            c_channel++;                                \
        }                                               \
        c_pixel += FLOAT_C_ob;                          \
    }

/**
 * @brief Multiply every element of the end tile by a constant value
 *
 * @param[out] c_cur   Offset into end tile where computation begins
 * @param[in]  scale   scale factor;
 * @param[in]  W_ob    The number of rows in the end tile.
 * @param[in]  C_ob    The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_tile[i][j] *= scale
 *
 * @todo Rename FLOAT_INPLACE_MUL_SCALAR_END_C
 */
#define FLOAT_DIV_END_C(c_cur, scale, W_ob, C_ob)     \
    float *c_pixel = c_cur;                           \
    for (uint32_t kk = 0; kk < W_ob; kk++)            \
    {                                                 \
        float *c_channel = c_pixel;                   \
        for (uint32_t jj = 0; jj < C_ob; jj++)        \
        {                                             \
            *(c_channel) *= scale;                    \
            c_channel++;                              \
        }                                             \
        c_pixel += C_ob;                              \
    }

//****************************************************************************
// Broadcast Addition kernels
//****************************************************************************

/**
 * @brief Add a constant scalar to every element of the full tile.
 *        The full tile is FLOAT_W_ob x FLOAT_C_ob.
 *
 * @param[in] scalar   Constant to add.
 *
 * foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] += scalar
 *
 * @todo Rename FLOAT_INPLACE_ADD_SCALAR_TILE_C
 */
#define FLOAT_EWISE_ADD_SCALAR_TILE_C(scalar)           \
    float *c_pixel = c_tile;                            \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)        \
    {                                                   \
        float *c_channel = c_pixel;                     \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)    \
        {                                               \
            *(c_channel) += scalar;                     \
            c_channel++;                                \
        }                                               \
        c_pixel += FLOAT_C_ob;                          \
    }

/**
 * @brief Add a constant scalar to every element of the end tile.
 *        The end tile is W_ob x C_ob.
 *
 * @param[out] c_cur   Offset into end tile where computation begins
 * @param[in]  scalar  Constant to add.
 * @param[in]  W_ob    The number of rows in the end tile.
 * @param[in]  C_ob    The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_tile[i][j] += scalar
 *
 * @todo Rename FLOAT_INPLACE_ADD_SCALAR_END_C
 */
#define FLOAT_EWISE_ADD_SCALAR_END_C(c_cur, scalar, W_ob, C_ob) \
    float *c_pixel = c_cur;                                     \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        float *c_channel = c_pixel;                             \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            *(c_channel) += scalar;                             \
            c_channel++;                                        \
        }                                                       \
        c_pixel += C_ob;                                        \
    }

//****************************************************************************
// Accumulate upsampling
//****************************************************************************

/**
 * @brief Accumulate a tile of an input I, repeating rows factor times, into
 *        the full tile
 *
 * @param[in] I       The input buffer containing FLOAT_W_ob//factor contiguous
 *                    rows of FLOAT_C_ob values each.
 * @param[in] factor  The repeat factor of rows of I, positive integer
 *
 * foreach i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] += I[i/factor][j]
 */
#define FLOAT_ACCUM_TILE_C_upsample(I, factor)                         \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                       \
    {                                                                  \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                   \
        {                                                              \
            c_tile[kk * FLOAT_C_ob + jj] += I[(kk / factor) * (FLOAT_C_ob) + jj]; \
        }                                                              \
    }


/**
 * @brief Accumulate a tile of an input I, repeating rows factor times, into
 *        the end tile
 *
 * @param[in] I       The input buffer containing FLOAT_W_ob//factor contiguous
 *                    rows of FLOAT_C_ob values each.
 * @param[in] factor  The repeat factor of rows of I, positive integer
 * @param[in] W_ob    The number of rows in the end tile.
 * @param[in] C_ob    The number of channels in each row of the end tile.
 *
 * foreach i in [0, W_ob), j in [0, C_ob)
 *    c_tile[i][j] += I[i/factor][j]
 */
#define FLOAT_ACCUM_END_C_upsample(I, factor, W_ob, C_ob)              \
    for (uint32_t kk = 0; kk < W_ob; kk++)                             \
    {                                                                  \
        for (uint32_t jj = 0; jj < C_ob; jj++)                         \
        {                                                              \
            c_tile[kk * C_ob + jj] += I[(kk / factor) * (C_ob) + jj];  \
        }                                                              \
    }

//****************************************************************************
// Accumulate channel dimension
//****************************************************************************

/**
 * @todo Consider adding more documentation about how FLOAT_SIMD_EPILOGUE
 *       affects this kernel.
 *
 * @brief Accumulate all elements along the channel dimension of the full tile
 *        into the first element of each row. Clear the rest.
 *
 * @param[in] O_w_left  The number of rows to accumulate across.
 * @param[in] C_ob      The number of channels elements in the result of
 *                      each row of the full tile.
 *
 * foreach i in [0, O_w_left)
 *    for j in [1, FLOAT_C_ob)
 *       c_tile[i][0] += c_tile[i][j]
 *       c_tile[i][j]  = 0.0
 */
#define FLOAT_REDUCE_CHANNEL_END_C(O_w_left, C_ob)                      \
    if constexpr (C_ob == 1 && C_ob != FLOAT_SIMD_EPILOGUE)             \
    {                                                                   \
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
// Ewise exponentiation (Softmax)
//****************************************************************************

/**
 * @brief Compute the exponentiated value of each element in strided rows of I
 *        and store in full tile.
 *
 * @param[in]  step  The offset corresponding to the stride between non-
 *                   contiguous rows in I, a positive integer, where
 *                   step = stride*_C_ib
 * @param[in]  I     The input containining FLOAT_W_ob*stride contiguous rows
 *                   of FLOAT_C_ob elements each
 *
 * foreach  i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] = exp(I[i*stride][j])
 */
#define FLOAT_EXP_TILE_C(step, I)                                       \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *I_pixel = I;                                        \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *I_channel = I_pixel;                            \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                    \
        {                                                               \
            *(c_channel) = std::exp(*I_channel);                        \
            c_channel++;                                                \
            I_channel++;                                                \
        }                                                               \
        I_pixel += step;                                                \
        c_pixel += FLOAT_C_ob;                                          \
    }

/**
 * @brief Compute the exponentiated value of each element in strided rows of I
 *        and store in end tile.
 *
 * @param[in]  step  The offset corresponding to the stride between non-
 *                   contiguous rows in I, a positive integer, where
 *                   step = stride*_C_ib
 * @param[in]  I     The input containining W_ob*stride contiguous rows
 *                   of C_ob elements each
 * @param[out] c_cur An offset pointer into the end tile (where to start)
 * @param[in]  W_ob  The number of rows in the end tile.
 * @param[in]  C_ob  The number of channels in each row of the end tile.
 *
 * foreach  i in [0, W_ob), j in [0, C_ob)
 *    c_cur[i][j] = exp(I[i*stride][j])
 */
#define FLOAT_EXP_END_C(step, I, c_cur, W_ob, C_ob)   \
    c_tile_t *c_pixel = c_cur;                        \
    c_tile_t const *I_pixel = I;                      \
    for (uint32_t kk = 0; kk < W_ob; kk++)            \
    {                                                 \
        c_tile_t *c_channel = c_pixel;                \
        c_tile_t const *I_channel = I_pixel;          \
        for (uint32_t jj = 0; jj < C_ob; jj++)        \
        {                                             \
            *(c_channel) = std::exp(*I_channel);      \
            c_channel++;                              \
            I_channel++;                              \
        }                                             \
        I_pixel += step;                              \
        c_pixel += C_ob;                              \
    }


/**
 * @brief Compute the exponentiated value of each element in-place in the
 *        full tile.
 *
 * foreach  i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] = exp(c_tile[i][j])
 */
#define FLOAT_FUSED_EXP_TILE_C                          \
    c_tile_t *c_pixel = c_tile;                         \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)        \
    {                                                   \
        c_tile_t *c_channel = c_pixel;                  \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)    \
        {                                               \
            *(c_channel) = std::exp(*c_channel);        \
            c_channel++;                                \
        }                                               \
        c_pixel += FLOAT_C_ob;                          \
    }

/**
 * @brief Compute the exponentiated value of each element in-place in the
 *        end tile.
 *
 * @param[out] c_cur An offset pointer into the end tile (where to start)
 * @param[in]  W_ob  The number of rows in the end tile.
 * @param[in]  C_ob  The number of channels in each row of the end tile.
 *
 * foreach  i in [0, W_ob), j in [0, C_ob)
 *    c_cur[i][j] = exp(c_cur[i][j])
 */
#define FLOAT_FUSED_EXP_END_C(c_cur, W_ob, C_ob)   \
    c_tile_t *c_pixel = c_cur;                     \
    for (uint32_t kk = 0; kk < W_ob; kk++)         \
    {                                              \
        c_tile_t *c_channel = c_pixel;             \
        for (uint32_t jj = 0; jj < C_ob; jj++)     \
        {                                          \
            *(c_channel) = std::exp(*c_channel);   \
            c_channel++;                           \
        }                                          \
        c_pixel += C_ob;                           \
    }

//****************************************************************************
// Ewise logarithm
//****************************************************************************

/**
 * @brief Compute the logarithm of each element in strided rows of I
 *        and store in full tile.
 *
 * @param[in]  step  The offset corresponding to the stride between non-
 *                   contiguous rows in I, a positive integer, where
 *                   step = stride*_C_ib
 * @param[in]  I     The input containining FLOAT_W_ob*stride contiguous rows
 *                   of FLOAT_C_ob elements each
 *
 * foreach  i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] = log(I[i*stride][j])
 */
#define FLOAT_LOG_TILE_C(step, I)                                       \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *I_pixel = I;                                        \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *I_channel = I_pixel;                            \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                    \
        {                                                               \
            *(c_channel) = std::log(*I_channel);                        \
            c_channel++;                                                \
            I_channel++;                                                \
        }                                                               \
        I_pixel += step;                                                \
        c_pixel += FLOAT_C_ob;                                          \
    }

/**
 * @brief Compute the logarithm of each element in strided rows of I
 *        and store in end tile.
 *
 * @param[in]  step  The offset corresponding to the stride between non-
 *                   contiguous rows in I, a positive integer, where
 *                   step = stride*_C_ib
 * @param[in]  I     The input containining W_ob*stride contiguous rows
 *                   of C_ob elements each
 * @param[out] c_cur An offset pointer into the end tile (where to start)
 * @param[in]  W_ob  The number of rows in the end tile.
 * @param[in]  C_ob  The number of channels in each row of the end tile.
 *
 * foreach  i in [0, W_ob), j in [0, C_ob)
 *    c_cur[i][j] = log(I[i*stride][j])
 */
#define FLOAT_LOG_END_C(step, I, c_cur, W_ob, C_ob)   \
    c_tile_t *c_pixel = c_cur;                        \
    c_tile_t const *I_pixel = I;                      \
    for (uint32_t kk = 0; kk < W_ob; kk++)            \
    {                                                 \
        c_tile_t *c_channel = c_pixel;                \
        c_tile_t const *I_channel = I_pixel;          \
        for (uint32_t jj = 0; jj < C_ob; jj++)        \
        {                                             \
            *(c_channel) = std::log(*I_channel);      \
            c_channel++;                              \
            I_channel++;                              \
        }                                             \
        I_pixel += step;                              \
        c_pixel += C_ob;                              \
    }

/**
 * @brief Compute the logarithm of each element in-place in the
 *        full tile.
 *
 * foreach  i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] = log(c_tile[i][j])
 */
#define FLOAT_FUSED_LOG_TILE_C                         \
    c_tile_t *c_pixel = c_tile;                        \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)       \
    {                                                  \
        c_tile_t *c_channel = c_pixel;                 \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)   \
        {                                              \
            *(c_channel) = std::log(*c_channel);       \
            c_channel++;                               \
        }                                              \
        c_pixel += FLOAT_C_ob;                         \
    }

/**
 * @brief Compute the logarithm of each element in-place in the
 *        end tile.
 *
 * @param[out] c_cur An offset pointer into the end tile (where to start)
 * @param[in]  W_ob  The number of rows in the end tile.
 * @param[in]  C_ob  The number of channels in each row of the end tile.
 *
 * foreach  i in [0, W_ob), j in [0, C_ob)
 *    c_cur[i][j] = log(c_cur[i][j])
 */
#define FLOAT_FUSED_LOG_END_C(c_cur, W_ob, C_ob)   \
    c_tile_t *c_pixel = c_cur;                     \
    for (uint32_t kk = 0; kk < W_ob; kk++)         \
    {                                              \
        c_tile_t *c_channel = c_pixel;             \
        for (uint32_t jj = 0; jj < C_ob; jj++)     \
        {                                          \
            *(c_channel) = std::log(*c_channel);   \
            c_channel++;                           \
        }                                          \
        c_pixel += C_ob;                           \
    }

//****************************************************************************
// Ewise Softsign
//****************************************************************************

/**
 * @brief Compute the softsign value of each element in strided rows of I
 *        and store in full tile.
 *
 * @param[in]  step  The offset corresponding to the stride between non-
 *                   contiguous rows in I, a positive integer, where
 *                   step = stride*_C_ib
 * @param[in]  I     The input containining W_ob*stride contiguous rows of
 *                   C_ob elements each
 *
 * foreach  i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] = I[i*stride][j] / (1.0 + | I[i*stride][j] |)
 */
#define FLOAT_SOFTSIGN_TILE_C(step, I)                                  \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *I_pixel = I;                                        \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *I_channel = I_pixel;                            \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                    \
        {                                                               \
            *(c_channel) = *(I_channel)/(1.0f + std::abs(*I_channel));  \
            c_channel++;                                                \
            I_channel++;                                                \
        }                                                               \
        I_pixel += step;                                                \
        c_pixel += FLOAT_C_ob;                                          \
    }

/**
 * @brief Compute the softsign value of each element in strided rows of I
 *        and store in end tile.
 *
 * @param[in]  step  The offset corresponding to the stride between non-
 *                   contiguous rows in I, a positive integer, where
 *                   step = stride*_C_ib
 * @param[in]  I     The input containining W_ob*stride contiguous rows
 *                   of C_ob elements each
 * @param[out] c_cur An offset pointer into the end tile (where to start)
 * @param[in]  W_ob  The number of rows in the end tile.
 * @param[in]  C_ob  The number of channels in each row of the end tile.
 *
 * foreach  i in [0, W_ob), j in [0, C_ob)
 *    c_cur[i][j] = I[i*stride][j] / (1.0 + | I[i*stride][j] |)
 */
#define FLOAT_SOFTSIGN_END_C(step, I, c_cur, W_ob, C_ob)                \
    c_tile_t *c_pixel = c_cur;                                          \
    c_tile_t const *I_pixel = I;                                        \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *I_channel = I_pixel;                            \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = *(I_channel) / (1.0f + std::abs(*I_channel)); \
            c_channel++;                                                \
            I_channel++;                                                \
        }                                                               \
        I_pixel += step;                                                \
        c_pixel += C_ob;                                                \
    }

/**
 * @brief Compute the softsign value of each element in-place in the
 *        full tile.
 *
 * foreach  i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] = c_tile[i*stride][j] / (1.0 + | c_tile[i*stride][j] |)
 */
#define FLOAT_FUSED_SOFTSIGN_TILE_C                                     \
    c_tile_t *c_pixel = c_tile;                                         \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                    \
        {                                                               \
            *(c_channel) = *(c_channel) / (1.0f + std::abs(*(c_channel))); \
            c_channel++;                                                \
        }                                                               \
        c_pixel += FLOAT_C_ob;                                          \
    }

/**
 * @brief Compute the softsign value of each element in-place in the
 *        end tile.
 *
 * @param[out] c_cur An offset pointer into the end tile (where to start)
 * @param[in]  W_ob  The number of rows in the end tile.
 * @param[in]  C_ob  The number of channels in each row of the end tile.
 *
 * foreach  i in [0, W_ob), j in [0, C_ob)
 *    c_cur[i][j] = c_cur[i*stride][j] / (1.0 + | c_cur[i*stride][j] |)
 */
#define FLOAT_FUSED_SOFTSIGN_END_C(c_cur, W_ob, C_ob)                   \
    c_tile_t *c_pixel = c_cur;                                          \
    for (uint32_t kk = 0; kk < W_ob; kk++)                              \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        for (uint32_t jj = 0; jj < C_ob; jj++)                          \
        {                                                               \
            *(c_channel) = *(c_channel) / (1.0f + std::abs(*(c_channel))); \
            c_channel++;                                                \
        }                                                               \
        c_pixel += C_ob;                                                \
    }

//****************************************************************************
// Ewise abs
//****************************************************************************

/**
 * @brief Compute the absolute value of each element in strided rows of I
 *        and store in full tile.
 *
 * @param[in]  step  The offset corresponding to the stride between non-
 *                   contiguous rows in I, a positive integer, where
 *                   step = stride*_C_ib
 * @param[in]  I     The input containining FLOAT_W_ob*stride contiguous rows
 *                   of FLOAT_C_ob elements each
 *
 * foreach  i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] = abs(I[i*stride][j])
 */
#define FLOAT_ABS_TILE_C(step, I)                                       \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *I_pixel = I;                                        \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *I_channel = I_pixel;                            \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                    \
        {                                                               \
            *(c_channel) = std::abs(*I_channel);                        \
            c_channel++;                                                \
            I_channel++;                                                \
        }                                                               \
        I_pixel += step;                                                \
        c_pixel += FLOAT_C_ob;                                          \
    }

/**
 * @brief Compute the absolute value of each element in strided rows of I
 *        and store in end tile.
 *
 * @param[in]  step  The offset corresponding to the stride between non-
 *                   contiguous rows in I, a positive integer, where
 *                   step = stride*_C_ib
 * @param[in]  I     The input containining W_ob*stride contiguous rows
 *                   of C_ob elements each
 * @param[out] c_cur An offset pointer into the end tile (where to start)
 * @param[in]  W_ob  The number of rows in the end tile.
 * @param[in]  C_ob  The number of channels in each row of the end tile.
 *
 * foreach  i in [0, W_ob), j in [0, C_ob)
 *    c_cur[i][j] = abs(I[i*stride][j])
 */
#define FLOAT_ABS_END_C(step, I, c_cur, W_ob, C_ob)             \
    c_tile_t *c_pixel = c_cur;                                  \
    c_tile_t const *I_pixel = I;                                \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        c_tile_t *c_channel = c_pixel;                          \
        c_tile_t const *I_channel = I_pixel;                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            *(c_channel) = std::abs(*I_channel);                \
            c_channel++;                                        \
            I_channel++;                                        \
        }                                                       \
        I_pixel += step;                                        \
        c_pixel += C_ob;                                        \
    }

//****************************************************************************
// Ewise div
//****************************************************************************

/**
 * @brief Compute the absolute value of each element in-place in the
 *        full tile.
 *
 * foreach  i in [0, FLOAT_W_ob), j in [0, FLOAT_C_ob)
 *    c_tile[i][j] = I[i][j]/(c_tile[i][j])
 */
#define FLOAT_FUSED_DIV_TILE_C(step, I)                                 \
    c_tile_t *c_pixel = c_tile;                                         \
    c_tile_t const *I_pixel = I;                                        \
    for (uint32_t kk = 0; kk < FLOAT_W_ob; kk++)                        \
    {                                                                   \
        c_tile_t *c_channel = c_pixel;                                  \
        c_tile_t const *I_channel = I_pixel;                            \
        for (uint32_t jj = 0; jj < FLOAT_C_ob; jj++)                    \
        {                                                               \
            *(c_channel) = *(I_channel) / *(c_channel);                 \
            c_channel++;                                                \
            I_channel++;                                                \
        }                                                               \
        I_pixel += step;                                                \
        c_pixel += FLOAT_C_ob;                                          \
    }

/**
 * @brief Compute the absolute value of each element in-place in the
 *        end tile.
 *
 * @param[out] c_cur An offset pointer into the end tile (where to start)
 * @param[in]  W_ob  The number of rows in the end tile.
 * @param[in]  C_ob  The number of channels in each row of the end tile.
 *
 * foreach  i in [0, W_ob), j in [0, C_ob)
 *    c_cur[i][j] = abs(c_cur[i][j])
 */
#define FLOAT_FUSED_DIV_END_C(step, I, c_cur, W_ob, C_ob)       \
    c_tile_t *c_pixel = c_cur;                                  \
    c_tile_t const *I_pixel = I;                                \
    for (uint32_t kk = 0; kk < W_ob; kk++)                      \
    {                                                           \
        c_tile_t *c_channel = c_pixel;                          \
        c_tile_t const *I_channel = I_pixel;                    \
        for (uint32_t jj = 0; jj < C_ob; jj++)                  \
        {                                                       \
            *(c_channel) = *(I_channel) / *(c_channel);         \
            c_channel++;                                        \
            I_channel++;                                        \
        }                                                       \
        I_pixel += step;                                        \
        c_pixel += C_ob;                                        \
    }
