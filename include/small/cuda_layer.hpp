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

namespace small
{

//============================================================================
#if defined(SMALL_HAS_CUDA_DOUBLE_SUPPORT)
template <>
void ReLUActivation<CudaDoubleBuffer>(int input_channels,
                                      int input_height, int input_width,
                                      CudaDoubleBuffer const &input_buf,
                                      CudaDoubleBuffer       &output_buf)
{
#if defined(RECORD_CALLS)
    std::cout << "ReLUActivation<cuda:double>(chans:" << input_channels
              << ",img:" << input_height << "x" << input_width
              << ",I,O)\n";
#endif

    // if (input_channels % CUDA_DOUBLE_C_ib == 0)
    {
        input_buf.copyToDevice();

        ReLU_ker<<<CUDA_DOUBLE_MAX_THREAD_BLOCKS, CUDA_DOUBLE_MAX_THREADS>>>(
            input_channels*input_height*input_width,
            input_buf.device_data(),
            output_buf.device_data());

        output_buf.copyFromDevice();
        cudaDeviceSynchronize();
    }
    // else
    // {
    //     throw std::invalid_argument(
    //         "ReLUActivation<cuda:double> ERROR: in_channels unsupported.");
    // }
}
#endif

} // small
