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

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <small.h>

#if !defined(SMALL_HAS_CUDA_DOUBLE_SUPPORT)
#error CUDA platform not supported.
#endif

//#include <cuda_runtime.h>


//****************************************************************************
int main()
{
    size_t const num_channels = 1;
    size_t const image_height = 512;
    size_t const image_width  = 512;
    size_t num_elts = num_channels * image_height * image_width;
    small::CudaDoubleBuffer in(num_elts);
    small::CudaDoubleBuffer out(num_elts);
    std::vector<double> check(num_elts);

    srand(42);
    double *cur_ptr = in.data();
    for (size_t i = 0; i != num_elts; ++i)
    {
        cur_ptr[i] = 2.0*((double) rand()/RAND_MAX) - 1.0;
        check[i] = (cur_ptr[i] > 0 ? cur_ptr[i] : 0.0);
    }

    small::ReLUActivation(num_channels, image_height, image_width,
                          in, out);

    bool correct = true;
    for (size_t i = 0; i != num_elts; ++i)
    {
        correct &= (fabs(out[i] - check[i]) < 1e-15);
    }

    std::cout << (correct ? "PASSED" : "FAILED") << std::endl;

    return 0;
}
